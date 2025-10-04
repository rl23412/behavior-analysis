#!/usr/bin/env python3
"""Two-camera anipose calibration helper.

This module provides a simplified calibration routine tailored for sessions with
exactly two cameras. It creates or updates the project's ``config.toml`` with
the required calibration, filtering, and triangulation settings and then runs
``anipose calibrate`` from the project root to produce
``<session>/calibration/calibration.toml``.
"""

import logging
import shutil
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import toml


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

try:
    from custom_calibration import ChArUcoBoard
    from frame_level_optimizer import FrameLevelAnalyzer

    DETECTION_AVAILABLE = True
except ImportError as exc:  # pragma: no cover - optional dependency
    DETECTION_AVAILABLE = False
    logger.warning("ChArUco detection modules unavailable: %s", exc)


class TwoCameraAniposeCalibrator:
    """Run anipose calibration for two-camera sessions."""

    def __init__(self) -> None:
        self.required_cameras = ["1", "2"]

        self._board_config = ChArUcoBoard() if DETECTION_AVAILABLE else None
        self._sample_interval = 3
        self._temp_root: Optional[Path] = None

        # Defaults mirror the values used elsewhere in the pipeline so that
        # later stages (filtering, triangulation, projections) behave
        # consistently.
        self._calibration_defaults: Dict[str, Dict] = {
            "calibration": {
                "board_type": "charuco",
                "board_size": [10, 7],
                "board_marker_bits": 4,
                "board_marker_dict_number": 50,
                "board_marker_length": 18.75,
                "board_square_side_length": 25.0,
                "fisheye": False,
            },
            "triangulation": {
                "triangulate": True,
                "cam_regex": "-cam([0-9]+)",
                "cam_align": "1",
                "ransac": True,
                "optim": True,
            },
            "filter": {
                "enabled": True,
                "type": "viterbi",
                "score_threshold": 0.4,
                "medfilt": 13,
                "offset_threshold": 25,
                "spline": True,
            },
            "filter3d": {
                "enabled": True,
                "medfilt": 7,
                "offset_threshold": 40,
                "n_back_track": 5,
                "score_threshold": 0.3,
                "spline": True,
            },
        }

    def _ensure_config(self, project_dir: Path) -> Path:
        """Create or update project-level ``config.toml`` with pipeline defaults."""

        config_path = project_dir / "config.toml"

        if config_path.exists():
            logger.info("Updating existing config.toml with two-camera defaults")
            config_data = toml.load(config_path)
        else:
            logger.info("Creating config.toml with two-camera defaults")
            config_data: Dict = {}

        # Basic project settings expected by anipose
        config_data.setdefault("nesting", 1)
        config_data.setdefault("video_extension", "mp4")

        # Merge defaults without clobbering any explicit overrides
        for section, defaults in self._calibration_defaults.items():
            section_data = config_data.get(section, {})
            for key, value in defaults.items():
                section_data.setdefault(key, value)
            config_data[section] = section_data

        config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(config_path, "w") as f:
            toml.dump(config_data, f)

        logger.info("Config ready at %s", config_path)
        return config_path

    def _find_calibration_videos(self, calibration_dir: Path) -> Dict[str, Path]:
        """Locate calibration videos for the required camera IDs."""

        calibration_videos: Dict[str, Path] = {}
        for camera_id in self.required_cameras:
            candidates = [
                calibration_dir / f"calib-cam{camera_id}.mp4",
                calibration_dir / f"cam{camera_id}-calib.mp4",
            ]
            video_path = next((p for p in candidates if p.exists()), None)
            if video_path is None:
                logger.error("Missing calibration video for camera %s", camera_id)
            else:
                calibration_videos[camera_id] = video_path
                logger.info("Found calibration video for camera %s: %s", camera_id, video_path.name)

        return calibration_videos

    def _find_synchronized_frames(self, calibration_videos: Dict[str, Path]) -> List[int]:
        """Find synchronized ChArUco frames across both cameras."""

        if not DETECTION_AVAILABLE or self._board_config is None:
            logger.error(
                "ChArUCo detection is unavailable. Install detection dependencies to enable synchronized frame extraction."
            )
            return []

        analyzer = FrameLevelAnalyzer(self._board_config)
        camera_analyses = {}

        for camera_id, video_path in calibration_videos.items():
            logger.info("Analyzing calibration video for camera %s", camera_id)
            analysis = analyzer.analyze_camera_frames(
                video_path,
                camera_id,
                sample_interval=self._sample_interval,
            )
            camera_analyses[camera_id] = analysis
            logger.info(
                "Camera %s: %d good frames detected",
                camera_id,
                len(analysis.good_frames),
            )

        synchronized = analyzer.find_synchronized_frames_for_pairs(camera_analyses)
        sync_frames = synchronized.get(("1", "2"), [])

        if sync_frames:
            logger.info(
                "Found %d synchronized frames for cameras 1 and 2 (range %d-%d)",
                len(sync_frames),
                sync_frames[0],
                sync_frames[-1],
            )
        else:
            logger.error("No synchronized frames found for cameras 1 and 2")

        return sync_frames

    def _extract_frames_to_video(self, source_video: Path, frame_numbers: List[int], output_video: Path) -> None:
        """Extract the specified frames from a video into a new MP4 file."""

        cap = cv2.VideoCapture(str(source_video))
        if not cap.isOpened():
            raise ValueError(f"Cannot open calibration video: {source_video}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        output_video.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(output_video), fourcc, fps, (width, height))

        logger.info("    Writing %d synchronized frames to %s", len(frame_numbers), output_video.name)

        for frame_num in frame_numbers:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ret, frame = cap.read()
            if ret:
                writer.write(frame)

        cap.release()
        writer.release()

    def _prepare_pair_project(
        self,
        session_dir: Path,
        calibration_videos: Dict[str, Path],
        synchronized_frames: List[int],
    ) -> Optional[Path]:
        """Create a temporary two-camera calibration project with synchronized videos."""

        if not synchronized_frames:
            logger.error("Cannot prepare calibration project without synchronized frames")
            return None

        temp_root = Path(
            tempfile.mkdtemp(
                prefix=f"two_cam_calib_{session_dir.name}_",
            )
        )
        self._temp_root = temp_root
        pair_project_dir = temp_root / "session_pair"
        calibration_output_dir = pair_project_dir / "calibration"

        logger.info("Creating temporary two-camera calibration project at %s", pair_project_dir)

        calibration_output_dir.mkdir(parents=True, exist_ok=True)

        for camera_id in self.required_cameras:
            source_video = calibration_videos[camera_id]
            output_video = calibration_output_dir / f"calib-cam{camera_id}.mp4"
            logger.info("  Extracting synchronized clip for camera %s", camera_id)
            self._extract_frames_to_video(source_video, synchronized_frames, output_video)

        # Ensure config exists inside the temporary project
        self._ensure_config(pair_project_dir)

        return pair_project_dir

    def _persist_trimmed_videos(self, session_dir: Path, temp_project_dir: Path) -> None:
        """Copy synchronized calibration clips into the session for future reference."""

        source_dir = temp_project_dir / "calibration"
        destination_dir = session_dir / "calibration" / "synchronized_clips"
        destination_dir.mkdir(parents=True, exist_ok=True)

        for camera_id in self.required_cameras:
            source_video = source_dir / f"calib-cam{camera_id}.mp4"
            if source_video.exists():
                dest_video = destination_dir / f"sync-calib-cam{camera_id}.mp4"
                shutil.copy2(source_video, dest_video)
                logger.info("Copied synchronized clip to %s", dest_video)

    def _run_pair_calibration(self, project_dir: Path) -> Optional[Path]:
        """Run anipose calibrate inside the temporary two-camera project."""

        logger.info("Running anipose calibrate in temporary project %s", project_dir)

        try:
            subprocess.run(["anipose", "calibrate"], cwd=project_dir, check=True)
        except FileNotFoundError:
            logger.error("anipose command not found. Is the dlc-anipose environment activated?")
            return None
        except subprocess.CalledProcessError as exc:
            logger.error("anipose calibrate failed in temp project (exit code %s)", exc.returncode)
            return None

        calibration_file = project_dir / "calibration" / "calibration.toml"
        if not calibration_file.exists():
            logger.error("Calibration output missing from temporary project: %s", calibration_file)
            return None

        logger.info("Temporary calibration file ready: %s", calibration_file)
        return calibration_file

    def _install_calibration_results(self, session_dir: Path, calibration_file: Path) -> None:
        """Copy the calibration results back into the target session."""

        destination_dir = session_dir / "calibration"
        destination_dir.mkdir(exist_ok=True)

        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        destination_file = destination_dir / "calibration.toml"

        if destination_file.exists():
            backup = destination_dir / f"calibration_{timestamp}.bak.toml"
            shutil.copy2(destination_file, backup)
            logger.info("Backed up existing calibration to %s", backup)

        shutil.copy2(calibration_file, destination_file)
        logger.info("Installed new calibration at %s", destination_file)

    def _resolve_project_and_session(self, session_dir: Path) -> Optional[Tuple[Path, Path]]:
        """Resolve provided path into (project_dir, session_dir)."""

        session_dir = session_dir.resolve()

        calibration_dir = session_dir / "calibration"
        if calibration_dir.exists():
            project_dir = session_dir.parent
            logger.info("Using provided session directory: %s", session_dir)
            return project_dir, session_dir

        # Maybe the user passed the project root. Look for a single session underneath.
        if not session_dir.exists():
            logger.error("Session directory not found: %s", session_dir)
            return None

        candidate_sessions = [
            child for child in session_dir.iterdir()
            if child.is_dir() and (child / "calibration").exists()
        ]

        if len(candidate_sessions) == 1:
            resolved_session = candidate_sessions[0]
            logger.info(
                "No calibration folder in %s; defaulting to nested session %s",
                session_dir,
                resolved_session.name,
            )
            return session_dir, resolved_session

        if len(candidate_sessions) > 1:
            logger.error(
                "Multiple candidate session directories found under %s: %s",
                session_dir,
                ", ".join(child.name for child in candidate_sessions),
            )
            logger.error("Please rerun with the explicit session directory path")
            return None

        logger.error(
            "Calibration directory not found at %s or within its subdirectories",
            session_dir,
        )
        return None

    def run_calibration(self, session_dir: Path) -> bool:
        """Execute the calibration routine for the provided session."""

        resolved = self._resolve_project_and_session(session_dir)
        if resolved is None:
            return False

        project_dir, session_dir = resolved

        calibration_dir = session_dir / "calibration"
        if not calibration_dir.exists():
            logger.error("Calibration directory not found: %s", calibration_dir)
            return False

        calibration_videos = self._find_calibration_videos(calibration_dir)
        if len(calibration_videos) != len(self.required_cameras):
            logger.error(
                "Expected calibration videos for cameras %s, found %d",
                ", ".join(self.required_cameras),
                len(calibration_videos),
            )
            return False

        sync_frames = self._find_synchronized_frames(calibration_videos)
        if not sync_frames:
            logger.error("Unable to continue without synchronized frames")
            return False

        temp_project_dir = self._prepare_pair_project(session_dir, calibration_videos, sync_frames)
        if temp_project_dir is None:
            return False

        calibration_file = self._run_pair_calibration(temp_project_dir)
        if calibration_file is None:
            return False

        self._persist_trimmed_videos(session_dir, temp_project_dir)

        # Ensure main project config matches downstream expectations
        config_path = self._ensure_config(project_dir)
        logger.info("Using project config at %s", config_path)

        self._install_calibration_results(session_dir, calibration_file)

        logger.info("Two-camera calibration completed successfully")
        if self._temp_root is not None:
            logger.info("Temporary calibration project kept at %s", self._temp_root)

        return True
