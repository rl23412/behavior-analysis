#!/usr/bin/env python3
"""Three-camera anipose calibration driven by shared ChArUco frames."""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import cv2
try:
    from custom_calibration import ChArUcoBoard
    from frame_level_optimizer import FrameLevelAnalyzer
    DETECTION_AVAILABLE = True
except ImportError as exc:  # pragma: no cover - optional dependency
    DETECTION_AVAILABLE = False
    logging.getLogger(__name__).warning("ChArUco detection unavailable: %s", exc)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class ThreeCameraAniposeCalibrator:
    """Run anipose calibration after extracting frames shared by all three cameras."""

    def __init__(self, temp_base_dir: Optional[Path] = None, sample_interval: int = 1) -> None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.temp_base_dir = Path(temp_base_dir) if temp_base_dir else Path("/tmp") / f"three_cam_calib_{timestamp}"
        self.sample_interval = sample_interval
        self.camera_ids = ["1", "2", "3"]
        self.board_config = ChArUcoBoard() if DETECTION_AVAILABLE else None

    def run_joint_calibration(self, session_dir: Path, cleanup: bool = False) -> bool:
        logger.info("🚀 STARTING SHARED-FRAME 3-CAMERA ANIPOSE CALIBRATION")
        logger.info("=" * 70)
        logger.info("Session directory: %s", session_dir)
        logger.info("Temporary directory: %s", self.temp_base_dir)
        if cleanup:
            logger.info("Temporary directory will be removed after calibration")
        else:
            logger.info("Temporary directory will be preserved for inspection")

        try:
            calibration_videos = self._find_calibration_videos(session_dir)
            if len(calibration_videos) < len(self.camera_ids):
                logger.error("Expected calibration videos for cameras %s", ", ".join(self.camera_ids))
                return False

            shared_frames = self._find_shared_frames(calibration_videos)
            if not shared_frames:
                logger.error("No shared ChArUco frames detected across all cameras")
                return False

            project_dir = self._prepare_project(calibration_videos, shared_frames)
            if project_dir is None:
                return False

            calib_file = self._run_anipose_calibrate(project_dir)
            if calib_file is None:
                return False

            self._persist_outputs(calib_file, session_dir, shared_frames)
            logger.info("🎉 Shared-frame calibration completed successfully")
            logger.info("Primary calibration saved to experiment root and session/calibration")
            if not cleanup:
                logger.info("Temporary project preserved at: %s", project_dir)
            return True
        except Exception as exc:  # pragma: no cover - defensive
            logger.exception("Shared-frame calibration failed: %s", exc)
            return False
        finally:
            if cleanup:
                self.cleanup_temp_directory()

    def cleanup_temp_directory(self) -> None:
        if self.temp_base_dir.exists():
            shutil.rmtree(self.temp_base_dir, ignore_errors=True)
            logger.info("Removed temporary directory: %s", self.temp_base_dir)

    def _find_calibration_videos(self, session_dir: Path) -> Dict[str, Path]:
        calibration_dir = session_dir / "calibration"
        if not calibration_dir.exists():
            logger.error("Calibration directory not found: %s", calibration_dir)
            return {}

        videos: Dict[str, Path] = {}
        for camera_id in self.camera_ids:
            candidates = [
                calibration_dir / f"calib-cam{camera_id}.mp4",
                calibration_dir / f"cam{camera_id}-calib.mp4",
            ]
            video_path = next((candidate for candidate in candidates if candidate.exists()), None)
            if video_path is None:
                logger.error("Missing calibration video for camera %s", camera_id)
            else:
                videos[camera_id] = video_path
                logger.info("Found calibration video for camera %s: %s", camera_id, video_path.name)
        return videos

    def _find_shared_frames(self, calibration_videos: Dict[str, Path]) -> List[int]:
        if not DETECTION_AVAILABLE or self.board_config is None:
            logger.error("ChArUco detection modules are required for shared-frame calibration")
            return []

        analyzer = FrameLevelAnalyzer(self.board_config)
        frame_sets: List[set[int]] = []

        for camera_id in self.camera_ids:
            video_path = calibration_videos.get(camera_id)
            if video_path is None:
                logger.error("Cannot analyze camera %s without a calibration video", camera_id)
                return []
            logger.info("Analyzing calibration frames for camera %s", camera_id)
            analysis = analyzer.analyze_camera_frames(
                video_path, camera_id, sample_interval=self.sample_interval
            )
            frame_set = analysis.get_frame_set()
            if not frame_set:
                logger.warning("No good ChArUco detections found for camera %s", camera_id)
            frame_sets.append(frame_set)

        shared_frames = sorted(set.intersection(*frame_sets)) if frame_sets else []
        if shared_frames:
            logger.info(
                "Shared frames detected: %d (range %d-%d)",
                len(shared_frames),
                shared_frames[0],
                shared_frames[-1],
            )
        return shared_frames

    def _prepare_project(
        self,
        calibration_videos: Dict[str, Path],
        shared_frames: List[int],
    ) -> Optional[Path]:
        if not shared_frames:
            logger.error("Cannot prepare project without shared frames")
            return None

        project_dir = self.temp_base_dir / "joint_session"
        if project_dir.exists():
            shutil.rmtree(project_dir, ignore_errors=True)
        project_dir.mkdir(parents=True, exist_ok=True)

        calib_dir = project_dir / "calibration"
        calib_dir.mkdir(parents=True, exist_ok=True)

        for camera_id in self.camera_ids:
            source_video = calibration_videos[camera_id]
            output_video = calib_dir / f"calib-cam{camera_id}.mp4"
            logger.info("Extracting %d frames for camera %s", len(shared_frames), camera_id)
            self._extract_frames_to_video(source_video, shared_frames, output_video)

        self._write_config(project_dir)
        return project_dir

    def _write_config(self, project_dir: Path) -> None:
        config_path = project_dir / "config.toml"
        config_content = """
nesting = 1
video_extension = 'mp4'

[calibration]
board_type = "charuco"
board_size = [10, 7]
board_marker_bits = 4
board_marker_dict_number = 50
board_marker_length = 18.75
board_square_side_length = 25.0
fisheye = false

[triangulation]
triangulate = true
cam_regex = '-cam([0-9]+)'
cam_align = "1"
ransac = false
optim = true

[pipeline]
calibration_videos = "calibration"
calibration_results = "calibration"
videos_raw = "videos-raw"
pose_2d = "pose-2d"
pose_3d = "pose-3d"

[filter]
enabled = false

[filter3d]
enabled = false
"""
        with open(config_path, "w", encoding="utf-8") as buffer:
            buffer.write(config_content.strip())
        logger.info("Config prepared at %s", config_path)

    def _extract_frames_to_video(self, source_video: Path, frame_numbers: List[int], output_video: Path) -> None:
        cap = cv2.VideoCapture(str(source_video))
        if not cap.isOpened():
            raise ValueError(f"Cannot open calibration video: {source_video}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(output_video), fourcc, fps, (width, height))

        for frame_index in frame_numbers:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = cap.read()
            if ok:
                writer.write(frame)

        cap.release()
        writer.release()

    def _run_anipose_calibrate(self, project_dir: Path) -> Optional[Path]:
        try:
            logger.info("Running anipose calibrate in %s", project_dir)
            subprocess.run(["anipose", "calibrate"], cwd=project_dir, check=True)
        except FileNotFoundError:
            logger.error("anipose command not found. Install anipose to continue.")
            return None
        except subprocess.CalledProcessError as exc:
            logger.error("anipose calibrate failed with exit code %s", exc.returncode)
            logger.error("Stdout: %s", exc.stdout)
            logger.error("Stderr: %s", exc.stderr)
            return None

        calib_dir = project_dir / "calibration" / "calibration.toml"
        if calib_dir.exists():
            return calib_dir

        fallback = project_dir / "calibration.toml"
        if fallback.exists():
            return fallback

        logger.error("Calibration file not produced by anipose")
        return None

    def _persist_outputs(
        self,
        calib_file: Path,
        session_dir: Path,
        shared_frames: List[int],
    ) -> None:
        experiment_root = session_dir.parent
        experiment_root.mkdir(parents=True, exist_ok=True)
        primary_output = experiment_root / "calibration.toml"
        shutil.copy2(calib_file, primary_output)
        logger.info("Saved calibration to %s", primary_output)

        session_calib_dir = session_dir / "calibration"
        session_calib_dir.mkdir(parents=True, exist_ok=True)
        session_output = session_calib_dir / "calibration.toml"
        shutil.copy2(calib_file, session_output)
        logger.info("Saved calibration to %s", session_output)

        clip_archive = session_calib_dir / "shared_clips"
        clip_archive.mkdir(exist_ok=True)
        project_clips_dir = self.temp_base_dir / "joint_session" / "calibration"
        for camera_id in self.camera_ids:
            clip_source = project_clips_dir / f"calib-cam{camera_id}.mp4"
            if clip_source.exists():
                clip_dest = clip_archive / f"shared-calib-cam{camera_id}.mp4"
                shutil.copy2(clip_source, clip_dest)

        shared_json = session_calib_dir / "shared_frames.json"
        with open(shared_json, "w", encoding="utf-8") as buffer:
            json.dump(shared_frames, buffer)
        logger.info("Stored shared frame indices at %s", shared_json)
