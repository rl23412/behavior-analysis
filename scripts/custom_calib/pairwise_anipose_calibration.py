#!/usr/bin/env python3
"""
Pairwise Anipose Calibration for 3-Camera System

This script uses anipose's calibration system in a pairwise manner:
1. Divide 3 cameras into pairs: (1,2), (1,3), (2,3) 
2. Find synchronized frames with ChArUco detection for each pair
3. Create temporary anipose project structure for each pair
4. Run anipose calibrate on each pair
5. Combine pairwise calibrations into final 3-camera calibration

Usage:
    python pairwise_anipose_calibration.py --session-dir /path/to/session
    python pairwise_anipose_calibration.py --session-dir /path/to/session --temp-dir /tmp/calib

Author: Pairwise Anipose Calibration
"""

import os
import sys
import cv2
import numpy as np
import toml
import json
import shutil
import subprocess
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import frame detection utilities
try:
    from custom_calibration import ChArUcoBoard, ChArUcoDetector
    from frame_level_optimizer import FrameLevelAnalyzer, CameraFrameAnalysis
    DETECTION_AVAILABLE = True
except ImportError as e:
    DETECTION_AVAILABLE = False
    logger.warning(f"Detection modules not available: {e}")

class PairwiseAniposeCalibrator:
    """Calibrate 3 cameras using pairwise anipose calibration"""
    
    def __init__(self, temp_base_dir: Path = None):
        self.board_config = ChArUcoBoard()
        self.temp_base_dir = temp_base_dir or Path("/tmp") / f"pairwise_calib_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.camera_pairs = [("1", "2"), ("1", "3"), ("2", "3")]
        
    def create_anipose_config_for_pair(self, pair_dir: Path, cam1: str, cam2: str) -> Path:
        """
        Create anipose config.toml for a camera pair
        
        Args:
            pair_dir: Directory for this pair's calibration
            cam1: First camera ID
            cam2: Second camera ID
            
        Returns:
            Path to created config.toml
        """
        config_path = pair_dir / "config.toml"
        
        config_content = f"""
# Anipose project configuration for camera pair ({cam1}, {cam2})
nesting = 1
video_extension = 'mp4'

[calibration]
# ChArUco board settings
board_type = "charuco"
board_size = [10, 7]
board_marker_bits = 4
board_marker_dict_number = 50
board_marker_length = 18.75  # mm
board_square_side_length = 25  # mm
fisheye = false


[triangulation]
# Enable triangulation
triangulate = true

# Camera regex for pair
cam_regex = '-cam([0-9]+)'

# Use first camera as alignment reference
cam_align = "{cam1}"

# Disable RANSAC for deterministic results
ransac = false

# Enable optimization
optim = true

[labeling]
# Simple labeling scheme for calibration
scheme = [
    ["point1", "point2"]
]

[pipeline]
# Pipeline folder structure
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
        
        with open(config_path, 'w') as f:
            f.write(config_content.strip())
        
        logger.info(f"Created anipose config for pair ({cam1}, {cam2}): {config_path}")
        return config_path
    
    def find_synchronized_charuco_frames(self, calibration_videos: Dict[str, Path]) -> Dict[Tuple[str, str], List[int]]:
        """
        Find frames where both cameras in each pair have good ChArUco detection
        
        Args:
            calibration_videos: Dict mapping camera_id to video path
            
        Returns:
            Dict mapping camera pairs to lists of synchronized frame numbers
        """
        logger.info("🔍 FINDING SYNCHRONIZED CHARUCO FRAMES FOR PAIRS")
        logger.info("="*60)
        
        if not DETECTION_AVAILABLE:
            logger.error("Detection modules not available")
            return {}
        
        # Analyze each camera to find good frames
        analyzer = FrameLevelAnalyzer(self.board_config)
        camera_analyses = {}
        
        for camera_id, video_path in calibration_videos.items():
            logger.info(f"Analyzing camera {camera_id} for ChArUco detection...")
            analysis = analyzer.analyze_camera_frames(video_path, camera_id, sample_interval=1)  # Inspect every frame
            camera_analyses[camera_id] = analysis
            
            if analysis.good_frames:
                logger.info(f"✅ Camera {camera_id}: {len(analysis.good_frames)} good frames")
            else:
                logger.warning(f"⚠️ Camera {camera_id}: No good frames found")
        
        # Find synchronized frames for each pair
        synchronized_pairs = {}
        
        for cam1, cam2 in self.camera_pairs:
            if cam1 in camera_analyses and cam2 in camera_analyses:
                logger.info(f"Finding synchronized frames for pair ({cam1}, {cam2})...")
                
                # Get good frame sets for each camera
                cam1_frames = camera_analyses[cam1].get_frame_set()
                cam2_frames = camera_analyses[cam2].get_frame_set()
                
                # Find intersection (synchronized frames where both have good detection)
                sync_frames = sorted(list(cam1_frames.intersection(cam2_frames)))
                
                if sync_frames:
                    synchronized_pairs[(cam1, cam2)] = sync_frames
                    logger.info(f"✅ Pair ({cam1}, {cam2}): {len(sync_frames)} synchronized frames")
                    logger.info(f"   Frame range: {min(sync_frames)}-{max(sync_frames)}")
                else:
                    logger.warning(f"⚠️ Pair ({cam1}, {cam2}): No synchronized frames found")
                    synchronized_pairs[(cam1, cam2)] = []
            else:
                logger.error(f"❌ Missing analysis for pair ({cam1}, {cam2})")
                synchronized_pairs[(cam1, cam2)] = []
        
        return synchronized_pairs
    
    def extract_pair_frames(self, calibration_videos: Dict[str, Path], 
                           pair_frames: Dict[Tuple[str, str], List[int]]) -> Dict[Tuple[str, str], Path]:
        """
        Extract synchronized frames for each camera pair and create anipose structure
        
        Args:
            calibration_videos: Dict mapping camera_id to video path
            pair_frames: Dict mapping pairs to synchronized frame lists
            
        Returns:
            Dict mapping pairs to their session directories
        """
        logger.info("📹 EXTRACTING FRAMES FOR EACH CAMERA PAIR")
        logger.info("="*60)
        
        pair_session_dirs = {}
        
        # Create temporary directory structure
        self.temp_base_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created temporary calibration directory: {self.temp_base_dir}")
        
        # Process each camera pair
        for session_idx, (cam1, cam2) in enumerate(self.camera_pairs, 1):
            if (cam1, cam2) not in pair_frames or not pair_frames[(cam1, cam2)]:
                logger.warning(f"Skipping pair ({cam1}, {cam2}) - no synchronized frames")
                continue
            
            sync_frames = pair_frames[(cam1, cam2)]
            logger.info(f"Processing pair ({cam1}, {cam2}) with {len(sync_frames)} frames...")
            
            # Create session directory for this pair
            session_dir = self.temp_base_dir / f"session{session_idx}"
            session_dir.mkdir(exist_ok=True)
            
            # Create calibration subdirectory
            calib_dir = session_dir / "calibration"
            calib_dir.mkdir(exist_ok=True)
            
            # Extract frames for both cameras in this pair
            for camera_id in [cam1, cam2]:
                if camera_id in calibration_videos:
                    video_path = calibration_videos[camera_id]
                    output_video = calib_dir / f"calib-cam{camera_id}.mp4"
                    
                    logger.info(f"  Extracting frames for camera {camera_id}...")
                    self._extract_frames_to_video(video_path, sync_frames, output_video)
                else:
                    logger.error(f"Video not found for camera {camera_id}")
            
            pair_session_dirs[(cam1, cam2)] = session_dir
            logger.info(f"✅ Created session for pair ({cam1}, {cam2}): {session_dir}")
        
        return pair_session_dirs
    
    def _extract_frames_to_video(self, source_video: Path, frame_numbers: List[int], output_video: Path):
        """
        Extract specific frames from source video and create new video
        
        Args:
            source_video: Source calibration video
            frame_numbers: List of frame numbers to extract
            output_video: Output video path
        """
        cap = cv2.VideoCapture(str(source_video))
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {source_video}")
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_video), fourcc, fps, (width, height))
        
        logger.info(f"    Extracting {len(frame_numbers)} frames to {output_video.name}")
        
        for frame_num in frame_numbers:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ret, frame = cap.read()
            if ret:
                out.write(frame)
        
        cap.release()
        out.release()
        
        logger.info(f"    ✅ Created calibration video: {output_video.name}")
    
    def create_pair_calibration_projects(self, calibration_videos: Dict[str, Path]) -> Dict[Tuple[str, str], Path]:
        """
        Create complete anipose project structure for each camera pair
        
        Args:
            calibration_videos: Dict mapping camera_id to video path
            
        Returns:
            Dict mapping camera pairs to their project directories
        """
        logger.info("🏗️  CREATING PAIRWISE ANIPOSE PROJECTS")
        logger.info("="*60)
        
        # Find synchronized frames for pairs
        pair_frames = self.find_synchronized_charuco_frames(calibration_videos)
        
        # Extract frames and create anipose structure
        pair_dirs = self.extract_pair_frames(calibration_videos, pair_frames)
        
        # Create anipose config for each pair project
        for (cam1, cam2), session_dir in pair_dirs.items():
            config_path = self.create_anipose_config_for_pair(self.temp_base_dir, cam1, cam2)
            logger.info(f"✅ Anipose project ready for pair ({cam1}, {cam2})")
        
        return pair_dirs
    
    def run_anipose_calibration_for_pairs(self, pair_dirs: Dict[Tuple[str, str], Path]) -> Dict[Tuple[str, str], Path]:
        """
        Run anipose calibrate command for each camera pair
        
        Args:
            pair_dirs: Dict mapping pairs to their session directories
            
        Returns:
            Dict mapping pairs to their calibration result files
        """
        logger.info("🔧 RUNNING ANIPOSE CALIBRATION FOR EACH PAIR")
        logger.info("="*60)
        
        calibration_results = {}
        
        for (cam1, cam2), session_dir in pair_dirs.items():
            session_name = session_dir.name
            logger.info(f"Running anipose calibration for pair ({cam1}, {cam2}) in {session_name}...")
            
            try:
                # Run anipose calibrate from WITHIN the session directory
                cmd = ['anipose', 'calibrate']
                logger.info(f"  Command: {' '.join(cmd)}")
                logger.info(f"  Working directory: {session_dir}")
                
                # Copy the config.toml into this session directory
                base_config = self.temp_base_dir / "config.toml"
                session_config = session_dir / "config.toml"
                if base_config.exists():
                    import shutil
                    shutil.copy2(base_config, session_config)
                
                result = subprocess.run(
                    cmd, cwd=session_dir, 
                    capture_output=True, text=True, check=True
                )
                
                logger.info(f"  ✅ Anipose calibration completed for pair ({cam1}, {cam2})")
                if result.stdout:
                    logger.debug(f"  Stdout: {result.stdout}")
                
                # Check for calibration output (anipose creates calibration.toml in calibration subdirectory)
                calib_file = session_dir / "calibration" / "calibration.toml"
                if calib_file.exists():
                    calibration_results[(cam1, cam2)] = calib_file
                    logger.info(f"  📁 Calibration file: {calib_file}")
                else:
                    # Also check in session directory (backup location)
                    calib_file_alt = session_dir / "calibration.toml"
                    if calib_file_alt.exists():
                        calibration_results[(cam1, cam2)] = calib_file_alt
                        logger.info(f"  📁 Calibration file: {calib_file_alt}")
                    else:
                        logger.error(f"  ❌ Calibration file not found for pair ({cam1}, {cam2})")
                        logger.info(f"  Checked: {calib_file} and {calib_file_alt}")
                        
                        # List what's actually in the session directory
                        logger.info(f"  Session directory contents:")
                        for item in session_dir.iterdir():
                            logger.info(f"    {item}")
                            if item.is_dir():
                                for subitem in item.iterdir():
                                    logger.info(f"      {subitem}")
                
            except subprocess.CalledProcessError as e:
                logger.error(f"❌ Anipose calibration failed for pair ({cam1}, {cam2})")
                logger.error(f"   Command: {' '.join(e.cmd)}")
                logger.error(f"   Exit code: {e.returncode}")
                logger.error(f"   Stderr: {e.stderr}")
                logger.error(f"   Stdout: {e.stdout}")
            except FileNotFoundError:
                logger.error(f"❌ Anipose command not found - is anipose installed?")
                logger.error("   Install with: pip install anipose")
        
        return calibration_results
    
    def load_anipose_pair_calibration(self, calib_file: Path) -> Dict[str, Dict]:
        """
        Load anipose calibration results for a camera pair
        
        Args:
            calib_file: Path to anipose calibration.toml
            
        Returns:
            Dict with camera parameters from anipose
        """
        with open(calib_file, 'r') as f:
            calib_data = toml.load(f)
        
        # Log the raw anipose calibration data
        logger.info(f"📋 Raw anipose calibration from {calib_file.name}:")
        
        # Extract and log metadata if present
        if 'metadata' in calib_data:
            metadata = calib_data['metadata']
            logger.info(f"  📊 Metadata:")
            if 'error' in metadata:
                logger.info(f"    Calibration error: {metadata['error']:.6f}")
            if 'adjusted' in metadata:
                logger.info(f"    Adjusted cameras: {metadata['adjusted']}")
            for key, value in metadata.items():
                if key not in ['error', 'adjusted']:
                    logger.info(f"    {key}: {value}")
        
        # Log camera parameters before conversion
        camera_keys = [k for k in calib_data.keys() if k.startswith('cam_')]
        logger.info(f"  📷 Found cameras: {camera_keys}")
        
        for key in camera_keys:
            data = calib_data[key]
            camera_id = key[4:]  # Remove 'cam_' prefix
            logger.info(f"  Camera {camera_id} original anipose data:")
            logger.info(f"    Size: {data.get('size', 'N/A')}")
            logger.info(f"    Matrix (K): {np.array(data['matrix']).tolist()}")
            logger.info(f"    Distortions: {data['distortions']}")
            logger.info(f"    Rotation (rvec): {data['rotation']}")
            logger.info(f"    Translation: {data['translation']}")
            
            # Convert and show rotation matrix
            rvec = np.array(data['rotation'])
            R, _ = cv2.Rodrigues(rvec)
            rot_deg = self._rotation_angle_degrees(R)
            trans_norm = np.linalg.norm(data['translation'])
            logger.info(f"    → Rotation from identity: {rot_deg:.2f}°")
            logger.info(f"    → Translation norm: {trans_norm:.2f}mm")
        
        # Convert anipose format to our format
        cameras = {}
        metadata = calib_data.get('metadata', {})
        rms_error = float(metadata.get('error')) if 'error' in metadata else None

        for key, data in calib_data.items():
            if key.startswith('cam_'):
                camera_id = key[4:]

                K = np.array(data['matrix'])
                dist = np.array(data['distortions'])
                rvec = np.array(data['rotation'])
                tvec = np.array(data['translation']).reshape(-1, 1)
                R, _ = cv2.Rodrigues(rvec)

                cameras[camera_id] = {
                    'K': K,
                    'R': R,
                    't': tvec,
                    'rvec': rvec,
                    'dist': dist,
                    'size': data['size'],
                    'metadata': {
                        'pair_error': rms_error
                    }
                }

        return cameras
    
    def _rotation_angle_degrees(self, R: np.ndarray) -> float:
        """Calculate rotation angle from identity matrix in degrees"""
        R_err = R @ np.eye(3).T
        val = (np.trace(R_err) - 1.0) * 0.5
        val = float(np.clip(val, -1.0, 1.0))
        return float(np.degrees(np.arccos(val)))
    
    def combine_pairwise_calibrations(self, pair_calibrations: Dict[Tuple[str, str], Dict[str, Dict]], \
                                    pair_errors: Dict[Tuple[str, str], float] = None) -> Dict[str, Dict]:
        """Merge pairwise anipose results into a 3-camera calibration."""
        logger.info("🔗 COMBINING PAIRWISE CALIBRATIONS INTO 3-CAMERA SYSTEM")
        logger.info("="*60)

        def rodrigues_vec_from_R(R: np.ndarray) -> np.ndarray:
            rvec, _ = cv2.Rodrigues(R)
            return rvec.reshape(3)

        def R_from_rodrigues_vec(rvec: np.ndarray) -> np.ndarray:
            R, _ = cv2.Rodrigues(rvec.reshape(3, 1))
            return R

        def average_rotations(R_list: List[np.ndarray], weights: List[float]) -> np.ndarray:
            assert len(R_list) == len(weights) and len(R_list) > 0
            total_w = max(sum(weights), 1e-8)
            r_sum = np.zeros(3)
            for R_i, w in zip(R_list, weights):
                r_sum += w * rodrigues_vec_from_R(R_i)
            r_mean = r_sum / total_w
            return R_from_rodrigues_vec(r_mean)

        def compose(R_ab: np.ndarray, t_ab: np.ndarray, R_bc: np.ndarray, t_bc: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
            R_ac = R_ab @ R_bc
            t_ac = (R_ab @ t_bc) + t_ab
            return R_ac, t_ac

        def invert(R_ab: np.ndarray, t_ab: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
            R_ba = R_ab.T
            t_ba = -R_ab.T @ t_ab
            return R_ba, t_ba

        def rot_err_deg(R_a: np.ndarray, R_b: np.ndarray) -> float:
            R_err = R_a @ R_b.T
            val = (np.trace(R_err) - 1.0) * 0.5
            val = float(np.clip(val, -1.0, 1.0))
            return float(np.degrees(np.arccos(val)))

        def build_camera_transform(cam_a: Dict[str, Dict], cam_b: Dict[str, Dict]) -> Tuple[np.ndarray, np.ndarray]:
            R_a = cam_a['R']
            t_a = cam_a['t']
            R_b = cam_b['R']
            t_b = cam_b['t']
            R_ab = R_a @ R_b.T
            t_ab = t_a - (R_a @ R_b.T @ t_b)
            return R_ab, t_ab

        def map_pair_to_global(pair: Tuple[str, str], pair_data: Dict[str, Dict]) -> Optional[Dict[str, str]]:
            local_metrics = []
            for key, cam_params in pair_data.items():
                rot = rot_err_deg(cam_params['R'], np.eye(3))
                trans = float(np.linalg.norm(cam_params['t']))
                local_metrics.append((rot + 1e-3 * trans, key))
            if len(local_metrics) != 2:
                logger.error(f"Unexpected camera entries for pair {pair}: {list(pair_data.keys())}")
                return None
            local_metrics.sort(key=lambda x: x[0])
            ref_local = local_metrics[0][1]
            other_local = local_metrics[1][1]
            if pair == ("1", "2"):
                mapping = {ref_local: "1", other_local: "2"}
            elif pair == ("1", "3"):
                mapping = {ref_local: "1", other_local: "3"}
            else:  # pair == ("2", "3")
                mapping = {ref_local: "2", other_local: "3"}
            logger.info(f"  Mapped local cameras {mapping}")
            return mapping

        transforms: Dict[Tuple[str, str], Dict[str, np.ndarray]] = {}
        cam_intrinsics: Dict[str, Dict] = {}
        cam_intrinsic_sources: Dict[str, Dict[str, Optional[float]]] = {}

        def register_intrinsics(global_cam_id: str, source_pair: Tuple[str, str], camera_data: Dict[str, Dict]):
            entry = {
                'K': camera_data['K'],
                'dist': camera_data['dist'],
                'size': camera_data['size']
            }
            error_val = camera_data.get('metadata', {}).get('pair_error')
            if global_cam_id not in cam_intrinsics:
                cam_intrinsics[global_cam_id] = entry
                cam_intrinsic_sources[global_cam_id] = {
                    'pair': f"{source_pair[0]}-{source_pair[1]}",
                    'error': float(error_val) if error_val is not None else None
                }
                return
            prev_error = cam_intrinsic_sources.get(global_cam_id, {}).get('error')
            if error_val is not None and (prev_error is None or error_val < prev_error):
                cam_intrinsics[global_cam_id] = entry
                cam_intrinsic_sources[global_cam_id] = {
                    'pair': f"{source_pair[0]}-{source_pair[1]}",
                    'error': float(error_val)
                }

        for pair in [("1", "2"), ("1", "3"), ("2", "3")]:
            if pair not in pair_calibrations:
                continue
            pair_data = pair_calibrations[pair]
            logger.info(f"🔍 Pair ({pair[0]},{pair[1]}) raw data: {list(pair_data.keys())}")
            for cam_name, cam_params in pair_data.items():
                rot_deg = rot_err_deg(cam_params['R'], np.eye(3))
                trans_norm = float(np.linalg.norm(cam_params['t']))
                logger.info(f"  Camera {cam_name}: rot={rot_deg:.2f}°, trans={trans_norm:.2f}mm")

            mapping = map_pair_to_global(pair, pair_data)
            if mapping is None:
                continue
            inverse_map = {global_id: local_id for local_id, global_id in mapping.items()}
            if pair[0] not in inverse_map or pair[1] not in inverse_map:
                logger.error(f"Failed to map pair {pair} into global camera ids")
                continue

            local_a = inverse_map[pair[0]]
            local_b = inverse_map[pair[1]]

            R_ab, t_ab = build_camera_transform(pair_data[local_a], pair_data[local_b])
            R_ba, t_ba = build_camera_transform(pair_data[local_b], pair_data[local_a])

            transforms[(pair[0], pair[1])] = {'R': R_ab, 't': t_ab}
            transforms[(pair[1], pair[0])] = {'R': R_ba, 't': t_ba}

            register_intrinsics(pair[0], pair, pair_data[local_a])
            register_intrinsics(pair[1], pair, pair_data[local_b])

        required_ids = {"1", "2", "3"}
        if not required_ids.issubset(cam_intrinsics.keys()):
            missing = required_ids - cam_intrinsics.keys()
            logger.error(f"Missing intrinsics for cameras: {missing}")
            return {}

        metrics: Dict[str, Dict] = {
            'pairwise_errors': {},
            'transforms': {},
            'cycle_consistency': {},
            'scale_adjustment': {},
            'intrinsics_source': cam_intrinsic_sources,
            'notes': {
                'transform_convention': 'world -> camera (opencv)',
                'units': 'mm translations',
                'angles': 'degrees'
            }
        }

        if pair_errors:
            for (cam1, cam2), error in pair_errors.items():
                metrics['pairwise_errors'][f"pair_{cam1}_{cam2}"] = float(error)
                logger.info(f"📊 Pair ({cam1}, {cam2}) calibration error: {error:.6f}")
        else:
            logger.warning("No pairwise errors provided to combine function")

        T12 = transforms.get(("1", "2"))
        T13 = transforms.get(("1", "3"))
        T23 = transforms.get(("2", "3"))

        if T12 is None or T13 is None:
            logger.error("Missing required transforms (1,2) or (1,3) for global assembly")
            return {}

        scale_factor = 1.0
        if T23 is not None:
            R12, t12 = T12['R'], T12['t']
            R13_direct, t13_direct = T13['R'], T13['t']
            R23, t23 = T23['R'], T23['t']
            R13_via, t13_via = compose(R12, t12, R23, t23)
            a_vec = t13_direct - t12
            b_vec = R12 @ t23
            denom = float(b_vec.T @ b_vec)
            if denom > 1e-8:
                s = float((b_vec.T @ a_vec) / denom)
                scale_factor = s
                applied = abs(s - 1.0) > 0.05
                metrics['scale_adjustment'] = {
                    'applied': applied,
                    'scale': s,
                    'source_pair': '2-3'
                }
                if applied:
                    logger.info(f"⚖️  Scaling pair (2,3) translations by factor {s:.4f} to enforce triangle closure")
                    scaled_t23 = s * t23
                    transforms[("2", "3")] = {'R': R23, 't': scaled_t23}
                    R32, t32 = invert(R23, scaled_t23)
                    transforms[("3", "2")] = {'R': R32, 't': t32}
                    T23 = transforms.get(("2", "3"))
            else:
                metrics['scale_adjustment'] = {
                    'applied': False,
                    'scale': 1.0,
                    'source_pair': '2-3',
                    'note': 'denominator too small'
                }
        else:
            metrics['scale_adjustment'] = {
                'applied': False,
                'scale': 1.0,
                'source_pair': '2-3',
                'note': 'pair (2,3) missing'
            }

        combined_calibration: Dict[str, Dict] = {
            '1': {
                'K': cam_intrinsics['1']['K'],
                'R': np.eye(3),
                't': np.zeros((3, 1)),
                'dist': cam_intrinsics['1']['dist'],
                'size': cam_intrinsics['1']['size']
            },
            '2': {
                'K': cam_intrinsics['2']['K'],
                'R': T12['R'],
                't': T12['t'],
                'dist': cam_intrinsics['2']['dist'],
                'size': cam_intrinsics['2']['size']
            },
            '3': {
                'K': cam_intrinsics['3']['K'],
                'R': T13['R'],
                't': T13['t'],
                'dist': cam_intrinsics['3']['dist'],
                'size': cam_intrinsics['3']['size']
            }
        }

        logger.info("✅ Camera 1: Set as reference (identity transform)")
        logger.info("✅ Camera 2: Using transform from pair (1,2)")
        logger.info("✅ Camera 3: Using transform from pair (1,3)")

        if T23 is not None:
            R12, t12 = combined_calibration['2']['R'], combined_calibration['2']['t']
            R13_direct, t13_direct = combined_calibration['3']['R'], combined_calibration['3']['t']
            R23, t23 = T23['R'], T23['t']
            R13_via, t13_via = compose(R12, t12, R23, t23)
            pre_rot_err = rot_err_deg(R13_direct, R13_via)
            pre_trans_err = float(np.linalg.norm(t13_direct - t13_via))
            metrics['cycle_consistency']['pre'] = {
                'pair': '1-3 direct vs 1-2 ∘ 2-3',
                'rotation_deg': float(pre_rot_err),
                'translation_mm': pre_trans_err
            }

            R13_refined = average_rotations([R13_direct, R13_via], [1.0, 1.0])
            t13_refined = 0.5 * (t13_direct + t13_via)

            R32, t32 = invert(R23, t23)
            R12_via, t12_via = compose(R13_direct, t13_direct, R32, t32)
            R12_refined = average_rotations([R12, R12_via], [1.0, 1.0])
            t12_refined = 0.5 * (t12 + t12_via)

            R13_via_refined, t13_via_refined = compose(R12_refined, t12_refined, R23, t23)
            post_rot_err = rot_err_deg(R13_refined, R13_via_refined)
            post_trans_err = float(np.linalg.norm(t13_refined - t13_via_refined))
            metrics['cycle_consistency']['post'] = {
                'pair': '1-3 direct vs refined 1-2 ∘ 2-3',
                'rotation_deg': float(post_rot_err),
                'translation_mm': post_trans_err
            }
            metrics['cycle_consistency']['refinement_applied'] = True

            combined_calibration['2']['R'] = R12_refined
            combined_calibration['2']['t'] = t12_refined
            combined_calibration['3']['R'] = R13_refined
            combined_calibration['3']['t'] = t13_refined
        else:
            metrics['cycle_consistency']['refinement_applied'] = False

        metrics['transforms']['T12'] = {
            'rotation_deg_from_identity': float(rot_err_deg(combined_calibration['2']['R'], np.eye(3))),
            'translation_mm_norm': float(np.linalg.norm(combined_calibration['2']['t']))
        }
        metrics['transforms']['T13'] = {
            'rotation_deg_from_identity': float(rot_err_deg(combined_calibration['3']['R'], np.eye(3))),
            'translation_mm_norm': float(np.linalg.norm(combined_calibration['3']['t']))
        }
        if T23 is not None:
            metrics['transforms']['T23'] = {
                'rotation_deg_from_identity': float(rot_err_deg(T23['R'], np.eye(3))),
                'translation_mm_norm': float(np.linalg.norm(T23['t'])),
                'scale_factor': scale_factor
            }

        self._metrics = metrics

        logger.info(f"✅ Combined calibration created for {len(combined_calibration)} cameras")
        return combined_calibration

    def save_metrics_report(self, output_dir: Path) -> Optional[Path]:
        """Save calibration metrics JSON report next to calibration file."""
        try:
            metrics = getattr(self, '_metrics', None)
            if metrics is None:
                logger.warning("No metrics available to save.")
                return None
            output_dir.mkdir(parents=True, exist_ok=True)
            metrics_path = output_dir / "calibration_metrics.json"
            with open(metrics_path, 'w') as f:
                json.dump(metrics, f, indent=2)
            logger.info(f"📈 Saved calibration metrics: {metrics_path}")
            # Brief console summary
            pre = metrics.get('cycle_consistency', {}).get('pre')
            post = metrics.get('cycle_consistency', {}).get('post')
            if pre and post:
                logger.info(f"Cycle consistency rot: pre {pre['rotation_deg']:.3f}° → post {post['rotation_deg']:.3f}°")
                logger.info(f"Cycle consistency trans: pre {pre['translation_mm']:.3f}mm → post {post['translation_mm']:.3f}mm")
            return metrics_path
        except Exception as e:
            logger.warning(f"Could not save metrics report: {e}")
            return None
    
    def save_combined_calibration(self, combined_calib: Dict[str, Dict], output_path: Path):
        """
        Save combined calibration in anipose format
        
        Args:
            combined_calib: Combined calibration parameters
            output_path: Path to save calibration.toml
        """
        logger.info(f"💾 SAVING COMBINED 3-CAMERA CALIBRATION")
        logger.info("="*50)
        
        # Convert to anipose TOML format
        toml_data = {}
        
        for camera_id, params in combined_calib.items():
            # Convert rotation matrix back to rotation vector for anipose format
            rvec, _ = cv2.Rodrigues(params['R'])
            
            toml_data[f'cam_{camera_id}'] = {
                'name': camera_id,
                'size': params['size'],
                'matrix': params['K'].tolist(),
                'distortions': params['dist'].tolist(),
                'rotation': rvec.flatten().tolist(),
                'translation': params['t'].flatten().tolist()
            }
        
        # Save to file
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            toml.dump(toml_data, f)
        
        logger.info(f"✅ Saved combined calibration: {output_path}")
        logger.info(f"   Contains {len(toml_data)} cameras")
        
        # Print calibration summary
        print("\n" + "="*60)
        print("PAIRWISE ANIPOSE CALIBRATION SUMMARY")
        print("="*60)
        print(f"Cameras calibrated: {list(combined_calib.keys())}")
        print(f"Output file: {output_path}")
        print("\nCamera parameters:")
        
        for camera_id, params in combined_calib.items():
            print(f"\nCamera {camera_id}:")
            print(f"  - Image size: {params['size']}")
            print(f"  - Focal length: fx={params['K'][0,0]:.2f}, fy={params['K'][1,1]:.2f}")
            print(f"  - Principal point: cx={params['K'][0,2]:.2f}, cy={params['K'][1,2]:.2f}")
            if camera_id != "1":  # Reference camera
                t_norm = np.linalg.norm(params['t'])
                print(f"  - Distance from camera 1: {t_norm:.2f}mm")
        
        print("="*60)
    
    def cleanup_temp_directory(self):
        """Clean up temporary calibration directory"""
        if self.temp_base_dir.exists():
            try:
                shutil.rmtree(self.temp_base_dir)
                logger.info(f"🧹 Cleaned up temporary directory: {self.temp_base_dir}")
            except Exception as e:
                logger.warning(f"Could not clean up temp directory: {e}")
    
    def run_calibration_on_existing_temp(self, temp_dir: Path, output_session_dir: Path) -> bool:
        """
        Run anipose calibration on existing temporary directory structure
        
        Args:
            temp_dir: Existing temporary directory with pairwise projects
            output_session_dir: Session directory to save final calibration
            
        Returns:
            True if calibration succeeded
        """
        logger.info("🔧 RUNNING ANIPOSE CALIBRATION ON EXISTING TEMP STRUCTURE")
        logger.info("="*70)
        logger.info(f"Temp directory: {temp_dir}")
        
        if not temp_dir.exists():
            logger.error(f"Temp directory not found: {temp_dir}")
            return False
        
        # Find existing session directories
        session_dirs = [d for d in temp_dir.iterdir() if d.is_dir() and d.name.startswith('session')]
        logger.info(f"Found {len(session_dirs)} session directories: {[d.name for d in session_dirs]}")
        
        # Map session directories to camera pairs
        pair_dirs = {}
        for i, (cam1, cam2) in enumerate(self.camera_pairs, 1):
            session_dir = temp_dir / f"session{i}"
            if session_dir.exists():
                pair_dirs[(cam1, cam2)] = session_dir
                logger.info(f"Pair ({cam1}, {cam2}) -> {session_dir.name}")
        
        # Run anipose calibration for each pair
        calibration_results = self.run_anipose_calibration_for_pairs(pair_dirs)
        
        if not calibration_results:
            logger.error("No calibration results obtained")
            return False
        
        # Load and combine pairwise calibrations
        pair_calibrations = {}
        pair_errors = {}  # Store calibration errors for each pair
        for (cam1, cam2), calib_file in calibration_results.items():
            logger.info(f"Loading calibration results for pair ({cam1}, {cam2})...")
            pair_calib = self.load_anipose_pair_calibration(calib_file)
            pair_calibrations[(cam1, cam2)] = pair_calib
            
            # Extract calibration error from the TOML file
            try:
                with open(calib_file, 'r') as f:
                    calib_data = toml.load(f)
                if 'metadata' in calib_data and 'error' in calib_data['metadata']:
                    pair_errors[(cam1, cam2)] = float(calib_data['metadata']['error'])
            except Exception as e:
                logger.warning(f"Could not extract error for pair ({cam1}, {cam2}): {e}")
        
        # Combine into unified 3-camera calibration
        combined_calib = self.combine_pairwise_calibrations(pair_calibrations, pair_errors)

        # Save final calibration to experiment root (parent of session)
        experiment_root = output_session_dir.parent
        output_path = experiment_root / "calibration.toml"
        self.save_combined_calibration(combined_calib, output_path)
        
        # Also save in session/calibration for backwards compatibility
        output_calib_dir = output_session_dir / "calibration"
        output_calib_dir.mkdir(parents=True, exist_ok=True)
        session_calib_path = output_calib_dir / "calibration.toml"
        self.save_combined_calibration(combined_calib, session_calib_path)
        
        # Save metrics next to main calibration
        self.save_metrics_report(experiment_root)
        
        logger.info("🎉 PAIRWISE ANIPOSE CALIBRATION COMPLETED SUCCESSFULLY!")
        return True

    def run_pairwise_calibration(self, session_dir: Path, cleanup: bool = False) -> bool:
        """
        Run complete pairwise anipose calibration process
        
        Args:
            session_dir: Session directory containing calibration videos
            cleanup: Whether to clean up temporary files
            
        Returns:
            True if calibration succeeded
        """
        logger.info("🚀 STARTING PAIRWISE ANIPOSE CALIBRATION")
        logger.info("="*70)
        logger.info(f"Session directory: {session_dir}")
        logger.info(f"Temporary directory: {self.temp_base_dir}")
        logger.info(f"Camera pairs: {self.camera_pairs}")
        if cleanup:
            logger.info("🗑️  Temporary files will be cleaned up after completion")
        else:
            logger.info(f"🗂️  Temporary files will be preserved at: {self.temp_base_dir}")
            logger.info("   Use --cleanup flag to remove temp files after calibration")
        
        try:
            # Find calibration videos
            calibration_dir = session_dir / "calibration"
            if not calibration_dir.exists():
                logger.error(f"Calibration directory not found: {calibration_dir}")
                return False
            
            calibration_videos = {}
            for camera_id in ["1", "2", "3"]:
                for pattern in [f"calib-cam{camera_id}.mp4", f"cam{camera_id}-calib.mp4"]:
                    video_path = calibration_dir / pattern
                    if video_path.exists():
                        calibration_videos[camera_id] = video_path
                        logger.info(f"Found calibration video: {camera_id} -> {pattern}")
                        break
            
            if len(calibration_videos) < 3:
                logger.error(f"Need 3 calibration videos, found {len(calibration_videos)}")
                return False
            
            # Create pairwise projects
            pair_dirs = self.create_pair_calibration_projects(calibration_videos)
            
            if not pair_dirs:
                logger.error("No pairwise projects created")
                return False
            
            # Run anipose calibration for each pair
            calibration_results = self.run_anipose_calibration_for_pairs(pair_dirs)
            
            if not calibration_results:
                logger.error("No calibration results obtained")
                return False
            
            # Load and combine pairwise calibrations
            pair_calibrations = {}
            pair_errors = {}  # Store calibration errors for each pair
            for (cam1, cam2), calib_file in calibration_results.items():
                logger.info(f"Loading calibration results for pair ({cam1}, {cam2})...")
                pair_calib = self.load_anipose_pair_calibration(calib_file)
                pair_calibrations[(cam1, cam2)] = pair_calib
                
                # Extract calibration error from the TOML file
                try:
                    with open(calib_file, 'r') as f:
                        calib_data = toml.load(f)
                    if 'metadata' in calib_data and 'error' in calib_data['metadata']:
                        pair_errors[(cam1, cam2)] = float(calib_data['metadata']['error'])
                except Exception as e:
                    logger.warning(f"Could not extract error for pair ({cam1}, {cam2}): {e}")
            
            # Combine into unified 3-camera calibration
            combined_calib = self.combine_pairwise_calibrations(pair_calibrations, pair_errors)
            
            # Save final calibration to experiment root (parent of session)
            experiment_root = session_dir.parent
            output_path = experiment_root / "calibration.toml"
            self.save_combined_calibration(combined_calib, output_path)
            
            # Also save in session/calibration for backwards compatibility
            session_calib_path = calibration_dir / "calibration.toml"
            self.save_combined_calibration(combined_calib, session_calib_path)
            
            # Save metrics next to main calibration
            self.save_metrics_report(experiment_root)
            
            logger.info("🎉 PAIRWISE ANIPOSE CALIBRATION COMPLETED SUCCESSFULLY!")
            logger.info(f"📁 Main calibration saved to experiment root: {output_path}")
            logger.info(f"📁 Backup calibration saved to: {session_calib_path}")
            logger.info("✅ Anipose commands should now be run from experiment root")
            
            if not cleanup:
                logger.info(f"🗂️  Temporary files preserved at: {self.temp_base_dir}")
                logger.info("   Contains pairwise anipose projects and intermediate calibrations")
            
            return True
            
        except Exception as e:
            logger.error(f"Pairwise calibration failed: {e}")
            import traceback
            traceback.print_exc()
            return False
            
        finally:
            if cleanup:
                self.cleanup_temp_directory()

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Pairwise Anipose Calibration for 3-Camera System")
    parser.add_argument("--session-dir", required=True, type=str,
                       help="Path to session directory with calibration videos")
    parser.add_argument("--temp-dir", type=str, default=None,
                       help="Temporary directory for pairwise calibration projects")
    parser.add_argument("--cleanup", action="store_true",
                       help="Clean up temporary files after calibration (default: keep temp files)")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Enable verbose logging")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Check dependencies
    if not DETECTION_AVAILABLE:
        logger.error("Detection modules not available")
        logger.error("Please ensure custom_calibration.py and frame_level_optimizer.py are in Python path")
        return 1
    
    # Parse arguments
    session_dir = Path(args.session_dir)
    if not session_dir.exists():
        logger.error(f"Session directory not found: {session_dir}")
        return 1
    
    temp_dir = Path(args.temp_dir) if args.temp_dir else None
    
    # Run pairwise calibration
    calibrator = PairwiseAniposeCalibrator(temp_dir)
    
    try:
        success = calibrator.run_pairwise_calibration(
            session_dir, 
            cleanup=args.cleanup
        )
        
        return 0 if success else 1
        
    except KeyboardInterrupt:
        logger.info("Calibration interrupted by user")
        if args.cleanup:
            calibrator.cleanup_temp_directory()
        else:
            logger.info(f"🗂️  Temporary files preserved at: {calibrator.temp_base_dir}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
