#!/usr/bin/env python3
"""
Calibration Integration Module

This module provides compatibility functions to integrate the new custom calibration system
with the existing s-DANNCE pipeline. It replaces the old anipose calibration loading functions
and provides the same interface for backward compatibility.

Author: Calibration Integration
"""

import numpy as np
import cv2
import toml
import os
from pathlib import Path
from typing import Dict, Any, Optional, Union, List
import logging

# Try to import the custom calibration module
try:
    from custom_calibration import load_calibration_from_toml, CalibrationData
    CUSTOM_CALIBRATION_AVAILABLE = True
except ImportError:
    CUSTOM_CALIBRATION_AVAILABLE = False
    logging.warning("Custom calibration module not available. Falling back to anipose calibration.")

logger = logging.getLogger(__name__)

def load_calibration_params(calib_path: Union[str, Path]) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Load calibration parameters from TOML file and convert to s-DANNCE format.
    
    This function provides backward compatibility with the existing pipeline
    while supporting both the old anipose format and the new custom format.
    
    Args:
        calib_path: Path to calibration TOML file
        
    Returns:
        Dictionary with camera parameters in s-DANNCE format:
        {
            'Camera1': {
                'K': 3x3 intrinsic matrix,
                'R': 3x3 rotation matrix, 
                't': 1x3 translation vector,
                'RDistort': 1x2 radial distortion,
                'TDistort': 1x2 tangential distortion
            },
            'Camera2': { ... },
            ...
        }
    """
    calib_path = Path(calib_path)
    
    if not calib_path.exists():
        raise FileNotFoundError(f"Calibration file not found: {calib_path}")
    
    # Load TOML data
    with open(calib_path, 'r') as f:
        calib_data = toml.load(f)
    
    # Check if this is a custom calibration format
    if 'metadata' in calib_data and calib_data['metadata'].get('calibration_type') == 'custom_multi_camera':
        return _load_custom_calibration_format(calib_data)
    else:
        # Fallback to old anipose format
        return _load_anipose_calibration_format(calib_data)

def _load_custom_calibration_format(calib_data: Dict[str, Any]) -> Dict[str, Dict[str, np.ndarray]]:
    """Load calibration from custom format"""
    
    if not CUSTOM_CALIBRATION_AVAILABLE:
        raise ImportError("Custom calibration module not available")
    
    params = {}
    metadata = calib_data.get('metadata', {})
    reference_camera = metadata.get('reference_camera')
    
    # Process each camera
    for key, cam_data in calib_data.items():
        if not key.startswith('cam_'):
            continue
            
        camera_id = key[4:]  # Remove 'cam_' prefix
        
        # Map camera ID to s-DANNCE camera names
        if camera_id == '1' or camera_id == 'cam1':
            cam_name = 'Camera1'
        elif camera_id == '2' or camera_id == 'cam2': 
            cam_name = 'Camera2'
        elif camera_id == '3' or camera_id == 'cam3':
            cam_name = 'Camera3'
        else:
            # For additional cameras, create sequential names
            cam_name = f'Camera{camera_id}'
        
        # Extract parameters
        K = np.array(cam_data['matrix'])  # 3x3 intrinsic matrix
        
        # Get rotation and translation
        if 'rotation_matrix' in cam_data:
            R = np.array(cam_data['rotation_matrix'])  # 3x3 rotation matrix
        else:
            # Convert from rotation vector
            rvec = np.array(cam_data['rotation'])
            R, _ = cv2.Rodrigues(rvec)

        # Optional transpose toggle for downstream compatibility testing
        # Enable by setting environment variable CUSTOM_CALIB_TRANSPOSE_R=1/true
        if os.getenv('CUSTOM_CALIB_TRANSPOSE_R', '0').lower() in ('1', 'true', 'yes', 'y'):
            R = R.T
        
        # Translation vector
        tvec = np.array(cam_data['translation'])
        if tvec.ndim == 1:
            tvec = tvec.reshape(1, 3)  # Ensure 1x3 shape
        
        # Distortion coefficients
        dist_coeffs = np.array(cam_data['distortions'])
        
        # Extract radial and tangential distortion components
        if 'RDistort' in cam_data and 'TDistort' in cam_data:
            RDistort = np.array(cam_data['RDistort'])
            TDistort = np.array(cam_data['TDistort'])
        else:
            # Extract from distortion vector
            if len(dist_coeffs) >= 4:
                RDistort = dist_coeffs[:2].reshape(1, 2)
                TDistort = dist_coeffs[2:4].reshape(1, 2)
            else:
                RDistort = np.zeros((1, 2))
                TDistort = np.zeros((1, 2))
        
        # Create parameter dictionary for this camera
        params[cam_name] = {
            'K': K,
            'R': R,
            't': tvec,
            'RDistort': RDistort,
            'TDistort': TDistort
        }
        
        logger.debug(f"Loaded custom calibration for {cam_name}")
    
    logger.info(f"Loaded custom calibration for {len(params)} cameras")
    return params

def _load_anipose_calibration_format(calib_data: Dict[str, Any]) -> Dict[str, Dict[str, np.ndarray]]:
    """Load calibration from old anipose format (backward compatibility)"""
    
    params = {}
    
    # Camera mapping for anipose format
    camera_mapping = {
        'cam_0': 'Camera1',
        'cam_1': 'Camera2',
        'cam_2': 'Camera3'
    }
    
    for cam_key, cam_name in camera_mapping.items():
        if cam_key in calib_data:
            cam_data = calib_data[cam_key]
            
            # Extract parameters with proper transformations for s-DANNCE
            K = np.array(cam_data['matrix']).T  # Transpose for s-DANNCE
            
            # Convert rotation vector to rotation matrix
            rvec = np.array(cam_data['rotation'])
            R_opencv, _ = cv2.Rodrigues(rvec)
            R = R_opencv.T  # Transpose for s-DANNCE
            
            # Translation vector
            tvec = np.array(cam_data['translation']).reshape(1, 3)
            
            # Distortion coefficients
            dist = np.array(cam_data['distortions'])
            RDistort = dist[:2].reshape(1, 2) if len(dist) >= 2 else np.zeros((1, 2))
            TDistort = dist[2:4].reshape(1, 2) if len(dist) >= 4 else np.zeros((1, 2))
            
            params[cam_name] = {
                'K': K,
                'R': R,
                't': tvec,
                'RDistort': RDistort,
                'TDistort': TDistort
            }
            
            logger.debug(f"Loaded anipose calibration for {cam_name}")
    
    logger.info(f"Loaded anipose calibration for {len(params)} cameras")
    return params

def get_camera_count(calib_path: Union[str, Path]) -> int:
    """
    Get the number of cameras in calibration file
    
    Args:
        calib_path: Path to calibration TOML file
        
    Returns:
        Number of cameras in calibration
    """
    calib_path = Path(calib_path)
    
    with open(calib_path, 'r') as f:
        calib_data = toml.load(f)
    
    # Count camera entries
    camera_count = 0
    for key in calib_data.keys():
        if key.startswith('cam_'):
            camera_count += 1
    
    return camera_count

def get_camera_names(calib_path: Union[str, Path]) -> list[str]:
    """
    Get list of camera names from calibration file
    
    Args:
        calib_path: Path to calibration TOML file
        
    Returns:
        List of camera names in s-DANNCE format
    """
    params = load_calibration_params(calib_path)
    return list(params.keys())

def convert_to_s_dannce_format(calibration_data: 'CalibrationData') -> Dict[str, Dict[str, np.ndarray]]:
    """
    Convert CalibrationData object to s-DANNCE parameter format
    
    Args:
        calibration_data: CalibrationData object from custom calibration
        
    Returns:
        Dictionary with s-DANNCE compatible parameters
    """
    if not CUSTOM_CALIBRATION_AVAILABLE:
        raise ImportError("Custom calibration module not available")
    
    params = {}
    
    for camera_id, intrinsic in calibration_data.intrinsics.items():
        # Map camera ID to s-DANNCE camera names
        if camera_id == '1' or camera_id == 'cam1':
            cam_name = 'Camera1'
        elif camera_id == '2' or camera_id == 'cam2':
            cam_name = 'Camera2'
        elif camera_id == '3' or camera_id == 'cam3':
            cam_name = 'Camera3'
        else:
            cam_name = f'Camera{camera_id}'
        
        # Get intrinsics
        K = intrinsic.camera_matrix
        
        # Get extrinsics or use identity for reference camera
        if camera_id in calibration_data.extrinsics:
            extrinsic = calibration_data.extrinsics[camera_id]
            R = extrinsic.rotation_matrix
            t = extrinsic.translation_vector.reshape(1, 3)
        else:
            # Reference camera
            R = np.eye(3)
            t = np.zeros((1, 3))
        
        # Distortion coefficients
        # Flatten before determining length; OpenCV often returns shape (1, N)
        dist_vec = np.asarray(intrinsic.distortion_coeffs).ravel()
        if dist_vec.size >= 4:
            RDistort = dist_vec[:2].reshape(1, 2)
            TDistort = dist_vec[2:4].reshape(1, 2)
        else:
            RDistort = np.zeros((1, 2))
            TDistort = np.zeros((1, 2))
        
        params[cam_name] = {
            'K': K,
            'R': R,
            't': t,
            'RDistort': RDistort,
            'TDistort': TDistort
        }
    
    return params

def visualize_reprojection_on_images(
    calib_path: Union[str, Path],
    images_by_camera: Dict[str, List[Union[str, Path]]],
    output_dir: Union[str, Path] = "reproj_debug",
    board_config: Optional[Dict[str, Any]] = None,
    max_images_per_cam: int = 10,
) -> Dict[str, float]:
    """
    Visualize ChArUco reprojection and compute per-camera RMS error on provided images.

    - Detect ChArUco corners in each image
    - Estimate board pose with PnP using loaded intrinsics
    - Reproject detected corners and draw overlays
    - Compute pixel residual RMS per camera

    Args:
        calib_path: Path to calibration TOML (custom or anipose format)
        images_by_camera: Mapping like {'1': [img1.jpg, ...], '2': [...]} or {'Camera1': [...], ...}
        output_dir: Directory to save overlay images
        board_config: Optional board config dict; falls back to defaults
        max_images_per_cam: Limit processed images per camera

    Returns:
        Dict mapping camera name to RMS reprojection error (pixels)
    """
    calib_path = Path(calib_path)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load calibration (K, R, t, distortions) in s-DANNCE format
    params = load_calibration_params(calib_path)

    # Lazy import detector/board to avoid hard dependency when unused
    try:
        from custom_calibration import ChArUcoBoard, ChArUcoDetector
    except Exception as e:
        raise ImportError(f"ChArUco utilities unavailable: {e}")

    # Create board
    if board_config is None:
        board = ChArUcoBoard()
    else:
        board = ChArUcoBoard(
            squares_x=board_config.get('squares_x', 10),
            squares_y=board_config.get('squares_y', 7),
            square_length=board_config.get('square_length', 25.0),
            marker_length=board_config.get('marker_length', 18.75),
            marker_dict=board_config.get('marker_dict', cv2.aruco.DICT_4X4_50),
        )
    charuco_board = board.create_board()
    detector = ChArUcoDetector(board)

    # Get board 3D corners (handle OpenCV API variants)
    try:
        board_corners = np.array(charuco_board.getChessboardCorners(), dtype=np.float32)
    except AttributeError:
        try:
            board_corners = np.array(charuco_board.chessboardCorners, dtype=np.float32)
        except AttributeError:
            # Manual calculation
            objp = np.zeros((board.squares_x * board.squares_y, 3), np.float32)
            objp[:, :2] = np.mgrid[0:board.squares_x, 0:board.squares_y].T.reshape(-1, 2)
            objp *= float(board.square_length)
            board_corners = objp

    def _norm_cam_key(k: str) -> str:
        k_low = str(k).lower()
        # accept 'camera1', 'cam1', '1'
        import re
        m = re.search(r'(?:camera|cam)?\s*(\d+)', k_low)
        if m:
            return f"Camera{int(m.group(1))}"
        # fallback: passthrough
        return str(k)

    per_cam_errors: Dict[str, float] = {}

    for user_key, img_list in images_by_camera.items():
        cam_name = _norm_cam_key(user_key)
        if cam_name not in params:
            logging.warning(f"No calibration for camera key '{user_key}' (normalized '{cam_name}'). Skipping.")
            continue

        K = np.asarray(params[cam_name]['K'], dtype=np.float64)
        dist = np.asarray(params[cam_name].get('RDistort', [[0, 0]]), dtype=np.float64).ravel()
        tdist = np.asarray(params[cam_name].get('TDistort', [[0, 0]]), dtype=np.float64).ravel()
        dist_vec = np.concatenate([dist, tdist], axis=0).reshape(-1, 1)

        img_errors = []
        cam_out = out_dir / cam_name
        cam_out.mkdir(exist_ok=True)

        for idx, img_path in enumerate(img_list[:max_images_per_cam]):
            img_path = Path(img_path)
            image = cv2.imread(str(img_path))
            if image is None:
                logging.warning(f"Could not read image: {img_path}")
                continue

            # Detect Charuco corners
            corners, ids = detector.detect_charuco_corners(image)
            if corners is None or ids is None or len(corners) < 6:
                logging.info(f"{cam_name}: insufficient ChArUco detections in {img_path.name}")
                continue

            ids_f = ids.flatten()
            img_pts = corners.reshape(-1, 2)

            # Build object points corresponding to detected IDs
            try:
                obj_pts = board_corners[ids_f]
            except Exception:
                # Filter valid ids within range
                valid = (ids_f >= 0) & (ids_f < len(board_corners))
                ids_valid = ids_f[valid]
                obj_pts = board_corners[ids_valid]
                img_pts = img_pts[valid]

            if len(obj_pts) < 6:
                continue

            # Estimate pose with PnP
            ok, rvec, tvec = cv2.solvePnP(
                obj_pts, img_pts, K, distCoeffs=dist_vec, flags=cv2.SOLVEPNP_ITERATIVE
            )
            if not ok:
                continue

            # Reproject and compute residuals
            proj, _ = cv2.projectPoints(obj_pts, rvec, tvec, K, dist_vec)
            proj = proj.reshape(-1, 2)
            err = np.linalg.norm(proj - img_pts, axis=1)
            rms = float(np.sqrt(np.mean(err**2))) if err.size else float('nan')
            img_errors.append(rms)

            # Draw overlay
            vis = image.copy()
            for p_meas, p_proj in zip(img_pts.astype(int), proj.astype(int)):
                cv2.circle(vis, tuple(p_meas), 3, (0, 255, 0), -1)  # measured: green
                cv2.circle(vis, tuple(p_proj), 3, (0, 0, 255), -1)  # projected: red
                cv2.line(vis, tuple(p_meas), tuple(p_proj), (0, 255, 255), 1)

            cv2.putText(
                vis,
                f"RMS: {rms:.2f}px  n: {len(obj_pts)}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
            )

            out_path = cam_out / f"reproj_{idx:03d}_{img_path.stem}.jpg"
            cv2.imwrite(str(out_path), vis)

        if img_errors:
            per_cam_errors[cam_name] = float(np.mean(img_errors))
        else:
            per_cam_errors[cam_name] = float('nan')

    return per_cam_errors

def create_multi_camera_config_toml(session_dir: str, num_cameras: int = 3, 
                                  board_config: Optional[Dict[str, Any]] = None):
    """
    Create config.toml for multi-camera setup in session directory.
    
    This function replaces the old create_config_toml function to support
    multiple cameras and the new calibration system.
    
    Args:
        session_dir: Session directory path
        num_cameras: Number of cameras to configure
        board_config: Custom board configuration (optional)
    """
    
    if board_config is None:
        board_config = {
            'board_type': 'charuco',
            'board_size': [10, 7],
            'board_marker_bits': 4,
            'board_marker_dict_number': 50,
            'board_marker_length': 18.75,  # mm
            'board_square_side_length': 25,  # mm
            'fisheye': False,
            'animal_calibration': True
        }
    
    # Generate camera regex pattern for multi-camera support
    cam_regex = '-cam([0-9]+)'
    
    # Generate camera names
    camera_names = [f'cam{i+1}' for i in range(num_cameras)]
    
    config_content = f"""
# Project settings
nesting = 1
video_extension = 'mp4'

[calibration]
# ChArUco board settings - Custom Calibration System
board_type = "{board_config['board_type']}"
board_size = {board_config['board_size']}
board_marker_bits = {board_config['board_marker_bits']}
board_marker_dict_number = {board_config['board_marker_dict_number']}
board_marker_length = {board_config['board_marker_length']}  # mm
board_square_side_length = {board_config['board_square_side_length']}  # mm
fisheye = {str(board_config['fisheye']).lower()}
animal_calibration = {str(board_config['animal_calibration']).lower()}

# Multi-camera setup
num_cameras = {num_cameras}
camera_names = {camera_names}

# Use custom calibration system instead of anipose
use_custom_calibration = true

[triangulation]
# Enable triangulation for 3D reconstruction
triangulate = true

# Regular expression to extract camera names from filenames
cam_regex = '{cam_regex}'

# Which camera to use for alignment (use cam1)
cam_align = "1"

# Use RANSAC for outlier rejection
ransac = false

# Enable optimization for 3D filtering
optim = true

# Define spatial constraints between connected body parts
constraints = [
   ["nose", "left_earbase"], ["nose", "right_earbase"],
   ["nose", "neck_end"], 
   ["neck_end", "back_middle"], ["back_base", "back_middle"],
   ["back_middle", "back_end"], ["back_end", "tail_base"],
   ["front_left_thai", "front_left_knee"], ["front_left_knee", "front_left_paw"],
   ["front_right_thai", "front_right_knee"], ["front_right_knee", "front_right_paw"],
   ["back_left_thai", "back_left_knee"], ["back_left_knee", "back_left_paw"],
   ["back_right_thai", "back_right_knee"], ["back_right_knee", "back_right_paw"]
]

[labeling]  
scheme = [  
    # Head connections  
    ["nose", "upper_jaw", "lower_jaw"],  
    ["right_eye", "nose", "left_eye"],  
    ["mouth_end_right", "nose", "mouth_end_left"],  
      
    # Ear connections  
    ["right_earbase", "right_earend"],  
    ["left_earbase", "left_earend"],  
      
    # Antler connections  
    ["right_antler_base", "right_antler_end"],  
    ["left_antler_base", "left_antler_end"],  
      
    # Neck and throat  
    ["neck_base", "neck_end"],  
    ["throat_base", "throat_end"],  
      
    # Back/spine  
    ["back_base", "back_middle", "back_end"],  
      
    # Tail  
    ["tail_base", "tail_end"],  
      
    # Front legs  
    ["front_left_thai", "front_left_knee", "front_left_paw"],  
    ["front_right_thai", "front_right_knee", "front_right_paw"],  
      
    # Back legs  
    ["back_left_thai", "back_left_knee", "back_left_paw"],  
    ["back_right_thai", "back_right_knee", "back_right_paw"],  
      
    # Body outline  
    ["neck_base", "body_middle_left", "belly_bottom", "body_middle_right", "neck_base"]  
]

# Smoothing and constraint parameters
scale_smooth = 7
scale_length = 20
scale_length_weak = 1
reproj_error_threshold = 5
score_threshold = 0.1
n_deriv_smooth = 3

[angles]
# Define angles you want to compute
front_left_knee = ["front_left_thai", "front_left_knee", "front_left_paw"]
front_right_knee = ["front_right_thai", "front_right_knee", "front_right_paw"]
back_left_knee = ["back_left_thai", "back_left_knee", "back_left_paw"]
back_right_knee = ["back_right_thai", "back_right_knee", "back_right_paw"]
neck_angle = ["neck_base", "neck_end", "back_base"]
back_angle = ["back_base", "back_middle", "back_end"]
tail_angle = ["back_end", "tail_base", "tail_end"]

[pipeline]
# Default pipeline folders - these match the created structure
calibration_videos = "calibration"
calibration_results = "calibration"
videos_raw = "videos-raw"
pose_2d = "pose-2d"
pose_2d_filter = "pose-2d-filtered"
pose_3d = "pose-3d"
pose_3d_filter = "pose-3d-filtered"

[filter]
enabled = false
type = "viterbi"
score_threshold = 0.2
medfilt = 13
offset_threshold = 25
spline = true

[filter3d]  
enabled = true  
medfilt = 7           
offset_threshold = 40  
"""
    
    config_path = Path(session_dir) / 'config.toml'
    with open(config_path, 'w') as f:
        f.write(config_content.strip())
    
    logger.info(f"Created multi-camera config.toml at: {config_path}")
    logger.info(f"Configured for {num_cameras} cameras with custom calibration system")

# Backward compatibility function names
def load_anipose_calibration(calib_path: Union[str, Path]) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Backward compatibility function that replaces the old load_anipose_calibration
    
    This function maintains the same interface as the original but supports
    both anipose and custom calibration formats.
    """
    return load_calibration_params(calib_path)

def create_config_toml(session_dir: str, num_cameras: int = 3):
    """
    Backward compatibility function that replaces the old create_config_toml
    
    This function maintains the same interface but creates a config that
    supports the new multi-camera calibration system.
    """
    create_multi_camera_config_toml(session_dir, num_cameras)

if __name__ == "__main__":
    # Test the calibration integration
    import tempfile
    
    # Create a test directory
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create a multi-camera config
        create_multi_camera_config_toml(temp_dir, num_cameras=4)
        
        config_path = Path(temp_dir) / 'config.toml'
        print(f"Created test config at: {config_path}")
        
        # Verify the config was created
        with open(config_path, 'r') as f:
            content = f.read()
            print("Config content preview:")
            print(content[:500] + "...")
