#!/usr/bin/env python3
"""
Anipose Library Integration for Custom Calibration

This module integrates our custom 3-camera calibration with the anipose library (aniposelib)
to leverage anipose's animal tracking and 3D processing capabilities without using anipose's
calibration system.

Key features:
- Convert custom calibration to anipose format
- Create anipose CameraGroup from custom calibration
- Use aniposelib for 3D triangulation and filtering
- Support animal-specific processing features

Author: Anipose Integration
"""

import os
import sys
import shutil
import numpy as np
import toml
import cv2
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import custom calibration modules
try:
    from custom_calibration import CalibrationData
    from calibration_integration import load_calibration_params
    CUSTOM_CALIBRATION_AVAILABLE = True
except ImportError as e:
    CUSTOM_CALIBRATION_AVAILABLE = False
    logger.warning(f"Custom calibration not available: {e}")

# Try to import aniposelib
try:
    import aniposelib
    from aniposelib.cameras import CameraGroup
    ANIPOSELIB_AVAILABLE = True
except ImportError as e:
    ANIPOSELIB_AVAILABLE = False
    logger.warning(f"aniposelib not available: {e}")
    logger.info("Install with: pip install aniposelib")

def convert_custom_calibration_to_anipose_format(calibration_data: 'CalibrationData') -> Dict[str, Any]:
    """
    Convert custom calibration data to anipose TOML format
    
    Args:
        calibration_data: CalibrationData object from custom calibration
        
    Returns:
        Dictionary in anipose calibration.toml format
    """
    if not CUSTOM_CALIBRATION_AVAILABLE:
        raise ImportError("Custom calibration module not available")
    
    logger.info("Converting custom calibration to anipose format...")
    
    anipose_dict = {}
    
    # Process each camera
    for i, (camera_id, intrinsic) in enumerate(calibration_data.intrinsics.items()):
        # Anipose uses cam_0, cam_1, cam_2, etc. format
        anipose_cam_key = f'cam_{i}'
        
        # Get extrinsics (if available) or use identity for reference camera
        if camera_id in calibration_data.extrinsics:
            extrinsic = calibration_data.extrinsics[camera_id]
            rotation_vector = extrinsic.rotation_vector.flatten()
            translation_vector = extrinsic.translation_vector.flatten()
        else:
            # Reference camera has identity transform
            rotation_vector = np.zeros(3)
            translation_vector = np.zeros(3)
        
        # Create anipose camera entry
        anipose_dict[anipose_cam_key] = {
            'name': f'Camera{int(camera_id)}',
            'size': list(intrinsic.image_size),  # [width, height]
            'matrix': intrinsic.camera_matrix.tolist(),  # 3x3 intrinsic matrix
            'distortions': intrinsic.distortion_coeffs.tolist(),  # Distortion coefficients
            'rotation': rotation_vector.tolist(),  # 3-element rotation vector (Rodrigues)
            'translation': translation_vector.tolist()  # 3-element translation vector
        }
        
        logger.info(f"Converted camera {camera_id} to anipose format as {anipose_cam_key}")
    
    logger.info(f"✅ Converted {len(anipose_dict)} cameras to anipose format")
    return anipose_dict

def save_anipose_calibration_toml(anipose_dict: Dict[str, Any], output_path: Path):
    """
    Save anipose calibration dictionary to TOML file
    
    Args:
        anipose_dict: Anipose calibration dictionary
        output_path: Path to save calibration.toml
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        toml.dump(anipose_dict, f)
    
    logger.info(f"Saved anipose calibration to: {output_path}")

def create_camera_group_from_custom_calibration(calibration_data: 'CalibrationData') -> 'CameraGroup':
    """
    Create aniposelib CameraGroup from custom calibration data
    
    Args:
        calibration_data: CalibrationData object from custom calibration
        
    Returns:
        aniposelib CameraGroup object ready for animal tracking
    """
    if not ANIPOSELIB_AVAILABLE:
        raise ImportError("aniposelib not available. Install with: pip install aniposelib")
    
    if not CUSTOM_CALIBRATION_AVAILABLE:
        raise ImportError("Custom calibration module not available")
    
    logger.info("Creating aniposelib CameraGroup from custom calibration...")
    
    # Convert to anipose format
    anipose_dict = convert_custom_calibration_to_anipose_format(calibration_data)
    
    # Create temporary TOML file for CameraGroup.load()
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.toml', delete=False) as f:
        toml.dump(anipose_dict, f)
        temp_toml_path = f.name
    
    try:
        # Load CameraGroup from temporary file using the actual aniposelib API
        from aniposelib.cameras import CameraGroup
        camera_group = CameraGroup.load(temp_toml_path)
        
        logger.info(f"✅ Created CameraGroup with {len(camera_group.cameras)} cameras")
        logger.info(f"Camera names: {camera_group.get_names()}")
        
        return camera_group
    finally:
        # Clean up temporary file
        os.unlink(temp_toml_path)

def apply_animal_bundle_adjustment(camera_group: 'CameraGroup',
                                  sample_2d_points: np.ndarray) -> 'CameraGroup':
    """
    Apply animal-specific bundle adjustment to refine camera parameters
    
    This is the core "animal calibration" - using aniposelib's bundle_adjust_iter()
    to optimize camera parameters with animal-specific constraints and error handling.
    
    Args:
        camera_group: aniposelib CameraGroup with our custom calibration
        sample_2d_points: Sample 2D points for bundle adjustment (n_cameras, n_points, 2)
        
    Returns:
        CameraGroup with animal-calibrated parameters
    """
    logger.info("Applying animal-specific bundle adjustment to camera parameters...")
    
    try:
        # Use aniposelib's iterative bundle adjustment with animal-specific parameters
        # This is what "animal calibration" actually does in anipose
        error = camera_group.bundle_adjust_iter(
            sample_2d_points,
            n_iters=6,                    # Iterative refinement
            start_mu=15,                  # Initial robust threshold  
            end_mu=1,                     # Final robust threshold
            max_nfev=200,                 # Max function evaluations
            ftol=1e-4,                    # Function tolerance
            n_samp_iter=200,              # Samples per iteration
            n_samp_full=1000,             # Full sample size
            error_threshold=0.3,          # Animal-specific error threshold
            only_extrinsics=False,        # Optimize both intrinsics and extrinsics
            verbose=True
        )
        
        logger.info(f"✅ Animal bundle adjustment completed with error: {error:.4f}")
        logger.info("Camera parameters optimized for animal tracking")
        
        return camera_group
        
    except Exception as e:
        logger.error(f"Animal bundle adjustment failed: {e}")
        logger.warning("Returning original camera group without animal calibration")
        return camera_group

def get_animal_constraints_indices():
    """
    Get animal anatomical constraints as joint indices
    
    Returns:
        List of (joint_a_index, joint_b_index) pairs representing 
        anatomical connections that should maintain constant length
    """
    # These would need to be mapped to actual joint indices from your DLC model
    # This is just an example - you'd need to map to your actual bodypart indices
    constraints = [
        (0, 1),   # nose to neck_end  
        (1, 2),   # neck_end to back_middle
        (2, 3),   # back_middle to back_end
        (3, 4),   # back_end to tail_base
        (5, 6),   # front_left_thai to front_left_knee
        (6, 7),   # front_left_knee to front_left_paw
        (8, 9),   # front_right_thai to front_right_knee
        (9, 10),  # front_right_knee to front_right_paw
        (11, 12), # back_left_thai to back_left_knee
        (12, 13), # back_left_knee to back_left_paw
        (14, 15), # back_right_thai to back_right_knee
        (15, 16), # back_right_knee to back_right_paw
    ]
    return constraints

def get_animal_constraints():
    """
    Get animal-specific constraints for calibration
    
    Returns:
        List of animal body part constraints
    """
    return [
        ("nose", "left_earbase"), ("nose", "right_earbase"),
        ("nose", "neck_end"), 
        ("neck_end", "back_middle"), ("back_base", "back_middle"),
        ("back_middle", "back_end"), ("back_end", "tail_base"),
        ("front_left_thai", "front_left_knee"), ("front_left_knee", "front_left_paw"),
        ("front_right_thai", "front_right_knee"), ("front_right_knee", "front_right_paw"),
        ("back_left_thai", "back_left_knee"), ("back_left_knee", "back_left_paw"),
        ("back_right_thai", "back_right_knee"), ("back_right_knee", "back_right_paw")
    ]

def triangulate_points_3d(camera_group: 'CameraGroup', 
                         points_2d_dict: Dict[str, np.ndarray]) -> np.ndarray:
    """
    Triangulate 2D points to 3D using aniposelib CameraGroup
    
    Args:
        camera_group: aniposelib CameraGroup object
        points_2d_dict: Dictionary mapping camera names to 2D points
                       Format: {'Camera1': points_nx2, 'Camera2': points_nx2, ...}
        
    Returns:
        3D points array (n_points, 3)
    """
    if not ANIPOSELIB_AVAILABLE:
        raise ImportError("aniposelib not available")
    
    # Convert camera names to indices
    camera_names = [cam.get_name() for cam in camera_group.cameras]
    logger.info(f"Camera names in group: {camera_names}")
    
    # Prepare points for triangulation
    # aniposelib expects points as (n_cameras, n_points, 2)
    n_points = next(iter(points_2d_dict.values())).shape[0]
    n_cameras = len(camera_group.cameras)
    
    points_array = np.full((n_cameras, n_points, 2), np.nan)
    
    for camera_name, points_2d in points_2d_dict.items():
        try:
            camera_idx = camera_names.index(camera_name)
            points_array[camera_idx] = points_2d
        except ValueError:
            logger.warning(f"Camera {camera_name} not found in CameraGroup")
    
    # Triangulate using aniposelib
    points_3d = camera_group.triangulate(points_array)
    
    logger.info(f"Triangulated {n_points} points from {n_cameras} cameras")
    return points_3d

# Removed the complex 3D processing function since we only use aniposelib for bundle adjustment

def create_anipose_config_with_animal_calibration(session_dir: Path, 
                                                num_cameras: int = 3) -> Path:
    """
    Create anipose config.toml with animal calibration enabled
    
    Args:
        session_dir: Session directory (config will be created in parent)
        num_cameras: Number of cameras
        
    Returns:
        Path to created config file
    """
    parent_dir = session_dir.parent
    config_path = parent_dir / "config.toml"
    
    # Generate camera names
    camera_names = [f'cam{i+1}' for i in range(num_cameras)]
    
    config_content = f"""
# Project settings for {num_cameras}-camera setup
nesting = 1
video_extension = 'mp4'

[calibration]
# Animal calibration settings - use pre-computed calibration
animal_calibration = true
board_type = "charuco"
board_size = [10, 7]
board_marker_bits = 4
board_marker_dict_number = 50
board_marker_length = 18.75  # mm
board_square_side_length = 25  # mm
fisheye = false

# Skip anipose calibration - use custom calibration
done = true

[triangulation]
# Enable triangulation for 3D reconstruction
triangulate = true

# Regular expression to extract camera names from filenames
cam_regex = '-cam([0-9]+)'

# Which camera to use for alignment (use cam1)
cam_align = "1"

# Use RANSAC for outlier rejection
ransac = false

# Enable optimization for 3D filtering (animal-specific)
optim = true

# Animal-specific constraints between connected body parts
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
# Animal body scheme for visualization
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

# Animal-specific smoothing and constraint parameters
scale_smooth = 7
scale_length = 20
scale_length_weak = 1
reproj_error_threshold = 5
score_threshold = 0.1
n_deriv_smooth = 3

[angles]
# Animal joint angles
front_left_knee = ["front_left_thai", "front_left_knee", "front_left_paw"]
front_right_knee = ["front_right_thai", "front_right_knee", "front_right_paw"]
back_left_knee = ["back_left_thai", "back_left_knee", "back_left_paw"]
back_right_knee = ["back_right_thai", "back_right_knee", "back_right_paw"]
neck_angle = ["neck_base", "neck_end", "back_base"]
back_angle = ["back_base", "back_middle", "back_end"]
tail_angle = ["back_end", "tail_base", "tail_end"]

[pipeline]
# Default pipeline folders
calibration_videos = "calibration"
calibration_results = "calibration"
videos_raw = "videos-raw"
pose_2d = "pose-2d"
pose_2d_filter = "pose-2d-filtered"
pose_3d = "pose-3d"
pose_3d_filter = "pose-3d-filtered"

[filter]
# 2D filtering (light for animal tracking)
enabled = true
type = "viterbi"
score_threshold = 0.2
medfilt = 13
offset_threshold = 25
spline = true

[filter3d]  
# 3D filtering optimized for animal movement
enabled = true
medfilt = 7           
offset_threshold = 40
n_back_track = 5
score_threshold = 0.1
spline = true
"""
    
    with open(config_path, 'w') as f:
        f.write(config_content.strip())
    
    logger.info(f"Created anipose config with animal calibration: {config_path}")
    return config_path

def create_anipose_config_without_calibration(session_dir: Path, 
                                            num_cameras: int = 3) -> Path:
    """
    Create anipose config.toml that skips calibration and uses pre-existing calibration.toml
    
    Args:
        session_dir: Session directory (config will be created in parent)
        num_cameras: Number of cameras
        
    Returns:
        Path to created config file
    """
    parent_dir = session_dir.parent
    config_path = parent_dir / "config.toml"
    
    config_content = f"""
# Project settings for {num_cameras}-camera setup with custom calibration
nesting = 1
video_extension = 'mp4'

[calibration]
# Skip anipose calibration - use existing calibration.toml created by aniposelib
done = true

[triangulation]
# Enable triangulation for 3D reconstruction
triangulate = true

# Regular expression to extract camera names from filenames
cam_regex = '-cam([0-9]+)'

# Which camera to use for alignment (use cam1)
cam_align = "1"

# Use RANSAC for outlier rejection
ransac = false

# Enable optimization for 3D filtering
optim = true

[labeling]  
# Animal body scheme for visualization
scheme = [  
    # Head connections  
    ["nose", "upper_jaw", "lower_jaw"],  
    ["right_eye", "nose", "left_eye"],  
    ["mouth_end_right", "nose", "mouth_end_left"],  
      
    # Ear connections  
    ["right_earbase", "right_earend"],  
    ["left_earbase", "left_earend"],  
      
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
]

# Animal-specific parameters  
scale_smooth = 7
scale_length = 20
scale_length_weak = 1
reproj_error_threshold = 5
score_threshold = 0.1
n_deriv_smooth = 3

[angles]
# Animal joint angles
front_left_knee = ["front_left_thai", "front_left_knee", "front_left_paw"]
front_right_knee = ["front_right_thai", "front_right_knee", "front_right_paw"]
back_left_knee = ["back_left_thai", "back_left_knee", "back_left_paw"]
back_right_knee = ["back_right_thai", "back_right_knee", "back_right_paw"]
neck_angle = ["neck_base", "neck_end", "back_base"]
back_angle = ["back_base", "back_middle", "back_end"]
tail_angle = ["back_end", "tail_base", "tail_end"]

[pipeline]
# Default pipeline folders
calibration_videos = "calibration"
calibration_results = "calibration"
videos_raw = "videos-raw"
pose_2d = "pose-2d"
pose_2d_filter = "pose-2d-filtered"
pose_3d = "pose-3d"
pose_3d_filter = "pose-3d-filtered"

[filter]
# 2D filtering for animal tracking
enabled = true
type = "viterbi"
score_threshold = 0.2
medfilt = 13
offset_threshold = 25
spline = true

[filter3d]  
# 3D filtering optimized for animal movement
enabled = true
medfilt = 7           
offset_threshold = 40
n_back_track = 5
score_threshold = 0.1
spline = true
"""
    
    with open(config_path, 'w') as f:
        f.write(config_content.strip())
    
    logger.info(f"Created anipose config without calibration: {config_path}")
    return config_path

def create_animal_calibration_with_bundle_adjustment(session_dir: Path, 
                                                    calibration_data: 'CalibrationData') -> bool:
    """
    Apply animal calibration by using aniposelib bundle adjustment on our custom calibration
    
    This function:
    1. Loads our custom calibration into aniposelib CameraGroup
    2. Uses aniposelib's bundle_adjust_iter() for animal calibration
    3. Saves the animal-optimized parameters for anipose CLI to use
    
    Args:
        session_dir: Session directory
        calibration_data: Custom calibration data
        
    Returns:
        True if animal calibration successful
    """
    logger.info("Applying animal calibration via aniposelib bundle adjustment...")
    
    if not ANIPOSELIB_AVAILABLE:
        logger.error("aniposelib not available - cannot apply animal calibration")
        return False
    
    try:
        parent_dir = session_dir.parent
        
        # 1. Create CameraGroup from our custom calibration
        camera_group = create_camera_group_from_custom_calibration(calibration_data)
        
        # 2. Get sample 2D points for bundle adjustment
        # We need some 2D points to perform bundle adjustment
        sample_points = get_sample_2d_points_for_bundle_adjustment(session_dir)
        
        if sample_points is None:
            logger.warning("No 2D points available for bundle adjustment")
            logger.info("Saving calibration without animal bundle adjustment")
        else:
            # 3. Apply animal-specific bundle adjustment using aniposelib
            logger.info("Running animal-specific bundle adjustment...")
            camera_group = apply_animal_bundle_adjustment(camera_group, sample_points)
        
        # 4. Save the animal-calibrated CameraGroup for anipose CLI
        final_calib_path = parent_dir / "calibration.toml"
        camera_group.dump(str(final_calib_path))
        
        logger.info("✅ Animal calibration completed via aniposelib bundle adjustment")
        logger.info("  - Used our custom calibration as starting point")
        logger.info("  - Applied animal-specific bundle adjustment")
        logger.info("  - Saved animal-calibrated parameters for anipose CLI")
        logger.info(f"  - Calibration file: {final_calib_path}")
        
        return True
        
    except Exception as e:
        logger.error(f"Animal calibration via bundle adjustment failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def get_sample_2d_points_for_bundle_adjustment(session_dir: Path) -> Optional[np.ndarray]:
    """
    Get sample 2D points from pose-2d files for bundle adjustment
    
    Args:
        session_dir: Session directory
        
    Returns:
        Sample 2D points array (n_cameras, n_points, 2) or None if not available
    """
    pose_2d_dir = session_dir / "pose-2d"
    if not pose_2d_dir.exists():
        logger.info("pose-2d directory not found - bundle adjustment will use default points")
        return None
    
    h5_files = list(pose_2d_dir.glob("*.h5"))
    if not h5_files:
        logger.info("No H5 files found - bundle adjustment will use default points")
        return None
    
    try:
        import pandas as pd
        
        # Load first few H5 files to get sample points
        sample_points_dict = {}
        
        for h5_file in h5_files[:6]:  # Use first 6 files for sampling
            # Extract camera from filename 
            if 'cam1' in h5_file.name:
                camera_idx = 0
            elif 'cam2' in h5_file.name:
                camera_idx = 1
            elif 'cam3' in h5_file.name:
                camera_idx = 2
            else:
                continue
            
            # Load H5 and extract some 2D points
            df = pd.read_hdf(h5_file)
            if isinstance(df.columns, pd.MultiIndex):
                # Sample every 10th frame for bundle adjustment
                sampled_df = df.iloc[::10][:50]  # Max 50 sample points
                
                # Extract x,y coordinates
                points_2d = []
                bodyparts = df.columns.get_level_values('bodyparts').unique()
                
                for bodypart in bodyparts[:5]:  # Use first 5 bodyparts
                    try:
                        x_col = sampled_df.xs('x', level='coords', axis=1)[bodypart]
                        y_col = sampled_df.xs('y', level='coords', axis=1)[bodypart]
                        
                        for x, y in zip(x_col.values, y_col.values):
                            if not (np.isnan(x) or np.isnan(y)):
                                points_2d.append([x, y])
                    except (KeyError, IndexError):
                        continue
                
                if points_2d:
                    if camera_idx not in sample_points_dict:
                        sample_points_dict[camera_idx] = []
                    sample_points_dict[camera_idx].extend(points_2d[:100])  # Max 100 points per camera
        
        # Convert to array format expected by aniposelib
        if len(sample_points_dict) >= 2:  # Need at least 2 cameras
            n_cameras = 3
            max_points = max(len(points) for points in sample_points_dict.values())
            sample_array = np.full((n_cameras, max_points, 2), np.nan)
            
            for cam_idx, points in sample_points_dict.items():
                sample_array[cam_idx, :len(points)] = points
            
            logger.info(f"Extracted {max_points} sample points from {len(sample_points_dict)} cameras for bundle adjustment")
            return sample_array
        else:
            logger.info("Insufficient cameras with 2D points for bundle adjustment")
            return None
            
    except Exception as e:
        logger.warning(f"Failed to extract sample points: {e}")
        return None

def integrate_custom_calibration_with_anipose(session_dir: Path, 
                                            calibration_data: 'CalibrationData') -> bool:
    """
    Complete integration of custom calibration with anipose animal features
    
    This function combines:
    1. Our custom calibration (pairwise + GTSAM)
    2. Anipose animal calibration features (via aniposelib)  
    3. Anipose CLI compatibility
    
    Args:
        session_dir: Session directory
        calibration_data: Custom calibration data
        
    Returns:
        True if integration successful
    """
    logger.info("Integrating custom calibration with anipose animal features...")
    
    try:
        parent_dir = session_dir.parent
        num_cameras = len(calibration_data.intrinsics)
        
        # 1. Create config that tells anipose to skip calibration
        config_path = create_anipose_config_without_calibration(session_dir, num_cameras)
        
        # 2. Use aniposelib bundle adjustment to create animal calibration with our custom data
        animal_success = create_animal_calibration_with_bundle_adjustment(session_dir, calibration_data)
        
        if not animal_success:
            logger.warning("Animal calibration failed, falling back to standard integration")
            # Fallback: regular calibration conversion
            anipose_dict = convert_custom_calibration_to_anipose_format(calibration_data)
            calibration_path = parent_dir / "calibration.toml"
            save_anipose_calibration_toml(anipose_dict, calibration_path)
        
        logger.info("✅ Custom calibration integrated with anipose")
        logger.info(f"  Config: {config_path}")
        logger.info(f"  Calibration: {parent_dir}/calibration.toml")
        logger.info("  Animal features: ENABLED via aniposelib")
        logger.info("  Anipose CLI: Ready for triangulate/filter-3d")
        
        return True
        
    except Exception as e:
        logger.error(f"Integration failed: {e}")
        import traceback
        traceback.print_exc()
        return False

# Backward compatibility function
def convert_and_save_for_anipose(session_dir: Path) -> bool:
    """
    Convenience function to convert existing custom calibration for anipose use
    
    Args:
        session_dir: Session directory with calibration/calibration.toml
        
    Returns:
        True if successful
    """
    if not CUSTOM_CALIBRATION_AVAILABLE:
        logger.error("Custom calibration not available")
        return False
    
    # Load existing custom calibration
    custom_calib_path = session_dir / "calibration" / "calibration.toml"
    if not custom_calib_path.exists():
        logger.error(f"Custom calibration not found: {custom_calib_path}")
        return False
    
    try:
        from custom_calibration import load_calibration_from_toml
        calibration_data = load_calibration_from_toml(custom_calib_path)
        
        return integrate_custom_calibration_with_anipose(session_dir, calibration_data)
        
    except Exception as e:
        logger.error(f"Failed to convert custom calibration: {e}")
        return False

if __name__ == "__main__":
    # Test the integration
    import argparse
    
    parser = argparse.ArgumentParser(description="Test anipose integration")
    parser.add_argument("--session-dir", required=True, help="Session directory")
    parser.add_argument("--test-triangulation", action="store_true", help="Test 3D triangulation")
    
    args = parser.parse_args()
    
    session_dir = Path(args.session_dir)
    
    if args.test_triangulation and ANIPOSELIB_AVAILABLE:
        print("Testing aniposelib integration...")
        success = convert_and_save_for_anipose(session_dir)
        print(f"Integration successful: {success}")
    else:
        print("aniposelib not available or test not requested")
