#!/usr/bin/env python3
"""
Custom Calibration Runner

This script performs multi-camera calibration using the custom calibration system.
It replaces the 'anipose calibrate' command and supports flexible multi-camera setups.

Usage:
    python run_custom_calibration.py --session-dir /path/to/session
    python run_custom_calibration.py --session-dir /path/to/session --cameras cam1,cam2,cam3

Author: Calibration Runner
"""

import os
import sys
import cv2
import argparse
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple, TYPE_CHECKING
import logging

if TYPE_CHECKING:
    from frame_level_optimizer import CameraFrameAnalysis

# Import our custom calibration modules
try:
    from custom_calibration import (
        MultiCameraCalibrator, 
        ChArUcoBoard, 
        CalibrationData
    )
    from calibration_integration import (
        load_calibration_params,
        create_multi_camera_config_toml
    )
    CALIBRATION_AVAILABLE = True
except ImportError as e:
    CALIBRATION_AVAILABLE = False
    print(f"Error importing calibration modules: {e}")
    print("Please ensure custom_calibration.py and calibration_integration.py are available")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def extract_frames_from_video(video_path: Path, output_dir: Path, 
                            max_frames: int = 200, skip_frames: int = 5) -> List[Path]:
    """
    Extract frames from calibration video for processing
    
    Args:
        video_path: Path to calibration video
        output_dir: Directory to save extracted frames
        max_frames: Maximum number of frames to extract (None = all frames)
        skip_frames: Number of frames to skip between extractions
        
    Returns:
        List of paths to extracted frame images
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    
    # Get total frame count
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    logger.info(f"Extracting frames from {video_path.name} (total: {total_frames} frames)...")
    
    frame_paths = []
    frame_count = 0
    extracted_count = 0
    
    # If max_frames is None or very large, extract all frames
    if max_frames is None or max_frames >= total_frames:
        max_frames = total_frames
        skip_frames = 1  # Extract every frame
        logger.info("Extracting ALL frames to ensure ChArUco board detection")
    
    while extracted_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Skip frames for better temporal distribution (unless extracting all)
        if frame_count % skip_frames == 0:
            frame_filename = f"frame_{frame_count:06d}.jpg"  # Use original frame number
            frame_path = output_dir / frame_filename
            
            cv2.imwrite(str(frame_path), frame)
            frame_paths.append(frame_path)
            extracted_count += 1
            
            if extracted_count % 100 == 0:
                logger.info(f"  Extracted {extracted_count} frames...")
        
        frame_count += 1
    
    cap.release()
    logger.info(f"Extracted {len(frame_paths)} frames from {video_path.name}")
    
    return frame_paths

def extract_frames_from_video_smart(video_path: Path, output_dir: Path) -> List[Path]:
    """
    Smart frame extraction that finds where ChArUco board is visible
    
    Based on diagnosis, the ChArUco board appears in later parts of videos.
    This function extracts frames from the middle and end portions where 
    boards are typically visible.
    
    Args:
        video_path: Path to calibration video
        output_dir: Directory to save extracted frames
        
    Returns:
        List of paths to extracted frame images
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    logger.info(f"Smart extraction from {video_path.name} (total: {total_frames} frames)")
    
    # Extract from different sections where board is likely visible
    # Skip first 25% where board might not be present
    start_frame = total_frames // 4
    
    # Extract from middle 50% and last 25% of video
    frame_ranges = [
        (start_frame, start_frame + total_frames//4, 5),           # Middle section (every 5th frame)
        (start_frame + total_frames//4, 3*total_frames//4, 10),   # Later middle (every 10th frame)  
        (3*total_frames//4, total_frames, 15),                    # End section (every 15th frame)
    ]
    
    frame_paths = []
    extracted_count = 0
    
    for start, end, skip in frame_ranges:
        logger.info(f"  Extracting from frames {start}-{end} (every {skip}th frame)")
        
        for frame_num in range(start, end, skip):
            if extracted_count >= 200:  # Limit total extractions
                break
                
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ret, frame = cap.read()
            if not ret:
                continue
            
            frame_filename = f"frame_{frame_num:06d}.jpg"
            frame_path = output_dir / frame_filename
            
            cv2.imwrite(str(frame_path), frame)
            frame_paths.append(frame_path)
            extracted_count += 1
    
    cap.release()
    logger.info(f"Smart extraction completed: {len(frame_paths)} frames from board-visible sections")
    
    return frame_paths

def extract_all_frames_if_no_detection(video_path: Path, output_dir: Path) -> List[Path]:
    """
    Extract ALL frames from video when initial detection fails
    
    Args:
        video_path: Path to calibration video
        output_dir: Directory to save extracted frames
        
    Returns:
        List of paths to extracted frame images
    """
    logger.info("No ChArUco detection in sampled frames - extracting ALL frames...")
    return extract_frames_from_video(video_path, output_dir, max_frames=None, skip_frames=1)

def collect_calibration_images_frame_optimized(calibration_dir: Path, 
                                              camera_ids: List[str]) -> Tuple[Dict[str, List[np.ndarray]], Dict[str, 'CameraFrameAnalysis']]:
    """
    Collect calibration images using frame-by-frame optimization with caching
    
    This function:
    1. Identifies ALL individual frames with good ChArUco detection for each camera
    2. Caches detection results to avoid redundant ChArUco detection
    3. Extracts optimal frames for intrinsic calibration (per camera)
    4. Stores synchronized frame information for pairwise calibration
    
    Args:
        calibration_dir: Directory containing calibration data
        camera_ids: List of camera IDs to process
        
    Returns:
        Tuple of:
        - Dictionary mapping camera ID to list of optimal calibration images
        - Dictionary mapping camera ID to CameraFrameAnalysis (with cached detection results)
    """
    logger.info("🎯 COLLECTING CALIBRATION IMAGES WITH FRAME-LEVEL OPTIMIZATION")
    logger.info("="*70)
    
    from custom_calibration import ChArUcoBoard
    
    # Try to import frame optimizer (required for this function)
    try:
        from frame_level_optimizer import optimize_calibration_frame_by_frame, extract_frames_by_numbers
        FRAME_OPTIMIZER_AVAILABLE = True
        logger.info("Frame optimizer imported successfully")
    except ImportError as e:
        logger.error(f"Frame optimizer not available: {e}")
        raise e  # Must have frame optimizer for this function
    
    # Find calibration videos
    calibration_videos = {}
    for camera_id in camera_ids:
        video_patterns = [
            f"calib-cam{camera_id}.mp4",
            f"cam{camera_id}-calib.mp4", 
            f"calibration-cam{camera_id}.mp4",
        ]
        
        for pattern in video_patterns:
            video_path = calibration_dir / pattern
            if video_path.exists():
                calibration_videos[camera_id] = video_path
                logger.info(f"Found calibration video for camera {camera_id}: {video_path.name}")
                break
        
        if camera_id not in calibration_videos:
            logger.error(f"No calibration video found for camera {camera_id}")
    
    if len(calibration_videos) < len(camera_ids):
        logger.error("Missing calibration videos for some cameras")
        return {}
    
    # Run frame-by-frame optimization
    board_config = ChArUcoBoard()
    logger.info("Starting frame-by-frame optimization...")
    
    if not FRAME_OPTIMIZER_AVAILABLE:
        logger.error("Frame optimizer not available - cannot proceed")
        raise ImportError("Frame optimizer required but not available")
    
    try:
        logger.info("Calling optimize_calibration_frame_by_frame...")
        result = optimize_calibration_frame_by_frame(calibration_videos, board_config)
        
        logger.info(f"Frame optimization returned: {type(result)}")
        if isinstance(result, tuple) and len(result) == 3:
            camera_images, pairwise_images, camera_analyses = result
            logger.info(f"Successfully unpacked: camera_images={len(camera_images)}, pairwise_images={len(pairwise_images)}, analyses={len(camera_analyses)}")
        else:
            logger.error(f"Unexpected return format from optimize_calibration_frame_by_frame: {type(result)}")
            if hasattr(result, '__len__'):
                logger.error(f"Result length: {len(result)}")
            raise ValueError("Incorrect return format - expected 3-tuple with cached detection results")
    except Exception as e:
        logger.error(f"Frame optimization failed: {e}")
        import traceback
        traceback.print_exc()
        raise
    
    # Store pairwise images for later use in pairwise calibration
    setattr(collect_calibration_images_frame_optimized, '_pairwise_images', pairwise_images)
    
    logger.info(f"✅ FRAME-LEVEL OPTIMIZATION COMPLETED")
    logger.info(f"   Cameras ready for intrinsic calibration: {list(camera_images.keys())}")
    logger.info(f"   Pairs ready for pairwise calibration: {list(pairwise_images.keys())}")
    logger.info("   ⚡ ChArUco detection results cached - no redundant detection needed!")
    
    # Return both images and frame analyses (with cached detection results)
    return camera_images, camera_analyses

def collect_calibration_images_simple(calibration_dir: Path, 
                                     camera_ids: List[str]) -> Dict[str, List[np.ndarray]]:
    """
    Simple calibration image collection (fallback method)
    """
    camera_images = {}
    
    for camera_id in camera_ids:
        logger.info(f"Collecting calibration data for camera {camera_id} (simple method)...")
        
        # Try different file naming patterns
        video_patterns = [
            f"calib-cam{camera_id}.mp4",
            f"cam{camera_id}-calib.mp4",
            f"calibration-cam{camera_id}.mp4",
            f"camera{camera_id}.mp4",
        ]
        
        # Try different directory patterns
        dir_patterns = [
            calibration_dir / f"cam{camera_id}",
            calibration_dir / f"camera{camera_id}",
            calibration_dir / f"Camera{camera_id}",
        ]
        
        images = []
        
        # First, try to find calibration video
        video_found = False
        for pattern in video_patterns:
            video_path = calibration_dir / pattern
            if video_path.exists():
                logger.info(f"Found calibration video: {video_path}")
                
                # Create temporary directory for extracted frames
                temp_dir = calibration_dir / f"temp_frames_cam{camera_id}"
                frame_paths = extract_frames_from_video_smart(video_path, temp_dir)
                images = load_images_from_paths(frame_paths)
                
                # Clean up temporary frames
                import shutil
                shutil.rmtree(temp_dir, ignore_errors=True)
                
                video_found = True
                break
        
        # If no video found, try to find image directory
        if not video_found:
            for dir_path in dir_patterns:
                if dir_path.exists() and dir_path.is_dir():
                    logger.info(f"Found calibration directory: {dir_path}")
                    
                    # Load images from directory
                    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
                    image_paths = []
                    
                    for ext in image_extensions:
                        image_paths.extend(dir_path.glob(ext))
                        image_paths.extend(dir_path.glob(ext.upper()))
                    
                    if image_paths:
                        images = load_images_from_paths(sorted(image_paths))
                        break
        
        # If still no images found, try the main calibration directory
        if not images:
            logger.warning(f"No calibration video or directory found for camera {camera_id}")
            logger.info(f"Trying to find images with camera {camera_id} in filename...")
            
            image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
            for ext in image_extensions:
                pattern_paths = calibration_dir.glob(f"*cam{camera_id}*{ext}")
                pattern_paths = list(pattern_paths) + list(calibration_dir.glob(f"*camera{camera_id}*{ext}"))
                
                if pattern_paths:
                    images = load_images_from_paths(sorted(pattern_paths))
                    break
        
        if images:
            camera_images[camera_id] = images
            logger.info(f"Collected {len(images)} calibration images for camera {camera_id}")
        else:
            logger.error(f"No calibration images found for camera {camera_id}")
    
    return camera_images

def collect_calibration_images(calibration_dir: Path, 
                             camera_ids: List[str]) -> Tuple[Dict[str, List[np.ndarray]], Dict[str, 'CameraFrameAnalysis']]:
    """
    Collect calibration images using frame-level optimization directly
    
    Args:
        calibration_dir: Directory containing calibration data
        camera_ids: List of camera IDs to process
        
    Returns:
        Tuple of:
        - Dictionary mapping camera ID to list of calibration images
        - Dictionary mapping camera ID to CameraFrameAnalysis (with cached detection results)
    """
    # Use frame-level optimization directly (no fallback)
    logger.info("Starting frame-level optimization for calibration image collection")
    
    try:
        result = collect_calibration_images_frame_optimized(calibration_dir, camera_ids)
        logger.info("✅ Frame-level optimization completed successfully")
        return result
    except Exception as e:
        logger.error(f"❌ Frame-level optimization failed: {e}")
        import traceback
        traceback.print_exc()
        
        logger.info("🔄 Falling back to simple collection method...")
        camera_images = collect_calibration_images_simple(calibration_dir, camera_ids)
        # Return empty analyses dict for fallback (no caching available)
        return camera_images, {}

def load_images_from_paths(image_paths: List[Path]) -> List[np.ndarray]:
    """
    Load images from file paths
    
    Args:
        image_paths: List of image file paths
        
    Returns:
        List of loaded images as numpy arrays
    """
    images = []
    for path in image_paths:
        image = cv2.imread(str(path))
        if image is not None:
            images.append(image)
        else:
            logger.warning(f"Could not load image: {path}")
    
    logger.info(f"Loaded {len(images)} images")
    return images

def discover_cameras(calibration_dir: Path) -> List[str]:
    """
    Automatically discover cameras from calibration directory structure
    
    Args:
        calibration_dir: Path to calibration directory
        
    Returns:
        List of discovered camera IDs
    """
    camera_ids = set()
    
    # Look for calibration videos with camera names
    for video_file in calibration_dir.glob("*.mp4"):
        # Try to extract camera ID from filename
        # Patterns: calib-cam1.mp4, cam1-calib.mp4, etc.
        filename = video_file.stem.lower()
        
        import re
        patterns = [
            r'cam(\d+)',
            r'camera(\d+)',
            r'c(\d+)',
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, filename)
            if matches:
                camera_ids.add(matches[0])
                break
    
    # Also look for subdirectories with camera names
    for subdir in calibration_dir.iterdir():
        if subdir.is_dir():
            dirname = subdir.name.lower()
            if dirname.startswith('cam') or dirname.startswith('camera'):
                # Extract number from directory name
                import re
                matches = re.findall(r'(\d+)', dirname)
                if matches:
                    camera_ids.add(matches[0])
    
    camera_list = sorted(list(camera_ids))
    logger.info(f"Discovered cameras: {camera_list}")
    
    return camera_list

# Removed duplicate function - using frame optimization version at line 358

def run_calibration(session_dir: Path, 
                   camera_ids: Optional[List[str]] = None,
                   reference_camera: Optional[str] = None,
                   board_config: Optional[Dict] = None) -> bool:
    """
    Run the complete calibration process
    
    Args:
        session_dir: Session directory containing calibration subdirectory
        camera_ids: List of camera IDs (auto-discover if None)
        reference_camera: Reference camera ID (first camera if None)
        board_config: Custom board configuration (default if None)
        
    Returns:
        True if calibration succeeded, False otherwise
    """
    if not CALIBRATION_AVAILABLE:
        logger.error("Calibration modules not available")
        return False
    
    # Locate calibration directory
    calibration_dir = session_dir / "calibration"
    if not calibration_dir.exists():
        logger.error(f"Calibration directory not found: {calibration_dir}")
        return False
    
    # Auto-discover cameras if not specified
    if camera_ids is None:
        camera_ids = discover_cameras(calibration_dir)
        if not camera_ids:
            logger.error("No cameras discovered in calibration directory")
            return False
    
    # Set reference camera
    if reference_camera is None:
        reference_camera = camera_ids[0]
    elif reference_camera not in camera_ids:
        logger.error(f"Reference camera {reference_camera} not in camera list {camera_ids}")
        return False
    
    logger.info(f"Starting calibration for cameras: {camera_ids}")
    logger.info(f"Reference camera: {reference_camera}")
    
    # Configure ChArUco board
    if board_config is None:
        board_config = ChArUcoBoard()
    else:
        board_config = ChArUcoBoard(**board_config)
    
    # Collect calibration images with frame-level optimization (now returns cached detection results too)
    camera_images, camera_analyses = collect_calibration_images(calibration_dir, camera_ids)
    
    if not camera_images:
        logger.error("No calibration images collected for any camera")
        return False
    
    # Get pairwise images if frame optimization was used
    pairwise_images = getattr(collect_calibration_images_frame_optimized, '_pairwise_images', None)
    
    # Verify all cameras have images
    missing_cameras = [cam_id for cam_id in camera_ids if cam_id not in camera_images]
    if missing_cameras:
        logger.error(f"Missing calibration images for cameras: {missing_cameras}")
        return False
    
    # Check minimum number of images per camera
    min_images = 5
    insufficient_cameras = [
        cam_id for cam_id, images in camera_images.items() 
        if len(images) < min_images
    ]
    
    if insufficient_cameras:
        logger.error(f"Insufficient images (need at least {min_images}) for cameras: {insufficient_cameras}")
        for cam_id in insufficient_cameras:
            logger.error(f"  Camera {cam_id}: {len(camera_images[cam_id])} images")
        return False
    
    # Initialize calibrator
    calibrator = MultiCameraCalibrator(board_config)
    
    try:
        # Perform frame-optimized calibration
        logger.info("Starting frame-optimized multi-camera calibration...")
        logger.info("   Intrinsic calibration: using optimal frames per camera")
        if pairwise_images:
            logger.info("   Pairwise calibration: using synchronized frames per pair")
            logger.info(f"   Available pairs: {list(pairwise_images.keys())}")
        else:
            logger.info("   Pairwise calibration: using all available images (fallback)")
        
        calibration_data = calibrator.calibrate_all_cameras(
            camera_images, 
            reference_camera=reference_camera,
            pairwise_images=pairwise_images,  # Pass synchronized frames for pairwise calibration
            camera_analyses=camera_analyses   # Pass cached detection results (avoids re-detection)
        )
        
        # Save calibration results
        output_path = calibration_dir / "calibration.toml"
        calibration_data.save_to_file(output_path)
        
        # Update session config to use custom calibration
        create_multi_camera_config_toml(str(session_dir), len(camera_ids))
        
        logger.info(f"Calibration completed successfully!")
        logger.info(f"Results saved to: {output_path}")
        
        # Print calibration summary
        print("\n" + "="*60)
        print("CALIBRATION SUMMARY")
        print("="*60)
        print(f"Cameras calibrated: {len(camera_images)}")
        print(f"Reference camera: {reference_camera}")
        print(f"Board configuration:")
        print(f"  - Size: {board_config.squares_x}x{board_config.squares_y}")
        print(f"  - Square length: {board_config.square_length}mm")
        print(f"  - Marker length: {board_config.marker_length}mm")
        print()
        
        # Print per-camera summary
        for camera_id, intrinsic in calibration_data.intrinsics.items():
            print(f"Camera {camera_id}:")
            print(f"  - Images used: {len(camera_images[camera_id])}")
            print(f"  - Image size: {intrinsic.image_size}")
            print(f"  - Calibration error: {intrinsic.calibration_error:.4f}")
            
            if camera_id in calibration_data.extrinsics:
                extrinsic = calibration_data.extrinsics[camera_id]
                print(f"  - Extrinsic error: {extrinsic.calibration_error:.4f}")
            print()
        
        print(f"Calibration saved to: {output_path}")
        print("="*60)
        
        return True
        
    except Exception as e:
        logger.error(f"Calibration failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def deploy_to_dcc():
    """Quick deployment function to sync to DCC cluster"""
    try:
        script_dir = Path(__file__).parent.parent
        deploy_script = script_dir / "deploy_custom_calib.py"
        
        if deploy_script.exists():
            logger.info("Deploying custom_calib to DCC cluster...")
            import subprocess
            result = subprocess.run([
                sys.executable, str(deploy_script), "--push", "--backup"
            ], check=True)
            logger.info("✅ Deployment completed!")
        else:
            logger.error(f"Deploy script not found: {deploy_script}")
            
    except Exception as e:
        logger.error(f"Deployment failed: {e}")

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Custom Multi-Camera Calibration")
    parser.add_argument("--session-dir", type=str, default=None,
                       help="Path to session directory")
    parser.add_argument("--cameras", type=str, default=None,
                       help="Comma-separated list of camera IDs (e.g., 1,2,3)")
    parser.add_argument("--reference-camera", type=str, default=None,
                       help="Reference camera ID")
    parser.add_argument("--board-squares-x", type=int, default=10,
                       help="Number of squares in X direction")
    parser.add_argument("--board-squares-y", type=int, default=7,
                       help="Number of squares in Y direction")
    parser.add_argument("--square-length", type=float, default=25.0,
                       help="Square length in mm")
    parser.add_argument("--marker-length", type=float, default=18.75,
                       help="Marker length in mm")
    parser.add_argument("--deploy", action="store_true",
                       help="Deploy custom_calib to DCC cluster")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Enable verbose logging")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Handle deployment action
    if args.deploy:
        logger.info("Deploying custom_calib to DCC cluster...")
        deploy_to_dcc()
        return 0
    
    # Check if session directory is provided for calibration
    if not args.session_dir:
        logger.error("--session-dir is required for calibration (or use --deploy to sync to DCC)")
        return 1
    
    # Parse session directory
    session_dir = Path(args.session_dir)
    if not session_dir.exists():
        logger.error(f"Session directory not found: {session_dir}")
        return 1
    
    # Parse camera list
    camera_ids = None
    if args.cameras:
        camera_ids = [cam.strip() for cam in args.cameras.split(',')]
    
    # Configure board
    board_config = {
        'squares_x': args.board_squares_x,
        'squares_y': args.board_squares_y,
        'square_length': args.square_length,
        'marker_length': args.marker_length
    }
    
    # Run calibration
    success = run_calibration(
        session_dir=session_dir,
        camera_ids=camera_ids,
        reference_camera=args.reference_camera,
        board_config=board_config
    )
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())
