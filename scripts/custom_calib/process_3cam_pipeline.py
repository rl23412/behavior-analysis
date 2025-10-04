#!/usr/bin/env python3
"""
3-Camera Pipeline with Custom Calibration

This script processes 3-camera setups using the custom calibration system.
It assumes videos are already synchronized and focuses on:
1. Custom pairwise calibration for 3 cameras
2. Global optimization of camera poses
3. 3D triangulation and filtering

Usage:
    python process_3cam_pipeline.py --session-dir /path/to/session
    python process_3cam_pipeline.py --session-dir /path/to/session --skip-dlc

Author: 3-Camera Pipeline
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path
from typing import List, Optional, Dict
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import custom calibration modules
try:
    from calibration_integration import (
        create_multi_camera_config_toml,
        load_calibration_params
    )
    from run_custom_calibration import run_calibration
    CUSTOM_CALIBRATION_AVAILABLE = True
except ImportError as e:
    CUSTOM_CALIBRATION_AVAILABLE = False
    logger.error(f"Custom calibration modules not available: {e}")

def print_header(title):
    """Print a formatted header"""
    print("\n" + "="*80)
    print(f"--- {title.upper()} ---")
    print("="*80)

def run_command(command, working_dir=None, check=True):
    """Execute a shell command with logging"""
    command_str = ' '.join(map(str, command))
    logger.info(f"RUNNING: {command_str}")
    if working_dir:
        logger.info(f"In directory: {working_dir}")
    
    try:
        process = subprocess.run(
            command, check=check, capture_output=True, text=True, cwd=working_dir
        )
        if process.stdout.strip():
            logger.info(f"STDOUT: {process.stdout.strip()}")
        if process.stderr.strip():
            logger.info(f"STDERR: {process.stderr.strip()}")
        logger.info("...SUCCESS.")
        return process
    except subprocess.CalledProcessError as e:
        logger.error(f"Command failed with exit code {e.returncode}")
        logger.error(f"  Command: {' '.join(e.cmd)}")
        logger.error(f"  Stderr: {e.stderr.strip()}")
        logger.error(f"  Stdout: {e.stdout.strip()}")
        raise

def setup_session_structure(session_dir: Path) -> bool:
    """
    Set up the directory structure for 3-camera processing
    
    Args:
        session_dir: Path to session directory
        
    Returns:
        True if setup successful
    """
    logger.info(f"Setting up session structure in {session_dir}")
    
    # Create main session directory
    session_dir.mkdir(parents=True, exist_ok=True)
    
    # Create required subdirectories
    subdirs = [
        "calibration",
        "videos-raw",
        "pose-2d",
        "pose-2d-filtered", 
        "pose-3d",
        "pose-3d-filtered"
    ]
    
    for subdir in subdirs:
        (session_dir / subdir).mkdir(exist_ok=True)
    
    logger.info("Session directory structure created successfully")
    return True

def verify_3camera_setup(session_dir: Path) -> Dict[str, List[str]]:
    """
    Verify that we have 3-camera setup with calibration and videos
    
    Args:
        session_dir: Path to session directory
        
    Returns:
        Dictionary with found files
    """
    logger.info("Verifying 3-camera setup...")
    
    found_files = {
        'calibration': [],
        'videos': []
    }
    
    # Check for calibration videos/images
    calibration_dir = session_dir / "calibration"
    if calibration_dir.exists():
        for cam_num in ['1', '2', '3']:
            # Look for different calibration file patterns
            patterns = [
                f"calib-cam{cam_num}.mp4",
                f"calibration-cam{cam_num}.mp4", 
                f"cam{cam_num}-calib.mp4"
            ]
            
            found = False
            for pattern in patterns:
                calib_file = calibration_dir / pattern
                if calib_file.exists():
                    found_files['calibration'].append(str(calib_file))
                    found = True
                    break
            
            if not found:
                logger.warning(f"No calibration file found for camera {cam_num}")
    
    # Check for experimental videos
    videos_dir = session_dir / "videos-raw"
    if videos_dir.exists():
        for video_file in videos_dir.glob("*.mp4"):
            if any(f"cam{i}" in video_file.name for i in ['1', '2', '3']):
                found_files['videos'].append(str(video_file))
    
    logger.info(f"Found {len(found_files['calibration'])} calibration files")
    logger.info(f"Found {len(found_files['videos'])} video files")
    
    # Check if we have all 3 cameras for calibration
    calib_cameras = set()
    for calib_file in found_files['calibration']:
        for cam_num in ['1', '2', '3']:
            if f"cam{cam_num}" in calib_file:
                calib_cameras.add(cam_num)
    
    if len(calib_cameras) < 3:
        logger.warning(f"Only found calibration for cameras: {sorted(calib_cameras)}")
        logger.warning("Need calibration files for all 3 cameras")
    else:
        logger.info("✅ All 3 cameras have calibration files")
    
    return found_files

def run_3camera_calibration(session_dir: Path, board_config: Optional[Dict] = None) -> bool:
    """
    Run custom 3-camera calibration with pairwise approach
    
    Args:
        session_dir: Path to session directory
        board_config: Optional custom board configuration
        
    Returns:
        True if calibration successful
    """
    if not CUSTOM_CALIBRATION_AVAILABLE:
        logger.error("Custom calibration system not available")
        return False
    
    print_header("Running 3-Camera Custom Calibration")
    
    # Camera configuration for 3 cameras
    camera_ids = ['1', '2', '3']
    reference_camera = '1'  # Use camera 1 as reference
    
    logger.info("3-Camera Calibration Configuration:")
    logger.info(f"  Camera IDs: {camera_ids}")
    logger.info(f"  Reference camera: {reference_camera}")
    logger.info("  Calibration approach: Pairwise with global optimization")
    logger.info("  Expected pairs: (1,2), (2,3), (1,3)")
    
    # Run the calibration
    try:
        success = run_calibration(
            session_dir=session_dir,
            camera_ids=camera_ids,
            reference_camera=reference_camera,
            board_config=board_config
        )
        
        if success:
            logger.info("✅ 3-Camera calibration completed successfully")
            
            # Verify calibration file was created
            calibration_file = session_dir / "calibration" / "calibration.toml"
            if calibration_file.exists():
                # Load and verify calibration
                calib_params = load_calibration_params(calibration_file)
                logger.info(f"Calibration verified: {len(calib_params)} cameras calibrated")
                logger.info(f"Camera names: {list(calib_params.keys())}")
                return True
            else:
                logger.error("Calibration file not found after calibration")
                return False
        else:
            logger.error("3-Camera calibration failed")
            return False
            
    except Exception as e:
        logger.error(f"Error during 3-camera calibration: {e}")
        import traceback
        traceback.print_exc()
        return False

def run_dlc_inference(session_dir: Path) -> bool:
    """
    Run DeepLabCut inference on 3-camera videos (processing one by one)
    
    Args:
        session_dir: Path to session directory
        
    Returns:
        True if DLC inference successful
    """
    print_header("Running DeepLabCut Inference (Individual Video Processing)")
    
    # Find all video files
    videos_dir = session_dir / "videos-raw"
    video_files = list(videos_dir.glob("*.mp4"))
    
    if not video_files:
        logger.error("No video files found for DLC inference")
        return False
    
    logger.info(f"Found {len(video_files)} videos for individual DLC processing")
    
    # Create pose-2d directory
    pose_2d_dir = session_dir / "pose-2d"
    pose_2d_dir.mkdir(exist_ok=True)
    
    try:
        import deeplabcut
        
        videotype = '.mp4'
        
        # Process each video individually
        successful_videos = 0
        failed_videos = []
        
        for i, video_file in enumerate(video_files, 1):
            video_name = video_file.name
            logger.info(f"Processing video {i}/{len(video_files)}: {video_name}")
            
            try:
                # Process single video
                deeplabcut.video_inference_superanimal(
                    [str(video_file)],  # Single video in list
                    superanimal_name="superanimal_quadruped",
                    model_name="hrnet_w32",
                    detector_name="fasterrcnn_resnet50_fpn_v2",
                    videotype=videotype,
                    video_adapt=True,
                    scale_list=[],
                    max_individuals=1,
                    dest_folder=str(pose_2d_dir),
                    batch_size=8,
                    detector_batch_size=8,
                    video_adapt_batch_size=4
                )
                
                successful_videos += 1
                logger.info(f"  ✅ Successfully processed {video_name}")
                
            except Exception as e:
                failed_videos.append(video_name)
                logger.error(f"  ❌ Failed to process {video_name}: {e}")
                # Continue with next video instead of stopping entirely
        
        # Report results
        if successful_videos > 0:
            logger.info(f"✅ DLC inference completed: {successful_videos}/{len(video_files)} videos successful")
            
            if failed_videos:
                logger.warning(f"⚠️  {len(failed_videos)} videos failed:")
                for failed_video in failed_videos:
                    logger.warning(f"    - {failed_video}")
            
            return True
        else:
            logger.error("❌ All videos failed DLC processing")
            return False
        
    except ImportError:
        logger.error("DeepLabCut not available")
        return False
    except Exception as e:
        logger.error(f"DLC inference setup failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def convert_h5_files(session_dir: Path) -> bool:
    """
    Convert H5 files to anipose-compatible format
    
    Args:
        session_dir: Path to session directory
        
    Returns:
        True if conversion successful
    """
    print_header("Converting H5 Files for Anipose Compatibility")
    
    pose_2d_dir = session_dir / "pose-2d"
    if not pose_2d_dir.exists():
        logger.error("pose-2d directory not found")
        return False
    
    # Get all H5 files
    h5_files = list(pose_2d_dir.glob("*.h5"))
    if not h5_files:
        logger.error("No H5 files found to convert")
        return False
    
    logger.info(f"Converting {len(h5_files)} H5 files...")
    
    try:
        import pandas as pd
        
        for h5_file in h5_files:
            logger.info(f"Processing: {h5_file.name}")
            
            # Load the H5 file
            df = pd.read_hdf(h5_file)
            
            # Check if it has multi-level columns
            if isinstance(df.columns, pd.MultiIndex):
                logger.info(f"  Original column levels: {df.columns.nlevels}")
                
                # Remove 'individuals' level if present
                if df.columns.nlevels == 4 and 'individuals' in df.columns.names:
                    individuals_level_idx = df.columns.names.index('individuals')
                    new_columns = df.columns.droplevel(individuals_level_idx)
                    df.columns = new_columns
                    
                    logger.info("  ✅ Removed 'individuals' level")
                    
                    # Save the modified dataframe
                    df.to_hdf(h5_file, key='df_with_missing', mode='w')
                    logger.info(f"  ✅ Saved converted file: {h5_file.name}")
                else:
                    logger.info(f"  ⏭️  Skipped: {h5_file.name} (unexpected structure)")
            else:
                logger.info(f"  ⏭️  Skipped: {h5_file.name} (not multi-level)")
            
            # Also rename to simple format
            simple_name = h5_file.name
            if '_superanimal' in simple_name:
                simple_name = simple_name.split('_superanimal')[0] + '.h5'
            elif '_snapshot-' in simple_name:
                simple_name = simple_name.split('_snapshot-')[0] + '.h5'
            elif 'DLC_' in simple_name:
                simple_name = simple_name.split('DLC_')[0] + '.h5'
            
            if simple_name != h5_file.name:
                new_path = pose_2d_dir / simple_name
                h5_file.rename(new_path)
                logger.info(f"  ✅ Renamed to: {simple_name}")
        
        logger.info("✅ H5 file conversion completed")
        return True
        
    except ImportError:
        logger.error("Pandas not available for H5 processing")
        return False
    except Exception as e:
        logger.error(f"H5 conversion failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def run_3d_processing(session_dir: Path, calibration_data: Optional = None) -> bool:
    """
    Run 3D triangulation and filtering using anipose with custom calibration integration
    
    Args:
        session_dir: Path to session directory
        calibration_data: Optional custom calibration data for aniposelib integration
        
    Returns:
        True if 3D processing successful
    """
    print_header("Running 3D Triangulation and Filtering with Animal Calibration")
    
    try:
        parent_dir = session_dir.parent
        session_name = session_dir.name
        
        logger.info(f"Session: {session_name}")
        logger.info(f"Parent directory: {parent_dir}")
        
        # Try to integrate custom calibration with anipose
        if calibration_data is not None:
            logger.info("Integrating custom calibration with anipose animal features...")
            try:
                from anipose_integration import integrate_custom_calibration_with_anipose
                integration_success = integrate_custom_calibration_with_anipose(session_dir, calibration_data)
                
                if integration_success:
                    logger.info("✅ Custom calibration integrated with anipose")
                else:
                    logger.warning("⚠️ Custom calibration integration failed, using command-line anipose")
                    
            except ImportError as e:
                logger.warning(f"⚠️ Anipose integration not available: {e}")
        
        # Run anipose commands from parent directory
        logger.info(f"Running anipose commands from: {parent_dir}")
        
        # Run anipose triangulation from parent directory
        logger.info("Running anipose triangulation with animal calibration...")
        run_command(['anipose', 'triangulate', session_name], working_dir=parent_dir)
        
        # Run 3D filtering from parent directory  
        logger.info("Running anipose 3D filtering with animal-specific parameters...")
        run_command(['anipose', 'filter-3d', session_name], working_dir=parent_dir)
        
        logger.info("✅ 3D processing completed successfully with animal calibration")
        return True
        
    except Exception as e:
        logger.error(f"3D processing failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def run_3camera_pipeline(session_dir: Path, skip_dlc: bool = False, 
                        board_config: Optional[Dict] = None) -> bool:
    """
    Run the complete 3-camera pipeline
    
    Args:
        session_dir: Path to session directory
        skip_dlc: Whether to skip DLC inference step
        board_config: Optional custom board configuration
        
    Returns:
        True if pipeline completed successfully
    """
    logger.info("Starting 3-Camera Processing Pipeline")
    logger.info(f"Session directory: {session_dir}")
    logger.info(f"Skip DLC: {skip_dlc}")
    
    # Step 1: Setup directory structure
    if not setup_session_structure(session_dir):
        logger.error("Failed to setup session structure")
        return False
    
    # Step 2: Verify 3-camera setup
    found_files = verify_3camera_setup(session_dir)
    if len(found_files['calibration']) < 3:
        logger.error("Insufficient calibration files for 3-camera setup")
        return False
    
    # Step 3: Create 3-camera config in parent directory (will be updated later with animal calibration)
    logger.info("Creating initial 3-camera configuration...")
    try:
        parent_dir = session_dir.parent
        session_name = session_dir.name
        logger.info(f"Config will be created in: {parent_dir}")
        logger.info(f"Session name: {session_name}")
        
        create_multi_camera_config_toml(str(parent_dir), num_cameras=3, 
                                      board_config=board_config)
        logger.info("✅ Initial configuration created (will be enhanced with animal calibration)")
    except Exception as e:
        logger.error(f"Failed to create configuration: {e}")
        return False
    
    # Step 4: Run custom 3-camera calibration (pairwise + optimization)
    if not run_3camera_calibration(session_dir, board_config):
        logger.error("3-camera calibration failed")
        return False
    
    # Step 5: Run DLC inference (optional)
    if not skip_dlc:
        if not run_dlc_inference(session_dir):
            logger.error("DLC inference failed")
            return False
    else:
        logger.info("Skipping DLC inference (--skip-dlc specified)")
    
    # Step 6: Convert H5 files (if DLC was run)
    if not skip_dlc:
        if not convert_h5_files(session_dir):
            logger.error("H5 conversion failed")
            return False
    
    # Step 7: Load calibration data and run 3D triangulation and filtering with animal calibration
    calibration_data = None
    try:
        # Load the custom calibration data for anipose integration
        custom_calib_path = session_dir / "calibration" / "calibration.toml"
        if custom_calib_path.exists():
            from custom_calibration import load_calibration_from_toml
            calibration_data = load_calibration_from_toml(custom_calib_path)
            logger.info("✅ Loaded custom calibration data for anipose integration")
        else:
            logger.warning("Custom calibration file not found, using standard anipose processing")
    except Exception as e:
        logger.warning(f"Could not load custom calibration: {e}")
    
    if not run_3d_processing(session_dir, calibration_data):
        logger.error("3D processing failed")
        return False
    
    print_header("3-Camera Pipeline Completed Successfully")
    parent_dir = session_dir.parent
    session_name = session_dir.name
    
    logger.info("Pipeline Summary:")
    logger.info("  ✅ 3-camera custom calibration (pairwise + global optimization)")
    if not skip_dlc:
        logger.info("  ✅ DeepLabCut inference (individual video processing)")
        logger.info("  ✅ H5 file conversion")
    logger.info("  ✅ Custom calibration integrated with anipose animal features")
    logger.info("  ✅ 3D triangulation (anipose with custom calibration)")
    logger.info("  ✅ 3D filtering (anipose with animal-specific parameters)")
    logger.info(f"Anipose config: {parent_dir}/config.toml")
    logger.info(f"Anipose calibration: {parent_dir}/calibration.toml")
    logger.info(f"Custom calibration: {session_dir}/calibration/calibration.toml")
    logger.info(f"Session data: {session_dir}")
    logger.info(f"Animal calibration features: ENABLED")
    
    return True

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="3-Camera Processing Pipeline")
    parser.add_argument("--session-dir", required=True, type=str,
                       help="Path to session directory")
    parser.add_argument("--skip-dlc", action="store_true",
                       help="Skip DeepLabCut inference step")
    parser.add_argument("--board-squares-x", type=int, default=10,
                       help="ChArUco board squares in X direction")
    parser.add_argument("--board-squares-y", type=int, default=7,
                       help="ChArUco board squares in Y direction")
    parser.add_argument("--square-length", type=float, default=25.0,
                       help="Square length in mm")
    parser.add_argument("--marker-length", type=float, default=18.75,
                       help="Marker length in mm")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Enable verbose logging")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Parse session directory
    session_dir = Path(args.session_dir).resolve()
    if not session_dir.exists():
        logger.error(f"Session directory not found: {session_dir}")
        return 1
    
    # Configure board
    board_config = {
        'squares_x': args.board_squares_x,
        'squares_y': args.board_squares_y,
        'square_length': args.square_length,
        'marker_length': args.marker_length
    }
    
    # Check if custom calibration is available
    if not CUSTOM_CALIBRATION_AVAILABLE:
        logger.error("Custom calibration modules not available!")
        logger.error("Please ensure custom_calibration.py and calibration_integration.py are in the Python path")
        return 1
    
    # Run the pipeline
    try:
        success = run_3camera_pipeline(session_dir, args.skip_dlc, board_config)
        return 0 if success else 1
        
    except KeyboardInterrupt:
        logger.info("Pipeline interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Pipeline failed with unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
