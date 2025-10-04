#!/usr/bin/env python3
"""
Full Pipeline Example with Custom Multi-Camera Calibration

This script demonstrates how to run the complete pipeline with the new custom
multi-camera calibration system. It replaces the traditional anipose workflow
and supports flexible camera configurations.

Usage:
    python full_pipeline_example.py --base-dir /path/to/data --num-cameras 3
    python full_pipeline_example.py --config example_config.json

Author: Pipeline Integration
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Any
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import the updated pipeline components
try:
    from calibration_integration import (
        load_calibration_params,
        create_multi_camera_config_toml,
        get_camera_count
    )
    from run_custom_calibration import run_calibration
    CUSTOM_CALIBRATION_AVAILABLE = True
except ImportError as e:
    CUSTOM_CALIBRATION_AVAILABLE = False
    logger.warning(f"Custom calibration modules not available: {e}")

def create_example_configuration() -> Dict[str, Any]:
    """Create an example configuration for the multi-camera pipeline"""
    
    config = {
        "project": {
            "name": "MultiCameraTracking",
            "description": "Custom multi-camera calibration and tracking pipeline",
            "base_output_dir": "/work/rl349/DeepLabCut/multi_camera_combined"
        },
        
        "cameras": {
            "num_cameras": 4,
            "camera_ids": ["1", "2", "3", "4"],
            "reference_camera": "1",
            "naming_pattern": "cam{id}",  # cam1, cam2, etc.
            "video_pattern": "vid{video_num}-cam{camera_id}.mp4"
        },
        
        "calibration": {
            "board_config": {
                "squares_x": 10,
                "squares_y": 7,
                "square_length": 25.0,  # mm
                "marker_length": 18.75,  # mm
                "marker_dict": "DICT_6X6_50"
            },
            "min_images_per_camera": 10,
            "max_frames_from_video": 30,
            "frame_skip": 30
        },
        
        "sessions": [
            {
                "name": "session1(baseline)",
                "source_dir": "/work/rl349/DeepLabCut/baseline_data",
                "cameras": ["1", "2", "3", "4"]
            },
            {
                "name": "session2(treatment)",
                "source_dir": "/work/rl349/DeepLabCut/treatment_data", 
                "cameras": ["1", "2", "3"]  # One camera might be missing
            },
            {
                "name": "session3(followup)",
                "source_dir": "/work/rl349/DeepLabCut/followup_data",
                "cameras": ["1", "2", "3", "4"]
            }
        ],
        
        "processing": {
            "dlc_config": "/path/to/dlc/config.yaml",
            "skeleton_type": "mouse14",  # or "rat23"
            "batch_size": 8,
            "use_gpu": True,
            "max_individuals": 1
        },
        
        "output": {
            "dannce_format": True,
            "export_videos": True,
            "export_labels": True,
            "create_visualization": True
        }
    }
    
    return config

def setup_session_structure(session_config: Dict[str, Any], base_dir: Path) -> Path:
    """
    Set up directory structure for a session
    
    Args:
        session_config: Session configuration
        base_dir: Base output directory
        
    Returns:
        Path to created session directory
    """
    session_name = session_config["name"]
    session_dir = base_dir / session_name
    
    # Create session directory structure
    session_dir.mkdir(parents=True, exist_ok=True)
    
    # Create session1 subdirectory (anipose structure)
    session1_dir = session_dir / "session1"
    session1_dir.mkdir(exist_ok=True)
    
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
        (session1_dir / subdir).mkdir(exist_ok=True)
    
    logger.info(f"Created session structure: {session_dir}")
    return session_dir

def copy_calibration_data(source_dir: Path, calibration_dir: Path, 
                         camera_ids: List[str]) -> bool:
    """
    Copy calibration videos/images from source to calibration directory
    
    Args:
        source_dir: Source directory containing calibration data
        calibration_dir: Target calibration directory
        camera_ids: List of camera IDs to process
        
    Returns:
        True if calibration data was found and copied
    """
    found_data = False
    
    # Look for calibration videos or images
    for camera_id in camera_ids:
        # Try different naming patterns
        patterns = [
            f"calib-cam{camera_id}.mp4",
            f"calibration-cam{camera_id}.mp4",
            f"cam{camera_id}-calib.mp4",
            f"abjust/cam{camera_id}/*.mp4",
            f"abjust/{camera_id}/*.MP4",
            f"calibration/cam{camera_id}/*",
        ]
        
        for pattern in patterns:
            source_files = list(source_dir.glob(pattern))
            if source_files:
                for source_file in source_files:
                    target_file = calibration_dir / f"calib-cam{camera_id}.mp4"
                    
                    if source_file.is_file():
                        import shutil
                        shutil.copy2(source_file, target_file)
                        logger.info(f"Copied calibration data: {source_file} -> {target_file}")
                        found_data = True
                        break
        
        if found_data:
            break  # Found at least one calibration file
    
    return found_data

def copy_video_data(source_dir: Path, videos_dir: Path, 
                   camera_ids: List[str], session_name: str) -> int:
    """
    Copy experimental videos from source to videos-raw directory
    
    Args:
        source_dir: Source directory containing videos
        videos_dir: Target videos-raw directory  
        camera_ids: List of camera IDs to process
        session_name: Name of the session for logging
        
    Returns:
        Number of videos copied
    """
    copied_count = 0
    
    # Look for experimental videos
    video_patterns = ["*.mp4", "*.MP4", "*.avi", "*.AVI"]
    
    for pattern in video_patterns:
        video_files = list(source_dir.glob(f"**/{pattern}"))
        
        for video_file in video_files:
            # Try to identify camera from filename
            filename = video_file.name.lower()
            
            for camera_id in camera_ids:
                if f"cam{camera_id}" in filename:
                    # Generate target filename
                    video_num = copied_count // len(camera_ids) + 1
                    target_name = f"vid{video_num}-cam{camera_id}.mp4"
                    target_path = videos_dir / target_name
                    
                    import shutil
                    shutil.copy2(video_file, target_path)
                    logger.info(f"Copied video: {video_file} -> {target_path}")
                    copied_count += 1
                    break
    
    logger.info(f"Copied {copied_count} videos for {session_name}")
    return copied_count

def run_pipeline_session(session_config: Dict[str, Any], 
                        pipeline_config: Dict[str, Any],
                        base_dir: Path) -> bool:
    """
    Run the complete pipeline for a single session
    
    Args:
        session_config: Configuration for this session
        pipeline_config: Overall pipeline configuration
        base_dir: Base output directory
        
    Returns:
        True if session processing succeeded
    """
    session_name = session_config["name"]
    logger.info(f"Processing session: {session_name}")
    
    try:
        # Setup session structure
        session_dir = setup_session_structure(session_config, base_dir)
        session1_dir = session_dir / "session1"
        
        # Get camera configuration
        camera_ids = session_config.get("cameras", pipeline_config["cameras"]["camera_ids"])
        num_cameras = len(camera_ids)
        
        # Create multi-camera config
        create_multi_camera_config_toml(str(session_dir), num_cameras)
        
        # Copy calibration data
        source_dir = Path(session_config["source_dir"])
        if source_dir.exists():
            calibration_dir = session1_dir / "calibration"
            found_calibration = copy_calibration_data(source_dir, calibration_dir, camera_ids)
            
            if not found_calibration:
                logger.warning(f"No calibration data found for {session_name}")
                return False
            
            # Copy video data
            videos_dir = session1_dir / "videos-raw"
            video_count = copy_video_data(source_dir, videos_dir, camera_ids, session_name)
            
            if video_count == 0:
                logger.warning(f"No videos found for {session_name}")
                return False
        else:
            logger.error(f"Source directory not found: {source_dir}")
            return False
        
        # Run custom calibration
        logger.info(f"Running calibration for {session_name}...")
        
        if CUSTOM_CALIBRATION_AVAILABLE:
            board_config = pipeline_config["calibration"]["board_config"]
            reference_camera = pipeline_config["cameras"]["reference_camera"]
            
            success = run_calibration(
                session_dir=session_dir,
                camera_ids=camera_ids,
                reference_camera=reference_camera,
                board_config=board_config
            )
            
            if not success:
                logger.error(f"Calibration failed for {session_name}")
                return False
        else:
            logger.warning("Custom calibration not available, skipping calibration step")
        
        # Verify calibration results
        calibration_file = session1_dir / "calibration" / "calibration.toml"
        if calibration_file.exists():
            try:
                # Load and verify calibration
                calib_params = load_calibration_params(calibration_file)
                camera_count = len(calib_params)
                
                logger.info(f"Calibration verified: {camera_count} cameras calibrated")
                
                # Print calibration summary
                print(f"\n{'='*50}")
                print(f"CALIBRATION SUMMARY - {session_name}")
                print(f"{'='*50}")
                print(f"Cameras calibrated: {list(calib_params.keys())}")
                print(f"Calibration file: {calibration_file}")
                print(f"{'='*50}\n")
                
            except Exception as e:
                logger.error(f"Failed to verify calibration: {e}")
                return False
        else:
            logger.warning(f"Calibration file not found: {calibration_file}")
        
        # TODO: Add DLC processing, triangulation, and export steps here
        # This would include:
        # 1. Run DLC inference on videos
        # 2. Convert to anipose format
        # 3. Run triangulation 
        # 4. Apply 3D filtering
        # 5. Export to s-DANNCE format
        
        logger.info(f"Session {session_name} processing completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"Session {session_name} processing failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def run_full_pipeline(config: Dict[str, Any]) -> bool:
    """
    Run the complete multi-camera pipeline
    
    Args:
        config: Pipeline configuration
        
    Returns:
        True if pipeline completed successfully
    """
    logger.info("Starting full multi-camera pipeline...")
    
    # Create base output directory
    base_dir = Path(config["project"]["base_output_dir"])
    base_dir.mkdir(parents=True, exist_ok=True)
    
    # Save configuration for reference
    config_file = base_dir / "pipeline_config.json"
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)
    logger.info(f"Pipeline configuration saved to: {config_file}")
    
    # Process each session
    successful_sessions = []
    failed_sessions = []
    
    for session_config in config["sessions"]:
        session_name = session_config["name"]
        
        try:
            success = run_pipeline_session(session_config, config, base_dir)
            
            if success:
                successful_sessions.append(session_name)
            else:
                failed_sessions.append(session_name)
                
        except Exception as e:
            logger.error(f"Unexpected error processing {session_name}: {e}")
            failed_sessions.append(session_name)
    
    # Print final summary
    print(f"\n{'='*80}")
    print("FULL PIPELINE SUMMARY")
    print(f"{'='*80}")
    print(f"Total sessions: {len(config['sessions'])}")
    print(f"Successful: {len(successful_sessions)}")
    print(f"Failed: {len(failed_sessions)}")
    print()
    
    if successful_sessions:
        print("✅ Successful sessions:")
        for session in successful_sessions:
            print(f"  - {session}")
        print()
    
    if failed_sessions:
        print("❌ Failed sessions:")
        for session in failed_sessions:
            print(f"  - {session}")
        print()
    
    print(f"Output directory: {base_dir}")
    print(f"Configuration: {config_file}")
    print(f"{'='*80}")
    
    return len(failed_sessions) == 0

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Full Multi-Camera Pipeline")
    parser.add_argument("--config", type=str, help="Path to configuration JSON file")
    parser.add_argument("--base-dir", type=str, 
                       help="Base output directory (overrides config)")
    parser.add_argument("--num-cameras", type=int, default=2,
                       help="Number of cameras (for example config)")
    parser.add_argument("--create-example-config", action="store_true",
                       help="Create example configuration file")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Enable verbose logging")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Create example configuration if requested
    if args.create_example_config:
        config = create_example_configuration()
        config_file = "example_pipeline_config.json"
        
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"Example configuration created: {config_file}")
        print("Edit this file and run with: python full_pipeline_example.py --config example_pipeline_config.json")
        return 0
    
    # Load configuration
    if args.config:
        with open(args.config, 'r') as f:
            config = json.load(f)
    else:
        # Create a minimal example configuration
        config = create_example_configuration()
        config["cameras"]["num_cameras"] = args.num_cameras
        config["cameras"]["camera_ids"] = [str(i+1) for i in range(args.num_cameras)]
    
    # Override base directory if provided
    if args.base_dir:
        config["project"]["base_output_dir"] = args.base_dir
    
    # Check if custom calibration is available
    if not CUSTOM_CALIBRATION_AVAILABLE:
        logger.error("Custom calibration modules not available!")
        logger.error("Please ensure custom_calibration.py and calibration_integration.py are in the Python path")
        return 1
    
    # Run the pipeline
    try:
        success = run_full_pipeline(config)
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
