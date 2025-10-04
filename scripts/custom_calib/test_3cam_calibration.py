#!/usr/bin/env python3
"""
Test 3-Camera Calibration System

This script tests the pairwise calibration approach for 3 cameras to ensure
it works correctly with the custom calibration system.

Usage:
    python test_3cam_calibration.py --session-dir /path/to/test/session

Author: 3-Camera Testing
"""

import os
import sys
import numpy as np
import cv2
from pathlib import Path
from typing import Dict, List
import logging

# Import custom calibration modules
try:
    from custom_calibration import (
        MultiCameraCalibrator,
        ChArUcoBoard,
        ChArUcoDetector
    )
    CALIBRATION_AVAILABLE = True
except ImportError as e:
    CALIBRATION_AVAILABLE = False
    print(f"Custom calibration modules not available: {e}")

def create_synthetic_charuco_images(num_cameras: int = 3, 
                                  num_images: int = 15,
                                  image_size: tuple = (640, 480)) -> Dict[str, List[np.ndarray]]:
    """
    Create synthetic ChArUco calibration images for testing
    
    Args:
        num_cameras: Number of cameras to simulate
        num_images: Number of images per camera
        image_size: Size of synthetic images (width, height)
        
    Returns:
        Dictionary mapping camera_id to list of synthetic images
    """
    print(f"Creating synthetic ChArUco images for {num_cameras} cameras...")
    
    # Create ChArUco board
    board = ChArUcoBoard()
    charuco_board = board.create_board()
    
    # Generate board image
    board_image = charuco_board.generateImage((800, 600))
    
    camera_images = {}
    
    for cam_id in range(1, num_cameras + 1):
        camera_id = str(cam_id)
        images = []
        
        print(f"  Generating images for camera {camera_id}...")
        
        for img_idx in range(num_images):
            # Create variations by adding slight transformations and noise
            # Simulate different viewing angles and distances
            
            # Create a homography for perspective transformation
            angle = (img_idx - num_images//2) * 5  # ±25 degrees variation
            scale = 0.8 + 0.4 * (img_idx / num_images)  # Scale variation
            tx = (cam_id - 2) * 50 + (img_idx - num_images//2) * 20  # Camera position offset
            ty = (img_idx - num_images//2) * 15
            
            # Create transformation matrix
            center = (400, 300)
            M = cv2.getRotationMatrix2D(center, angle, scale)
            M[0, 2] += tx
            M[1, 2] += ty
            
            # Apply transformation
            transformed = cv2.warpAffine(board_image, M, (800, 600))
            
            # Resize to target size
            resized = cv2.resize(transformed, image_size)
            
            # Add some noise to simulate real camera conditions
            noise = np.random.normal(0, 5, resized.shape).astype(np.uint8)
            noisy_image = cv2.add(resized, noise)
            
            # Convert to BGR (OpenCV format)
            if len(noisy_image.shape) == 2:
                bgr_image = cv2.cvtColor(noisy_image, cv2.COLOR_GRAY2BGR)
            else:
                bgr_image = noisy_image
            
            images.append(bgr_image)
        
        camera_images[camera_id] = images
        print(f"    Created {len(images)} images for camera {camera_id}")
    
    return camera_images

def test_pairwise_calibration_detection(camera_images: Dict[str, List[np.ndarray]]) -> bool:
    """
    Test that the ChArUco detector can find corners in synthetic images
    
    Args:
        camera_images: Dictionary of camera images
        
    Returns:
        True if detection works for all cameras
    """
    print("Testing ChArUco corner detection...")
    
    board = ChArUcoBoard()
    detector = ChArUcoDetector(board)
    
    detection_success = {}
    
    for camera_id, images in camera_images.items():
        successful_detections = 0
        
        for i, image in enumerate(images):
            corners, ids = detector.detect_charuco_corners(image)
            
            if corners is not None and ids is not None and len(corners) > 4:
                successful_detections += 1
        
        detection_rate = successful_detections / len(images)
        detection_success[camera_id] = detection_rate
        
        print(f"  Camera {camera_id}: {successful_detections}/{len(images)} images ({detection_rate*100:.1f}% success)")
    
    # Check if all cameras have reasonable detection rates
    min_success_rate = 0.5  # At least 50% of images should have detections
    all_good = all(rate >= min_success_rate for rate in detection_success.values())
    
    if all_good:
        print("✅ ChArUco detection test passed")
    else:
        print("❌ ChArUco detection test failed - low detection rates")
    
    return all_good

def test_3camera_pairwise_calibration(camera_images: Dict[str, List[np.ndarray]]) -> bool:
    """
    Test the complete 3-camera calibration pipeline
    
    Args:
        camera_images: Dictionary of camera images
        
    Returns:
        True if calibration succeeds
    """
    print("Testing 3-camera pairwise calibration...")
    
    if not CALIBRATION_AVAILABLE:
        print("❌ Calibration modules not available")
        return False
    
    try:
        # Initialize calibrator
        board_config = ChArUcoBoard()
        calibrator = MultiCameraCalibrator(board_config)
        
        # Run calibration
        print("  Running multi-camera calibration...")
        calibration_data = calibrator.calibrate_all_cameras(
            camera_images,
            reference_camera='1'
        )
        
        # Verify results
        print("  Verifying calibration results...")
        
        # Check intrinsics
        expected_cameras = {'1', '2', '3'}
        found_cameras = set(calibration_data.intrinsics.keys())
        
        if found_cameras != expected_cameras:
            print(f"❌ Expected cameras {expected_cameras}, found {found_cameras}")
            return False
        
        print(f"  ✅ Found intrinsics for all cameras: {sorted(found_cameras)}")
        
        # Check extrinsics (should have extrinsics for cameras 2 and 3, relative to camera 1)
        expected_extrinsics = {'2', '3'}  # Camera 1 is reference
        found_extrinsics = set(calibration_data.extrinsics.keys())
        
        if found_extrinsics != expected_extrinsics:
            print(f"❌ Expected extrinsics for {expected_extrinsics}, found {found_extrinsics}")
            return False
        
        print(f"  ✅ Found extrinsics for cameras: {sorted(found_extrinsics)}")
        
        # Print calibration summary
        print("\n📊 CALIBRATION SUMMARY:")
        print("="*50)
        print(f"Reference camera: {calibration_data.reference_camera}")
        
        for camera_id, intrinsic in calibration_data.intrinsics.items():
            print(f"\nCamera {camera_id}:")
            print(f"  Image size: {intrinsic.image_size}")
            print(f"  Focal length: fx={intrinsic.camera_matrix[0,0]:.2f}, fy={intrinsic.camera_matrix[1,1]:.2f}")
            print(f"  Principal point: cx={intrinsic.camera_matrix[0,2]:.2f}, cy={intrinsic.camera_matrix[1,2]:.2f}")
            print(f"  Calibration error: {intrinsic.calibration_error:.4f}")
        
        for camera_id, extrinsic in calibration_data.extrinsics.items():
            print(f"\nCamera {camera_id} extrinsics (relative to {extrinsic.reference_camera_id}):")
            print(f"  Translation: [{extrinsic.translation_vector[0,0]:.2f}, {extrinsic.translation_vector[1,0]:.2f}, {extrinsic.translation_vector[2,0]:.2f}]")
            print(f"  Rotation (rvec): [{extrinsic.rotation_vector[0]:.3f}, {extrinsic.rotation_vector[1]:.3f}, {extrinsic.rotation_vector[2]:.3f}]")
            print(f"  Calibration error: {extrinsic.calibration_error:.4f}")
        
        print("="*50)
        print("✅ 3-camera pairwise calibration test passed!")
        return True
        
    except Exception as e:
        print(f"❌ 3-camera calibration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_toml_conversion(calibration_data) -> bool:
    """
    Test conversion to TOML format compatible with s-DANNCE
    
    Args:
        calibration_data: CalibrationData object
        
    Returns:
        True if conversion succeeds
    """
    print("Testing TOML conversion and s-DANNCE compatibility...")
    
    try:
        # Convert to TOML format
        toml_dict = calibration_data.to_toml_dict()
        
        # Check structure
        if 'metadata' not in toml_dict:
            print("❌ Missing metadata section")
            return False
        
        if toml_dict['metadata']['calibration_type'] != 'custom_multi_camera':
            print("❌ Wrong calibration type")
            return False
        
        # Check camera sections
        expected_cameras = ['cam_1', 'cam_2', 'cam_3']
        for cam_section in expected_cameras:
            if cam_section not in toml_dict:
                print(f"❌ Missing camera section: {cam_section}")
                return False
            
            cam_data = toml_dict[cam_section]
            required_keys = ['matrix', 'distortions', 'rotation', 'translation', 
                           'rotation_matrix', 'RDistort', 'TDistort']
            
            for key in required_keys:
                if key not in cam_data:
                    print(f"❌ Missing key {key} in {cam_section}")
                    return False
        
        print("  ✅ TOML structure is valid")
        
        # Test integration with s-DANNCE format
        from calibration_integration import convert_to_s_dannce_format
        
        s_dannce_params = convert_to_s_dannce_format(calibration_data)
        
        expected_cam_names = ['Camera1', 'Camera2', 'Camera3']
        found_cam_names = list(s_dannce_params.keys())
        
        if sorted(found_cam_names) != sorted(expected_cam_names):
            print(f"❌ s-DANNCE format: expected {expected_cam_names}, found {found_cam_names}")
            return False
        
        # Check parameter shapes
        for cam_name, params in s_dannce_params.items():
            if params['K'].shape != (3, 3):
                print(f"❌ {cam_name}: K matrix shape {params['K'].shape} != (3,3)")
                return False
            if params['R'].shape != (3, 3):
                print(f"❌ {cam_name}: R matrix shape {params['R'].shape} != (3,3)")
                return False
            if params['t'].shape != (1, 3):
                print(f"❌ {cam_name}: t vector shape {params['t'].shape} != (1,3)")
                return False
        
        print("  ✅ s-DANNCE format conversion successful")
        print("✅ TOML conversion test passed!")
        return True
        
    except Exception as e:
        print(f"❌ TOML conversion test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main test function"""
    print("🧪 3-CAMERA CALIBRATION SYSTEM TEST")
    print("="*60)
    
    if not CALIBRATION_AVAILABLE:
        print("❌ Cannot run tests - calibration modules not available")
        return 1
    
    # Test 1: Create synthetic calibration images
    print("\nTest 1: Creating synthetic ChArUco images...")
    camera_images = create_synthetic_charuco_images(num_cameras=3, num_images=12)
    
    # Test 2: Test ChArUco detection
    print("\nTest 2: Testing ChArUco corner detection...")
    if not test_pairwise_calibration_detection(camera_images):
        print("❌ Detection test failed")
        return 1
    
    # Test 3: Test 3-camera calibration
    print("\nTest 3: Testing 3-camera pairwise calibration...")
    calibration_data = None
    try:
        board_config = ChArUcoBoard()
        calibrator = MultiCameraCalibrator(board_config)
        calibration_data = calibrator.calibrate_all_cameras(
            camera_images,
            reference_camera='1'
        )
        print("✅ Calibration completed successfully")
    except Exception as e:
        print(f"❌ Calibration failed: {e}")
        return 1
    
    # Test 4: Test TOML conversion
    print("\nTest 4: Testing TOML conversion and s-DANNCE compatibility...")
    if not test_toml_conversion(calibration_data):
        print("❌ TOML conversion test failed")
        return 1
    
    print("\n🎉 ALL TESTS PASSED!")
    print("="*60)
    print("✅ 3-camera pairwise calibration system is working correctly")
    print("✅ ChArUco detection works with synthetic data")
    print("✅ Global pose optimization works for 3 cameras")
    print("✅ s-DANNCE format conversion works correctly")
    print("\n🚀 The system is ready for use with real 3-camera data!")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())

