#!/usr/bin/env python3
"""
Custom Multi-Camera Calibration System

This module implements a custom calibration pipeline using:
- OpenCV for ChArUco tag detection and individual camera calibration
- Pairwise extrinsic calibration between camera pairs
- Global pose graph optimization using GTSAM
- Flexible multi-camera support (not limited to 2 cameras)
- Storage in calibration.toml format with proper parameter shapes

Author: Calibration System
"""

import os
import cv2
import numpy as np
import toml
import gtsam
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, TYPE_CHECKING
from dataclasses import dataclass
from collections import defaultdict
import logging

if TYPE_CHECKING:
    from frame_level_optimizer import CameraFrameAnalysis

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ChArUcoBoard:
    """ChArUco board configuration"""
    squares_x: int = 10
    squares_y: int = 7
    square_length: float = 25.0  # mm
    marker_length: float = 18.75  # mm
    marker_dict: int = cv2.aruco.DICT_4X4_50  # Fixed: user specs say bits=4, number=50
    
    def create_board(self):
        """Create ChArUco board object (portable across OpenCV builds)"""
        dictionary = cv2.aruco.getPredefinedDictionary(self.marker_dict)
        try:
            # OpenCV 4.8+ constructor API
            return cv2.aruco.CharucoBoard(
                (self.squares_x, self.squares_y),
                self.square_length,
                self.marker_length,
                dictionary
            )
        except (AttributeError, TypeError):
            try:
                # Older OpenCV factory method
                return cv2.aruco.CharucoBoard_create(
                    self.squares_x, self.squares_y,
                    self.square_length, self.marker_length,
                    dictionary
                )
            except AttributeError:
                # Fallback: create manually if needed
                raise RuntimeError("Cannot create ChArUco board with this OpenCV version")

@dataclass
class CameraIntrinsics:
    """Camera intrinsic parameters"""
    camera_id: str
    image_size: Tuple[int, int]  # (width, height)
    camera_matrix: np.ndarray   # 3x3 intrinsic matrix K
    distortion_coeffs: np.ndarray  # distortion coefficients
    calibration_error: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for TOML storage"""
        return {
            'camera_id': self.camera_id,
            'image_size': list(self.image_size),
            'camera_matrix': self.camera_matrix.tolist(),
            'distortion_coeffs': self.distortion_coeffs.tolist(),
            'calibration_error': self.calibration_error
        }

@dataclass
class CameraExtrinsics:
    """Camera extrinsic parameters relative to reference camera"""
    camera_id: str
    reference_camera_id: str
    rotation_matrix: np.ndarray  # 3x3 rotation matrix R
    translation_vector: np.ndarray  # 3x1 translation vector t
    rotation_vector: np.ndarray  # 3x1 rotation vector (rvec)
    calibration_error: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for TOML storage"""
        return {
            'camera_id': self.camera_id,
            'reference_camera_id': self.reference_camera_id,
            'rotation_matrix': self.rotation_matrix.tolist(),
            'translation_vector': self.translation_vector.tolist(),
            'rotation_vector': self.rotation_vector.tolist(),
            'calibration_error': self.calibration_error
        }

@dataclass
class CalibrationData:
    """Complete calibration data for all cameras"""
    intrinsics: Dict[str, CameraIntrinsics]
    extrinsics: Dict[str, CameraExtrinsics]
    board_config: ChArUcoBoard
    reference_camera: str
    
    def to_toml_dict(self) -> Dict[str, Any]:
        """Convert to TOML-compatible dictionary with proper parameter shapes"""
        toml_dict = {
            'metadata': {
                'calibration_type': 'custom_multi_camera',
                'reference_camera': self.reference_camera,
                'num_cameras': len(self.intrinsics),
                'board_config': {
                    'squares_x': self.board_config.squares_x,
                    'squares_y': self.board_config.squares_y,
                    'square_length': self.board_config.square_length,
                    'marker_length': self.board_config.marker_length,
                    'marker_dict': self.board_config.marker_dict
                }
            }
        }
        
        # Add camera parameters in the format expected by s-DANNCE
        for cam_id, intrinsic in self.intrinsics.items():
            # Camera matrix (3x3)
            K = intrinsic.camera_matrix
            
            # Get extrinsics (if available) or use identity for reference camera
            if cam_id in self.extrinsics:
                extrinsic = self.extrinsics[cam_id]
                R = extrinsic.rotation_matrix
                t = extrinsic.translation_vector.reshape(1, 3)  # 1x3 row vector
                rvec = extrinsic.rotation_vector
            else:
                # Reference camera has identity transform
                R = np.eye(3)
                t = np.zeros((1, 3))
                rvec = np.zeros(3)
            
            # Distortion coefficients
            # NOTE: OpenCV typically returns dist coeffs with shape (1, N).
            # Using len() on a Nx1 array returns the first dimension (often 1),
            # so always flatten before checking size.
            dist_vec = np.asarray(intrinsic.distortion_coeffs).ravel()
            n_dist = int(dist_vec.size)
            if n_dist >= 5:
                # Standard camera model: k1, k2, p1, p2, k3
                RDistort = dist_vec[:2].reshape(1, 2)  # 1x2
                TDistort = dist_vec[2:4].reshape(1, 2)  # 1x2
                additional_dist = dist_vec[4:].tolist() if n_dist > 4 else []
            elif n_dist >= 4:
                # Fisheye/limited model: k1, k2, p1, p2
                RDistort = dist_vec[:2].reshape(1, 2)
                TDistort = dist_vec[2:4].reshape(1, 2)
                additional_dist = []
            else:
                # Fallback: pad with zeros
                RDistort = np.zeros((1, 2))
                TDistort = np.zeros((1, 2))
                additional_dist = []
            
            # Create camera section
            toml_dict[f'cam_{cam_id}'] = {
                'name': cam_id,
                'size': list(intrinsic.image_size),
                'matrix': K.tolist(),  # 3x3 intrinsic matrix
                'distortions': dist_vec.tolist(),  # Full distortion vector
                'rotation': rvec.tolist(),  # 3-element rotation vector (rvec)
                'translation': t.flatten().tolist(),  # 3-element translation vector (tvec)
                'rotation_matrix': R.tolist(),  # 3x3 rotation matrix
                'calibration_error': intrinsic.calibration_error,
                # Additional fields for s-DANNCE compatibility
                'RDistort': RDistort.tolist(),  # 1x2 radial distortion
                'TDistort': TDistort.tolist(),  # 1x2 tangential distortion
            }
            
            if additional_dist:
                toml_dict[f'cam_{cam_id}']['additional_distortion'] = additional_dist
        
        return toml_dict
    
    def save_to_file(self, filepath: Path):
        """Save calibration to TOML file"""
        toml_dict = self.to_toml_dict()
        with open(filepath, 'w') as f:
            toml.dump(toml_dict, f)
        logger.info(f"Calibration saved to {filepath}")

class ChArUcoDetector:
    """ChArUco marker detector and pose estimator"""
    
    def __init__(self, board: ChArUcoBoard):
        self.board_config = board
        self.board = board.create_board()
        self.detector_params = cv2.aruco.DetectorParameters()
        
        # Optimize detector parameters for better detection (more robust for real videos)
        self.detector_params.adaptiveThreshWinSizeMin = 3
        self.detector_params.adaptiveThreshWinSizeMax = 53  # Increased for larger boards
        self.detector_params.adaptiveThreshWinSizeStep = 4   # Smaller steps for better detection
        self.detector_params.adaptiveThreshConstant = 7
        self.detector_params.minMarkerPerimeterRate = 0.01  # More permissive
        self.detector_params.maxMarkerPerimeterRate = 6.0   # More permissive  
        self.detector_params.polygonalApproxAccuracyRate = 0.05  # More permissive
        self.detector_params.minCornerDistanceRate = 0.03   # More permissive
        self.detector_params.minDistanceToBorder = 1        # More permissive
        self.detector_params.minMarkerDistanceRate = 0.03   # More permissive
        self.detector_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        self.detector_params.cornerRefinementWinSize = 7    # Larger window
        self.detector_params.cornerRefinementMaxIterations = 50  # More iterations
        self.detector_params.cornerRefinementMinAccuracy = 0.05  # More permissive
        
    def detect_charuco_corners(self, image: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Detect ChArUco corners in image
        
        Returns:
            charuco_corners: Detected corner coordinates (Nx1x2)
            charuco_ids: IDs of detected corners (Nx1)
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        # Try modern OpenCV 4.8+ CharucoDetector API first
        try:
            charuco_params = cv2.aruco.CharucoParameters()
            # Use our optimized detector params
            aruco_detector_params = cv2.aruco.DetectorParameters()
            aruco_detector_params.adaptiveThreshWinSizeMax = 53
            aruco_detector_params.minMarkerPerimeterRate = 0.01
            aruco_detector_params.maxMarkerPerimeterRate = 6.0
            aruco_detector_params.polygonalApproxAccuracyRate = 0.05
            aruco_detector_params.minCornerDistanceRate = 0.03
            aruco_detector_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
            
            charuco_detector = cv2.aruco.CharucoDetector(self.board, charuco_params, aruco_detector_params)
            
            # detectBoard returns 4 values in OpenCV 4.12.0
            charuco_corners, charuco_ids, marker_corners, marker_ids = charuco_detector.detectBoard(gray)
            
            if charuco_corners is not None and len(charuco_corners) > 4:
                logger.debug(f"✅ Modern CharucoDetector found {len(charuco_corners)} corners")
                return charuco_corners, charuco_ids
            else:
                # Debug what CharucoDetector actually found
                corner_count = len(charuco_corners) if charuco_corners is not None else 0
                marker_count = len(marker_corners) if marker_corners is not None else 0
                logger.debug(f"Modern CharucoDetector: {corner_count} corners, {marker_count} markers")
                
                # If we have markers but no corners, there might be an interpolation issue
                if marker_count > 0 and corner_count == 0:
                    logger.debug("Markers detected but no ChArUco corners - trying manual interpolation")
                    # Continue to fallback methods
                elif corner_count > 0:
                    logger.debug(f"Found {corner_count} corners but need >4, continuing to fallback")
                else:
                    logger.debug("No markers or corners detected with CharucoDetector")
        except Exception as e:
            logger.debug(f"Modern CharucoDetector failed: {e}")
            pass
        
        # Fallback: Traditional ArUco detection + interpolation
        detector = cv2.aruco.ArucoDetector(
            cv2.aruco.getPredefinedDictionary(self.board_config.marker_dict),
            self.detector_params
        )
        
        marker_corners, marker_ids, _ = detector.detectMarkers(gray)
        
        if len(marker_corners) > 0:
            # Try different OpenCV ChArUco interpolation APIs
            try:
                # OpenCV 4.7+ API
                num_detected, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
                    marker_corners, marker_ids, gray, self.board
                )
                if num_detected > 4:
                    return charuco_corners, charuco_ids
            except AttributeError:
                pass
            
            try:
                # OpenCV 4.5-4.6 API (handles different return formats)
                result = cv2.aruco.interpolateCornersCharuco(
                    marker_corners, marker_ids, gray, self.board
                )
                if len(result) == 2:
                    charuco_corners, charuco_ids = result
                elif len(result) == 3:
                    num_detected, charuco_corners, charuco_ids = result
                else:
                    charuco_corners, charuco_ids = result[0], result[1]
                
                if charuco_corners is not None and len(charuco_corners) > 4:
                    return charuco_corners, charuco_ids
            except (AttributeError, ValueError):
                pass
            
            try:
                # OpenCV 4.0-4.4 API
                retval, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
                    marker_corners, marker_ids, gray, self.board
                )
                if retval > 4:
                    return charuco_corners, charuco_ids
            except AttributeError:
                pass
            
            # If ChArUco interpolation fails, return None (don't use invalid data)
            logger.warning("ChArUco interpolation not available and all APIs failed")
            logger.warning("This frame will be skipped for calibration")
            return None, None
        
        return None, None
    
    def collect_calibration_data(self, images: List[np.ndarray]) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Collect calibration data from multiple images
        
        Args:
            images: List of calibration images
            
        Returns:
            all_corners: List of detected corner arrays
            all_ids: List of detected ID arrays
        """
        all_corners = []
        all_ids = []
        
        logger.info(f"Processing {len(images)} calibration images...")
        
        for i, image in enumerate(images):
            corners, ids = self.detect_charuco_corners(image)
            
            if corners is not None and ids is not None:
                all_corners.append(corners)
                all_ids.append(ids)
                logger.debug(f"Image {i}: Detected {len(corners)} corners")
            else:
                logger.warning(f"Image {i}: No ChArUco corners detected")
        
        logger.info(f"Successfully processed {len(all_corners)} images with valid detections")
        return all_corners, all_ids

class SingleCameraCalibrator:
    """Calibrate individual camera intrinsics using ChArUco board"""
    
    def __init__(self, board: ChArUcoBoard):
        self.board = board
        self.detector = ChArUcoDetector(board)
    
    def calibrate(self, images: List[np.ndarray], camera_id: str, 
                 cached_corners: List[np.ndarray] = None, cached_ids: List[np.ndarray] = None) -> CameraIntrinsics:
        """
        Calibrate camera intrinsics
        
        Args:
            images: List of calibration images
            camera_id: Unique camera identifier
            cached_corners: Optional pre-computed corner detections (avoids re-detection)
            cached_ids: Optional pre-computed corner IDs (avoids re-detection)
            
        Returns:
            CameraIntrinsics object with calibrated parameters
        """
        logger.info(f"Calibrating camera {camera_id}...")
        
        # Use cached detection results if available, otherwise detect
        if cached_corners is not None and cached_ids is not None:
            logger.info(f"Using cached ChArUco detection results ({len(cached_corners)} frames)")
            all_corners, all_ids = cached_corners, cached_ids
        else:
            logger.info("Running ChArUco detection on calibration images...")
            all_corners, all_ids = self.detector.collect_calibration_data(images)
        
        if len(all_corners) < 5:
            logger.error(f"Insufficient ChArUco detections for camera {camera_id}. Got {len(all_corners)}, need at least 5")
            logger.error("ChArUco calibration requires frames with well-distributed corners.")
            logger.error("Please ensure calibration videos contain clear, well-lit ChArUco board views.")
            raise ValueError(f"Insufficient ChArUco detections for camera {camera_id}: {len(all_corners)}/5 minimum")
        
        # Validate calibration data quality
        if not self._validate_calibration_data(all_corners, all_ids, images[0].shape):
            logger.error(f"ChArUco calibration data quality insufficient for camera {camera_id}")
            logger.error("Detected corners do not meet distribution requirements for stable calibration.")
            raise ValueError(f"Poor ChArUco detection quality for camera {camera_id}")
        
        # Get image size
        image_size = (images[0].shape[1], images[0].shape[0])  # (width, height)
        
        # Calibrate camera (use proper calibration flags and termination criteria)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6)
        
        try:
            # Additional validation before calling OpenCV calibration
            logger.info(f"Running ChArUco calibration with {len(all_corners)} high-quality detections...")
            
            ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.aruco.calibrateCameraCharuco(
                all_corners, all_ids, self.board.create_board(), image_size,
                None, None, flags=0, criteria=criteria
            )
            
            # Validate calibration result
            if ret > 5.0:  # High reprojection error indicates poor calibration
                logger.warning(f"High calibration error {ret:.4f} for camera {camera_id}")
                logger.warning("This may indicate insufficient or poor quality ChArUco detections")
                
        except cv2.error as e:
            logger.error(f"ChArUco calibration failed for camera {camera_id}: {e}")
            logger.error("This typically indicates:")
            logger.error("  1. Insufficient number of well-distributed corner detections")
            logger.error("  2. Degenerate corner patterns (collinear or poorly distributed)")
            logger.error("  3. Incorrect ChArUco board parameters")
            logger.error("Try improving calibration video quality with better lighting and more board positions")
            raise ValueError(f"ChArUco calibration failed for camera {camera_id}: {e}")
        
        logger.info(f"Camera {camera_id} calibration completed with RMS error: {ret:.4f}")
        
        return CameraIntrinsics(
            camera_id=camera_id,
            image_size=image_size,
            camera_matrix=camera_matrix,
            distortion_coeffs=dist_coeffs,
            calibration_error=ret
        )
    
    def _validate_calibration_data(self, all_corners: List[np.ndarray], all_ids: List[np.ndarray], 
                                  image_shape: Tuple[int, int, int]) -> bool:
        """
        Validate ChArUco calibration data quality before attempting calibration
        
        Args:
            all_corners: List of corner arrays from all frames
            all_ids: List of ID arrays from all frames
            image_shape: Image dimensions
            
        Returns:
            True if calibration data meets quality requirements
        """
        if len(all_corners) < 5:
            logger.warning("Insufficient number of frames with ChArUco detections")
            return False
        
        height, width = image_shape[:2]
        total_corners = 0
        valid_frames = 0
        
        # Check each frame's corner distribution
        for i, (corners, ids) in enumerate(zip(all_corners, all_ids)):
            if corners is None or ids is None:
                continue
                
            corner_count = len(corners)
            total_corners += corner_count
            
            if corner_count < 8:  # Need at least 8 corners per frame (more lenient)
                continue
            
            # Check corner distribution within this frame (more lenient)
            corners_reshaped = corners.reshape(-1, 2)
            
            # Ensure corners span reasonable portion of image (more lenient)
            x_span = np.max(corners_reshaped[:, 0]) - np.min(corners_reshaped[:, 0])
            y_span = np.max(corners_reshaped[:, 1]) - np.min(corners_reshaped[:, 1])
            
            if x_span < 0.2 * width or y_span < 0.2 * height:
                continue  # Only reject very poor coverage
            
            # Only check for severely degenerate patterns
            if len(corners) >= 6:
                try:
                    centered_corners = corners_reshaped - np.mean(corners_reshaped, axis=0)
                    _, s, _ = np.linalg.svd(centered_corners, full_matrices=False)
                    condition_number = s[0] / s[-1] if s[-1] > 1e-10 else np.inf
                    if condition_number > 200:  # Only reject severely degenerate patterns
                        continue
                except:
                    pass  # If SVD fails, accept the frame
            
            valid_frames += 1
        
        avg_corners_per_frame = total_corners / len(all_corners) if all_corners else 0
        valid_frame_ratio = valid_frames / len(all_corners) if all_corners else 0
        
        logger.info(f"Calibration data validation:")
        logger.info(f"  Total frames: {len(all_corners)}")
        logger.info(f"  Valid frames: {valid_frames} ({valid_frame_ratio:.1%})")
        logger.info(f"  Avg corners/frame: {avg_corners_per_frame:.1f}")
        
        # Need at least 3 valid frames and 30% valid frame ratio (much more lenient)
        if valid_frames < 3:
            logger.warning(f"Too few valid frames: {valid_frames}/3 minimum")
            return False
            
        if valid_frame_ratio < 0.3:
            logger.warning(f"Too low valid frame ratio: {valid_frame_ratio:.1%} (need >30%)")
            return False
        
        # Additional check: ensure corners are distributed across multiple unique IDs (more lenient)
        all_unique_ids = set()
        for ids in all_ids:
            if ids is not None:
                all_unique_ids.update(ids.flatten())
        
        if len(all_unique_ids) < 10:  # Need some diverse corner IDs (was 20)
            logger.warning(f"Insufficient unique corner IDs: {len(all_unique_ids)}/10 minimum")
            return False
        
        logger.info("✅ Calibration data validation passed")
        return True

class PairwiseCalibrator:
    """Calibrate extrinsics between pairs of cameras"""
    
    def __init__(self, board: ChArUcoBoard):
        self.board_config = board  # Keep config for fallbacks
        self.board = board.create_board()
        self.detector = ChArUcoDetector(board)
    
    def calibrate_stereo_pair(self, 
                            images1: List[np.ndarray], 
                            images2: List[np.ndarray],
                            intrinsics1: CameraIntrinsics,
                            intrinsics2: CameraIntrinsics) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Calibrate stereo pair extrinsics
        
        Args:
            images1: Images from first camera
            images2: Images from second camera (synchronized with images1)
            intrinsics1: Intrinsics of first camera
            intrinsics2: Intrinsics of second camera
            
        Returns:
            R: Rotation matrix from camera1 to camera2
            t: Translation vector from camera1 to camera2
            error: Stereo calibration error
        """
        logger.info(f"Calibrating stereo pair: {intrinsics1.camera_id} -> {intrinsics2.camera_id}")
        
        # Collect corresponding points from both cameras (only synchronized frames with common IDs)
        board = self.board
        
        # Get board corners (handle different OpenCV versions)
        try:
            # OpenCV 4.8+ API
            board_corners = np.array(board.getChessboardCorners(), dtype=np.float32)  # (N,3)
        except AttributeError:
            try:
                # Older OpenCV API
                board_corners = np.array(board.chessboardCorners, dtype=np.float32)  # (N,3)
            except AttributeError:
                # Manual calculation if needed
                squares_x, squares_y = self.board_config.squares_x, self.board_config.squares_y
                objp = np.zeros((squares_x * squares_y, 3), np.float32)
                objp[:,:2] = np.mgrid[0:squares_x, 0:squares_y].T.reshape(-1,2)
                objp *= self.board_config.square_length
                board_corners = objp
        object_points, image_points1, image_points2 = [], [], []

        for img1, img2 in zip(images1, images2):
            # Detect corners in both images
            corners1, ids1 = self.detector.detect_charuco_corners(img1)
            corners2, ids2 = self.detector.detect_charuco_corners(img2)

            if corners1 is None or corners2 is None or ids1 is None or ids2 is None:
                continue

            ids1_f = ids1.flatten()
            ids2_f = ids2.flatten()
            common = np.intersect1d(ids1_f, ids2_f)
            if len(common) < 6:
                continue

            obj_pts, img_pts1, img_pts2 = [], [], []
            for cid in common:
                i1 = np.where(ids1_f == cid)[0][0]
                i2 = np.where(ids2_f == cid)[0][0]
                # 3D ChArUco corner (planar board, z=0 in board frame)
                obj_pts.append(board_corners[int(cid)])
                img_pts1.append(corners1[i1][0])
                img_pts2.append(corners2[i2][0])

            object_points.append(np.array(obj_pts, dtype=np.float32))
            image_points1.append(np.array(img_pts1, dtype=np.float32))
            image_points2.append(np.array(img_pts2, dtype=np.float32))
        
        if len(object_points) < 3:
            raise ValueError(f"Not enough corresponding points for stereo calibration. Got {len(object_points)} valid pairs")
        
        logger.info(f"Found {len(object_points)} valid stereo pairs")
        
        # Verify image sizes match
        if intrinsics1.image_size != intrinsics2.image_size:
            logger.error(f"Image size mismatch: cam1={intrinsics1.image_size}, cam2={intrinsics2.image_size}")
            raise ValueError("Camera image sizes must match for stereo calibration")
        
        # Perform stereo calibration with better flags
        ret, _, _, _, _, R, t, E, F = cv2.stereoCalibrate(
            object_points, image_points1, image_points2,
            intrinsics1.camera_matrix, intrinsics1.distortion_coeffs,
            intrinsics2.camera_matrix, intrinsics2.distortion_coeffs,
            intrinsics1.image_size,
            flags=cv2.CALIB_FIX_INTRINSIC | cv2.CALIB_USE_INTRINSIC_GUESS,
            criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6)
        )
        
        logger.info(f"Stereo calibration completed with RMS error: {ret:.4f}")
        
        return R, t, ret

class GlobalOptimizer:
    """Global pose graph optimization using GTSAM"""

    def __init__(self):
        self.graph = gtsam.NonlinearFactorGraph()
        self.initial_estimate = gtsam.Values()
        self._id: Dict[str, int] = {}  # camera_id -> int index
        # Base rotation ~0.02 rad, base translation ~5 mm
        self.base_noise = gtsam.noiseModel.Diagonal.Sigmas(
            np.array([0.02, 0.02, 0.02, 5.0, 5.0, 5.0], dtype=np.float64)
        )

    def _key(self, cid: str) -> int:
        if cid not in self._id:
            self._id[cid] = len(self._id)
        return gtsam.symbol('c', self._id[cid])

    def add_camera_pose(self, camera_id: str, pose: gtsam.Pose3):
        key = self._key(camera_id)
        self.initial_estimate.insert(key, pose)
        return key

    def add_relative_pose_constraint(self, key1: int, key2: int, rel: gtsam.Pose3, pair_rms: float):
        # Scale translation noise by pair RMS (clipped)
        s = float(np.clip(pair_rms, 0.5, 5.0))
        noise = gtsam.noiseModel.Diagonal.Sigmas(
            np.array([0.02, 0.02, 0.02, 5.0 * s, 5.0 * s, 5.0 * s], dtype=np.float64)
        )
        robust = gtsam.noiseModel.Robust.Create(
            gtsam.noiseModel.mEstimator.Huber(1.345), noise
        )
        self.graph.add(gtsam.BetweenFactorPose3(key1, key2, rel, robust))

    def add_prior_constraint(self, key: int, pose: gtsam.Pose3):
        noise = gtsam.noiseModel.Diagonal.Sigmas(
            np.array([0.01, 0.01, 0.01, 0.5, 0.5, 0.5], dtype=np.float64)
        )
        self.graph.add(gtsam.PriorFactorPose3(key, pose, noise))

    def optimize(self) -> gtsam.Values:
        optimizer = gtsam.LevenbergMarquardtOptimizer(self.graph, self.initial_estimate)
        result = optimizer.optimize()
        logger.info("Global pose graph optimization completed")
        return result

class MultiCameraCalibrator:
    """Main multi-camera calibration system"""
    
    def __init__(self, board_config: ChArUcoBoard = None):
        if board_config is None:
            board_config = ChArUcoBoard()
        
        self.board_config = board_config
        self.single_calibrator = SingleCameraCalibrator(board_config)
        self.pairwise_calibrator = PairwiseCalibrator(board_config)
        
    def calibrate_all_cameras(self, 
                            camera_images: Dict[str, List[np.ndarray]], 
                            reference_camera: str = None,
                            pairwise_images: Dict[Tuple[str, str], Dict[str, List[np.ndarray]]] = None,
                            camera_analyses: Dict[str, 'CameraFrameAnalysis'] = None) -> CalibrationData:
        """
        Calibrate all cameras with frame-level optimization and global optimization
        
        Args:
            camera_images: Dictionary mapping camera_id to list of optimal intrinsic calibration images
            reference_camera: ID of reference camera (first camera if None)
            pairwise_images: Optional dictionary of synchronized images for pairwise calibration
            camera_analyses: Optional cached ChArUco detection results to avoid redundant detection
            
        Returns:
            CalibrationData object with complete calibration
        """
        camera_ids = list(camera_images.keys())
        
        if len(camera_ids) < 1:
            raise ValueError("Need at least 1 camera for calibration")
        
        if reference_camera is None:
            reference_camera = camera_ids[0]
        elif reference_camera not in camera_ids:
            raise ValueError(f"Reference camera {reference_camera} not found in provided cameras")
        
        logger.info(f"Starting frame-optimized multi-camera calibration with {len(camera_ids)} cameras")
        logger.info(f"Reference camera: {reference_camera}")
        
        # Step 1: Calibrate individual camera intrinsics using optimal frames
        logger.info("Step 1: Calibrating individual camera intrinsics with optimal frames...")
        intrinsics = {}
        for camera_id, images in camera_images.items():
            logger.info(f"   Calibrating camera {camera_id} with {len(images)} optimal images")
            
            # Use cached detection results if available (avoids redundant ChArUco detection)
            if camera_analyses and camera_id in camera_analyses:
                cached_corners, cached_ids = camera_analyses[camera_id].get_all_detection_results()
                logger.info(f"   ⚡ Using cached ChArUco detections for camera {camera_id} (no re-detection needed)")
                intrinsics[camera_id] = self.single_calibrator.calibrate(
                    images, camera_id, cached_corners=cached_corners, cached_ids=cached_ids
                )
            else:
                logger.info(f"   Running ChArUco detection for camera {camera_id} (no cache available)")
                intrinsics[camera_id] = self.single_calibrator.calibrate(images, camera_id)
        
        # Step 2: Pairwise stereo calibration using synchronized frames
        logger.info("Step 2: Performing pairwise stereo calibration with synchronized frames...")
        pairwise_extrinsics = {}
        
        # Use synchronized frames if available, otherwise fall back to all images
        if pairwise_images:
            logger.info("   Using synchronized frame sets for optimal pairwise calibration")
            for (cam1, cam2), pair_data in pairwise_images.items():
                if len(pair_data) == 2 and cam1 in pair_data and cam2 in pair_data:
                    try:
                        logger.info(f"   Calibrating pair ({cam1}, {cam2}) with {len(pair_data[cam1])} synchronized frames")
                        
                        R, t, error = self.pairwise_calibrator.calibrate_stereo_pair(
                            pair_data[cam1], pair_data[cam2],  # Synchronized images
                            intrinsics[cam1], intrinsics[cam2]
                        )
                        
                        pairwise_extrinsics[(cam1, cam2)] = {
                            'R': R, 't': t, 'error': error
                        }
                        
                        logger.info(f"   ✅ Pair ({cam1}, {cam2}): calibrated with error {error:.4f}")
                        
                    except Exception as e:
                        logger.warning(f"   ❌ Could not calibrate pair {cam1}-{cam2} with synchronized frames: {e}")
                else:
                    logger.warning(f"   ⚠️ Insufficient synchronized data for pair ({cam1}, {cam2})")
        else:
            # Fallback: try to calibrate all possible pairs with all available images
            logger.info("   No synchronized frames available, using all images for pairwise calibration")
            for i, cam1 in enumerate(camera_ids):
                for j, cam2 in enumerate(camera_ids[i+1:], i+1):
                    try:
                        R, t, error = self.pairwise_calibrator.calibrate_stereo_pair(
                            camera_images[cam1], camera_images[cam2],
                            intrinsics[cam1], intrinsics[cam2]
                        )
                        
                        pairwise_extrinsics[(cam1, cam2)] = {
                            'R': R, 't': t, 'error': error
                        }
                        
                    except Exception as e:
                        logger.warning(f"Could not calibrate pair {cam1}-{cam2}: {e}")
        
        if len(pairwise_extrinsics) == 0:
            logger.warning("No successful pairwise calibrations. Using single camera calibrations only.")
            extrinsics = {}
        else:
            # Step 3: Global optimization
            logger.info("Step 3: Global pose graph optimization...")
            logger.info(f"   Using {len(pairwise_extrinsics)} successful pairwise calibrations")
            extrinsics = self._global_optimization(camera_ids, pairwise_extrinsics, reference_camera)
        
        # Create calibration data
        calibration_data = CalibrationData(
            intrinsics=intrinsics,
            extrinsics=extrinsics,
            board_config=self.board_config,
            reference_camera=reference_camera
        )
        
        logger.info("Frame-optimized multi-camera calibration completed successfully!")
        return calibration_data
    
    def _global_optimization(self, 
                           camera_ids: List[str], 
                           pairwise_extrinsics: Dict[Tuple[str, str], Dict], 
                           reference_camera: str) -> Dict[str, CameraExtrinsics]:
        """
        Perform global pose graph optimization
        
        Args:
            camera_ids: List of all camera IDs
            pairwise_extrinsics: Pairwise calibration results
            reference_camera: Reference camera ID
            
        Returns:
            Dictionary of optimized extrinsics relative to reference camera
        """
        optimizer = GlobalOptimizer()
        camera_keys = {}
        
        # Ensure graph connectivity to reference (avoid floating subgraphs)
        self._ensure_connected(camera_ids, pairwise_extrinsics, reference_camera)

        # Add camera poses to graph
        for camera_id in camera_ids:
            # Start with identity pose for all cameras
            pose = gtsam.Pose3()
            key = optimizer.add_camera_pose(camera_id, pose)
            camera_keys[camera_id] = key
        
        # Add prior constraint for reference camera (fix it at origin)
        reference_pose = gtsam.Pose3()
        optimizer.add_prior_constraint(camera_keys[reference_camera], reference_pose)
        
        # Add pairwise constraints
        for (cam1, cam2), extrinsic_data in pairwise_extrinsics.items():
            R = extrinsic_data['R']
            t = extrinsic_data['t'].flatten()
            error = extrinsic_data['error']
            
            # Log stereo calibration quality
            if error > 10.0:
                logger.warning(f"High stereo RMS error for pair ({cam1},{cam2}): {error:.2f} pixels")
                logger.warning("This may indicate poor synchronization or detection quality")
            elif error > 5.0:
                logger.info(f"Moderate stereo RMS error for pair ({cam1},{cam2}): {error:.2f} pixels")
            else:
                logger.info(f"Good stereo RMS error for pair ({cam1},{cam2}): {error:.2f} pixels")
            
            # Convert to GTSAM pose
            relative_pose = gtsam.Pose3(gtsam.Rot3(R), gtsam.Point3(t))
            optimizer.add_relative_pose_constraint(
                camera_keys[cam1], camera_keys[cam2], relative_pose, pair_rms=float(error)
            )
        
        # Optimize
        try:
            result = optimizer.optimize()
            
            # Extract optimized poses
            extrinsics = {}
            for camera_id in camera_ids:
                if camera_id != reference_camera:
                    optimized_pose = result.atPose3(camera_keys[camera_id])
                    reference_pose = result.atPose3(camera_keys[reference_camera])
                    
                    # Compute relative pose from reference to this camera
                    relative_pose = reference_pose.inverse().compose(optimized_pose)
                    
                    R = relative_pose.rotation().matrix()
                    t = relative_pose.translation()
                    rvec, _ = cv2.Rodrigues(R)
                    
                    extrinsics[camera_id] = CameraExtrinsics(
                        camera_id=camera_id,
                        reference_camera_id=reference_camera,
                        rotation_matrix=R,
                        translation_vector=t.reshape(-1, 1),
                        rotation_vector=rvec.flatten()
                    )
            
            return extrinsics
            
        except Exception as e:
            logger.error(f"Global optimization failed: {e}")
            logger.info("Falling back to pairwise calibrations...")
            
            # Fallback: use pairwise calibrations directly
            return self._fallback_extrinsics(camera_ids, pairwise_extrinsics, reference_camera)

    def _ensure_connected(self,
                          camera_ids: List[str],
                          pairwise_extrinsics: Dict[Tuple[str, str], Dict],
                          reference_camera: str) -> None:
        # Union-Find to ensure all cameras connect to reference via edges
        parent = {c: c for c in camera_ids}

        def find(x: str) -> str:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a: str, b: str) -> None:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[rb] = ra

        for (a, b) in pairwise_extrinsics.keys():
            union(a, b)

        root_ref = find(reference_camera)
        disconnected = [c for c in camera_ids if find(c) != root_ref]
        if disconnected:
            raise RuntimeError(f"Disconnected cameras (no chain to {reference_camera}): {disconnected}")
    
    def _fallback_extrinsics(self, 
                           camera_ids: List[str], 
                           pairwise_extrinsics: Dict[Tuple[str, str], Dict], 
                           reference_camera: str) -> Dict[str, CameraExtrinsics]:
        """
        Fallback method: use pairwise calibrations without global optimization
        """
        extrinsics = {}
        
        for camera_id in camera_ids:
            if camera_id == reference_camera:
                continue
            
            # Find direct pairwise calibration with reference camera
            pair_key = None
            is_inverted = False
            
            if (reference_camera, camera_id) in pairwise_extrinsics:
                pair_key = (reference_camera, camera_id)
                is_inverted = False
            elif (camera_id, reference_camera) in pairwise_extrinsics:
                pair_key = (camera_id, reference_camera)
                is_inverted = True
            
            if pair_key:
                R = pairwise_extrinsics[pair_key]['R']
                t = pairwise_extrinsics[pair_key]['t']
                
                if is_inverted:
                    # Invert the transformation
                    R = R.T
                    t = -R @ t
                
                rvec, _ = cv2.Rodrigues(R)
                
                extrinsics[camera_id] = CameraExtrinsics(
                    camera_id=camera_id,
                    reference_camera_id=reference_camera,
                    rotation_matrix=R,
                    translation_vector=t.reshape(-1, 1),
                    rotation_vector=rvec.flatten(),
                    calibration_error=pairwise_extrinsics[pair_key]['error']
                )
            else:
                logger.warning(f"No direct calibration found for camera {camera_id} with reference {reference_camera}")
        
        return extrinsics

def load_calibration_from_toml(filepath: Path) -> CalibrationData:
    """
    Load calibration data from TOML file
    
    Args:
        filepath: Path to calibration.toml file
        
    Returns:
        CalibrationData object
    """
    with open(filepath, 'r') as f:
        data = toml.load(f)
    
    # Load metadata
    metadata = data.get('metadata', {})
    reference_camera = metadata.get('reference_camera')
    board_config_data = metadata.get('board_config', {})
    
    # Reconstruct board configuration
    board_config = ChArUcoBoard(
        squares_x=board_config_data.get('squares_x', 10),
        squares_y=board_config_data.get('squares_y', 7),
        square_length=board_config_data.get('square_length', 25.0),
        marker_length=board_config_data.get('marker_length', 18.75),
        marker_dict=board_config_data.get('marker_dict', cv2.aruco.DICT_6X6_50)
    )
    
    # Load camera intrinsics and extrinsics
    intrinsics = {}
    extrinsics = {}
    
    for key, value in data.items():
        if key.startswith('cam_'):
            camera_id = key[4:]  # Remove 'cam_' prefix
            
            # Load intrinsics
            intrinsics[camera_id] = CameraIntrinsics(
                camera_id=camera_id,
                image_size=tuple(value['size']),
                camera_matrix=np.array(value['matrix']),
                distortion_coeffs=np.array(value['distortions']),
                calibration_error=value.get('calibration_error', 0.0)
            )
            
            # Load extrinsics (if not reference camera)
            if camera_id != reference_camera and 'rotation_matrix' in value:
                extrinsics[camera_id] = CameraExtrinsics(
                    camera_id=camera_id,
                    reference_camera_id=reference_camera,
                    rotation_matrix=np.array(value['rotation_matrix']),
                    translation_vector=np.array(value['translation']).reshape(-1, 1),
                    rotation_vector=np.array(value['rotation'])
                )
    
    return CalibrationData(
        intrinsics=intrinsics,
        extrinsics=extrinsics,
        board_config=board_config,
        reference_camera=reference_camera
    )

# Example usage and testing functions
def example_calibration():
    """Example of how to use the multi-camera calibration system"""
    
    # Configure ChArUco board
    board_config = ChArUcoBoard(
        squares_x=10,
        squares_y=7,
        square_length=25.0,  # mm
        marker_length=18.75,  # mm
        marker_dict=cv2.aruco.DICT_6X6_50
    )
    
    # Initialize calibrator
    calibrator = MultiCameraCalibrator(board_config)
    
    # Load calibration images for each camera
    # In practice, you would load actual images here
    camera_images = {
        'cam1': [],  # List of calibration images for camera 1
        'cam2': [],  # List of calibration images for camera 2
        'cam3': [],  # List of calibration images for camera 3
        # Add more cameras as needed
    }
    
    # Perform calibration
    try:
        calibration_data = calibrator.calibrate_all_cameras(
            camera_images, 
            reference_camera='cam1'
        )
        
        # Save calibration to file
        output_path = Path('calibration/calibration.toml')
        output_path.parent.mkdir(exist_ok=True)
        calibration_data.save_to_file(output_path)
        
        print(f"Calibration completed and saved to {output_path}")
        
    except Exception as e:
        logger.error(f"Calibration failed: {e}")

if __name__ == "__main__":
    example_calibration()
