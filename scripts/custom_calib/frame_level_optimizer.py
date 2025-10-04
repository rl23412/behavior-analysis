#!/usr/bin/env python3
"""
Frame-Level Calibration Optimizer

This module identifies individual frames with successful ChArUco detection
and optimizes calibration by using only the best frames:

1. Scan entire video to find ALL frames with good ChArUco detection
2. For intrinsics: Use all good frames from each camera individually
3. For pairwise: Use only synchronized frames where BOTH cameras have good detection

Author: Frame-Level Optimizer
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Set
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class FrameDetection:
    """Information about ChArUco detection in a single frame"""
    frame_number: int
    corner_count: int
    marker_count: int
    quality_score: float  # Combined metric of corner count and detection confidence
    corners: np.ndarray  # Actual corner coordinates (Nx1x2)
    ids: np.ndarray      # Corner IDs (Nx1)
    
    def __repr__(self):
        return f"Frame {self.frame_number}: {self.corner_count} corners, quality={self.quality_score:.2f}"

@dataclass
class CameraFrameAnalysis:
    """Complete frame-by-frame analysis for a single camera"""
    camera_id: str
    video_path: Path
    total_frames: int
    good_frames: List[FrameDetection]  # All frames with successful detection
    
    def get_best_frames_for_intrinsics(self, max_frames: int = None) -> List[int]:
        """Get best frame numbers for intrinsic calibration"""
        if not self.good_frames:
            return []
        
        # Use ALL available good frames if max_frames is None, otherwise limit
        if max_frames is None:
            selected = self.good_frames  # Use ALL good frames
        else:
            # Sort by quality score and take the best frames
            sorted_frames = sorted(self.good_frames, key=lambda f: f.quality_score, reverse=True)
            selected = sorted_frames[:max_frames]
        
        # Return frame numbers sorted by frame order (not quality)
        return sorted([f.frame_number for f in selected])
    
    def get_frame_set(self) -> Set[int]:
        """Get set of all good frame numbers for intersection operations"""
        return {f.frame_number for f in self.good_frames}
    
    def get_detection_results_for_frames(self, frame_numbers: List[int]) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Get cached ChArUco detection results for specific frames
        
        Args:
            frame_numbers: List of frame numbers to get results for
            
        Returns:
            Tuple of (all_corners, all_ids) for calibration use
        """
        # Create lookup dict for fast access
        frame_detections = {f.frame_number: f for f in self.good_frames}
        
        all_corners = []
        all_ids = []
        
        for frame_num in frame_numbers:
            if frame_num in frame_detections:
                detection = frame_detections[frame_num]
                all_corners.append(detection.corners)
                all_ids.append(detection.ids)
            else:
                logger.warning(f"Frame {frame_num} not found in cached detections for camera {self.camera_id}")
        
        logger.info(f"Retrieved {len(all_corners)} cached detections for camera {self.camera_id}")
        return all_corners, all_ids
    
    def get_all_detection_results(self) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Get all cached ChArUco detection results
        
        Returns:
            Tuple of (all_corners, all_ids) for all good frames
        """
        all_corners = [f.corners for f in self.good_frames]
        all_ids = [f.ids for f in self.good_frames]
        
        logger.info(f"Retrieved all {len(all_corners)} cached detections for camera {self.camera_id}")
        return all_corners, all_ids

class FrameLevelAnalyzer:
    """Analyzes videos frame-by-frame to find optimal calibration frames"""
    
    def __init__(self, board_config):
        from custom_calibration import ChArUcoDetector
        self.detector = ChArUcoDetector(board_config)
        self.board_config = board_config
    
    def find_synchronized_frames_for_pairs(self, 
                                          camera_analyses: Dict[str, CameraFrameAnalysis]) -> Dict[Tuple[str, str], List[int]]:
        """
        Find synchronized frames where both cameras in each pair have good ChArUco detection
        
        Args:
            camera_analyses: Dictionary of camera analyses
            
        Returns:
            Dictionary mapping camera pairs to lists of synchronized good frame numbers
        """
        logger.info("🔗 FINDING SYNCHRONIZED FRAMES FOR PAIRWISE CALIBRATION")
        logger.info("="*60)
        
        camera_ids = list(camera_analyses.keys())
        synchronized_frames = {}
        
        # Check all camera pairs
        for i, cam1 in enumerate(camera_ids):
            for j, cam2 in enumerate(camera_ids[i+1:], i+1):
                logger.info(f"Analyzing pair ({cam1}, {cam2})...")
                
                # Get good frame sets for each camera
                cam1_frames = camera_analyses[cam1].get_frame_set()
                cam2_frames = camera_analyses[cam2].get_frame_set()
                
                # Find intersection (frames where BOTH cameras have good detection)
                synchronized_good_frames = cam1_frames.intersection(cam2_frames)
                
                if synchronized_good_frames:
                    # Use ALL synchronized frames for maximum calibration accuracy
                    frame_list = sorted(list(synchronized_good_frames))
                    
                    logger.info(f"   Using ALL {len(frame_list)} synchronized frames (no artificial limits)")
                    logger.info(f"   Maximum frames = maximum calibration accuracy")
                    
                    synchronized_frames[(cam1, cam2)] = frame_list
                    
                    logger.info(f"   ✅ Found {len(frame_list)} synchronized frames")
                    logger.info(f"      Frame range: {min(frame_list)}-{max(frame_list)}")
                    logger.info(f"      Sample frames: {frame_list[:5]}...{frame_list[-5:] if len(frame_list) > 5 else []}")
                else:
                    logger.warning(f"   ❌ No synchronized frames found for pair ({cam1}, {cam2})")
                    synchronized_frames[(cam1, cam2)] = []
        
        return synchronized_frames
    
    def analyze_camera_frames(self, video_path: Path, camera_id: str, 
                             sample_interval: int = 3) -> CameraFrameAnalysis:
        """
        Analyze every Nth frame in video to find all frames with good ChArUco detection
        
        Args:
            video_path: Path to calibration video
            camera_id: Camera identifier
            sample_interval: Analyze every Nth frame (3 = every 3rd frame)
            
        Returns:
            CameraFrameAnalysis with all good frames identified
        """
        logger.info(f"🔍 Analyzing ALL frames for camera {camera_id}...")
        logger.info(f"   Video: {video_path.name}")
        
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        logger.info(f"   Total frames: {total_frames} ({total_frames/fps:.1f}s at {fps:.1f}fps)")
        if sample_interval == 1:
            logger.info(f"   Analyzing ALL {total_frames} frames for maximum accuracy...")
        else:
            logger.info(f"   Analyzing every {sample_interval} frames...")
        
        good_frames = []
        frames_analyzed = 0
        
        # Analyze every Nth frame throughout the entire video for global optimization
        for frame_num in range(0, total_frames, sample_interval):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ret, frame = cap.read()
            if not ret:
                continue
            
            frames_analyzed += 1
            
            # Test ChArUco detection on this frame
            corners, ids = self.detector.detect_charuco_corners(frame)
            
            # Moderately stricter requirements for calibration-ready frames
            if corners is not None and ids is not None and len(corners) >= 8:  # Need >=8 corners for stable calibration
                # Calculate quality score based on corner count and distribution
                quality_score = self._calculate_frame_quality(corners, ids, frame.shape)

                # Additional validation: check corner distribution
                if self._validate_corner_distribution(corners, frame.shape):
                    detection = FrameDetection(
                        frame_number=frame_num,
                        corner_count=len(corners),
                        marker_count=len(ids),
                        quality_score=quality_score,
                        corners=corners.copy(),  # Store actual corner coordinates
                        ids=ids.copy()          # Store actual corner IDs
                    )

                    good_frames.append(detection)

                    if len(good_frames) <= 5:  # Log first few detections
                        logger.info(f"   ✅ Frame {frame_num}: {len(corners)} corners, quality={quality_score:.2f}")
            
            # Progress reporting - more frequent for all-frame analysis
            if sample_interval == 1:
                # Every 1000 frames when analyzing all frames
                if frames_analyzed % 1000 == 0:
                    logger.info(f"   Analyzed {frames_analyzed}/{total_frames} frames, found {len(good_frames)} good frames...")
            else:
                # Every 300 frames when sampling
                if frames_analyzed % 300 == 0:
                    logger.info(f"   Analyzed {frames_analyzed} frames, found {len(good_frames)} good frames...")
        
        cap.release()
        
        # Final progress report
        if sample_interval == 1:
            logger.info(f"   ✅ Completed analysis of ALL {frames_analyzed} frames")
        
        detection_rate = len(good_frames) / frames_analyzed if frames_analyzed > 0 else 0
        
        logger.info(f"✅ Camera {camera_id} analysis complete:")
        logger.info(f"   Frames analyzed: {frames_analyzed}")
        logger.info(f"   Good frames found: {len(good_frames)} ({detection_rate:.1%} success rate)")
        
        if good_frames:
            avg_quality = np.mean([f.quality_score for f in good_frames])
            best_frame = max(good_frames, key=lambda f: f.quality_score)
            logger.info(f"   Average quality: {avg_quality:.2f}")
            logger.info(f"   Best frame: {best_frame}")
        
        return CameraFrameAnalysis(
            camera_id=camera_id,
            video_path=video_path,
            total_frames=total_frames,
            good_frames=good_frames
        )
    
    def _calculate_frame_quality(self, corners: np.ndarray, ids: np.ndarray, 
                                image_shape: Tuple[int, int, int]) -> float:
        """
        Calculate quality score for a frame based on ChArUco detection
        
        Args:
            corners: Detected ChArUco corners
            ids: Corner IDs
            image_shape: Image dimensions
            
        Returns:
            Quality score (higher = better, max ~3.0 for excellent frames)
        """
        corner_count = len(corners)
        
        # Base score from corner count - higher threshold for good detection
        corner_score = min(corner_count / 25.0, 2.0)  # Cap at 2.0 for 25+ corners
        
        # Distribution score - corners should be spread across image
        if corner_count >= 8:
            corners_reshaped = corners.reshape(-1, 2)
            
            # Calculate spread (standard deviation of corner positions)
            x_spread = np.std(corners_reshaped[:, 0])
            y_spread = np.std(corners_reshaped[:, 1])
            
            # Normalize by image size
            height, width = image_shape[:2]
            spread_score = (x_spread / width + y_spread / height)
            
            # Coverage score - how much of image area is covered
            x_range = np.max(corners_reshaped[:, 0]) - np.min(corners_reshaped[:, 0])
            y_range = np.max(corners_reshaped[:, 1]) - np.min(corners_reshaped[:, 1])
            coverage_score = (x_range / width) * (y_range / height)
            
        else:
            spread_score = 0.0
            coverage_score = 0.0
        
        # Combined quality score (corner_score + distribution bonuses)
        quality_score = corner_score * (1.0 + spread_score + coverage_score)
        
        return quality_score

    def _validate_corner_distribution(self, corners: np.ndarray, image_shape: Tuple[int, int, int]) -> bool:
        """
        Validate that ChArUco corners are reasonably distributed for stable calibration
        
        Args:
            corners: Detected ChArUco corners
            image_shape: Image dimensions
            
        Returns:
            True if corners are reasonably distributed for calibration
        """
        if len(corners) < 8:  # Minimum corners for any calibration
            return False
        
        height, width = image_shape[:2]
        corners_reshaped = corners.reshape(-1, 2)
        
        # Check coverage of image regions (more lenient)
        # Divide image into 2x2 grid (simpler than 3x3)
        x_bins = np.linspace(0, width, 3)
        y_bins = np.linspace(0, height, 3)
        
        regions_with_corners = 0
        for i in range(2):
            for j in range(2):
                x_min, x_max = x_bins[i], x_bins[i+1]
                y_min, y_max = y_bins[j], y_bins[j+1]
                
                # Count corners in this region
                in_region = np.sum(
                    (corners_reshaped[:, 0] >= x_min) & (corners_reshaped[:, 0] < x_max) &
                    (corners_reshaped[:, 1] >= y_min) & (corners_reshaped[:, 1] < y_max)
                )
                
                if in_region > 0:
                    regions_with_corners += 1
        
        # Need corners in at least 2 out of 4 regions (more lenient)
        if regions_with_corners < 2:
            return False
        
        # Check that corners span reasonable portion of image (more lenient)
        x_span = np.max(corners_reshaped[:, 0]) - np.min(corners_reshaped[:, 0])
        y_span = np.max(corners_reshaped[:, 1]) - np.min(corners_reshaped[:, 1])
        
        # Need to span at least 30% of image in both dimensions (was 60%)
        if x_span < 0.3 * width or y_span < 0.3 * height:
            return False
        
        # Only check for severely degenerate cases
        if len(corners) >= 6:
            try:
                centered_corners = corners_reshaped - np.mean(corners_reshaped, axis=0)
                _, s, _ = np.linalg.svd(centered_corners, full_matrices=False)
                
                # Much more lenient condition number check
                condition_number = s[0] / s[-1] if s[-1] > 1e-10 else np.inf
                if condition_number > 500:  # Only reject severely degenerate patterns (was 100)
                    return False
            except:
                pass  # If SVD fails, just accept the frame
        
        return True

    def find_synchronized_frames_for_pairs_gated(self,
                                                 camera_analyses: Dict[str, CameraFrameAnalysis],
                                                 min_quality: float = 0.7,
                                                 min_frames: int = 20) -> Dict[Tuple[str, str], List[int]]:
        """
        Quality-gated synchronized frame selection for pairwise calibration.

        - Starts from frame intersections across two cameras
        - Keeps frames where BOTH cameras' quality_score >= min_quality
        - Relaxes threshold gradually until at least min_frames are available

        Args:
            camera_analyses: Dict of per-camera analyses
            min_quality: Min per-camera quality score (adaptive)
            min_frames: Target minimum number of synchronized frames per pair

        Returns:
            Dict mapping (cam1, cam2) -> sorted list of synchronized frame numbers
        """
        logger.info("🔗 FINDING SYNCHRONIZED FRAMES (quality-gated)")
        logger.info("="*60)

        camera_ids = list(camera_analyses.keys())
        synchronized_frames: Dict[Tuple[str, str], List[int]] = {}

        # Frame -> detection lookups
        det_by_cam: Dict[str, Dict[int, FrameDetection]] = {
            cam_id: {fd.frame_number: fd for fd in analysis.good_frames}
            for cam_id, analysis in camera_analyses.items()
        }

        for i, cam1 in enumerate(camera_ids):
            for j, cam2 in enumerate(camera_ids[i+1:], i+1):
                logger.info(f"Analyzing pair ({cam1}, {cam2})...")

                inter = set(det_by_cam[cam1].keys()).intersection(det_by_cam[cam2].keys())
                if not inter:
                    logger.warning(f"   No synchronized frames found for pair ({cam1}, {cam2})")
                    synchronized_frames[(cam1, cam2)] = []
                    continue

                thr = float(min_quality)
                selected: List[int] = []
                while thr >= 0.35 and len(selected) < min_frames:
                    gated = [
                        fr for fr in inter
                        if det_by_cam[cam1][fr].quality_score >= thr and det_by_cam[cam2][fr].quality_score >= thr
                    ]
                    selected = sorted(gated)
                    if len(selected) < min_frames:
                        thr *= 0.9

                if not selected:
                    # Fallback: use intersection as-is
                    selected = sorted(list(inter))
                    logger.warning(f"   Using ungated {len(selected)} synchronized frames (quality too low)")
                else:
                    logger.info(f"   Selected {len(selected)} synchronized frames (min_quality≈{thr:.2f})")

                synchronized_frames[(cam1, cam2)] = selected

                if selected:
                    logger.info(f"      Frame range: {selected[0]}-{selected[-1]}")
                    logger.info(f"      Sample: {selected[:5]}...{selected[-5:] if len(selected) > 5 else selected}")

        return synchronized_frames
    def find_synchronized_frames_for_pairs(self, 
                                          camera_analyses: Dict[str, CameraFrameAnalysis]) -> Dict[Tuple[str, str], List[int]]:
        """
        Find synchronized frames where both cameras in each pair have good ChArUco detection
        
        Args:
            camera_analyses: Dictionary of camera analyses
            
        Returns:
            Dictionary mapping camera pairs to lists of synchronized good frame numbers
        """
        logger.info("🔗 FINDING SYNCHRONIZED FRAMES FOR PAIRWISE CALIBRATION")
        logger.info("="*60)
        
        camera_ids = list(camera_analyses.keys())
        synchronized_frames = {}
        
        # Check all camera pairs
        for i, cam1 in enumerate(camera_ids):
            for j, cam2 in enumerate(camera_ids[i+1:], i+1):
                logger.info(f"Analyzing pair ({cam1}, {cam2})...")
                
                # Get good frame sets for each camera
                cam1_frames = camera_analyses[cam1].get_frame_set()
                cam2_frames = camera_analyses[cam2].get_frame_set()
                
                # Find intersection (frames where BOTH cameras have good detection)
                synchronized_good_frames = cam1_frames.intersection(cam2_frames)
                
                if synchronized_good_frames:
                    # Use ALL synchronized frames for maximum calibration accuracy
                    frame_list = sorted(list(synchronized_good_frames))
                    
                    logger.info(f"   Using ALL {len(frame_list)} synchronized frames (no artificial limits)")
                    logger.info(f"   Maximum frames = maximum calibration accuracy")
                    
                    synchronized_frames[(cam1, cam2)] = frame_list
                    
                    logger.info(f"   ✅ Found {len(frame_list)} synchronized frames")
                    logger.info(f"      Frame range: {min(frame_list)}-{max(frame_list)}")
                    logger.info(f"      Sample frames: {frame_list[:5]}...{frame_list[-5:] if len(frame_list) > 5 else []}")
                else:
                    logger.warning(f"   ❌ No synchronized frames found for pair ({cam1}, {cam2})")
                    synchronized_frames[(cam1, cam2)] = []
        
        return synchronized_frames

def extract_frames_by_numbers(video_path: Path, frame_numbers: List[int], 
                             output_dir: Path, prefix: str = "") -> List[Path]:
    """
    Extract specific frames by frame numbers
    
    Args:
        video_path: Path to video
        frame_numbers: Specific frame numbers to extract
        output_dir: Output directory
        prefix: Prefix for saved frame files
        
    Returns:
        List of extracted frame paths
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    
    frame_paths = []
    
    logger.info(f"Extracting {len(frame_numbers)} specific frames...")
    
    for i, frame_num in enumerate(frame_numbers):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        if ret:
            frame_filename = f"{prefix}frame_{frame_num:06d}.jpg"
            frame_path = output_dir / frame_filename
            cv2.imwrite(str(frame_path), frame)
            frame_paths.append(frame_path)
            
            if (i + 1) % 20 == 0:
                logger.info(f"  Extracted {i + 1}/{len(frame_numbers)} frames...")
    
    cap.release()
    logger.info(f"✅ Extracted {len(frame_paths)} frames by frame numbers")
    
    return frame_paths

def check_frame_synchronization(calibration_videos: Dict[str, Path], 
                               board_config) -> Dict[Tuple[str, str], int]:
    """
    Check if calibration videos are truly synchronized by analyzing detection patterns
    
    Returns:
        Dictionary mapping camera pairs to estimated frame offset
    """
    logger.info("🔍 CHECKING FRAME SYNCHRONIZATION")
    logger.info("="*50)
    
    analyzer = FrameLevelAnalyzer(board_config)
    camera_analyses = {}
    
    # Analyze each camera - use ALL frames for synchronization check
    for camera_id, video_path in calibration_videos.items():
        analysis = analyzer.analyze_camera_frames(video_path, camera_id, sample_interval=1)  # Every frame
        camera_analyses[camera_id] = analysis
    
    # Check synchronization between pairs
    offsets = {}
    camera_ids = list(camera_analyses.keys())
    
    for i, cam1 in enumerate(camera_ids):
        for j, cam2 in enumerate(camera_analyses.keys()):
            if i >= j:
                continue
                
            logger.info(f"Checking synchronization: {cam1} vs {cam2}")
            
            # Get detection patterns (boolean arrays)
            cam1_frames = {f.frame_number for f in camera_analyses[cam1].good_frames}
            cam2_frames = {f.frame_number for f in camera_analyses[cam2].good_frames}
            
            # Find common frame range
            all_frames = sorted(cam1_frames.union(cam2_frames))
            if not all_frames:
                logger.warning(f"No common frames between {cam1} and {cam2}")
                continue
                
            # Create detection arrays
            max_frame = max(all_frames)
            cam1_detection = np.zeros(max_frame + 1, dtype=bool)
            cam2_detection = np.zeros(max_frame + 1, dtype=bool)
            
            for frame in cam1_frames:
                if frame <= max_frame:
                    cam1_detection[frame] = True
            for frame in cam2_frames:
                if frame <= max_frame:
                    cam2_detection[frame] = True
            
            # Cross-correlation to find offset
            correlation = np.correlate(cam1_detection.astype(float), cam2_detection.astype(float), mode='full')
            offset = np.argmax(correlation) - len(cam2_detection) + 1
            
            offsets[(cam1, cam2)] = offset
            
            if abs(offset) > 10:
                logger.warning(f"Large frame offset detected: {cam1} vs {cam2} = {offset} frames")
                logger.warning("Videos may not be properly synchronized!")
            else:
                logger.info(f"Frame offset: {cam1} vs {cam2} = {offset} frames")
    
    return offsets

def optimize_calibration_frame_by_frame(calibration_videos: Dict[str, Path], 
                                       board_config) -> Tuple[Dict[str, List[np.ndarray]], 
                                                            Dict[Tuple[str, str], Dict[str, List[np.ndarray]]],
                                                            Dict[str, CameraFrameAnalysis]]:
    """
    Frame-by-frame optimization for multi-camera calibration with caching
    
    This function:
    1. Finds ALL good frames for each camera individually  
    2. Caches ChArUco detection results to avoid redundant detection
    3. Finds synchronized good frames for each camera pair
    4. Extracts optimal frames for both intrinsic and pairwise calibration
    
    Args:
        calibration_videos: Dict mapping camera_id to video path
        board_config: ChArUco board configuration
        
    Returns:
        Tuple of:
        - camera_images: Dict[camera_id, List[images]] for intrinsic calibration
        - pairwise_images: Dict[(cam1,cam2), Dict[camera_id, List[images]]] for pairwise calibration  
        - camera_analyses: Dict[camera_id, CameraFrameAnalysis] with cached detection results
    """
    logger.info("🎯 FRAME-BY-FRAME CALIBRATION OPTIMIZATION")
    logger.info("="*70)
    
    # First check if videos are properly synchronized
    logger.info("🔍 STEP 0: CHECKING VIDEO SYNCHRONIZATION")
    logger.info("="*50)
    logger.info("⚠️  Analyzing ALL frames for maximum synchronization accuracy...")
    logger.info("   This may take several minutes but ensures perfect frame alignment")
    sync_offsets = check_frame_synchronization(calibration_videos, board_config)
    
    # Warn about large offsets
    for (cam1, cam2), offset in sync_offsets.items():
        if abs(offset) > 10:
            logger.error(f"❌ CRITICAL: Large frame offset {offset} between {cam1} and {cam2}")
            logger.error("This will cause high stereo RMS errors!")
            logger.error("Videos need to be properly synchronized before calibration")
        elif abs(offset) > 5:
            logger.warning(f"⚠️ Moderate frame offset {offset} between {cam1} and {cam2}")
    
    analyzer = FrameLevelAnalyzer(board_config)
    
    # Step 1: Analyze each camera to find ALL good frames
    logger.info("📹 STEP 1: ANALYZING INDIVIDUAL CAMERAS")
    logger.info("="*50)
    
    camera_analyses = {}
    for camera_id, video_path in calibration_videos.items():
        analysis = analyzer.analyze_camera_frames(video_path, camera_id, sample_interval=1)  # Every frame
        camera_analyses[camera_id] = analysis
        
        if not analysis.good_frames:
            logger.error(f"❌ Camera {camera_id}: No good frames found!")
        else:
            logger.info(f"✅ Camera {camera_id}: {len(analysis.good_frames)} good frames found")
    
    # Step 2: Find synchronized frames for pairwise calibration
    logger.info("\n🔗 STEP 2: FINDING SYNCHRONIZED FRAMES FOR PAIRS")
    logger.info("="*50)
    
    # Use quality-gated synchronized frame selection
    synchronized_frames = analyzer.find_synchronized_frames_for_pairs_gated(camera_analyses)
    
    # Step 3: Extract images for intrinsic calibration (per camera)
    logger.info("\n📸 STEP 3: EXTRACTING INTRINSIC CALIBRATION IMAGES")
    logger.info("="*50)
    
    camera_images = {}
    
    for camera_id, analysis in camera_analyses.items():
        if not analysis.good_frames:
            logger.error(f"Skipping camera {camera_id}: no good frames")
            continue
        
        logger.info(f"Extracting intrinsic images for camera {camera_id}...")
        
        # Get ALL good frames for this camera's intrinsic calibration  
        best_frame_numbers = analysis.get_best_frames_for_intrinsics(max_frames=None)  # Use ALL frames
        
        if not best_frame_numbers:
            logger.error(f"No optimal frames for camera {camera_id}")
            continue
        
        # Extract these specific frames
        temp_dir = Path(f"temp_intrinsic_cam{camera_id}")
        frame_paths = extract_frames_by_numbers(
            analysis.video_path, best_frame_numbers, temp_dir, f"intrinsic_cam{camera_id}_"
        )
        
        # Load images
        images = load_images_from_paths(frame_paths)
        
        if images:
            camera_images[camera_id] = images
            logger.info(f"✅ Camera {camera_id}: {len(images)} optimal intrinsic images")
            logger.info(f"   Frame range: {min(best_frame_numbers)}-{max(best_frame_numbers)}")
        
        # Clean up
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    # Step 4: Extract images for pairwise calibration
    logger.info("\n🔗 STEP 4: EXTRACTING PAIRWISE CALIBRATION IMAGES") 
    logger.info("="*50)
    
    pairwise_images = {}
    
    for (cam1, cam2), sync_frames in synchronized_frames.items():
        if not sync_frames:
            logger.warning(f"No synchronized frames for pair ({cam1}, {cam2})")
            continue
        
        logger.info(f"Extracting pairwise images for ({cam1}, {cam2})...")
        logger.info(f"   Using {len(sync_frames)} synchronized frames")
        logger.info(f"   Frame range: {min(sync_frames)}-{max(sync_frames)}")
        
        pair_images = {}
        
        # Extract synchronized frames from both cameras
        for camera_id in [cam1, cam2]:
            video_path = camera_analyses[camera_id].video_path
            
            temp_dir = Path(f"temp_pairwise_{cam1}_{cam2}_cam{camera_id}")
            frame_paths = extract_frames_by_numbers(
                video_path, sync_frames, temp_dir, f"pairwise_{cam1}{cam2}_cam{camera_id}_"
            )
            
            images = load_images_from_paths(frame_paths)
            
            if images:
                pair_images[camera_id] = images
                logger.info(f"   ✅ Camera {camera_id}: {len(images)} synchronized images")
            
            # Clean up
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)
        
        if len(pair_images) == 2:  # Both cameras successful
            pairwise_images[(cam1, cam2)] = pair_images
            logger.info(f"✅ Pair ({cam1}, {cam2}): Ready for pairwise calibration")
        else:
            logger.error(f"❌ Pair ({cam1}, {cam2}): Failed to extract images from both cameras")
    
    # Summary
    logger.info("\n📊 FRAME-LEVEL OPTIMIZATION SUMMARY")
    logger.info("="*50)
    logger.info(f"Cameras with intrinsic images: {list(camera_images.keys())}")
    logger.info(f"Pairs ready for calibration: {list(pairwise_images.keys())}")
    logger.info(f"Cached detection results: {list(camera_analyses.keys())}")
    
    for camera_id, images in camera_images.items():
        logger.info(f"   Camera {camera_id}: {len(images)} intrinsic images")
        if camera_id in camera_analyses:
            logger.info(f"      ⚡ {len(camera_analyses[camera_id].good_frames)} cached detections available")
    
    for (cam1, cam2), pair_data in pairwise_images.items():
        logger.info(f"   Pair ({cam1},{cam2}): {len(pair_data[cam1])} synchronized images each")
    
    return camera_images, pairwise_images, camera_analyses

def load_images_from_paths(image_paths: List[Path]) -> List[np.ndarray]:
    """Load images from file paths"""
    images = []
    for path in image_paths:
        image = cv2.imread(str(path))
        if image is not None:
            images.append(image)
        else:
            logger.warning(f"Could not load image: {path}")
    
    return images

if __name__ == "__main__":
    # Test with local videos
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    
    calibration_videos = {
        "1": Path(r"C:\Users\Runda\Desktop\9.12\session1\synced_COMPLETE\calib\cam1-calib.mp4"),
        "2": Path(r"C:\Users\Runda\Desktop\9.12\session1\synced_COMPLETE\calib\cam2-calib.mp4"),
        "3": Path(r"C:\Users\Runda\Desktop\9.12\session1\synced_COMPLETE\calib\cam3-calib.mp4"),
    }
    
    available_videos = {k: v for k, v in calibration_videos.items() if v.exists()}
    
    if available_videos:
        print(f"🧪 Testing frame-level optimization with {len(available_videos)} videos")
        
        from custom_calibration import ChArUcoBoard
        board_config = ChArUcoBoard()
        
        camera_images, pairwise_images = optimize_calibration_frame_by_frame(
            available_videos, board_config
        )
        
        print(f"\n✅ Frame-level optimization completed!")
        print(f"   Intrinsic calibration ready for: {list(camera_images.keys())}")
        print(f"   Pairwise calibration ready for: {list(pairwise_images.keys())}")
    else:
        print("❌ No test videos available")
