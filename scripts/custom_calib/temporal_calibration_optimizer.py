#!/usr/bin/env python3
"""
Temporal Calibration Optimizer

This module finds optimal time periods for calibration in synchronized multi-camera videos:
1. Analyzes each camera video to find periods with good ChArUco board visibility
2. Scores the quality of ChArUco detection over time
3. Finds optimal periods for intrinsic calibration (per camera)
4. Finds synchronized overlapping periods for pairwise calibration
5. Extracts frames from optimal periods for best calibration quality

Author: Temporal Optimizer
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging
from dataclasses import dataclass
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

@dataclass
class DetectionPeriod:
    """A period of good ChArUco detection in a video"""
    start_frame: int
    end_frame: int
    quality_score: float
    detection_rate: float
    avg_corners: float
    frame_count: int
    
    def duration_frames(self) -> int:
        return self.end_frame - self.start_frame + 1
    
    def __repr__(self):
        return f"Period[{self.start_frame}-{self.end_frame}]: {self.detection_rate:.1%} detection, {self.avg_corners:.1f} corners"

@dataclass
class CameraCalibrationPeriods:
    """Calibration periods for a single camera"""
    camera_id: str
    video_path: Path
    total_frames: int
    detection_periods: List[DetectionPeriod]
    best_period: Optional[DetectionPeriod] = None
    
    def get_best_frames_for_intrinsics(self, max_frames: int = 50) -> List[int]:
        """Get best frame numbers for intrinsic calibration"""
        if not self.best_period:
            return []
        
        period = self.best_period
        total_available = period.duration_frames()
        
        if total_available <= max_frames:
            # Use all frames in the period
            return list(range(period.start_frame, period.end_frame + 1, 1))
        else:
            # Sample evenly from the period
            return list(range(period.start_frame, period.end_frame + 1, 
                            max(1, total_available // max_frames)))

@dataclass
class PairwisePeriod:
    """A synchronized period good for pairwise calibration"""
    camera1_id: str
    camera2_id: str
    start_frame: int
    end_frame: int
    quality_score: float  # Combined quality of both cameras
    cam1_detection_rate: float
    cam2_detection_rate: float
    
    def duration_frames(self) -> int:
        return self.end_frame - self.start_frame + 1
    
    def get_frame_numbers(self, max_frames: int = 30) -> List[int]:
        """Get synchronized frame numbers for pairwise calibration"""
        total_available = self.duration_frames()
        
        if total_available <= max_frames:
            return list(range(self.start_frame, self.end_frame + 1, 1))
        else:
            return list(range(self.start_frame, self.end_frame + 1,
                            max(1, total_available // max_frames)))

class TemporalCalibrationAnalyzer:
    """Analyzes videos to find optimal calibration periods"""
    
    def __init__(self, board_config, window_size: int = 100, overlap: int = 50):
        """
        Initialize temporal analyzer
        
        Args:
            board_config: ChArUco board configuration
            window_size: Size of analysis window in frames
            overlap: Overlap between windows in frames
        """
        from custom_calibration import ChArUcoDetector
        self.detector = ChArUcoDetector(board_config)
        self.window_size = window_size
        self.overlap = overlap
        
    def analyze_video_temporal(self, video_path: Path, camera_id: str) -> CameraCalibrationPeriods:
        """
        Analyze a single video to find periods with good ChArUco visibility
        
        Args:
            video_path: Path to camera calibration video
            camera_id: Camera identifier
            
        Returns:
            CameraCalibrationPeriods with detected periods
        """
        logger.info(f"Analyzing temporal ChArUco visibility for camera {camera_id}...")
        
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        logger.info(f"  Video: {video_path.name} ({total_frames} frames)")
        
        # Analyze video in sliding windows
        detection_periods = []
        
        window_start = 0
        while window_start < total_frames:
            window_end = min(window_start + self.window_size, total_frames)
            
            # Analyze this window
            period_info = self._analyze_window(cap, window_start, window_end, camera_id)
            
            if period_info and period_info.detection_rate > 0.1:  # At least 10% detection
                detection_periods.append(period_info)
                logger.debug(f"  Found good period: {period_info}")
            
            window_start += (self.window_size - self.overlap)
        
        cap.release()
        
        # Merge overlapping periods and find the best one
        merged_periods = self._merge_overlapping_periods(detection_periods)
        best_period = max(merged_periods, key=lambda p: p.quality_score) if merged_periods else None
        
        logger.info(f"  Camera {camera_id}: Found {len(merged_periods)} good periods")
        if best_period:
            logger.info(f"  Best period: {best_period}")
        
        return CameraCalibrationPeriods(
            camera_id=camera_id,
            video_path=video_path,
            total_frames=total_frames,
            detection_periods=merged_periods,
            best_period=best_period
        )
    
    def _analyze_window(self, cap: cv2.VideoCapture, start_frame: int, end_frame: int, 
                       camera_id: str) -> Optional[DetectionPeriod]:
        """Analyze a window of frames for ChArUco detection quality"""
        
        # Sample frames from this window (every 5th frame)
        sample_frames = list(range(start_frame, end_frame, 5))
        
        detections = []
        corner_counts = []
        
        for frame_num in sample_frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ret, frame = cap.read()
            if not ret:
                continue
            
            # Detect ChArUco corners
            corners, ids = self.detector.detect_charuco_corners(frame)
            
            if corners is not None and ids is not None and len(corners) > 4:
                detections.append(True)
                corner_counts.append(len(corners))
            else:
                detections.append(False)
                corner_counts.append(0)
        
        if not detections:
            return None
        
        detection_rate = np.mean(detections)
        avg_corners = np.mean(corner_counts) if corner_counts else 0
        
        # Quality score combines detection rate and corner count
        quality_score = detection_rate * avg_corners
        
        if detection_rate > 0:
            return DetectionPeriod(
                start_frame=start_frame,
                end_frame=end_frame,
                quality_score=quality_score,
                detection_rate=detection_rate,
                avg_corners=avg_corners,
                frame_count=len(sample_frames)
            )
        
        return None
    
    def _merge_overlapping_periods(self, periods: List[DetectionPeriod]) -> List[DetectionPeriod]:
        """Merge overlapping detection periods"""
        if not periods:
            return []
        
        # Sort by start frame
        sorted_periods = sorted(periods, key=lambda p: p.start_frame)
        merged = [sorted_periods[0]]
        
        for current in sorted_periods[1:]:
            last = merged[-1]
            
            # Check for overlap (with small gap tolerance)
            if current.start_frame <= last.end_frame + 50:  # 50 frame gap tolerance
                # Merge periods
                merged_period = DetectionPeriod(
                    start_frame=last.start_frame,
                    end_frame=max(last.end_frame, current.end_frame),
                    quality_score=(last.quality_score + current.quality_score) / 2,
                    detection_rate=(last.detection_rate + current.detection_rate) / 2,
                    avg_corners=(last.avg_corners + current.avg_corners) / 2,
                    frame_count=last.frame_count + current.frame_count
                )
                merged[-1] = merged_period
            else:
                merged.append(current)
        
        return merged
    
    def find_pairwise_overlaps(self, cam1_periods: CameraCalibrationPeriods, 
                              cam2_periods: CameraCalibrationPeriods) -> List[PairwisePeriod]:
        """
        Find synchronized overlapping periods between two cameras
        
        Since videos are synchronized, we can use the same frame numbers
        for both cameras to find periods where both have good detection.
        
        Args:
            cam1_periods: Calibration periods for camera 1
            cam2_periods: Calibration periods for camera 2
            
        Returns:
            List of synchronized periods good for pairwise calibration
        """
        logger.info(f"Finding pairwise overlaps: {cam1_periods.camera_id} & {cam2_periods.camera_id}")
        
        pairwise_periods = []
        
        # Check all combinations of periods from both cameras
        for period1 in cam1_periods.detection_periods:
            for period2 in cam2_periods.detection_periods:
                # Find overlap in frame numbers (since videos are synchronized)
                overlap_start = max(period1.start_frame, period2.start_frame)
                overlap_end = min(period1.end_frame, period2.end_frame)
                
                if overlap_end > overlap_start:
                    # We have an overlap
                    overlap_duration = overlap_end - overlap_start + 1
                    
                    # Only consider substantial overlaps
                    if overlap_duration >= 50:  # At least 50 frames (~1.7 seconds at 30fps)
                        # Combined quality score
                        quality_score = (period1.quality_score + period2.quality_score) / 2
                        
                        pairwise_period = PairwisePeriod(
                            camera1_id=cam1_periods.camera_id,
                            camera2_id=cam2_periods.camera_id,
                            start_frame=overlap_start,
                            end_frame=overlap_end,
                            quality_score=quality_score,
                            cam1_detection_rate=period1.detection_rate,
                            cam2_detection_rate=period2.detection_rate
                        )
                        
                        pairwise_periods.append(pairwise_period)
                        logger.debug(f"  Found overlap: frames {overlap_start}-{overlap_end} "
                                   f"(quality: {quality_score:.2f})")
        
        # Sort by quality and return best periods
        pairwise_periods.sort(key=lambda p: p.quality_score, reverse=True)
        
        logger.info(f"  Found {len(pairwise_periods)} overlapping periods")
        if pairwise_periods:
            best = pairwise_periods[0] 
            logger.info(f"  Best overlap: frames {best.start_frame}-{best.end_frame} "
                       f"({best.duration_frames()} frames)")
        
        return pairwise_periods

def extract_frames_from_optimal_periods(video_path: Path, frame_numbers: List[int], 
                                       output_dir: Path) -> List[Path]:
    """
    Extract specific frames from optimal detection periods
    
    Args:
        video_path: Path to video
        frame_numbers: Specific frame numbers to extract  
        output_dir: Output directory for frames
        
    Returns:
        List of extracted frame paths
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    
    frame_paths = []
    
    logger.info(f"Extracting {len(frame_numbers)} frames from optimal periods...")
    
    for i, frame_num in enumerate(frame_numbers):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        if ret:
            frame_filename = f"optimal_frame_{frame_num:06d}.jpg"
            frame_path = output_dir / frame_filename
            cv2.imwrite(str(frame_path), frame)
            frame_paths.append(frame_path)
            
            if (i + 1) % 20 == 0:
                logger.info(f"  Extracted {i + 1}/{len(frame_numbers)} optimal frames...")
    
    cap.release()
    logger.info(f"✅ Extracted {len(frame_paths)} frames from optimal periods")
    
    return frame_paths

def create_temporal_analysis_plot(camera_periods_dict: Dict[str, CameraCalibrationPeriods], 
                                 pairwise_periods: Dict[Tuple[str, str], List[PairwisePeriod]],
                                 output_path: Path = None):
    """
    Create a visualization of detection periods across cameras
    
    Args:
        camera_periods_dict: Dictionary of camera periods
        pairwise_periods: Dictionary of pairwise periods
        output_path: Path to save plot
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
        
        fig, axes = plt.subplots(len(camera_periods_dict), 1, figsize=(15, 8), sharex=True)
        if len(camera_periods_dict) == 1:
            axes = [axes]
        
        colors = ['blue', 'green', 'red', 'purple', 'orange']
        
        for i, (camera_id, periods) in enumerate(camera_periods_dict.items()):
            ax = axes[i]
            
            # Plot detection periods for this camera
            for period in periods.detection_periods:
                width = period.duration_frames()
                height = period.quality_score
                
                rect = patches.Rectangle(
                    (period.start_frame, 0), width, height,
                    linewidth=1, edgecolor='black', facecolor=colors[i], alpha=0.6
                )
                ax.add_patch(rect)
            
            # Highlight best period
            if periods.best_period:
                best = periods.best_period
                rect = patches.Rectangle(
                    (best.start_frame, 0), best.duration_frames(), best.quality_score,
                    linewidth=3, edgecolor='red', facecolor=colors[i], alpha=0.8
                )
                ax.add_patch(rect)
            
            ax.set_ylabel(f'Camera {camera_id}\nQuality Score')
            ax.set_xlim(0, periods.total_frames)
            ax.set_ylim(0, max(p.quality_score for p in periods.detection_periods) * 1.1 if periods.detection_periods else 1)
            ax.grid(True, alpha=0.3)
        
        axes[-1].set_xlabel('Frame Number')
        plt.title('ChArUco Detection Periods Across Cameras')
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            logger.info(f"Saved temporal analysis plot: {output_path}")
        
        return fig
        
    except ImportError:
        logger.warning("Matplotlib not available - skipping plot creation")
        return None

def optimize_calibration_periods(calibration_videos: Dict[str, Path], 
                               board_config) -> Tuple[Dict[str, CameraCalibrationPeriods], 
                                                    Dict[Tuple[str, str], List[PairwisePeriod]]]:
    """
    Main function to optimize calibration periods for all cameras
    
    Args:
        calibration_videos: Dict mapping camera_id to video path
        board_config: ChArUco board configuration
        
    Returns:
        Tuple of (camera_periods, pairwise_periods)
    """
    logger.info("🔍 OPTIMIZING CALIBRATION PERIODS FOR ALL CAMERAS")
    logger.info("="*60)
    
    analyzer = TemporalCalibrationAnalyzer(board_config)
    
    # Step 1: Analyze each camera individually
    camera_periods = {}
    
    for camera_id, video_path in calibration_videos.items():
        periods = analyzer.analyze_video_temporal(video_path, camera_id)
        camera_periods[camera_id] = periods
        
        if periods.best_period:
            logger.info(f"✅ Camera {camera_id}: Best period {periods.best_period}")
        else:
            logger.warning(f"⚠️ Camera {camera_id}: No good periods found")
    
    # Step 2: Find pairwise overlapping periods
    logger.info("\n🔗 FINDING PAIRWISE OVERLAPPING PERIODS")
    logger.info("="*60)
    
    pairwise_periods = {}
    camera_ids = list(calibration_videos.keys())
    
    for i, cam1 in enumerate(camera_ids):
        for j, cam2 in enumerate(camera_ids[i+1:], i+1):
            if cam1 in camera_periods and cam2 in camera_periods:
                overlaps = analyzer.find_pairwise_overlaps(
                    camera_periods[cam1], camera_periods[cam2]
                )
                
                if overlaps:
                    pairwise_periods[(cam1, cam2)] = overlaps
                    best_overlap = overlaps[0]
                    logger.info(f"✅ Pair ({cam1},{cam2}): {len(overlaps)} overlaps, "
                              f"best: frames {best_overlap.start_frame}-{best_overlap.end_frame}")
                else:
                    logger.warning(f"⚠️ Pair ({cam1},{cam2}): No overlapping periods found")
    
    # Step 3: Create visualization
    try:
        plot_path = Path("calibration_periods_analysis.png")
        create_temporal_analysis_plot(camera_periods, pairwise_periods, plot_path)
    except Exception as e:
        logger.warning(f"Could not create analysis plot: {e}")
    
    return camera_periods, pairwise_periods

if __name__ == "__main__":
    # Test with the user's calibration videos
    calibration_videos = {
        "1": Path(r"C:\Users\Runda\Desktop\9.12\session1\synced_COMPLETE\calib\cam1-calib.mp4"),
        "2": Path(r"C:\Users\Runda\Desktop\9.12\session1\synced_COMPLETE\calib\cam2-calib.mp4"), 
        "3": Path(r"C:\Users\Runda\Desktop\9.12\session1\synced_COMPLETE\calib\cam3-calib.mp4"),
    }
    
    # Check if videos exist
    available_videos = {k: v for k, v in calibration_videos.items() if v.exists()}
    
    if not available_videos:
        print("❌ No calibration videos found for testing")
    else:
        print(f"🎯 Testing temporal optimization with {len(available_videos)} videos")
        
        # Import board config
        sys.path.insert(0, str(Path(__file__).parent))
        from custom_calibration import ChArUcoBoard
        
        board_config = ChArUcoBoard()
        
        # Run optimization
        camera_periods, pairwise_periods = optimize_calibration_periods(
            available_videos, board_config
        )
        
        print(f"\n📊 OPTIMIZATION RESULTS:")
        print("="*60)
        for camera_id, periods in camera_periods.items():
            print(f"Camera {camera_id}: {len(periods.detection_periods)} periods")
            if periods.best_period:
                print(f"  Best: frames {periods.best_period.start_frame}-{periods.best_period.end_frame}")
        
        for (cam1, cam2), overlaps in pairwise_periods.items():
            print(f"Pair ({cam1},{cam2}): {len(overlaps)} synchronized overlaps")

