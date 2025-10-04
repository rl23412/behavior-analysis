#!/usr/bin/env python
# =============================================================================
#
#       MULTI-SESSION COMBINED ANIPOSE & DEEPLABCUT PROCESSING PIPELINE
#
# Description:
# This script combines videos from multiple regions and sessions for DLC adaptation
# to ensure consistency across all data while maintaining separate calibrations per session.
# It processes from 2D labeling through 3D triangulation and filtering.
#
# Key Features:
# - Combines videos from multiple regions/sessions for DLC adaptation
# - Maintains separate calibrations per session
# - Creates mapping file for traceability
# - Runs pipeline from 2D processing to 3D filtering
#
# Usage:
# python 23.py --dlc-config-yaml /path/to/config.yaml
# =============================================================================

import os
import sys
import shutil
import subprocess
import re
import time
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd
import soundfile as sf
from scipy import signal, io as scipy_io
import deeplabcut
import json
from datetime import datetime

# ==========================================================================
# --- FIXED SETTINGS ---
# =============================================================================

DEFAULT_DLC_FILTER_CONFIG_YAML = "/home/users/rl349/object/config.yaml"
DEFAULT_MAT_OUTPUT_DIR = "/home/users/rl349/object/"
COMBINED_OUTPUT_BASE = "/work/rl349/DeepLabCut/2weekcombine"
WEEK4_SOURCE_DIR = "/work/rl349/temp/4week"  # New week4 source directory
SESSION4_TBI_SOURCE_DIR = "/work/rl349/DeepLabCut/2weekcombine/session4(TBI)"  # Session4 TBI source directory

# Data Conversion & Video Settings
VOXEL_VOLUME_BOUNDS_MIN = -120
VOXEL_VOLUME_BOUNDS_MAX = 120
VOXEL_N_VOXELS = 80
TARGET_RESOLUTION_HEIGHT = 720
TARGET_FPS = 30
MAX_SYNC_DURATION_S = 1800  # 30 minutes max sync duration
SYNC_SNIPPET_DURATION_S = 60  # 60 seconds for audio analysis

# =============================================================================
# --- HELPER FUNCTIONS ---
# =============================================================================

def print_header(title):
    """Prints a formatted header to the console."""
    print("\n" + "="*80)
    print(f"--- {title.upper()} ---")
    print("="*80)

def run_command(command, working_dir=None, check=True):
    """Executes a shell command, prints status, and handles errors."""
    command_str = ' '.join(map(str, command))
    print(f"RUNNING: {command_str}")
    if working_dir:
        print(f"In directory: {working_dir}")
    try:
        process = subprocess.run(
            command, check=check, capture_output=True, text=True, cwd=working_dir
        )
        # Only print stdout/stderr if they contain content, to reduce log clutter
        if process.stdout.strip():
            print(f"STDOUT: {process.stdout.strip()}")
        if process.stderr.strip():
            print(f"STDERR: {process.stderr.strip()}")
        print("...SUCCESS.")
        return process
    except subprocess.CalledProcessError as e:
        print(f"ERROR: Command failed with exit code {e.returncode}")
        print(f"  Command: {' '.join(e.cmd)}")
        print(f"  Stderr: {e.stderr.strip()}")
        print(f"  Stdout: {e.stdout.strip()}")
        raise





def create_config_toml(session_dir, num_cameras=3):
    """Create config.toml for multi-camera custom calibration in the session directory."""
    
    try:
        # Import the new calibration integration module
        from calibration_integration import create_multi_camera_config_toml
        create_multi_camera_config_toml(session_dir, num_cameras)
    except ImportError:
        # Fallback to old method if new module not available
        print("Warning: Custom calibration module not available, using fallback config")
        create_legacy_config_toml(session_dir)

def create_legacy_config_toml(session_dir):
    """Legacy config creation (fallback for backward compatibility)"""
    config_content = """
# Project settings
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
animal_calibration = true

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
    
    config_path = os.path.join(session_dir, 'config.toml')
    with open(config_path, 'w') as f:
        f.write(config_content.strip())
    print(f"Created legacy config.toml at: {config_path}")


def check_ffmpeg_availability():
    """Check if FFmpeg is available and working"""
    try:
        result = subprocess.run(['ffmpeg', '-version'], capture_output=True, text=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"FFmpeg check failed: {e}")
        return False

class FFmpegLibraryError(Exception):
    """Custom exception for FFmpeg library issues"""
    pass

def get_audio_offset_fft(ref_file, target_file):
    """
    Calculates audio offset using a fast FFT-based cross-correlation.
    Extracts a short snippet of audio to a temporary WAV file to avoid memory issues.
    """
    print(f"  Calculating audio offset between {os.path.basename(ref_file)} and {os.path.basename(target_file)}...")
    
    ref_dir = os.path.dirname(ref_file)
    target_dir = os.path.dirname(target_file)
    ref_wav = os.path.join(ref_dir, "temp_ref_audio.wav")
    target_wav = os.path.join(target_dir, "temp_target_audio.wav")
    
    try:
        # Extract a short snippet of audio using ffmpeg
        cmd_ref = ['ffmpeg', '-y', '-i', ref_file, '-t', str(SYNC_SNIPPET_DURATION_S), '-vn', '-acodec', 'pcm_s16le', ref_wav]
        cmd_target = ['ffmpeg', '-y', '-i', target_file, '-t', str(SYNC_SNIPPET_DURATION_S), '-vn', '-acodec', 'pcm_s16le', target_wav]
        
        run_command(cmd_ref)
        run_command(cmd_target)
        
        # Read the small, manageable WAV snippets
        ref_audio, ref_sr = sf.read(ref_wav, dtype='float32', always_2d=True)
        target_audio, target_sr = sf.read(target_wav, dtype='float32', always_2d=True)
        
        if ref_sr != target_sr:
            print(f"Warning: Sample rates differ ({ref_sr} vs {target_sr}).")
        
        # Use mono audio for correlation
        ref_mono = np.mean(ref_audio, axis=1)
        target_mono = np.mean(target_audio, axis=1)

        # Use Scipy's optimized FFT-based correlation - this is the key speedup
        correlation = signal.correlate(ref_mono, target_mono, mode='full', method='fft')
        lag = np.argmax(correlation) - (len(target_mono) - 1)
        
        offset_seconds = lag / ref_sr
        print(f"  --> Calculated offset: {offset_seconds:.4f} seconds.")
        return offset_seconds
        
    except Exception as e:
        print(f"Could not calculate audio offset. Error: {e}")
        return None
        
    finally:
        # Clean up temporary WAV files
        if os.path.exists(ref_wav): os.remove(ref_wav)
        if os.path.exists(target_wav): os.remove(target_wav)







def synchronize_video_group(video_group):
    """
    Synchronizes a group of videos based on audio, trims to the overlapping part,
    and overwrites the original files.
    """
    group_name = os.path.basename(os.path.splitext(video_group[0])[0])
    print(f"\n--- STEP 3: Synchronizing Group: '{group_name}' ---")
    if len(video_group) < 2:
        print("  Only one video in group, skipping synchronization.")
        return

    ref_video = video_group[0]
    
    offsets = {ref_video: 0.0}
    for target_video in video_group[1:]:
        offset = get_audio_offset_fft(ref_video, target_video)
        if offset is None:
            print(f"Cannot synchronize group '{group_name}'. Skipping.")
            return
        offsets[target_video] = offset

    durations = {}
    for video_path in video_group:
        try:
            result = subprocess.run([
                'ffprobe', '-v', 'error', '-show_entries', 'format=duration',
                '-of', 'default=noprint_wrappers=1:nokey=1', video_path
            ], capture_output=True, text=True, check=True)
            durations[video_path] = float(result.stdout)
        except Exception as e:
            print(f"Could not get duration for {video_path}. Skipping group '{group_name}'. Error: {e}")
            return
            
    start_times_on_ref_timeline = [offsets[p] for p in video_group]
    end_times_on_ref_timeline = [offsets[p] + durations[p] for p in video_group]
    
    sync_start = max(start_times_on_ref_timeline)
    sync_end = min(end_times_on_ref_timeline)
    sync_duration = sync_end - sync_start

    if sync_duration <= 0:
        print(f"  No overlapping audio found for group '{group_name}'. Skipping.")
        return

    print(f"  Overlap found. Duration: {sync_duration:.2f}s")
    
    if sync_duration > MAX_SYNC_DURATION_S:
        sync_duration = MAX_SYNC_DURATION_S
        print(f"  Overlap exceeds {MAX_SYNC_DURATION_S}s. Trimming to first {MAX_SYNC_DURATION_S}s of overlap.")
    
    for video_path in video_group:
        trim_start_seconds = sync_start - offsets[video_path]

        if trim_start_seconds < 0:
             print(f"Warning: Calculated trim start for {os.path.basename(video_path)} is negative ({trim_start_seconds:.2f}s). Trimming from 0.")
             trim_start_seconds = 0
        
        temp_output_path = video_path.replace('.mp4', '_synced_temp.mp4')
        
        command = [
            'ffmpeg', '-y', '-ss', str(trim_start_seconds), '-i', video_path,
            '-t', str(sync_duration), '-c', 'copy', temp_output_path
        ]
        
        try:
            run_command(command)
            shutil.move(temp_output_path, video_path)
            print(f"  Successfully synced and replaced {os.path.basename(video_path)}")
        except Exception as e:
            print(f"  Failed to trim and sync {os.path.basename(video_path)}. Error: {e}")
            if os.path.exists(temp_output_path):
                os.remove(temp_output_path)

def process_week4_videos(week4_source_dir, session3_dir):
    """Process week4 videos: organize, synchronize, and prepare for DLC processing."""
    print_header("Processing Week4 Videos for Session3")
    
    # Create session3 structure
    session3_path = Path(session3_dir)
    session3_path.mkdir(parents=True, exist_ok=True)
    
    session1_subdir = session3_path / "session1"
    session1_subdir.mkdir(exist_ok=True)
    
    # Create required subdirectories
    videos_raw_dir = session1_subdir / "videos-raw"
    calibration_dir = session1_subdir / "calibration"
    pose_2d_dir = session1_subdir / "pose-2d"
    pose_2d_filtered_dir = session1_subdir / "pose-2d-filtered"
    pose_3d_dir = session1_subdir / "pose-3d"
    pose_3d_filtered_dir = session1_subdir / "pose-3d-filtered"
    
    for dir_path in [videos_raw_dir, calibration_dir, pose_2d_dir, pose_2d_filtered_dir, pose_3d_dir, pose_3d_filtered_dir]:
        dir_path.mkdir(exist_ok=True)
    
    # Create config.toml
    create_config_toml(str(session3_path))
    
    # Process and organize videos
    week4_path = Path(week4_source_dir)
    regions = ['DRG', 'SC', 'SNI']
    
    # Count existing videos to continue indexing
    existing_video_indices = set()
    session_dirs = [
        os.path.join(COMBINED_OUTPUT_BASE, "session1(2week)"),
        os.path.join(COMBINED_OUTPUT_BASE, "session2(2weeksession2)"),
        os.path.join(COMBINED_OUTPUT_BASE, "session3(week4)")
    ]
    
    for session_dir in session_dirs:
        videos_dir = os.path.join(session_dir, "session1", "videos-raw")
        if os.path.exists(videos_dir):
            for f in os.listdir(videos_dir):
                if f.endswith('.mp4'):
                    # Extract video index from filename (e.g., "vid1-cam1.mp4" -> 1)
                    match = re.match(r'vid(\d+)', f)
                    if match:
                        existing_video_indices.add(int(match.group(1)))
    
    starting_index = max(existing_video_indices) + 1 if existing_video_indices else 1
    print(f"Starting video index from: {starting_index}")
    
    # Collect videos by group for synchronization
    video_groups = []
    video_mapping = []
    
    for region_idx, region in enumerate(regions):
        region_path = week4_path / region
        if not region_path.exists():
            print(f"Warning: Region {region} not found in week4 data")
            continue
            
        # Process each numbered folder (video group)
        for video_num in ['1', '2', '3']:
            group_videos = []
            
            # Collect cam1 and cam2 videos for this group
            for cam_num in ['1', '2']:
                cam_path = region_path / cam_num / f"{video_num}.MP4"
                if cam_path.exists():
                    group_videos.append((cam_path, cam_num))
                    
            if len(group_videos) == 2:  # We have both cameras
                video_idx = starting_index + region_idx * 3 + int(video_num) - 1
                video_name = f"vid{video_idx}"
                
                video_groups.append({
                    'name': video_name,
                    'videos': group_videos,
                    'region': region,
                    'original_num': video_num
                })
                
                video_mapping.append({
                    'video_index': video_idx,
                    'video_name': video_name,
                    'region': region,
                    'session': 'session3(week4)',
                    'original_video_num': video_num
                })
    
    # Process calibration videos with synchronization
    abjust_path = week4_path / "abjust"
    if abjust_path.exists():
        print("\nProcessing calibration videos...")
        
        # Collect calibration videos for synchronization
        calib_videos = []
        calib_temp_videos = []
        
        for cam_num in ['1', '2']:
            calib_video = abjust_path / cam_num / "0.MP4"
            if calib_video.exists():
                calib_videos.append((calib_video, cam_num))
        
        if len(calib_videos) == 2:  # We have both camera calibration videos
            print("Converting and synchronizing calibration videos...")
            
            # First convert to temp files
            for calib_video, cam_num in calib_videos:
                temp_path = calibration_dir / f"calib-cam{cam_num}_temp.mp4"
                print(f"Converting calibration video for cam{cam_num}...")
                
                command = [
                    'ffmpeg', '-y', '-i', str(calib_video),
                    '-vf', f'scale=-2:{TARGET_RESOLUTION_HEIGHT}', '-r', str(TARGET_FPS),
                    '-c:v', 'libx264', '-preset', 'fast', '-crf', '23',
                    '-c:a', 'aac', '-b:a', '192k',
                    str(temp_path)
                ]
                try:
                    run_command(command)
                    calib_temp_videos.append(str(temp_path))
                except Exception as e:
                    print(f"  Warning: Calibration video conversion failed: {e}")
                    # Fallback: copy original
                    shutil.copy2(calib_video, temp_path)
                    calib_temp_videos.append(str(temp_path))
            
            # Synchronize calibration videos if we have both
            if len(calib_temp_videos) == 2:
                print("Synchronizing calibration videos...")
                try:
                    synchronize_video_group(calib_temp_videos)
                except Exception as e:
                    print(f"  Warning: Calibration video synchronization failed: {e}")
                    print("  Calibration videos will be used without synchronization")
            
            # Rename to final calibration names
            for i, temp_video in enumerate(calib_temp_videos):
                temp_path = Path(temp_video)
                cam_match = re.search(r'-cam(\d+)', temp_path.name)
                if cam_match:
                    cam_num = cam_match.group(1)
                    final_name = f"calib-cam{cam_num}.mp4"
                    final_path = calibration_dir / final_name
                    
                    if temp_path.exists():
                        temp_path.rename(final_path)
                        print(f"  Created synchronized calibration: {final_name}")
        
        else:
            print(f"Warning: Found {len(calib_videos)} calibration videos, expected 2")
            # Process individually without sync
            for calib_video, cam_num in calib_videos:
                dest_path = calibration_dir / f"calib-cam{cam_num}.mp4"
                print(f"Processing single calibration video for cam{cam_num}...")
                try:
                    shutil.copy2(calib_video, dest_path)
                except Exception as e:
                    print(f"  Warning: Could not copy calibration video: {e}")
    
    # Check FFmpeg availability first
    print("\nChecking FFmpeg availability...")
    ffmpeg_available = check_ffmpeg_availability()
    if not ffmpeg_available:
        print("Warning: FFmpeg not properly installed or missing dependencies")
        print("Attempting to process videos with fallback methods...")
    
    # Process and synchronize video groups
    print("\nProcessing and synchronizing video groups...")
    for group in video_groups:
        print(f"\nProcessing {group['name']} from region {group['region']}...")
        
        # First, convert videos to standard format
        temp_videos = []
        for video_path, cam_num in group['videos']:
            temp_path = videos_raw_dir / f"{group['name']}-cam{cam_num}_temp.mp4"
            
            if ffmpeg_available:
                command = [
                    'ffmpeg', '-y', '-i', str(video_path),
                    '-vf', f'scale=-2:{TARGET_RESOLUTION_HEIGHT}', '-r', str(TARGET_FPS),
                    '-c:v', 'libx264', '-preset', 'fast', '-crf', '23',
                    '-c:a', 'aac', '-b:a', '192k',
                    str(temp_path)
                ]
                try:
                    run_command(command)
                    temp_videos.append(str(temp_path))
                except FFmpegLibraryError:
                    print(f"  FFmpeg library issue, copying original video for {group['name']}-cam{cam_num}")
                    # Fallback: just copy the original file
                    shutil.copy2(video_path, temp_path)
                    temp_videos.append(str(temp_path))
                except Exception as e:
                    print(f"  Video conversion failed for {group['name']}-cam{cam_num}: {e}")
                    # Fallback: just copy the original file
                    shutil.copy2(video_path, temp_path)
                    temp_videos.append(str(temp_path))
            else:
                print(f"  FFmpeg not available, copying original video for {group['name']}-cam{cam_num}")
                shutil.copy2(video_path, temp_path)
                temp_videos.append(str(temp_path))
        
        # Synchronize the videos (if possible)
        if len(temp_videos) >= 2:
            print(f"Synchronizing {group['name']}...")
            try:
                synchronize_video_group(temp_videos)
            except Exception as e:
                print(f"  Warning: Synchronization failed for {group['name']}: {e}")
                print(f"  Videos will be used without synchronization")
        
        # Rename synchronized videos to final names
        for temp_video in temp_videos:
            temp_path = Path(temp_video)
            cam_match = re.search(r'-cam(\d+)', temp_path.name)
            if cam_match:
                cam_num = cam_match.group(1)
                final_name = f"{group['name']}-cam{cam_num}.mp4"
                final_path = videos_raw_dir / final_name
                
                # Remove the _temp suffix
                if temp_path.exists():
                    temp_path.rename(final_path)
                    print(f"  Created: {final_name}")
    
    print(f"\nProcessed {len(video_groups)} video groups for session3")
    return video_mapping

def process_session4_tbi_videos(session4_tbi_source_dir, session4_dir):
    """Process session4 TBI videos: organize, convert resolution, and prepare for DLC processing.
    
    Videos are already synced, so no synchronization is needed.
    Video indices will continue from the last index of session3.
    """
    print_header("Processing Session4 TBI Videos")
    
    # Create session4 structure
    session4_path = Path(session4_dir)
    session4_path.mkdir(parents=True, exist_ok=True)
    
    session1_subdir = session4_path / "session1"
    session1_subdir.mkdir(exist_ok=True)
    
    # Create required subdirectories
    videos_raw_dir = session1_subdir / "videos-raw"
    calibration_dir = session1_subdir / "calibration"
    pose_2d_dir = session1_subdir / "pose-2d"
    pose_2d_filtered_dir = session1_subdir / "pose-2d-filtered"
    pose_3d_dir = session1_subdir / "pose-3d"
    pose_3d_filtered_dir = session1_subdir / "pose-3d-filtered"
    
    for dir_path in [videos_raw_dir, calibration_dir, pose_2d_dir, pose_2d_filtered_dir, pose_3d_dir, pose_3d_filtered_dir]:
        dir_path.mkdir(exist_ok=True)
    
    # Create config.toml
    create_config_toml(str(session4_path))
    
    # Process and organize videos
    session4_path_source = Path(session4_tbi_source_dir)
    regions = ['DRG', 'SC', 'SNI']  # TBI typically uses these regions
    
    # Count existing videos to continue indexing (include all previous sessions)
    existing_video_indices = set()
    session_dirs = [
        os.path.join(COMBINED_OUTPUT_BASE, "session1(2week)"),
        os.path.join(COMBINED_OUTPUT_BASE, "session2(2weeksession2)"),
        os.path.join(COMBINED_OUTPUT_BASE, "session3(week4)")
    ]
    
    for session_dir in session_dirs:
        videos_dir = os.path.join(session_dir, "session1", "videos-raw")
        if os.path.exists(videos_dir):
            for f in os.listdir(videos_dir):
                if f.endswith('.mp4'):
                    # Extract video index from filename (e.g., "vid1-cam1.mp4" -> 1)
                    match = re.match(r'vid(\d+)', f)
                    if match:
                        existing_video_indices.add(int(match.group(1)))
    
    starting_index = max(existing_video_indices) + 1 if existing_video_indices else 22
    print(f"Starting video index from: {starting_index}")
    
    # Collect videos for processing (no grouping for synchronization since they're already synced)
    video_groups = []
    video_mapping = []
    
    # Check if videos are directly in the session4(TBI) folder or in a subfolder structure
    # Look for videos that might be in session1/videos-raw already
    existing_videos_path = session4_path_source / "session1" / "videos-raw"
    if existing_videos_path.exists():
        print("Found existing videos-raw directory in session4(TBI)")
        # Process existing videos that need to be reindexed
        video_files = [f for f in os.listdir(existing_videos_path) if f.endswith('.mp4')]
        video_files.sort()  # Ensure consistent ordering
        
        print(f"Found {len(video_files)} MP4 files: {video_files}")
        
        if not video_files:
            print("No MP4 files found in videos-raw directory")
        else:
            # Group videos by their base name (ignoring camera numbers)
            video_dict = {}
            for video_file in video_files:
                print(f"Processing file: {video_file}")
                
                # Try multiple patterns to match video files
                patterns = [
                    r'(vid\d+)-cam(\d+)\.mp4',  # vid1-cam1.mp4
                    r'(vid\d+)_cam(\d+)\.mp4',  # vid1_cam1.mp4  
                    r'(\d+)-cam(\d+)\.mp4',     # 1-cam1.mp4
                    r'(\d+)_cam(\d+)\.mp4',     # 1_cam1.mp4
                    r'vid(\d+)-(\d+)\.mp4',     # vid1-1.mp4
                    r'(\d+)-(\d+)\.mp4'         # 1-1.mp4
                ]
                
                matched = False
                for pattern in patterns:
                    match = re.match(pattern, video_file)
                    if match:
                        base_part = match.group(1)
                        cam_num = match.group(2)
                        
                        # Normalize base name to vid format
                        if base_part.startswith('vid'):
                            base_name = base_part
                        else:
                            base_name = f"vid{base_part}"
                        
                        print(f"  Matched pattern '{pattern}': {base_name} cam{cam_num}")
                        
                        if base_name not in video_dict:
                            video_dict[base_name] = {}
                        video_dict[base_name][cam_num] = existing_videos_path / video_file
                        matched = True
                        break
                
                if not matched:
                    print(f"  No match for any expected pattern: {video_file}")
            
            print(f"Video dictionary: {list(video_dict.keys())} ({len(video_dict)} video groups)")
            
            # Process each video group and renumber them
            for old_vid_name, cameras in sorted(video_dict.items()):
                print(f"Processing video group: {old_vid_name} with cameras: {list(cameras.keys())}")
                
                if len(cameras) == 2 and '1' in cameras and '2' in cameras:  # We have both cameras
                    new_video_idx = starting_index + len(video_groups)
                    new_video_name = f"vid{new_video_idx}"
                    
                    group_videos = [(cameras['1'], '1'), (cameras['2'], '2')]
                    
                    print(f"  Renumbering {old_vid_name} -> {new_video_name}")
                    
                    video_groups.append({
                        'name': new_video_name,
                        'videos': group_videos,
                        'region': 'TBI',  # Default region for TBI videos
                        'original_name': old_vid_name
                    })
                    
                    video_mapping.append({
                        'video_index': new_video_idx,
                        'video_name': new_video_name,
                        'region': 'TBI',
                        'session': 'session4(TBI)',
                        'original_video_name': old_vid_name
                    })
                else:
                    print(f"  Warning: {old_vid_name} doesn't have both cameras (found: {list(cameras.keys())})")
    else:
        # Look for region-based structure
        print("Looking for region-based video structure...")
        for region_idx, region in enumerate(regions):
            region_path = session4_path_source / region
            if not region_path.exists():
                print(f"Warning: Region {region} not found in session4 TBI data")
                continue
                
            # Process each numbered folder (video group)
            for video_num in ['1', '2', '3']:
                group_videos = []
                
                # Collect cam1 and cam2 videos for this group
                for cam_num in ['1', '2']:
                    cam_path = region_path / cam_num / f"{video_num}.MP4"
                    if cam_path.exists():
                        group_videos.append((cam_path, cam_num))
                        
                if len(group_videos) == 2:  # We have both cameras
                    video_idx = starting_index + region_idx * 3 + int(video_num) - 1
                    video_name = f"vid{video_idx}"
                    
                    video_groups.append({
                        'name': video_name,
                        'videos': group_videos,
                        'region': region,
                        'original_num': video_num
                    })
                    
                    video_mapping.append({
                        'video_index': video_idx,
                        'video_name': video_name,
                        'region': region,
                        'session': 'session4(TBI)',
                        'original_video_num': video_num
                    })
    
    # Check if FFmpeg is available
    ffmpeg_available = True
    try:
        run_command(['ffmpeg', '-version'])
    except:
        ffmpeg_available = False
        print("Warning: FFmpeg not available. Videos will be copied without conversion.")
    
    # Process and convert video groups (no synchronization needed)
    print(f"\nProcessing and converting {len(video_groups)} video groups...")
    for group in video_groups:
        print(f"\nProcessing {group['name']} (region: {group.get('region', 'TBI')})...")
        
        # Convert videos to standard format
        for video_path, cam_num in group['videos']:
            output_path = videos_raw_dir / f"{group['name']}-cam{cam_num}.mp4"
            
            if ffmpeg_available:
                command = [
                    'ffmpeg', '-y', '-i', str(video_path),
                    '-vf', f'scale=-2:{TARGET_RESOLUTION_HEIGHT}', '-r', str(TARGET_FPS),
                    '-c:v', 'libx264', '-preset', 'fast', '-crf', '23',
                    '-c:a', 'aac', '-b:a', '192k',
                    str(output_path)
                ]
                try:
                    run_command(command)
                    print(f"  ✅ Converted: {group['name']}-cam{cam_num}")
                except Exception as e:
                    print(f"  ⚠️ Video conversion failed for {group['name']}-cam{cam_num}: {e}")
                    print("  📁 Copying original file as fallback")
                    shutil.copy2(video_path, output_path)
            else:
                print(f"  📁 Copying {group['name']}-cam{cam_num} (no FFmpeg)")
                shutil.copy2(video_path, output_path)
    
    # Process calibration videos if they exist
    # Look for calibration in various possible locations
    calib_paths = [
        session4_path_source / "session1" / "calibration",
        session4_path_source / "calibration",
        session4_path_source / "abjust"
    ]
    
    for calib_path in calib_paths:
        if calib_path.exists():
            print(f"\nProcessing calibration videos from {calib_path}...")
            
            # Look for calibration videos
            calib_videos = []
            for cam_num in ['1', '2']:
                # Try different naming patterns
                possible_calib_files = [
                    calib_path / f"calib-cam{cam_num}.mp4",
                    calib_path / cam_num / "0.MP4",
                    calib_path / f"cam{cam_num}" / "0.MP4"
                ]
                
                for calib_file in possible_calib_files:
                    if calib_file.exists():
                        calib_videos.append((calib_file, cam_num))
                        break
            
            # Convert calibration videos
            for calib_video, cam_num in calib_videos:
                output_path = calibration_dir / f"calib-cam{cam_num}.mp4"
                
                if ffmpeg_available:
                    command = [
                        'ffmpeg', '-y', '-i', str(calib_video),
                        '-vf', f'scale=-2:{TARGET_RESOLUTION_HEIGHT}', '-r', str(TARGET_FPS),
                        '-c:v', 'libx264', '-preset', 'fast', '-crf', '23',
                        '-c:a', 'aac', '-b:a', '192k',
                        str(output_path)
                    ]
                    try:
                        run_command(command)
                        print(f"  ✅ Converted calibration: cam{cam_num}")
                    except Exception as e:
                        print(f"  ⚠️ Calibration conversion failed for cam{cam_num}: {e}")
                        shutil.copy2(calib_video, output_path)
                else:
                    print(f"  📁 Copying calibration: cam{cam_num}")
                    shutil.copy2(calib_video, output_path)
            
            break  # Use first found calibration directory
    
    print(f"\nProcessed {len(video_groups)} video groups for session4(TBI)")
    print(f"Videos start from index {starting_index}")
    return video_mapping

def update_combined_mapping(session3_mapping):
    """Update the combined mapping file with session3 videos."""
    print_header("Updating Combined Video Mapping")
    
    mapping_file = os.path.join(COMBINED_OUTPUT_BASE, "video_source_mapping.json")
    
    # Load existing mapping if it exists
    existing_mapping = []
    if os.path.exists(mapping_file):
        with open(mapping_file, 'r') as f:
            existing_mapping = json.load(f)
        print(f"Loaded existing mapping with {len(existing_mapping)} videos")
    
    # Add session3 mapping
    combined_mapping = existing_mapping + session3_mapping
    
    # Save updated mapping
    with open(mapping_file, 'w') as f:
        json.dump(combined_mapping, f, indent=2)
    
    # Also update the text file
    txt_file = os.path.join(COMBINED_OUTPUT_BASE, "video_source_mapping.txt")
    with open(txt_file, 'w') as f:
        f.write("COMBINED VIDEO SOURCE MAPPING\n")
        f.write("="*80 + "\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total videos: {len(combined_mapping)}\n")
        f.write("="*80 + "\n\n")
        
        # Group by session
        by_session = defaultdict(list)
        for mapping in combined_mapping:
            by_session[mapping['session']].append(mapping)
        
        for session, videos in sorted(by_session.items()):
            f.write(f"\n{session}:\n")
            f.write("-"*40 + "\n")
            for video in sorted(videos, key=lambda x: x['video_index']):
                f.write(f"  {video['video_name']} <- {video['region']} video {video['original_video_num']}\n")
    
    print(f"Updated mapping files with {len(session3_mapping)} new videos")
    print(f"Total videos in combined mapping: {len(combined_mapping)}")

def convert_h5_remove_individuals_level(session1_dir, session2_dir, session3_dir, session4_dir=None):
    """Remove the 'individuals' level (Level 1) from H5 files' multi-level columns."""
    print_header("Converting H5 Files - Removing Individuals Level")
    
    session_dirs = {
        "session1(2week)": session1_dir,
        "session2(2weeksession2)": session2_dir,
        "session3(week4)": session3_dir
    }
    
    if session4_dir:
        session_dirs["session4(TBI)"] = session4_dir
    
    for session_name, session_dir in session_dirs.items():
        pose_2d_dir = os.path.join(session_dir, "session1", "pose-2d")
        
        if not os.path.exists(pose_2d_dir):
            print(f"Warning: pose-2d directory not found for {session_name}")
            continue
            
        print(f"\nProcessing session: {session_name}")
        print(f"Pose-2d directory: {pose_2d_dir}")
        
        # Get all H5 files in the directory
        h5_files = [f for f in os.listdir(pose_2d_dir) if f.endswith('.h5')]
        
        if not h5_files:
            print(f"  No H5 files found in {session_name}")
            continue
            
        print(f"  Found {len(h5_files)} H5 files to convert")
        
        for h5_file in h5_files:
            h5_path = os.path.join(pose_2d_dir, h5_file)
            
            try:
                print(f"  Processing: {h5_file}")
                
                # Load the H5 file
                df = pd.read_hdf(h5_path)
                
                # Check if it has multi-level columns
                if isinstance(df.columns, pd.MultiIndex):
                    print(f"    Original column levels: {df.columns.nlevels}")
                    print(f"    Original level names: {df.columns.names}")
                    
                    # Check if we have the expected 4-level structure
                    if df.columns.nlevels == 4 and 'individuals' in df.columns.names:
                        # Remove the 'individuals' level (Level 1)
                        # Get the level index for 'individuals'
                        individuals_level_idx = df.columns.names.index('individuals')
                        
                        # Drop the individuals level
                        new_columns = df.columns.droplevel(individuals_level_idx)
                        df.columns = new_columns
                        
                        print(f"    ✅ Removed 'individuals' level")
                        print(f"    New column levels: {df.columns.nlevels}")
                        print(f"    New level names: {df.columns.names}")
                        
                        # Save the modified dataframe back to the same file
                        df.to_hdf(h5_path, key='df_with_missing', mode='w')
                        print(f"    ✅ Saved converted file: {h5_file}")
                        
                    else:
                        print(f"    ⏭️  Skipped: {h5_file} (unexpected column structure)")
                        print(f"       Levels: {df.columns.nlevels}, Names: {df.columns.names}")
                else:
                    print(f"    ⏭️  Skipped: {h5_file} (not multi-level columns)")
                    
            except Exception as e:
                print(f"    ❌ Failed to convert {h5_file}: {e}")
    for session_name, session_dir in session_dirs.items():
        pose_2d_dir = os.path.join(session_dir, "session2", "pose-2d")
        
        if not os.path.exists(pose_2d_dir):
            print(f"Warning: pose-2d directory not found for {session_name}")
            continue
            
        print(f"\nProcessing session: {session_name}")
        print(f"Pose-2d directory: {pose_2d_dir}")
        
        # Get all H5 files in the directory
        h5_files = [f for f in os.listdir(pose_2d_dir) if f.endswith('.h5')]
        
        if not h5_files:
            print(f"  No H5 files found in {session_name}")
            continue
            
        print(f"  Found {len(h5_files)} H5 files to convert")
        
        for h5_file in h5_files:
            h5_path = os.path.join(pose_2d_dir, h5_file)
            
            try:
                print(f"  Processing: {h5_file}")
                
                # Load the H5 file
                df = pd.read_hdf(h5_path)
                
                # Check if it has multi-level columns
                if isinstance(df.columns, pd.MultiIndex):
                    print(f"    Original column levels: {df.columns.nlevels}")
                    print(f"    Original level names: {df.columns.names}")
                    
                    # Check if we have the expected 4-level structure
                    if df.columns.nlevels == 4 and 'individuals' in df.columns.names:
                        # Remove the 'individuals' level (Level 1)
                        # Get the level index for 'individuals'
                        individuals_level_idx = df.columns.names.index('individuals')
                        
                        # Drop the individuals level
                        new_columns = df.columns.droplevel(individuals_level_idx)
                        df.columns = new_columns
                        
                        print(f"    ✅ Removed 'individuals' level")
                        print(f"    New column levels: {df.columns.nlevels}")
                        print(f"    New level names: {df.columns.names}")
                        
                        # Save the modified dataframe back to the same file
                        df.to_hdf(h5_path, key='df_with_missing', mode='w')
                        print(f"    ✅ Saved converted file: {h5_file}")
                        
                    else:
                        print(f"    ⏭️  Skipped: {h5_file} (unexpected column structure)")
                        print(f"       Levels: {df.columns.nlevels}, Names: {df.columns.names}")
                else:
                    print(f"    ⏭️  Skipped: {h5_file} (not multi-level columns)")
                    
            except Exception as e:
                print(f"    ❌ Failed to convert {h5_file}: {e}")

def rename_h5_files_to_simple_format(session1_dir, session2_dir, session3_dir, session4_dir=None):
    """Rename H5 files in pose-2d directories to simplified format (e.g., vid1-cam1.h5)."""
    print_header("Renaming H5 Files to Simplified Format")
    
    session_dirs = {
        "session1(2week)": session1_dir,
        "session2(2weeksession2)": session2_dir,
        "session3(week4)": session3_dir
    }
    
    if session4_dir:
        session_dirs["session4(TBI)"] = session4_dir
    
    for session_name, session_dir in session_dirs.items():
        pose_2d_dir = os.path.join(session_dir, "session1", "pose-2d")
        
        if not os.path.exists(pose_2d_dir):
            print(f"Warning: pose-2d directory not found for {session_name}")
            continue
            
        print(f"\nProcessing session: {session_name}")
        print(f"Pose-2d directory: {pose_2d_dir}")
        
        # Get all H5 files in the directory
        h5_files = [f for f in os.listdir(pose_2d_dir) if f.endswith('.h5')]
        
        if not h5_files:
            print(f"  No H5 files found in {session_name}")
            continue
            
        print(f"  Found {len(h5_files)} H5 files to rename")
        
        for h5_file in h5_files:
            # Extract the video name part (everything before _superanimal or similar patterns)
            original_path = os.path.join(pose_2d_dir, h5_file)
            
            # Split at common DLC output patterns
            if '_superanimal' in h5_file:
                simple_name = h5_file.split('_superanimal')[0] + '.h5'
            elif '_snapshot-' in h5_file:
                simple_name = h5_file.split('_snapshot-')[0] + '.h5'
            elif 'DLC_' in h5_file:
                simple_name = h5_file.split('DLC_')[0] + '.h5'
            else:
                # If no known pattern, keep the original name
                simple_name = h5_file
                
            new_path = os.path.join(pose_2d_dir, simple_name)
            
            # Only rename if the name would actually change
            if simple_name != h5_file:
                try:
                    os.rename(original_path, new_path)
                    print(f"  ✅ Renamed: {h5_file} -> {simple_name}")
                except OSError as e:
                    print(f"  ❌ Failed to rename {h5_file}: {e}")
            else:
                print(f"  ⏭️  Skipped: {h5_file} (already in simple format)")

def run_session_calibration(session_dir):
    """Run calibration for a specific session using custom multi-camera calibration."""
    print_header(f"Running Custom Calibration for {os.path.basename(session_dir)}")
    
    try:
        # Try to use the new custom calibration system
        import subprocess
        import sys
        import os
        
        # Get the directory where this script is located
        script_dir = os.path.dirname(os.path.abspath(__file__))
        calibration_script = os.path.join(script_dir, "run_custom_calibration.py")
        
        if os.path.exists(calibration_script):
            print("Using custom multi-camera calibration system...")
            
            # Run the custom calibration script
            command = [sys.executable, calibration_script, "--session-dir", session_dir, "--verbose"]
            run_command(command)
            
            print("✅ Custom calibration completed successfully")
        else:
            print("Warning: Custom calibration script not found, falling back to anipose...")
            # Fallback to anipose calibration
            run_command(['anipose', 'calibrate'], working_dir=session_dir)
            
    except Exception as e:
        print(f"Custom calibration failed: {e}")
        print("Falling back to anipose calibration...")
        try:
            run_command(['anipose', 'calibrate'], working_dir=session_dir)
        except Exception as e2:
            print(f"Anipose calibration also failed: {e2}")
            raise e2

def run_session_processing(session_dir):
    """Run the full anipose processing pipeline for a session."""
    session_name = os.path.basename(session_dir)
    print_header(f"Running Anipose Processing for {session_name}")
    
    # Run anipose filter
    print("Running anipose filter...")
    run_command(['anipose', 'filter'], working_dir=session_dir)
    
    # Run calibration after filtering
    print("Running calibration after filtering...")
    run_command(['anipose', 'calibrate'], working_dir=session_dir)
    
    # Run triangulation
    print("Running triangulation...")
    run_command(['anipose', 'triangulate'], working_dir=session_dir)
    
    # Run 3D filtering
    print("Running 3D filtering...")
    run_command(['anipose', 'filter-3d'], working_dir=session_dir)
    
    # Run labeling
    print("Running labeling...")
    run_command(['anipose', 'label-2d-filter'], working_dir=session_dir)
    run_command(['anipose', 'project-2d'], working_dir=session_dir)
    run_command(['anipose', 'label-2d-proj'], working_dir=session_dir)


def maintbi():
    """Main execution function - Process week4 videos only."""
    
    print_header("Processing Week4 Videos as Session3")
    print(f"Combined Output: {COMBINED_OUTPUT_BASE}")
    
    # Set session directories
    session1_dir = os.path.join(COMBINED_OUTPUT_BASE, "session1(2week)")
    session2_dir = os.path.join(COMBINED_OUTPUT_BASE, "session2(2weeksession2)")
    session3_dir = os.path.join(COMBINED_OUTPUT_BASE, "session3(week4)")
    
    print(f"Session 3 directory: {session3_dir}")
    print(f"Week4 source: {WEEK4_SOURCE_DIR}")
    
    # Step 1: Process week4 videos
    print_header("Processing Week4 Videos")
    #session3_mapping = process_week4_videos(WEEK4_SOURCE_DIR, session3_dir)
    #update_combined_mapping(session3_mapping)
    
    # Step 2: Run DLC inference on session3 videos only
    print_header("Running DLC Inference on Session3 Videos")
    
    # Collect session3 video paths
    session3_video_paths = []
    videos_dir = os.path.join(session3_dir, "session1", "videos-raw")
    if os.path.exists(videos_dir):
        for video_file in os.listdir(videos_dir):
            if video_file.endswith('.mp4'):
                session3_video_paths.append(os.path.join(videos_dir, video_file))
    
    if session3_video_paths:
        print(f"Found {len(session3_video_paths)} videos in session3 for DLC inference")
        
        # Create pose-2d directory
        pose_2d_dir = os.path.join(session3_dir, "session1", "pose-2d")
        
        # Get videotype from first video
        videotype = Path(session3_video_paths[0]).suffix
        
        print("Running DLC inference on session3 videos...")
        
        # Run DLC on session3 videos
        deeplabcut.video_inference_superanimal(
            session3_video_paths,
            superanimal_name="superanimal_quadruped",
            model_name="hrnet_w32",
            detector_name="fasterrcnn_resnet50_fpn_v2",
            videotype=videotype,
            video_adapt=True,
            scale_list=[],
            max_individuals=1,
            dest_folder=pose_2d_dir,
            batch_size=8,
            detector_batch_size=8,
            video_adapt_batch_size=4
        )
        
        print("✅ Finished DLC inference on session3 videos")
        
        # Step 3: Convert H5 files - Remove individuals level for anipose compatibility
        print_header("Converting H5 Files for Anipose Compatibility")
        convert_h5_remove_individuals_level(session1_dir, session2_dir, session3_dir)
        
        # Step 4: Rename H5 files to simplified format for session3
        rename_h5_files_to_simple_format(session1_dir, session2_dir, session3_dir)
    
    # Step 5: Run calibration for session3
    print_header("Running Calibration for Session3")
    run_session_calibration(session3_dir)
    
    # Step 6: Run the complete anipose processing pipeline for session3
    print_header("Running Anipose Processing for Session3")
    run_session_processing(session3_dir)
    
    print_header("Week4 Processing Completed Successfully")
    print(f"Session3 results available in: {session3_dir}")

def main_session4_tbi():
    """Main execution function for processing session4(TBI) videos."""
    
    print_header("Processing Session4(TBI) Videos")
    print(f"Combined Output: {COMBINED_OUTPUT_BASE}")
    
    # Set session directories
    session1_dir = os.path.join(COMBINED_OUTPUT_BASE, "session1(2week)")
    session2_dir = os.path.join(COMBINED_OUTPUT_BASE, "session2(2weeksession2)")
    session3_dir = os.path.join(COMBINED_OUTPUT_BASE, "session3(week4)")
    session4_dir = os.path.join(COMBINED_OUTPUT_BASE, "session4(TBI)")
    
    print(f"Session 4 directory: {session4_dir}")
    print(f"Session4 TBI source: {SESSION4_TBI_SOURCE_DIR}")
    
    # Check if session4(TBI) source exists
    if not os.path.exists(SESSION4_TBI_SOURCE_DIR):
        print(f"❌ Error: Session4(TBI) source directory not found: {SESSION4_TBI_SOURCE_DIR}")
        print("Please ensure the session4(TBI) videos are located in the expected directory.")
        return
    
    # Step 1: Process session4(TBI) videos
    print_header("Processing Session4(TBI) Videos")
    session4_mapping = process_session4_tbi_videos(SESSION4_TBI_SOURCE_DIR, session4_dir)
    
    #if not session4_mapping:
        #print("❌ No videos were processed for session4(TBI). Check the input directory structure.")
        #return
    
    # Step 2: Update combined mapping with session4 videos
    print_header("Updating Combined Video Mapping")
    update_combined_mapping(session4_mapping)
    
    # Step 3: Run DLC inference on session4 videos only
    print_header("Running DLC Inference on Session4(TBI) Videos")
    
    # Collect session4 video paths
    session4_video_paths = []
    videos_dir = os.path.join(session4_dir, "session1", "videos-raw")
    if os.path.exists(videos_dir):
        for video_file in os.listdir(videos_dir):
            if video_file.endswith('.mp4'):
                session4_video_paths.append(os.path.join(videos_dir, video_file))
    
    if session4_video_paths and False:
        print(f"Found {len(session4_video_paths)} videos in session4(TBI) for DLC inference")
        
        # Create pose-2d directory
        pose_2d_dir = os.path.join(session4_dir, "session1", "pose-2d")
        
        # Get videotype from first video
        videotype = Path(session4_video_paths[0]).suffix
        
        print("Running DLC inference on session4(TBI) videos...")
        
        try:
            #Run DLC on session4 videos
            deeplabcut.video_inference_superanimal(
                session4_video_paths,
                superanimal_name="superanimal_quadruped",
                model_name="hrnet_w32",
                detector_name="fasterrcnn_resnet50_fpn_v2",
                videotype=videotype,
                video_adapt=True,
                scale_list=[],
                max_individuals=1,
                dest_folder=pose_2d_dir,
                batch_size=8,
                detector_batch_size=8,
                video_adapt_batch_size=4
            )
            
            print("✅ Finished DLC inference on session4(TBI) videos")
            
        except Exception as e:
            print(f"❌ Error during DLC inference: {e}")
            print("Continuing with the pipeline...")
    
    # Step 4: Convert H5 files - Remove individuals level for anipose compatibility
    print_header("Converting H5 Files for Anipose Compatibility") 
    #convert_h5_remove_individuals_level(session1_dir, session2_dir, session3_dir, session4_dir)
    
    # Step 5: Rename H5 files to simplified format for all sessions including session4
    print_header("Renaming H5 Files to Simplified Format")
    rename_h5_files_to_simple_format(session1_dir, session2_dir, session3_dir, session4_dir)

    print_header("Converting H5 Files for Anipose Compatibility") 
    convert_h5_remove_individuals_level(session1_dir, session2_dir, session3_dir, session4_dir)

    # Step 6: Run calibration for session4
    print_header("Running Calibration for Session4(TBI)")
    try:
        run_session_calibration(session4_dir)
        print("✅ Calibration completed for session4(TBI)")
    except Exception as e:
        print(f"❌ Error during calibration: {e}")
        print("Continuing with the pipeline...")
    
    # Step 7: Run the complete anipose processing pipeline for session4
    print_header("Running Anipose Processing for Session4(TBI)")
    try:
        run_session_processing(session4_dir)
        print("✅ Anipose processing completed for session4(TBI)")
    except Exception as e:
        print(f"❌ Error during anipose processing: {e}")
        print("Pipeline completed with errors.")
    
    print_header("Session4(TBI) Processing Completed")
    print(f"Session4(TBI) results available in: {session4_dir}")
    
    # Print summary of processed videos
    print(f"\nProcessed {len(session4_mapping)} videos for session4(TBI):")
    for mapping in session4_mapping:
        print(f"  {mapping['video_name']} <- {mapping.get('region', 'TBI')} (originally: {mapping.get('original_video_name', mapping.get('original_video_num', 'N/A'))})")

def convert_existing_h5_files():
    """Standalone function to convert existing H5 files across all sessions."""
    print_header("Converting Existing H5 Files - Removing Individuals Level")
    
    # Set session directories
    session1_dir = os.path.join(COMBINED_OUTPUT_BASE, "session1(2week)")
    session2_dir = os.path.join(COMBINED_OUTPUT_BASE, "session2(2weeksession2)")
    session3_dir = os.path.join(COMBINED_OUTPUT_BASE, "session3(week4)")
    session4_dir = os.path.join(COMBINED_OUTPUT_BASE, "session4(TBI)")
    
    # Convert all H5 files
    convert_h5_remove_individuals_level(session1_dir, session2_dir, session3_dir, session4_dir)
    
    print_header("H5 Conversion Completed")
    print("All H5 files have been processed to remove the 'individuals' level.")

if __name__ == "__main__":
    import sys
    main_session4_tbi()