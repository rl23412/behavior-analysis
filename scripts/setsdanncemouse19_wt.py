#!/usr/bin/env python3

"""
Three-camera WT and CP(full) Anipose to s-DANNCE conversion.

Creates s-DANNCE-ready directories that mirror the control-set layout but
targets both WT and CP(full) recordings (vid1–vid5, three cameras each) and
upgrades the output skeleton to the requested 19-joint definition.
"""

import argparse
import numpy as np
import pandas as pd
import scipy.io as sio
import toml
import os
import shutil
import yaml
import json
from pathlib import Path
import cv2
from collections import defaultdict
from datetime import datetime

# WT uses the expanded 19-joint mouse skeleton requested by the user.
TARGET_MOUSE19_JOINTS = [
    "Snout",
    "EarL",
    "EarR",
    "NeckB",
    "SpineF",
    "SpineM",
    "Tail(base)",
    "ForShdL",
    "ForEbwlL",
    "ForepawL",
    "ForeShdR",
    "ForEbwlR",
    "ForepawR",
    "HindShdL",
    "HindKneeL",
    "HindpawL",
    "HindShdR",
    "HindKneeR",
    "HindpawR",
]

# Mapping from the new skeleton to the closest WT 2D/3D landmark names.
joint_mapping = {
    "Snout": "nose",
    "EarL": "left_earend",
    "EarR": "right_earend",
    "NeckB": "neck_base",
    "SpineF": "back_base",
    "SpineM": "back_middle",
    "Tail(base)": "tail_base",
    "ForShdL": "front_left_thai",
    "ForEbwlL": "front_left_knee",
    "ForepawL": "front_left_paw",
    "ForeShdR": "front_right_thai",
    "ForEbwlR": "front_right_knee",
    "ForepawR": "front_right_paw",
    "HindShdL": "back_left_thai",
    "HindKneeL": "back_left_knee",
    "HindpawL": "back_left_paw",
    "HindShdR": "back_right_thai",
    "HindKneeR": "back_right_knee",
    "HindpawR": "back_right_paw",
}

# Canonical camera naming used across WT outputs.
CAMERA_NAME_MAP = {
    'cam_0': 'Camera1',
    'cam_1': 'Camera2',
    'cam_2': 'Camera3',
}

VIDEO_TO_CAMERA = {
    'cam1': 'Camera1',
    'cam2': 'Camera2',
    'cam3': 'Camera3',
}

CAMERA_ORDER = ['Camera1', 'Camera2', 'Camera3']

MOUSE19_LEFT_KEYPOINTS = [1, 7, 8, 9, 13, 14, 15]
MOUSE19_RIGHT_KEYPOINTS = [2, 10, 11, 12, 16, 17, 18]

CAMERA_TO_VIDEO = {v: k for k, v in VIDEO_TO_CAMERA.items()}

POSE_2D_SCORE_THRESHOLD = 0.75


def detect_image_size(video_mapping):
    """Infer (height, width) from the first available calibration file."""
    for mapping in video_mapping:
        calib_path = mapping.get('calibration_file')
        if not calib_path:
            continue
        try:
            calib = toml.load(calib_path)
        except (FileNotFoundError, toml.TomlDecodeError) as exc:
            print(f"Warning: Unable to read calibration {calib_path}: {exc}")
            continue

        for cam_key in CAMERA_NAME_MAP.keys():
            cam_data = calib.get(cam_key)
            if cam_data and 'size' in cam_data:
                width, height = cam_data['size']
                return int(height), int(width)

    # Default fallback if calibration size not found
    print("Warning: Could not determine image size from calibration; defaulting to 720x1280")
    return 720, 1280


DATASET_CONFIGS = {
    'wt': {
        'description': 'WT (3-camera)',
        'source_configs': [
            {
                'type': 'single_session',
                'path': '/work/rl349/DeepLabCut/WT',
                'name': 'WT',
                'region': 'WT'
            }
        ],
        'output_base_dir': '/work/rl349/dannce/wt_mouse19'
    },
    'cpfull': {
        'description': 'CP(full) (3-camera)',
        'source_configs': [
            {
                'type': 'single_session',
                'path': '/work/rl349/DeepLabCut/CP(full)',
                'name': 'CP_full',
                'region': 'CP'
            }
        ],
        'output_base_dir': '/work/rl349/dannce/cpfull_mouse19'
    }
}

def scan_multiple_sources(source_configs):
    """Scan multiple source directories with different structures"""
    
    all_sessions_info = []
    
    for source_config in source_configs:
        source_type = source_config['type']
        source_path = source_config['path']
        
        print(f"\n{'='*60}")
        print(f"Processing source: {source_path}")
        print(f"Type: {source_type}")
        print(f"{'='*60}")
        
        if source_type == 'multi_region':
            # Handle directories with region subdirectories
            sessions_info = scan_multi_region_structure(source_path, source_config['name'])
        elif source_type == 'single_session':
            # Handle direct session directories
            sessions_info = scan_single_session_source(source_path, source_config['name'], source_config['region'])
        else:
            print(f"Unknown source type: {source_type}")
            continue
            
        all_sessions_info.extend(sessions_info)
    
    return all_sessions_info

def scan_multi_region_structure(source_base_dir, source_name):
    """Scan source directory with multiple region subdirectories"""
    
    source_dir = Path(source_base_dir)
    sessions_info = []
    
    # List all subdirectories as potential regions
    region_dirs = [d for d in source_dir.iterdir() if d.is_dir()]
    
    for region_dir in region_dirs:
        region = region_dir.name
        print(f"\nFound region: {region}")
        
        # Look for session folders
        session_dirs = [d for d in region_dir.iterdir() if d.is_dir() and d.name.startswith('session')]
        
        for session_dir in session_dirs:
            session_info = scan_session_directory(session_dir, region, source_name)
            if session_info:
                sessions_info.append(session_info)
    
    return sessions_info

def scan_single_session_source(source_path, source_name, region):
    """Scan a directory that contains session folders"""
    
    source_dir = Path(source_path)
    sessions_info = []
    
    if source_dir.exists():
        # Look for session folders under this directory
        session_dirs = [d for d in source_dir.iterdir() if d.is_dir() and d.name.startswith('session')]
        
        if session_dirs:
            print(f"  Found {len(session_dirs)} session(s) in {source_dir}")
            for session_dir in session_dirs:
                region_label = f"{region}_{session_dir.name}" if region else session_dir.name
                session_info = scan_session_directory(session_dir, region_label, source_name)
                if session_info:
                    sessions_info.append(session_info)
        else:
            # If no session folders found, try treating it as a direct session directory
            print(f"  No session folders found, checking if {source_dir} is a session directory itself")
            session_info = scan_session_directory(source_dir, region or source_dir.name, source_name)
            if session_info:
                sessions_info.append(session_info)
    else:
        print(f"Warning: Source directory not found: {source_dir}")
    
    return sessions_info

def scan_session_directory(session_dir, region, source_name=None):
    """Scan a single session directory for videos and data"""
    
    print(f"  Scanning session: {session_dir}")
    
    # Look for key directories
    calibration_dir = session_dir / "calibration"
    videos_raw_dir = session_dir / "videos-raw"
    pose_3d_filtered_dir = session_dir / "pose-3d-filtered"
    pose_3d_dir = session_dir / "pose-3d"
    pose_2d_filtered_dir = session_dir / "pose-2d-filtered"
    pose_2d_dir = session_dir / "pose-2d"
    
    if not calibration_dir.exists():
        print(f"    Warning: No calibration directory found in {session_dir}")
        return None
    
    if not videos_raw_dir.exists():
        print(f"    Warning: No videos-raw directory found in {session_dir}")
        return None
        
    if pose_3d_dir.exists():
        pose3d_source = pose_3d_dir
    elif pose_3d_filtered_dir.exists():
        print(f"    Info: Using pose-3d-filtered directory as fallback for {session_dir}")
        pose3d_source = pose_3d_filtered_dir
    else:
        print(f"    Warning: No pose-3d[-filtered] directory found in {session_dir}")
        return None
    
    # Find calibration file
    calib_files = list(calibration_dir.glob("*.toml"))
    if not calib_files:
        print(f"    Warning: No calibration TOML file found in {calibration_dir}")
        return None
    
    calib_file = calib_files[0]  # Use first toml file found
    print(f"    Found calibration: {calib_file.name}")
    
    # Find videos and corresponding CSV files
    videos_info = []
    
    # Look for video files in videos-raw
    video_files = sorted(videos_raw_dir.glob("*.mp4"))
    
    # Group videos by video name (vid1, vid2, etc.)
    video_groups = defaultdict(list)
    for video_file in video_files:
        # Parse video name (e.g., vid1-cam1.mp4 -> vid1)
        parts = video_file.stem.split('-')
        if len(parts) >= 2:
            vid_name = parts[0]  # vid1, vid2, etc.
            cam_name = parts[1]  # cam1, cam2, cam3
            video_groups[vid_name].append({
                'cam': cam_name,
                'file': video_file
            })
    
    # For each video group, find corresponding CSV file
    required_cameras = {'cam1', 'cam2', 'cam3'}

    for vid_name, cameras in video_groups.items():
        available = {cam['cam'] for cam in cameras}
        missing = sorted(required_cameras - available)
        if missing:
            print(f"    Warning: {vid_name} missing cameras: {missing} (found {sorted(available)})")
            continue
            
        # Find corresponding CSV file in pose-3d-filtered
        csv_candidates = list(pose3d_source.glob(f"{vid_name}*.csv"))

        if not csv_candidates:
            print(f"    Warning: No CSV file found for {vid_name}")
            continue

        # Find largest CSV file (as requested)
        largest_csv = max(csv_candidates, key=lambda f: f.stat().st_size)

        # Sort cameras to ensure consistent order
        cameras = [cam for cam in cameras if cam['cam'] in required_cameras]
        cameras.sort(key=lambda x: x['cam'])

        # Gather 2D h5 files
        h5_files_2d = {}
        for cam_info in cameras:
            cam_name = cam_info['cam']
            h5_path = None

            if pose_2d_filtered_dir.exists():
                candidate = pose_2d_filtered_dir / f"{vid_name}-{cam_name}.h5"
                if candidate.exists():
                    h5_path = candidate

            if h5_path is None and pose_2d_dir.exists():
                candidate = pose_2d_dir / f"{vid_name}-{cam_name}.h5"
                if candidate.exists():
                    h5_path = candidate

            if h5_path is not None:
                h5_files_2d[cam_name] = h5_path
            else:
                print(f"      Warning: No 2D h5 file found for {vid_name}-{cam_name}")

        video_info = {
            'vid_name': vid_name,
            'csv_file': largest_csv,
            'cameras': cameras,
            'region': region,
            'session_dir': session_dir,
            'source_name': source_name or str(session_dir.parent.name),
            'h5_files_2d': h5_files_2d
        }

        videos_info.append(video_info)
        print(f"    Found video: {vid_name} with CSV {largest_csv.name} ({largest_csv.stat().st_size} bytes)")
    
    session_info = {
        'region': region,
        'session_dir': session_dir,
        'calibration_file': calib_file,
        'videos': videos_info,
        'source_name': source_name or str(session_dir.parent.name)
    }
    
    return session_info

def load_anipose_calibration(calib_path):
    """Load Anipose calibration and convert to s-DANNCE format"""
    with open(calib_path, 'r') as f:
        calib = toml.load(f)
    
    print(f"    Loading calibration from: {calib_path.name}")
    
    params = {}

    for cam_key, cam_name in CAMERA_NAME_MAP.items():
        if cam_key in calib:
            cam_data = calib[cam_key]
            
            # --- FIX 1: Transpose the intrinsic matrix K ---
            K = np.array(cam_data['matrix']).T
            
            # --- CORRECT ROTATION CONVERSION ---
            rvec = np.array(cam_data['rotation'])
            R_opencv, _ = cv2.Rodrigues(rvec)
            R = R_opencv.T
            
            # --- FIX 2: Reshape translation vector t to a 1x3 row vector ---
            tvec = np.array(cam_data['translation']).reshape(1, 3)
            
            # Extract distortion coefficients
            dist = np.array(cam_data['distortions'])
            # Ensure these are also 1xN row vectors for consistency
            RDistort = dist[:2].reshape(1, 2) if len(dist) >= 2 else np.zeros((1, 2))
            TDistort = dist[2:4].reshape(1, 2) if len(dist) >= 4 else np.zeros((1, 2))
            
            # Create parameter dictionary for this camera
            params[cam_name] = {
                'K': K,
                'R': R,
                't': tvec,
                'RDistort': RDistort,
                'TDistort': TDistort
            }
    
    return params

def load_anipose_3d_labels(csv_path):
    """Load Anipose 3D tracking results from CSV and map to the mouse19 skeleton."""
    # Read the CSV file
    df = pd.read_csv(csv_path)
    
    print(f"    CSV shape: {df.shape}")
    
    # Extract frame numbers
    if 'fnum' in df.columns:
        frames = df['fnum'].values
    else:
        frames = np.arange(len(df))
    
    n_frames = len(df)
    n_keypoints = len(TARGET_MOUSE19_JOINTS)
    data_3d = np.zeros((n_frames, n_keypoints * 3))
    
    # Load directly mapped joints - no interpolation needed since we remap joints
    for i, target_joint in enumerate(TARGET_MOUSE19_JOINTS):
        if target_joint in joint_mapping:
            source_joint = joint_mapping[target_joint]
            # Check if source joint columns exist in CSV
            cols = [f"{source_joint}_x", f"{source_joint}_y", f"{source_joint}_z"]
            if all(c in df.columns for c in cols):
                coords = df[cols].values
                data_3d[:, i*3:(i+1)*3] = np.array(coords)
            else:
                print(f"      Warning: Source joint '{source_joint}' columns not found for '{target_joint}'")
                data_3d[:, i*3:(i+1)*3] = np.nan
        else:
            print(f"      Warning: No mapping found for '{target_joint}'")
            data_3d[:, i*3:(i+1)*3] = np.nan
    
    # Extract scores and other metadata if available
    scores = np.ones((n_frames, n_keypoints))
    ncams = np.ones((n_frames, n_keypoints)) * 3  # WT recordings use three cameras
    
    # Try to extract scores for the mapped joints
    for i, target_joint in enumerate(TARGET_MOUSE19_JOINTS):
        if target_joint in joint_mapping:
            source_joint = joint_mapping[target_joint]
            score_col = f"{source_joint}_score"
            ncams_col = f"{source_joint}_ncams"
            
            if score_col in df.columns:
                scores[:, i] = np.array(df[score_col].values)
            if ncams_col in df.columns:
                ncams[:, i] = np.array(df[ncams_col].values)
    
    return {
        'data_3d': data_3d,
        'frames': frames,
        'keypoint_names': TARGET_MOUSE19_JOINTS,
        'scores': scores,
        'ncams': ncams
    }

def load_anipose_2d_labels(h5_path, threshold=POSE_2D_SCORE_THRESHOLD):
    """Load 2D pose estimates from an h5 file and apply a score threshold."""
    print(f"    Loading 2D data from: {h5_path.name}")

    try:
        df = pd.read_hdf(h5_path)
    except Exception as exc:
        print(f"    Error reading {h5_path}: {exc}")
        return None

    print(f"    2D H5 shape: {df.shape}")
    print(f"    Column levels: {df.columns.nlevels}")

    n_frames = len(df)
    n_keypoints = len(TARGET_MOUSE19_JOINTS)
    data_2d = np.full((n_frames, n_keypoints * 2), np.nan)

    def apply_threshold(x_coords, y_coords, likelihoods):
        mask = likelihoods >= threshold
        return np.where(mask, x_coords, np.nan), np.where(mask, y_coords, np.nan), mask.sum()

    # Default frame index from DataFrame; fallback to range if missing
    if isinstance(df.index, pd.RangeIndex):
        frames = df.index.values
    else:
        frames = np.arange(n_frames)

    if df.columns.nlevels == 1:
        # Flat columns: expect suffixes _x/_y/_likelihood
        for i, target_joint in enumerate(TARGET_MOUSE19_JOINTS):
            if target_joint not in joint_mapping:
                continue
            source_joint = joint_mapping[target_joint]
            x_col = f"{source_joint}_x"
            y_col = f"{source_joint}_y"
            likelihood_col = f"{source_joint}_likelihood"
            if all(col in df.columns for col in (x_col, y_col, likelihood_col)):
                x_coords = df[x_col].values
                y_coords = df[y_col].values
                likelihoods = df[likelihood_col].values
                x_coords, y_coords, valid = apply_threshold(x_coords, y_coords, likelihoods)
                data_2d[:, i*2] = x_coords
                data_2d[:, i*2+1] = y_coords
                print(f"      {target_joint} ({source_joint}): {valid}/{n_frames} frames above threshold {threshold}")
            else:
                print(f"      Warning: Missing flat columns for '{source_joint}' in {h5_path.name}")
        return {
            'data_2d': data_2d,
            'frames': frames
        }

    # MultiIndex handling
    cols = df.columns

    def find_column(source_joint, coord, likelihood=False):
        coord_key = 'likelihood' if likelihood else coord
        for col in cols:
            col_parts = [str(part).lower() for part in (col if isinstance(col, tuple) else (col,))]
            if source_joint.lower() in col_parts and coord_key.lower() in col_parts:
                return col
        return None

    for i, target_joint in enumerate(TARGET_MOUSE19_JOINTS):
        if target_joint not in joint_mapping:
            print(f"      Warning: No mapping found for '{target_joint}'")
            continue
        source_joint = joint_mapping[target_joint]

        x_col = find_column(source_joint, 'x')
        y_col = find_column(source_joint, 'y')
        likelihood_col = find_column(source_joint, 'likelihood', likelihood=True)

        if x_col is None or y_col is None or likelihood_col is None:
            print(f"      Warning: Could not resolve 2D columns for '{source_joint}'")
            continue

        x_coords = df[x_col].values
        y_coords = df[y_col].values
        likelihoods = df[likelihood_col].values
        x_coords, y_coords, valid = apply_threshold(x_coords, y_coords, likelihoods)
        data_2d[:, i*2] = x_coords
        data_2d[:, i*2+1] = y_coords
        print(f"      {target_joint} ({source_joint}): {valid}/{n_frames} frames above threshold {threshold}")

    return {
        'data_2d': data_2d,
        'frames': frames
    }
def create_label3d_mat(label_data, params, output_path, exp_name, h5_files_2d=None):
    """Create Label3D mat file in correct s-DANNCE format."""

    n_frames = len(label_data['frames'])
    n_keypoints = len(label_data['keypoint_names'])

    camnames = [name for name in CAMERA_ORDER if name in params]
    if not camnames:
        raise ValueError("No camera parameters provided; expected at least one camera.")
    n_cameras = len(camnames)

    labelData = np.empty((n_cameras, 1), dtype=object)
    for i, cam_name in enumerate(camnames):
        data_2d = np.full((n_frames, n_keypoints * 2), np.nan)
        video_cam_key = CAMERA_TO_VIDEO.get(cam_name)

        if h5_files_2d and video_cam_key in h5_files_2d:
            h5_path = h5_files_2d[video_cam_key]
            try:
                label_data_2d = load_anipose_2d_labels(h5_path, threshold=POSE_2D_SCORE_THRESHOLD)
                if label_data_2d:
                    frames_2d = len(label_data_2d['frames'])
                    if frames_2d != n_frames:
                        print(f"    Warning: 2D/3D frame mismatch for {cam_name}: 3D={n_frames}, 2D={frames_2d}")
                        min_frames = min(frames_2d, n_frames)
                        data_2d[:min_frames, :] = label_data_2d['data_2d'][:min_frames, :]
                    else:
                        data_2d = label_data_2d['data_2d']
                    print(f"    Loaded 2D data for {cam_name}: {data_2d.shape}")
                else:
                    print(f"    Warning: 2D data unavailable for {cam_name}, using NaN")
            except Exception as exc:
                print(f"    Error loading 2D data for {cam_name}: {exc}")

        cam_label_data = {
            'data_3d': label_data['data_3d'],
            'data_2d': data_2d,
            'data_frame': label_data['frames'].reshape(-1, 1),
            'data_sampleID': np.arange(n_frames).reshape(1, -1),
            'data_scores': label_data['scores'],
            'data_ncams': label_data['ncams']
        }
        labelData[i, 0] = cam_label_data

    params_array = np.empty((n_cameras, 1), dtype=object)
    for i, cam_name in enumerate(camnames):
        cam_params = params[cam_name]
        params_array[i, 0] = {
            'K': cam_params['K'],
            'R': cam_params['R'],
            'r': cam_params['R'],
            't': cam_params['t'],
            'RDistort': cam_params['RDistort'],
            'TDistort': cam_params['TDistort']
        }

    sync = np.empty((n_cameras, 1), dtype=object)
    for i, cam_name in enumerate(camnames):
        video_cam_key = CAMERA_TO_VIDEO.get(cam_name)
        if h5_files_2d and video_cam_key in h5_files_2d:
            data_2d_sync = labelData[i, 0]['data_2d']
        else:
            data_2d_sync = np.full((n_frames, n_keypoints * 2), np.nan)

        sync[i, 0] = {
            'data_frame': label_data['frames'].reshape(-1, 1),
            'data_sampleID': np.arange(n_frames).reshape(1, -1),
            'data_2d': data_2d_sync,
            'data_3d': np.full((n_frames, n_keypoints * 3), np.nan)
        }

    camnames_cell = np.empty((1, n_cameras), dtype=object)
    for idx, cam_name in enumerate(camnames):
        camnames_cell[0, idx] = cam_name

    mat_data = {
        'camnames': camnames_cell,
        'labelData': labelData,
        'params': params_array,
        'sync': sync,
        'jointnames': label_data['keypoint_names'],
        'expname': exp_name,
        'n_frames': n_frames,
        'n_keypoints': n_keypoints
    }

    sio.savemat(output_path, mat_data, do_compression=False, oned_as='column')
    print(f"    Saved Label3D file: {output_path}")

def generate_com_from_label3d(label3d_path, output_path):
    """Generate COM data from Label3D file.
    If no points detected in current frame, use nearby frames.
    
    Args:
        label3d_path: Path to the Label3D file
        output_path: Path where to save the com3d.mat file
    """
    # Load Label3D data
    labels = sio.loadmat(str(label3d_path))
    
    # Extract labelData
    if 'labelData' in labels:
        labelData = labels['labelData']
    else:
        print(f"    Warning: No labelData found in {label3d_path}")
        return None
    
    # Process the first camera's data (all cameras should have same 3D data)
    if labelData.size > 0:
        label_data = labelData[0, 0]  # First camera data
        
        # Extract sample IDs and 3D data
        if 'data_sampleID' in label_data.dtype.names:
            sample_ids = label_data['data_sampleID'][0, 0]
        else:
            # Fallback to frame indices
            sample_ids = label_data['data_frame'][0, 0].flatten()
        
        data_3d = label_data['data_3d'][0, 0]
        
        # Debug outputs
        print(f"    Debug: sample_ids shape: {sample_ids.shape}, dtype: {sample_ids.dtype}")
        print(f"    Debug: data_3d shape: {data_3d.shape}")
        
        # Flatten sample_ids if it's 2D
        if sample_ids.ndim == 2:
            sample_ids = sample_ids.flatten()
            print(f"    Debug: Flattened sample_ids to shape: {sample_ids.shape}")
        
        # Skip if no data
        if data_3d.size == 0:
            print("    No 3D data found")
            return None
        else:
            # Reshape data_3d from (n_frames, n_keypoints*3) to (n_frames, 3, n_keypoints)
            if len(data_3d.shape) == 2:
                n_frames = data_3d.shape[0]
                n_coords = data_3d.shape[1]
                n_keypoints = n_coords // 3
                data_3d = np.transpose(
                    np.reshape(data_3d, [n_frames, n_keypoints, 3]), [0, 2, 1]
                )
                print(f"    Debug: Reshaped data_3d to shape: {data_3d.shape}")
            
            # Compute COM as mean across keypoints (axis=2)
            com3d_dict = {}
            for i, sample_id in enumerate(sample_ids):
                if i < data_3d.shape[0]:  # Safety check
                    # sample_id should now be a scalar
                    sample_id_scalar = int(sample_id)
                    
                    # Check if all points in this frame are NaN
                    frame_data = data_3d[i]  # Shape: (3, n_keypoints)
                    if np.all(np.isnan(frame_data)):
                        # If all points are NaN, look for nearby frames
                        com_value = None
                        search_range = 5  # Search within 5 frames
                        
                        # Search nearby frames
                        for offset in range(1, search_range + 1):
                            # Check previous frame
                            prev_idx = i - offset
                            if prev_idx >= 0 and prev_idx < data_3d.shape[0]:
                                prev_frame_data = data_3d[prev_idx]
                                if not np.all(np.isnan(prev_frame_data)):
                                    com_value = np.nanmean(prev_frame_data, axis=1, keepdims=True)
                                    print(f"    Frame {sample_id_scalar}: Using frame {int(sample_ids[prev_idx])} (offset -{offset})")
                                    break
                            
                            # Check next frame
                            next_idx = i + offset
                            if next_idx < data_3d.shape[0]:
                                next_frame_data = data_3d[next_idx]
                                if not np.all(np.isnan(next_frame_data)):
                                    com_value = np.nanmean(next_frame_data, axis=1, keepdims=True)
                                    print(f"    Frame {sample_id_scalar}: Using frame {int(sample_ids[next_idx])} (offset +{offset})")
                                    break
                        
                        if com_value is None:
                            # If no nearby frames found, set COM to NaN
                            com3d_dict[sample_id_scalar] = np.array([[np.nan], [np.nan], [np.nan]])
                            print(f"    Frame {sample_id_scalar}: No nearby frames available, using NaN")
                        else:
                            com3d_dict[sample_id_scalar] = com_value
                    else:
                        # Calculate mean, ignoring NaN values
                        com3d_dict[sample_id_scalar] = np.nanmean(frame_data, axis=1, keepdims=True)
            
            # Prepare data for saving
            samples_keys = list(com3d_dict.keys())
            c3d = np.zeros((len(samples_keys), 3))
            
            for i, key in enumerate(samples_keys):
                c3d[i] = com3d_dict[key].squeeze()
            
            # Save as .mat file
            sio.savemat(str(output_path), {
                "sampleID": np.array(samples_keys),
                "com": c3d,
                "metadata": {}  # Add any metadata if needed
            })
            
            print(f"    Generated COM file with {len(samples_keys)} frames")
            return output_path
    else:
        print(f"    Warning: Empty labelData in {label3d_path}")
        return None

def organize_videos_sequentially(sessions_info, output_base_dir):
    """Organize videos sequentially across all regions (vid1, vid2, vid3...)"""
    
    output_base = Path(output_base_dir)
    videos_dir = output_base / "videos"
    videos_dir.mkdir(parents=True, exist_ok=True)
    
    # Collect all videos from all sessions
    all_videos = []
    for session_info in sessions_info:
        for video_info in session_info['videos']:
            video_info['calibration_file'] = session_info['calibration_file']
            all_videos.append(video_info)
    
    # Sort videos to ensure consistent ordering
    all_videos.sort(key=lambda x: (x['source_name'], x['region'], x['vid_name']))
    
    # Process each video sequentially
    video_mapping = []
    
    for i, video_info in enumerate(all_videos, 1):
        new_vid_name = f"vid{i}"
        print(f"\nProcessing {video_info['source_name']}/{video_info['region']}/{video_info['vid_name']} -> {new_vid_name}")
        
        # Create video directory
        vid_dir = videos_dir / new_vid_name
        vid_dir.mkdir(exist_ok=True)
        
        # Load calibration
        params = load_anipose_calibration(video_info['calibration_file'])
        
        # Load and convert 3D labels
        print(f"  Converting 3D labels from: {video_info['csv_file'].name}")
        label_data = load_anipose_3d_labels(video_info['csv_file'])
        
        # Create Label3D mat file
        mat_file = vid_dir / f"{new_vid_name}_Label3D_dannce.mat"
        create_label3d_mat(label_data, params, mat_file, new_vid_name, h5_files_2d=video_info.get('h5_files_2d'))
        
        # Generate COM from Label3D file
        com_dir = vid_dir / "COM" / "predict01"
        com_dir.mkdir(parents=True, exist_ok=True)
        com_file = com_dir / "com3d.mat"
        generate_com_from_label3d(mat_file, com_file)
        
        # Create video directories and copy videos
        vid_videos_dir = vid_dir / "videos"
        vid_videos_dir.mkdir(exist_ok=True)
        
        for cam_info in video_info['cameras']:
            cam_label = VIDEO_TO_CAMERA.get(cam_info['cam'])
            if cam_label is None:
                print(f"    Warning: Unknown camera key '{cam_info['cam']}' for {video_info['vid_name']}")
                continue

            cam_dir = vid_videos_dir / cam_label
            cam_dir.mkdir(exist_ok=True)
            
            # Copy video as 0.mp4
            src_video = cam_info['file']
            dst_video = cam_dir / "0.mp4"
            
            if not dst_video.exists():
                print(f"    Copying {src_video.name} -> {dst_video}")
                shutil.copy2(src_video, dst_video)
            else:
                print(f"    Video already exists: {dst_video}")
        
        # Store mapping info for config files and tracking
        video_mapping.append({
            'vid_name': new_vid_name,
            'original_source': video_info['source_name'],
            'original_region': video_info['region'],
            'original_name': video_info['vid_name'],
            'original_path': str(video_info['session_dir']),
            'calibration_file': str(video_info['calibration_file']),
            'mat_file': mat_file,
            'vid_dir': vid_dir,
            'com_file': com_file  # Add COM file path
        })
    
    return video_mapping

def save_video_mapping(video_mapping, output_base_dir):
    """Save video mapping to JSON and text files for tracking"""
    
    output_base = Path(output_base_dir)
    
    # Save as JSON for programmatic access
    json_path = output_base / "video_source_mapping.json"
    with open(json_path, 'w') as f:
        json_mapping = []
        for vm in video_mapping:
            json_mapping.append({
                'dannce_name': vm['vid_name'],
                'original_source': vm['original_source'],
                'original_region': vm['original_region'],
                'original_video': vm['original_name'],
                'original_path': vm['original_path']
            })
        json.dump(json_mapping, f, indent=2)
    
    # Save as readable text file
    txt_path = output_base / "video_source_mapping.txt"
    with open(txt_path, 'w') as f:
        f.write("DANNCE Video Source Mapping\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*80 + "\n\n")
        
        # Group by source
        current_source = None
        for vm in video_mapping:
            if vm['original_source'] != current_source:
                if current_source is not None:
                    f.write("\n")
                current_source = vm['original_source']
                f.write(f"Source: {current_source}\n")
                f.write("-"*60 + "\n")
            
            f.write(f"{vm['vid_name']:8} <- {vm['original_region']:4} / {vm['original_name']:8} (from {vm['original_path']})\n")
    
    print(f"\nSaved video mapping to:")
    print(f"  - {json_path}")
    print(f"  - {txt_path}")

def generate_io_yaml(video_mapping, output_base_dir):
    """Generate io.yaml configuration file"""
    
    output_base = Path(output_base_dir)
    io_yaml_path = output_base / "io.yaml"
    
    # Create COM and DANNCE directories
    com_train_dir = output_base / "COM" / "train00"
    com_predict_dir = output_base / "COM" / "predict01"
    dannce_train_dir = output_base / "DANNCE" / "train00"
    dannce_predict_dir = output_base / "DANNCE" / "predict00"
    
    com_train_dir.mkdir(parents=True, exist_ok=True)
    com_predict_dir.mkdir(parents=True, exist_ok=True)
    dannce_train_dir.mkdir(parents=True, exist_ok=True)
    dannce_predict_dir.mkdir(parents=True, exist_ok=True)
    
    # Build configuration
    config = {
        'com_train_dir': str(com_train_dir),
        'com_predict_dir': str(com_predict_dir),
        'com_predict_weights': '/usr/project/xtmp/rl349/project-dannce/37training/COM/train00/checkpoint.pth',
        'com_exp': [],
        'dannce_train_dir': str(dannce_train_dir),
        'dannce_predict_dir': str(dannce_predict_dir),
        'exp': [],
        'com_fromlabels': True
    }
    
    # Add entries for each video
    for video_info in video_mapping:
        label3d_file = str(video_info['mat_file'])
        viddir = str(video_info['vid_dir'] / "videos")
        com_file = f"./videos/{video_info['vid_name']}/COM/predict01/com3d.mat"
        
        # COM experiment entry
        config['com_exp'].append({
            'label3d_file': label3d_file,
            'viddir': viddir
        })
        
        # DANNCE experiment entry
        config['exp'].append({
            'label3d_file': label3d_file,
            'com_file': com_file,
            'viddir': viddir
        })
    
    # Write YAML file
    with open(io_yaml_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    
    print(f"\nGenerated io.yaml: {io_yaml_path}")
    print(f"  COM experiments: {len(config['com_exp'])}")
    print(f"  DANNCE experiments: {len(config['exp'])}")

def generate_sdannce_config(video_mapping, output_base_dir):
    """Generate sdannce_config_custom.yaml configuration file"""

    output_base = Path(output_base_dir)
    config_path = output_base / "sdannce_config_custom.yaml"

    if config_path.exists():
        print(f"Config exists, skipping overwrite: {config_path}")
        return

    img_height, img_width = detect_image_size(video_mapping)

    config = {
        # Basic settings
        'io_config': 'io.yaml',
        'dataset': 'label3d',
        'camnames': CAMERA_ORDER,
        'n_views': len(CAMERA_ORDER),
        'n_instances': 1,
        'random_seed': 1024,

        # Data settings
        'crop_height': [0, img_height],
        'crop_width': [0, img_width],
        'vmin': -120,
        'vmax': 120,
        'nvox': 80,
        'interp': 'nearest',
        'unlabeled_sampling': 'equal',
        'COM_augmentation': False,
        
        # Training settings
        'batch_size': 4,
        'epochs': 2,
        'lr': 0.0001,
        'train_mode': 'finetune',  # Changed from 'finetune' to 'new'
        'dannce_finetune_weights': './DANNCE/train00/checkpoint-epoch2.pth',  # No pretrained weights for new training
        'num_validation_per_exp': 50,
        'save_period': 1,
        'data_split_seed': 1024,
        'num_train_per_exp': 10000,
        
        # Learning rate scheduler
        'lr_scheduler': {
            'type': 'MultiStepLR',
            'args': {
                'milestones': [20, 35],
                'gamma': 0.5
            }
        },
        
        # Architecture tuned for mouse19/WT recordings
        'expval': True,
        'net_type': 'dannce',
        'n_channels_in': 3,
        'n_channels_out': len(TARGET_MOUSE19_JOINTS),
        'new_n_channels_out': len(TARGET_MOUSE19_JOINTS),
        'n_views': len(CAMERA_ORDER),
        'multi_gpu_train': False,
        'gpu_id': [0, 1, 2, 3],
        
        # Graph configuration
        'graph_cfg': {
            'model': 'PoseGCN',
            'n_instances': 1,
            'hidden_dim': 128,
            'n_layers': 3,
            'dropout': 0.2,
            'use_residual': False,
            'predict_diff': True,
            'use_features': True,
            'fuse_dim': 128
        },
        
        # Loss configuration - updated to use mouse19 profile
        'metric': ['euclidean_distance_3D'],
        'loss': {
            'L1Loss': {
                'loss_weight': 1.0
            },
            'BoneLengthLoss': {
                'loss_weight': 0.5,
                'compute_priors_from_data': True,
                'body_profile': 'mouse19',
                'mask': [2],
                'relative_scale': True,
                'ref_loss_weight': 0.0
            },
            'ConsistencyLoss': {
                'copies_per_sample': 4,
                'loss_weight': 0.1
            }
        },
        
        # Data augmentation - use mouse19 left/right groupings
        'medfilt_window': 30,
        'rand_view_replace': True,
        'n_rand_views': len(CAMERA_ORDER),
        'mirror_augmentation': True,
        'left_keypoints': MOUSE19_LEFT_KEYPOINTS,
        'right_keypoints': MOUSE19_RIGHT_KEYPOINTS,
        'augment_hue': False,
        'augment_brightness': False,
        'augment_bright_val': 0.01,
        
        # Skeleton profile
        'skeleton': 'mouse19',
        'body_profile': 'mouse19',

        # Predictions
        'max_num_samples': 'max',
        'dannce_train_dir': './SDANNCE/my_custom_training',
    }
    
    # Write YAML file
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    
    print(f"Generated sdannce_config_custom.yaml: {config_path}")
    print(f"  - Configured for mouse19 skeleton ({len(TARGET_MOUSE19_JOINTS)} keypoints)")
    print(f"  - Train mode set to 'finetune' with three-camera input")
    print(f"  - BoneLengthLoss set to compute priors from data")

def generate_dannce_config(video_mapping, output_base_dir):
    """Generate dannce_config_custom.yaml configuration file"""

    output_base = Path(output_base_dir)
    config_path = output_base / "dannce_config_custom.yaml"

    if config_path.exists():
        print(f"Config exists, skipping overwrite: {config_path}")
        return

    img_height, img_width = detect_image_size(video_mapping)

    config = {
        '### basic ###': None,
        'io_config': 'io.yaml',
        'dataset': 'label3d',
        'camnames': CAMERA_ORDER,
        'random_seed': 1024,

        '### data ###': None,
        'crop_height': [0, img_height],
        'crop_width': [0, img_width],
        
        # volumetric representation - adjust based on your animal size
        'vmin': -120,
        'vmax': 120,
        'nvox': 80,
        'interp': 'nearest',

        '### train ###': None,
        'batch_size': 2,
        'epochs': 7,
        'lr': 0.0001,
        'train_mode': 'finetune',  # Changed from 'finetune' to 'new'
        'channel_combo': None,
        'dannce_finetune_weights': './DANNCE/train00/checkpoint-epoch2.pth',  # No pretrained weights for new training
        'dannce_train_dir': './DANNCE/my_custom_training',
        'COM_augmentation': True,
        'COM_aug_iters': 2,  # Augment training samples
        'multi_gpu_train': False,
        'gpu_id': [0, 1, 2, 3],

        'num_validation_per_exp': 500,
        'save_period': 1,
        'data_split_seed': 1024,
        'num_train_per_exp': 10000,
        
        '### architecture ###': None,
        'expval': True,
        'net_type': 'dannce',
        'n_channels_in': 3,
        'n_channels_out': len(TARGET_MOUSE19_JOINTS),
        'new_n_channels_out': len(TARGET_MOUSE19_JOINTS),
        'n_views': len(CAMERA_ORDER),
        'use_npy': True,
        
        '### loss ###': None,
        'metric': ['euclidean_distance_3D'],
        'loss': {
            'L1Loss': {
                'loss_weight': 1.0
            }
        },
        
        '### data augmentation ###': None,
        'medfilt_window': 30,
        'rand_view_replace': True,
        'n_rand_views': len(CAMERA_ORDER),
        'mirror_augmentation': True,
        'left_keypoints': MOUSE19_LEFT_KEYPOINTS,
        'right_keypoints': MOUSE19_RIGHT_KEYPOINTS,
        
        'augment_hue': False,
        'augment_brightness': False,
        'augment_bright_val': 0.01,
        
        '### skeleton ###': None,
        'skeleton': 'mouse19',
        'body_profile': 'mouse19',

        '### prediction ###': None,
        'max_num_samples': 'max'
    }
    
    # Custom YAML writer to handle section headers
    with open(config_path, 'w') as f:
        for key, value in config.items():
            if key.startswith('###'):
                f.write(f"\n{key}\n")
            else:
                if isinstance(value, dict):
                    f.write(f"{key}:\n")
                    for k, v in value.items():
                        if isinstance(v, dict):
                            f.write(f"    {k}:\n")
                            for kk, vv in v.items():
                                f.write(f"      {kk}: {vv}\n")
                        else:
                            f.write(f"    {k}: {v}\n")
                elif value is None:
                    f.write(f"{key}: null\n")
                else:
                    f.write(f"{key}: {value}\n")
    
    print(f"Generated dannce_config_custom.yaml: {config_path}")
    print(f"  - Configured for mouse19 skeleton ({len(TARGET_MOUSE19_JOINTS)} keypoints)")
    print(f"  - Three-camera training settings written")

def convert_dataset(dataset_key, dataset_cfg):
    """Execute conversion for a configured dataset."""

    description = dataset_cfg['description']
    source_configs = dataset_cfg['source_configs']
    output_base_dir = dataset_cfg['output_base_dir']

    print(f"=== {description} to s-DANNCE Conversion (mouse19, 3 cameras) ===\n")

    output_base = Path(output_base_dir)
    output_base.mkdir(parents=True, exist_ok=True)

    print(f"Output directory: {output_base_dir}")

    print("\n" + "="*50)
    print("STEP 1: Scanning Source Session(s)")
    print("="*50)

    all_sessions_info = scan_multiple_sources(source_configs)

    if not all_sessions_info:
        print("Error: No valid sessions found!")
        return

    print(f"\n\nTotal sessions found: {len(all_sessions_info)}")
    total_videos = 0
    for session in all_sessions_info:
        source_name = session.get('source_name', 'unknown')
        region = session['region']
        n_videos = len(session['videos'])
        total_videos += n_videos
        print(f"  {source_name}/{region}: {n_videos} videos")

    print(f"Total videos to process: {total_videos}")

    print("\n" + "="*50)
    print("STEP 2: Converting and Organizing Videos")
    print("="*50)

    video_mapping = organize_videos_sequentially(all_sessions_info, output_base_dir)

    save_video_mapping(video_mapping, output_base_dir)

    print("\n" + "="*50)
    print("STEP 3: Generating Configuration Files")
    print("="*50)

    generate_io_yaml(video_mapping, output_base_dir)
    generate_sdannce_config(video_mapping, output_base_dir)
    generate_dannce_config(video_mapping, output_base_dir)

    print("\n" + "="*50)
    print("CONVERSION COMPLETE")
    print("="*50)

    print(f"\nOutput structure created in: {output_base_dir}")
    print(f"Videos processed: {len(video_mapping)}")

    print("\nVideo mapping summary:")
    source_counts = defaultdict(lambda: defaultdict(int))
    for mapping in video_mapping:
        source_counts[mapping['original_source']][mapping['original_region']] += 1

    for source, regions in source_counts.items():
        print(f"\n{source}:")
        for region, count in regions.items():
            print(f"  {region}: {count} videos")

    print("\nGenerated files:")
    print("  - videos/vid1/, videos/vid2/, ... (sequential video folders)")
    print("  - videos/vidN/vidN_Label3D_dannce.mat (converted labels)")
    print("  - videos/vidN/COM/predict01/com3d.mat (generated COM from Label3D)")
    print("  - videos/vidN/videos/Camera1/0.mp4, videos/vidN/videos/Camera2/0.mp4, videos/vidN/videos/Camera3/0.mp4")
    print("  - io.yaml (experiment configuration)")
    print("  - sdannce_config_custom.yaml (s-DANNCE training configuration)")
    print("  - dannce_config_custom.yaml (DANNCE training configuration)")
    print("  - video_source_mapping.json (machine-readable mapping)")
    print("  - video_source_mapping.txt (human-readable mapping)")

    print("\nNext steps:")
    print("1. Review the video_source_mapping.txt file to verify all videos were found")
    print("2. Check the generated configuration files")
    print("3. Adjust paths in sdannce_config_custom.yaml if needed")
    print("4. Run s-DANNCE training with the generated config")
    print("\nNote: COM files are regenerated from each Label3D file.")
    print(f"Note: Using mouse19 skeleton with {len(TARGET_MOUSE19_JOINTS)} keypoints and three cameras")


def main():
    parser = argparse.ArgumentParser(description="Convert WT/CP(full) three-camera datasets to s-DANNCE format")
    parser.add_argument(
        '--datasets',
        nargs='+',
        default=list(DATASET_CONFIGS.keys()),
        choices=sorted(DATASET_CONFIGS.keys()),
        help='Datasets to process (default: both wt and cpfull)'
    )
    args = parser.parse_args()

    for dataset_key in args.datasets:
        dataset_cfg = DATASET_CONFIGS[dataset_key]
        convert_dataset(dataset_key, dataset_cfg)


if __name__ == "__main__":
    main()
