#!/usr/bin/env python3
"""
Reproject mouse19 DANNCE predictions to Camera1 2D keypoints.

Reads 3D predictions from `/work/rl349/dannce_predictions_train200/mouse19` for
both cpfull and wt cohorts, loads the original camera calibration from the raw
video directories under `/work/rl349/dannce/`, and projects the 19 joints onto
Camera1. The resulting per-video arrays are stored in a `.npy` dictionary keyed
as `vid1`..`vid10` (cpfull -> vid1-vid5, wt -> vid6-vid10) so downstream overlay
scripts can reuse the legacy naming convention.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
from scipy.io import loadmat

from dannce.engine.utils import vis

PREDICTION_ROOT = Path("/work/rl349/dannce_predictions_train200/mouse19")
EXPERIMENT_ROOT = Path("/work/rl349/dannce")

# Map legacy video ids to prediction groups and source folders.
VIDEO_MAP: List[Dict[str, object]] = [
    {"key": f"vid{i}", "group": "cpfull", "video_id": i} for i in range(1, 6)
] + [
    {"key": f"vid{i+5}", "group": "wt", "video_id": i} for i in range(1, 6)
]


def _load_predictions(group: str, video_id: int) -> np.ndarray:
    pred_path = (
        PREDICTION_ROOT
        / group
        / f"save_data_AVG_{group}_vid{video_id}.mat"
    )
    if not pred_path.exists():
        raise FileNotFoundError(f"Prediction file not found: {pred_path}")

    mat = loadmat(pred_path, simplify_cells=True)
    if "pred" not in mat:
        raise KeyError(f"'pred' field missing in {pred_path}")

    pred = mat["pred"]  # shape: (frames, 3, 19)
    if pred.ndim != 3 or pred.shape[1] != 3:
        raise ValueError(
            f"Unexpected prediction shape {pred.shape} in {pred_path}; "
            "expected (frames, 3, joints)"
        )
    return pred.astype(np.float32, copy=False)


def _load_camera(group: str, video_id: int, camera_name: str) -> Dict[str, np.ndarray]:
    # Raw videos + calibration live under Dannce experiment root
    exp_dir = EXPERIMENT_ROOT / f"{group}_mouse19" / "videos" / f"vid{video_id}"
    label_file = exp_dir / f"vid{video_id}_Label3D_dannce.mat"
    if not label_file.exists():
        raise FileNotFoundError(f"Camera calibration file missing: {label_file}")

    cameras = vis.load_cameras(str(label_file))
    if camera_name not in cameras:
        raise KeyError(
            f"Camera {camera_name} not found in {label_file}. "
            f"Available cameras: {list(cameras.keys())}"
        )
    return cameras[camera_name]




def _project_to_2d_single_camera(pts: np.ndarray, K: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Project 3D points to 2D using DANNCE camera conventions."""
    M = np.concatenate((R, t), axis=0) @ K
    homogeneous = np.concatenate((pts, np.ones((pts.shape[0], 1), dtype=pts.dtype)), axis=1)
    proj = homogeneous @ M
    proj[:, :2] /= proj[:, 2:3]
    return proj


def _reproject_single(
    pred: np.ndarray,
    camera: Dict[str, np.ndarray],
) -> np.ndarray:
    """Project 3D predictions to 2D for a single camera."""

    n_frames, _, n_joints = pred.shape
    # Rearrange to (frames, joints, xyz)
    pose = np.transpose(pred, (0, 2, 1))
    flat_pose = pose.reshape((-1, 3))  # (frames * joints, 3)

    K = camera["K"].astype(np.float64, copy=False)
    R = camera["r"].astype(np.float64, copy=False)
    # camera['t'] arrives as (1, 3); ensure row vector
    t = np.squeeze(camera["t"], axis=0).reshape(1, 3).astype(np.float64, copy=False)
    radial = np.squeeze(camera["RDistort"]).astype(np.float64, copy=False)
    tangential = np.squeeze(camera["TDistort"]).astype(np.float64, copy=False)

    proj = _project_to_2d_single_camera(flat_pose, K, R, t)[:, :2]
    proj = vis.distortPoints(proj, K, radial, tangential).T  # -> (n_pts, 2)
    proj = proj.astype(np.float32, copy=False)
    coords = proj.reshape(n_frames, n_joints, 2)
    return coords


def reproject_camera1(output_path: Path) -> Dict[str, np.ndarray]:
    results: Dict[str, np.ndarray] = {}
    for entry in VIDEO_MAP:
        key = str(entry["key"])
        group = str(entry["group"])
        video_id = int(entry["video_id"])

        pred = _load_predictions(group, video_id)
        camera = _load_camera(group, video_id, camera_name="Camera1")
        coords = _reproject_single(pred, camera)
        results[key] = coords
        print(
            f"Reprojected {group} vid{video_id}: "
            f"frames={coords.shape[0]}, joints={coords.shape[1]}"
        )

    np.save(output_path, results)
    print(f"Saved coordinates to {output_path}")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("mouse19_pipeline/coordinates_2D_mouse19.npy"),
        help="Destination .npy path for the 2D coordinate dictionary.",
    )
    args = parser.parse_args()

    output_path = args.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    reproject_camera1(output_path)


if __name__ == "__main__":
    main()
