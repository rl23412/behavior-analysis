#!/usr/bin/env python3
"""Render mouse19 overlays using imageio for frame access."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, Tuple

import cv2
import imageio
import numpy as np


MOUSE19_CONNECTIONS: Tuple[Tuple[int, int], ...] = (
    (0, 1), (0, 2), (1, 2),
    (0, 3), (3, 4), (4, 5), (5, 6),
    (4, 7), (7, 8), (8, 9),
    (4, 10), (10, 11), (11, 12),
    (5, 13), (13, 14), (14, 15),
    (5, 16), (16, 17), (17, 18),
)


def _load_coordinates(path: Path, key: str | None) -> np.ndarray:
    data = np.load(path, allow_pickle=True)
    if isinstance(data, dict):
        coords_dict = data
    elif data.dtype == object and data.shape == ():
        coords_dict = data.item()
    else:
        coords_dict = None

    if coords_dict is not None:
        if key is None:
            raise ValueError("Coordinate file contains multiple entries; supply --coords-key")
        if key not in coords_dict:
            raise KeyError(f"Key '{key}' not found in coordinates file")
        coords = np.asarray(coords_dict[key])
    else:
        coords = np.asarray(data)

    if coords.ndim != 3 or coords.shape[2] != 2:
        raise ValueError(f"Unexpected coordinate shape {coords.shape}; expected (frames, joints, 2)")
    return coords.astype(np.float32, copy=False)


def _compute_centroid(coords: np.ndarray) -> np.ndarray | None:
    if coords.size == 0:
        return None
    valid = ~np.isnan(coords).any(axis=1)
    if not np.any(valid):
        return None
    return coords[valid].mean(axis=0)


def _overlay_skeleton(frame_bgr: np.ndarray, coords: np.ndarray, connections: Iterable[Tuple[int, int]]) -> np.ndarray:
    annotated = frame_bgr.copy()
    for idx_a, idx_b in connections:
        if idx_a >= len(coords) or idx_b >= len(coords):
            continue
        pt_a = coords[idx_a]
        pt_b = coords[idx_b]
        if np.isnan(pt_a).any() or np.isnan(pt_b).any():
            continue
        ax, ay = int(round(pt_a[0])), int(round(pt_a[1]))
        bx, by = int(round(pt_b[0])), int(round(pt_b[1]))
        cv2.line(annotated, (ax, ay), (bx, by), (0, 255, 255), 2)

    for joint_idx, (x, y) in enumerate(coords):
        if np.isnan(x) or np.isnan(y):
            continue
        cx, cy = int(round(x)), int(round(y))
        cv2.circle(annotated, (cx, cy), 4, (0, 255, 0), -1)
        cv2.putText(
            annotated,
            str(joint_idx + 1),
            (cx + 5, cy + 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )
    return annotated


def _crop_around_centroid(frame_bgr: np.ndarray, centroid: np.ndarray | None, crop_size: int) -> np.ndarray:
    if centroid is None or np.isnan(centroid).any():
        return cv2.resize(frame_bgr, (crop_size, crop_size))
    cx, cy = int(round(centroid[0])), int(round(centroid[1]))
    half = crop_size // 2
    x1 = max(0, cx - half)
    y1 = max(0, cy - half)
    x2 = min(frame_bgr.shape[1], cx + half)
    y2 = min(frame_bgr.shape[0], cy + half)
    cropped = frame_bgr[y1:y2, x1:x2]
    if cropped.size == 0:
        return np.zeros((crop_size, crop_size, 3), dtype=np.uint8)
    return cv2.resize(cropped, (crop_size, crop_size))


def render_overlay(
    video_path: Path,
    coords: np.ndarray,
    output_full: Path,
    output_crop: Path | None,
    crop_size: int,
    fps_override: float | None,
    title: str,
    connections: Iterable[Tuple[int, int]] = MOUSE19_CONNECTIONS,
) -> Dict[str, int]:
    reader = imageio.get_reader(video_path)
    meta = reader.get_meta_data()
    fps = fps_override or float(meta.get("fps", 30.0))

    writer_full = imageio.get_writer(output_full, fps=fps)
    writer_crop = imageio.get_writer(output_crop, fps=fps) if output_crop else None

    frames_written = 0
    for idx, frame_rgb in enumerate(reader):
        if idx >= len(coords):
            break
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        coords_frame = coords[idx]
        centroid = _compute_centroid(coords_frame)
        overlay_bgr = _overlay_skeleton(frame_bgr, coords_frame, connections)

        cv2.putText(
            overlay_bgr,
            f"{title} frame {idx}",
            (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 255),
            2,
        )
        overlay_rgb = cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB)
        writer_full.append_data(overlay_rgb)

        if writer_crop:
            cropped_bgr = _crop_around_centroid(overlay_bgr, centroid, crop_size)
            cv2.putText(
                cropped_bgr,
                f"{title} frame {idx}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )
            writer_crop.append_data(cv2.cvtColor(cropped_bgr, cv2.COLOR_BGR2RGB))

        frames_written += 1

    reader.close()
    writer_full.close()
    if writer_crop:
        writer_crop.close()
    return {"frames_written": frames_written}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path, help="Path to the source video (mp4)")
    parser.add_argument("coordinates", type=Path, help="Path to coordinates npy file")
    parser.add_argument("output", type=Path, help="Destination path for full-size overlay mp4")
    parser.add_argument(
        "--coords-key",
        type=str,
        default=None,
        help="Dictionary key inside the coordinates npy (if applicable)",
    )
    parser.add_argument(
        "--crop-output",
        type=Path,
        default=None,
        help="Optional output path for cropped overlay mp4",
    )
    parser.add_argument(
        "--crop-size",
        type=int,
        default=400,
        help="Crop size for centroid view (default: 400)",
    )
    parser.add_argument(
        "--fps-override",
        type=float,
        default=None,
        help="Override FPS value for the output writer",
    )
    parser.add_argument(
        "--title",
        type=str,
        default="overlay",
        help="Text prefix to print on each frame",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    coords = _load_coordinates(args.coordinates, args.coords_key)
    stats = render_overlay(
        video_path=args.video,
        coords=coords,
        output_full=args.output,
        output_crop=args.crop_output,
        crop_size=args.crop_size,
        fps_override=args.fps_override,
        title=args.title,
    )
    print(f"Rendered {stats['frames_written']} frames")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
