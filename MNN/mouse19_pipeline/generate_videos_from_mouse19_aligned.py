#!/usr/bin/env python3
"""Convenience runner for the mouse19-aligned embedding outputs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from create_vid import generate_videos_from_matlab_csv


MOUSE19_VID_MAPPING = [
    {"dannce_name": f"vid{i}", "original_region": "cpfull"} for i in range(1, 6)
] + [
    {"dannce_name": f"vid{i+5}", "original_region": "wt"} for i in range(1, 6)
]

def parse_args(argv: list[str]) -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base",
        dest="base_dir",
        default=str(repo_root / "combined_videos_mouse19"),
        help="Directory containing the vid*/ raw videos (default: %(default)s)",
    )
    parser.add_argument(
        "--analysis",
        dest="analysis_outputs_dir",
        default=str(repo_root / "analysis_outputs_mouse19_aligned"),
        help="Directory with MATLAB CSV outputs (default: %(default)s)",
    )
    parser.add_argument(
        "--output",
        dest="output_dir",
        default=str(repo_root / "region_video_overlays_mouse19_aligned"),
        help="Destination directory for rendered videos (default: %(default)s)",
    )
    parser.add_argument(
        "--coordinates",
        dest="coordinates_path",
        default=str(repo_root / 'coordinates_2D_mouse19.npy'),
        help="Optional override for the 2D coordinates .npy file (default: %(default)s)",
    )
    parser.add_argument(
        "--max-regions",
        type=int,
        default=None,
        help="Optional cap on how many watershed regions to process.",
    )
    parser.add_argument(
        "--regions",
        nargs="*",
        type=int,
        default=None,
        help="Explicit list of watershed region indices to render (overrides --max-regions).",
    )

    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])

    kwargs = dict(
        base_dir=args.base_dir,
        analysis_outputs_dir=str(args.analysis_outputs_dir),
        output_dir=str(args.output_dir),
        skeleton_type="mouse19",
        vid_mapping=MOUSE19_VID_MAPPING,
    )

    if args.coordinates_path:
        kwargs["coordinates_path"] = args.coordinates_path
    if args.max_regions is not None:
        kwargs["max_watershed_regions"] = args.max_regions
    if args.regions:
        kwargs["watershed_regions"] = args.regions

    success = generate_videos_from_matlab_csv(**kwargs)
    return 0 if success else 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
