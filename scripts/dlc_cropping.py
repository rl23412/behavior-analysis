"""Utilities for camera-specific DLC cropping configuration."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, Optional, Tuple

CAMERA_CROPPING: Dict[str, Tuple[int, int, int, int]] = {
    "cam2": (180, 900, 120, 650),
}


def parse_camera_from_name(path: Path | str) -> Optional[str]:
    """Return camera label such as 'cam2' parsed from a video or H5 path."""

    stem = Path(path).stem.lower()
    normalized = stem.replace("_", "-")
    for part in normalized.split("-"):
        if re.fullmatch(r"cam\d+", part):
            return part
    return None


def get_crop_bounds(camera: str) -> Optional[Tuple[int, int, int, int]]:
    """Lookup cropping bounds for the given camera label."""

    return CAMERA_CROPPING.get(camera)


def requires_crop_adjustment(path: Path | str) -> bool:
    """True when the asset comes from a camera with cropping applied."""

    camera = parse_camera_from_name(path)
    return bool(camera and camera in CAMERA_CROPPING)
