#!/usr/bin/env python
"""Run DANNCE predictions for cpfull/wt (train200 checkpoints), build overlays,
reproject 3D poses into 2D, and bundle the outputs under /work/rl349.

Usage:
    python predict_and_package_train200.py

Prereqs:
    - Run inside the sdannce conda env or ensure PATH points to the sdannce bin.
    #!/bin/bash
#SBATCH --job-name=danncetrain                    # Name shown by squeue / sacct
#SBATCH --partition=scavenger-gpu                  # Any GPU partition you’re eligible for
#SBATCH --nodes=1                               # All resources on one node
#SBATCH --ntasks=1                              # One task (MPI rank)
#SBATCH --cpus-per-task=4                       # 8 CPU cores
#SBATCH --mem=20G                               # 80 GB system RAM
#SBATCH --gres=gpu:1                            # 4 GPUs on that node
#SBATCH --time=7-00:00:00                       # HH:MM:SS  (1 day here)

# ------------  Software setup  ------------
            # or the Anaconda module your cluster provides
eval "$($HOME/.local/bin/micromamba shell hook -s bash)"

# 2) Activate env
micromamba activate /work/rl349/miniconda3/envs/sdannce  
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import scipy.io as sio
import yaml

from dannce.engine.utils.projection import load_cameras, project_to_2d, distortPoints

os.environ.setdefault("SETUPTOOLS_USE_DISTUTILS", "stdlib")

# Absolute paths
SDANNCE_BIN = Path("/work/rl349/miniconda3/envs/sdannce/bin")
DANNCE_CLI = SDANNCE_BIN / "dannce"
PYTHON_BIN = SDANNCE_BIN / "python"

DEST_ROOT = Path("/work/rl349/dannce_predictions_train200")
DEST_ROOT.mkdir(parents=True, exist_ok=True)


@dataclass
class Experiment:
    name: str
    run_tag: str
    exp_root: Path
    predict_workspace: Path
    config_path: Path
    io_config_path: Path
    skeleton: str = "mouse19"
    n_animals: int = 1

    def label(self) -> str:
        return f"{self.name}_{self.run_tag}"

    @property
    def predict_outputs(self) -> Path:
        return self.predict_workspace / "predict_outputs"


EXPERIMENTS: List[Experiment] = [
    Experiment(
        name="cpfull",
        run_tag="train200",
        exp_root=Path("/work/rl349/dannce/cpfull_mouse19"),
        predict_workspace=Path("/hpc/home/rl349/cpfull_predict_train200"),
        config_path=Path("/hpc/home/rl349/cpfull_predict_train200/config_predict_train200.yaml"),
        io_config_path=Path("/hpc/home/rl349/cpfull_predict_train200/io_predict_vid1.yaml"),
    ),
    Experiment(
        name="wt",
        run_tag="train200",
        exp_root=Path("/work/rl349/dannce/wt_mouse19"),
        predict_workspace=Path("/hpc/home/rl349/wt_predict_train200"),
        config_path=Path("/hpc/home/rl349/wt_predict_train200/config_predict_train200.yaml"),
        io_config_path=Path("/hpc/home/rl349/wt_predict_train200/io_predict_vid1.yaml"),
    ),
]


def run(cmd: List[str], cwd: Path | None = None) -> None:
    print(f"[cmd] {' '.join(cmd)} (cwd={cwd})")
    subprocess.run(cmd, check=True, cwd=cwd)


def load_camnames(config_path: Path) -> List[str]:
    config = yaml.safe_load(config_path.read_text())
    camnames = config.get("camnames")
    if not camnames:
        raise ValueError(f"No camnames found in {config_path}")
    return camnames


def get_label_file(io_config_path: Path) -> Path:
    io_cfg = yaml.safe_load(io_config_path.read_text())
    try:
        label_file = Path(io_cfg["exp"][0]["label3d_file"])
    except Exception as exc:  # pylint: disable=broad-except
        raise ValueError(f"Could not parse label3d_file from {io_config_path}") from exc
    if not label_file.is_absolute():
        label_file = (io_config_path.parent / label_file).resolve()
    return label_file


def generate_reprojection(exp: Experiment, camnames: List[str], output_dir: Path) -> Path:
    save_data_path = exp.predict_outputs / "save_data_AVG.mat"
    mat = sio.loadmat(save_data_path)
    poses = mat["pred"]  # [N, n_instances, 3, n_markers]
    poses = poses[:, 0].transpose(0, 2, 1)  # [N, n_markers, 3]
    sample_ids = mat.get("sampleID")
    if sample_ids is not None:
        sample_ids = np.squeeze(sample_ids).astype(int)
    else:
        sample_ids = np.arange(poses.shape[0])

    label_file = get_label_file(exp.io_config_path)
    cameras = load_cameras(str(label_file))

    coords_per_cam = []
    for cam in camnames:
        params = cameras[cam]
        pts_flat = poses.reshape(-1, 3)
        proj = project_to_2d(pts_flat, params["K"], params["r"], params["t"])[:, :2]
        proj = distortPoints(
            proj,
            params["K"],
            np.squeeze(params["RDistort"]),
            np.squeeze(params["TDistort"]),
        )
        proj = proj.T.reshape(poses.shape[0], poses.shape[1], 2)
        coords_per_cam.append(proj.astype(np.float32))

    coords = np.stack(coords_per_cam, axis=1)  # [N, n_cams, n_markers, 2]
    repro_path = output_dir / f"{exp.label()}_reprojected_coords.npy"
    np.save(
        repro_path,
        {
            "camera_order": camnames,
            "sample_ids": sample_ids,
            "coords": coords,
        },
    )
    return repro_path


def write_structure_md(dest_dir: Path, exp: Experiment, repro_path: Path, camnames: List[str], coords_shape: tuple[int, ...]) -> None:
    md_path = dest_dir / f"{exp.label()}_structure.md"
    content = f"""# {exp.label()} deliverables

* `save_data_AVG_{exp.label()}.mat` — raw DANNCE prediction volume (3D poses).
* `com3d_used_{exp.label()}.mat` — COM positions used during inference.
* `stats_dannce_predict_{exp.label()}.log` — runtime log.
* `{exp.label()}_overlay.mp4` — three-view 2D visualization.
* `{repro_path.name}` — NumPy array (object) with:
  * `camera_order`: list of camera names ({', '.join(camnames)}).
  * `sample_ids`: int array matching frame indices used during inference.
  * `coords`: float32 array of shape {coords_shape} = (frames, cameras, markers, 2) containing pixel coordinates (x, y).

Load example:
```python
import numpy as np
payload = np.load('{repro_path.name}', allow_pickle=True).item()
coords = payload['coords']
```
"""
    md_path.write_text(content)


def copy_outputs(exp: Experiment, camnames: List[str]) -> None:
    dest_dir = DEST_ROOT / exp.label()
    dest_dir.mkdir(parents=True, exist_ok=True)

    mapping = {
        exp.predict_outputs / "save_data_AVG.mat": dest_dir / f"save_data_AVG_{exp.label()}.mat",
        exp.predict_outputs / "com3d_used.mat": dest_dir / f"com3d_used_{exp.label()}.mat",
        exp.predict_outputs / "stats_dannce_predict.log": dest_dir / f"stats_dannce_predict_{exp.label()}.log",
    }
    for src, dst in mapping.items():
        shutil.copy2(src, dst)

    vis_dir = exp.predict_outputs / "vis"
    overlay_mp4s = sorted(vis_dir.glob("*.mp4"))
    if overlay_mp4s:
        shutil.copy2(overlay_mp4s[0], dest_dir / f"{exp.label()}_overlay.mp4")

    repro_path = generate_reprojection(exp, camnames, dest_dir)
    coords_shape = np.load(repro_path, allow_pickle=True).item()["coords"].shape
    write_structure_md(dest_dir, exp, repro_path, camnames, coords_shape)

    # Persist metadata summary for quick reference
    summary = {
        "experiment": exp.name,
        "run_tag": exp.run_tag,
        "camnames": camnames,
        "predict_workspace": str(exp.predict_workspace),
        "dest_dir": str(dest_dir),
    }
    with open(dest_dir / f"summary_{exp.label()}.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)


def run_prediction(exp: Experiment, camnames: List[str]) -> None:
    run([str(DANNCE_CLI), "predict", "dannce", str(exp.config_path)], cwd=exp.exp_root)

    # Determine frame count from output file for the visualization step.
    mat = sio.loadmat(exp.predict_outputs / "save_data_AVG.mat")
    n_frames = mat["pred"].shape[0]
    cameras_arg = ",".join(camnames)
    run(
        [
            str(PYTHON_BIN),
            "-m",
            "dannce.engine.utils.vis",
            "--root",
            str(exp.exp_root / "videos" / "vid1"),
            "--pred",
            str(exp.predict_outputs),
            "--datafile",
            "save_data_AVG.mat",
            "--skeleton",
            exp.skeleton,
            "--n_animals",
            str(exp.n_animals),
            "--cameras",
            cameras_arg,
            "--n_frames",
            str(n_frames),
            "--start_frame",
            "0",
        ]
    )


def main() -> None:
    for exp in EXPERIMENTS:
        camnames = load_camnames(exp.config_path)
        run_prediction(exp, camnames)
        copy_outputs(exp, camnames)

    print(f"All outputs staged in {DEST_ROOT}")


if __name__ == "__main__":
    main()
