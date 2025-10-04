#!/usr/bin/env python
"""Automate DANNCE train200 predictions for wt and cpfull datasets.

This script generates per-video prediction configs, launches predictions via
SLURM ``srun`` with GPU resources, renders overlay videos with additional CPU
cores, and copies the resulting artifacts into dataset-specific folders with
clear filenames.

Usage:
    python scripts/run_train200_predictions.py --datasets wt,cpfull --videos all

The command expects access to micromamba at ``$HOME/.local/bin/micromamba`` and
assumes the sdannce environment lives at ``/work/rl349/miniconda3/envs/sdannce``.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import numpy as np
import scipy.io as sio
import yaml

MICROMAMBA = Path.home() / ".local/bin/micromamba"
ENV_PATH = Path("/work/rl349/miniconda3/envs/sdannce")
MICROMAMBA_RUN = f"{MICROMAMBA} run -p {ENV_PATH}"
DANNCE_BIN = ENV_PATH / "bin/dannce"
PYTHON_BIN = ENV_PATH / "bin/python"

GPU_PARTITION = "scavenger-gpu"
CPU_PARTITION = "scavenger"

CONTROLLER_LOG = Path("/work/rl349/dannce/predict_train200_controller.log")
CONTROLLER_LOG.parent.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(CONTROLLER_LOG, mode="a", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
LOGGER = logging.getLogger("train200_controller")


@dataclass
class DatasetSpec:
    name: str
    root: Path
    skeleton: str = "mouse19"
    n_animals: int = 1

    @property
    def train_params_path(self) -> Path:
        return self.root / "DANNCE/train200/params.yaml"

    @property
    def train_checkpoint(self) -> Path:
        return self.root / "DANNCE/train200/checkpoint.pth"

    @property
    def io_path(self) -> Path:
        return self.root / "io.yaml"

    @property
    def predict_base(self) -> Path:
        return self.root / "DANNCE/predict200"

    @property
    def com_train_dir(self) -> Path:
        return self.root / "COM/train00"


@dataclass
class VideoSpec:
    dataset: DatasetSpec
    vid_name: str  # e.g., "vid1"

    @property
    def label3d_file(self) -> Path:
        return self.dataset.root / "videos" / self.vid_name / f"{self.vid_name}_Label3D_dannce.mat"

    @property
    def com_file(self) -> Path:
        return self.dataset.root / "videos" / self.vid_name / "COM/predict01/com3d.mat"

    @property
    def viddir(self) -> Path:
        return self.dataset.root / "videos" / self.vid_name / "videos"

    @property
    def workspace(self) -> Path:
        return self.dataset.predict_base / self.vid_name

    @property
    def predict_dir(self) -> Path:
        return self.workspace / "predict_outputs"

    @property
    def com_predict_dir(self) -> Path:
        return self.workspace / "com_predict"

    @property
    def config_path(self) -> Path:
        return self.workspace / f"config_predict_{self.dataset.name}_{self.vid_name}.yaml"

    @property
    def io_config_path(self) -> Path:
        return self.workspace / f"io_predict_{self.dataset.name}_{self.vid_name}.yaml"

    @property
    def overlay_dest(self) -> Path:
        return self.dataset.predict_base / f"overlay_{self.dataset.name}_{self.vid_name}.mp4"

    @property
    def save_data_dest(self) -> Path:
        return self.dataset.predict_base / f"save_data_AVG_{self.dataset.name}_{self.vid_name}.mat"

    @property
    def com3d_dest(self) -> Path:
        return self.dataset.predict_base / f"com3d_used_{self.dataset.name}_{self.vid_name}.mat"

    @property
    def stats_dest(self) -> Path:
        return self.dataset.predict_base / f"stats_dannce_predict_{self.dataset.name}_{self.vid_name}.log"


def run_cmd(command: Sequence[str]) -> None:
    subprocess.run(list(command), check=True)


def _detect_cpu_limit() -> int | None:
    """Return the CPU limit from the current SLURM allocation, if available."""
    env_order = ["SLURM_CPUS_PER_TASK", "SLURM_JOB_CPUS_PER_NODE", "SLURM_CPUS_ON_NODE"]
    for var in env_order:
        val = os.environ.get(var)
        if not val:
            continue
        try:
            # Handle formats like "8(x1)" or "4,4" by extracting the first integer.
            cleaned = ''.join(ch if ch.isdigit() or ch in ', ' else ' ' for ch in val)
            token = cleaned.replace(' ', '').split(',')[0]
            if token:
                return int(token)
        except ValueError:
            continue
    return None


def launch_srun(
    *,
    command: str,
    job_name: str,
    partition: str,
    cpus: int,
    time_limit: str,
    gres: str | None = None,
) -> subprocess.Popen[str]:
    available = _detect_cpu_limit()
    effective_cpus = max(1, min(cpus, available)) if available else max(1, cpus)
    if available and effective_cpus < cpus:
        LOGGER.warning(
            "Reducing CPU request for job '%s' from %s to %s to respect SLURM allocation (%s)",
            job_name, cpus, effective_cpus, available
        )
    srun_cmd: List[str] = [
        "srun",
        "--partition",
        partition,
        "--cpus-per-task",
        str(effective_cpus),
        "--time",
        time_limit,
        "--job-name",
        job_name,
    ]
    if gres:
        srun_cmd.extend(["--gres", gres])
    srun_cmd.extend(["bash", "-lc", command])
    return subprocess.Popen(srun_cmd)


def wait_for_jobs(jobs: Sequence[Tuple[str, subprocess.Popen[str]]], phase: str) -> None:
    for label, proc in jobs:
        ret = proc.wait()
        if ret != 0:
            LOGGER.error("%s job '%s' failed with exit code %s", phase, label, ret)
            raise RuntimeError(f"{phase} job '{label}' failed with exit code {ret}")
        LOGGER.info("%s job '%s' completed", phase.capitalize(), label)


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def dump_yaml(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)


def prepare_video_workspace(video: VideoSpec, train_params: dict, io_defaults: dict) -> None:
    LOGGER.info("Preparing workspace for %s %s", video.dataset.name, video.vid_name)
    # Ensure required folders exist and are clean
    if video.predict_dir.exists():
        for child in video.predict_dir.iterdir():
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()
    else:
        video.predict_dir.mkdir(parents=True, exist_ok=True)

    if video.com_predict_dir.exists():
        shutil.rmtree(video.com_predict_dir)
    video.com_predict_dir.mkdir(parents=True, exist_ok=True)

    # Compose IO config with a single experiment entry
    com_predict_weights = io_defaults.get("com_predict_weights")
    if com_predict_weights is None:
        raise RuntimeError(f"Missing com_predict_weights in {video.dataset.io_path}")

    io_payload = {
        "com_train_dir": str(video.dataset.com_train_dir),
        "com_predict_dir": str(video.com_predict_dir),
        "com_predict_weights": com_predict_weights,
        "com_exp": [
            {
                        "viddir": str(video.viddir),
                "com_file": str(video.com_file),
            }
        ],
        "dannce_train_dir": str(video.dataset.root / "DANNCE/train200"),
        "dannce_predict_dir": str(video.predict_dir),
        "exp": [
            {
                        "com_file": str(video.com_file),
                "viddir": str(video.viddir),
            }
        ],
        "com_fromlabels": True,
    }
    dump_yaml(video.io_config_path, io_payload)

    # Compose prediction config using training params as template
    config_payload = {
        "batch_size": min(4, int(train_params.get("batch_size", 4))),
        "camnames": train_params.get("camnames", ["Camera1", "Camera2", "Camera3"]),
        "com_file": str(video.com_file),
        "crop_height": train_params.get("crop_height", [0, 720]),
        "crop_width": train_params.get("crop_width", [0, 1280]),
        "dannce_predict_dir": str(video.predict_dir),
        "dannce_predict_model": str(video.dataset.train_checkpoint),
        "dataset": train_params.get("dataset", "label3d"),
        "expval": bool(train_params.get("expval", True)),
        "immode": train_params.get("immode", "vid"),
        "interp": train_params.get("interp", "nearest"),
        "io_config": str(video.io_config_path),
        "label3d_index": 0,
        "max_num_samples": "max",
        "mirror": bool(train_params.get("mirror", False)),
        "mono": bool(train_params.get("mono", False)),
        "n_channels_in": int(train_params.get("n_channels_in", 3)),
        "n_channels_out": int(train_params.get("n_channels_out", 19)),
        "n_views": len(train_params.get("camnames", [])) or 3,
        "net": train_params.get("net", "dannce"),
        "net_type": train_params.get("net_type", "dannce"),
        "new_n_channels_out": int(train_params.get("new_n_channels_out", train_params.get("n_channels_out", 19))),
        "nvox": int(train_params.get("nvox", 80)),
        "random_seed": int(train_params.get("random_seed", 1024)),
        "sigma": int(train_params.get("sigma", 10)),
        "start_batch": 0,
        "use_npy": bool(train_params.get("use_npy", True)),
        "viddir": str(video.viddir),
        "vmax": int(train_params.get("vmax", 120)),
        "vmin": int(train_params.get("vmin", -120)),
        "apply_2d_distortion": train_params.get("apply_2d_distortion", True),
    }
    dump_yaml(video.config_path, config_payload)


def launch_prediction_job(video: VideoSpec) -> subprocess.Popen[str]:
    command = (
        f"cd {video.dataset.root} && {MICROMAMBA_RUN} {DANNCE_BIN} predict dannce {video.config_path}"
    )
    LOGGER.info("Launching prediction job for %s %s", video.dataset.name, video.vid_name)
    return launch_srun(
        command=command,
        job_name=f"{video.dataset.name}_{video.vid_name}_pred",
        partition=GPU_PARTITION,
        cpus=8,
        time_limit="1-00:00:00",
        gres="gpu:1",
    )


def launch_visualization_job(
    video: VideoSpec, n_frames: int, camnames: Sequence[str]
) -> subprocess.Popen[str]:
    cameras_arg = ",".join([name.replace("Camera", "") for name in camnames])
    command = (
        f"cd {video.dataset.root} && {MICROMAMBA_RUN} {PYTHON_BIN} -m dannce.engine.utils.vis "
        f"--root {video.dataset.root / 'videos' / video.vid_name} "
        f"--pred {video.predict_dir} "
        f"--datafile save_data_AVG.mat "
        f"--skeleton {video.dataset.skeleton} "
        f"--n_animals {video.dataset.n_animals} "
        f"--cameras {cameras_arg} "
        f"--n_frames {n_frames} "
        f"--start_frame 0 "
        f"--fps 30"
    )
    LOGGER.info("Launching visualization job for %s %s", video.dataset.name, video.vid_name)
    return launch_srun(
        command=command,
        job_name=f"{video.dataset.name}_{video.vid_name}_vis",
        partition=CPU_PARTITION,
        cpus=12,
        time_limit="1-00:00:00",
    )


def copy_artifacts(video: VideoSpec) -> dict:
    save_data_src = video.predict_dir / "save_data_AVG.mat"
    com3d_src = video.predict_dir / "com3d_used.mat"
    stats_src = video.predict_dir / "stats_dannce_predict.log"

    for path in (save_data_src, com3d_src, stats_src):
        if not path.exists():
            raise FileNotFoundError(f"Expected prediction artifact missing: {path}")

    shutil.copy2(save_data_src, video.save_data_dest)
    shutil.copy2(com3d_src, video.com3d_dest)
    shutil.copy2(stats_src, video.stats_dest)

    vis_dir = video.predict_dir / "vis"
    overlay_src = next(iter(sorted(vis_dir.glob("*.mp4"))), None)
    overlay_dest = None
    if overlay_src is not None:
        overlay_dest = video.overlay_dest
        shutil.copy2(overlay_src, overlay_dest)

    manifest = {
        "save_data": str(video.save_data_dest),
        "com3d": str(video.com3d_dest),
        "stats": str(video.stats_dest),
        "overlay": str(overlay_dest) if overlay_dest else None,
    }
    with (video.workspace / "artifact_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    LOGGER.info('Collected artifacts for %s %s', video.dataset.name, video.vid_name)
    return manifest


def process_videos(
    videos: List[VideoSpec],
    train_params: dict,
    io_defaults: dict,
) -> dict:
    summary = {}

    # Prepare workspaces
    for video in videos:
        LOGGER.info("=== Prepare %s %s ===", video.dataset.name, video.vid_name)
        prepare_video_workspace(video, train_params, io_defaults)

    # Launch predictions concurrently
    pred_jobs: List[Tuple[str, subprocess.Popen[str]]] = []
    for video in videos:
        proc = launch_prediction_job(video)
        pred_jobs.append((f"pred_{video.dataset.name}_{video.vid_name}", proc))
    wait_for_jobs(pred_jobs, "prediction")

    # Launch visualizations (after predictions finished)
    camnames = train_params.get("camnames", ["Camera1", "Camera2", "Camera3"])
    vis_jobs: List[Tuple[str, subprocess.Popen[str]]] = []
    frame_counts = {}
    for video in videos:
        save_data_path = video.predict_dir / "save_data_AVG.mat"
        mat = sio.loadmat(save_data_path)
        n_frames = int(mat["pred"].shape[0])
        frame_counts[video.vid_name] = n_frames
        proc = launch_visualization_job(video, n_frames, camnames)
        vis_jobs.append((f"vis_{video.dataset.name}_{video.vid_name}", proc))
    wait_for_jobs(vis_jobs, "visualization")

    # Collect artifacts
    for video in videos:
        manifest = copy_artifacts(video)
        manifest["n_frames"] = frame_counts[video.vid_name]
        summary[video.vid_name] = manifest

    return summary


def gather_videos(dataset: DatasetSpec, selected: Iterable[str] | None) -> List[VideoSpec]:
    available = sorted(p.name for p in (dataset.root / "videos").glob("vid*"))
    if selected is None:
        chosen = available
    else:
        chosen = [vid for vid in available if vid in selected]
    return [VideoSpec(dataset=dataset, vid_name=vid) for vid in chosen]


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run DANNCE train200 predictions for wt/cpfull datasets")
    parser.add_argument(
        "--datasets",
        default="wt,cpfull",
        help="Comma-separated dataset keys to run (wt, cpfull).",
    )
    parser.add_argument(
        "--videos",
        default="all",
        help="Comma-separated video labels (e.g., vid2,vid3) or 'all'.",
    )
    args = parser.parse_args(argv)

    dataset_map = {
        "wt": DatasetSpec(name="wt", root=Path("/work/rl349/dannce/wt_mouse19")),
        "cpfull": DatasetSpec(name="cpfull", root=Path("/work/rl349/dannce/cpfull_mouse19")),
    }

    requested_datasets = [key.strip() for key in args.datasets.split(",") if key.strip()]
    for key in requested_datasets:
        if key not in dataset_map:
            raise ValueError(f"Unknown dataset key: {key}")

    selected_videos = None if args.videos.strip().lower() == "all" else [v.strip() for v in args.videos.split(",") if v.strip()]

    LOGGER.info('Datasets requested: %s', ','.join(requested_datasets))
    LOGGER.info('Video selection: %s', 'all' if selected_videos is None else ','.join(selected_videos))

    summary = {}
    for key in requested_datasets:
        dataset = dataset_map[key]
        LOGGER.info('===== Dataset: %s =====', dataset.name)
        train_params = load_yaml(dataset.train_params_path)
        io_defaults = load_yaml(dataset.io_path)
        videos = gather_videos(dataset, selected_videos)
        summary[dataset.name] = process_videos(videos, train_params, io_defaults)

    summary_path = Path("/work/rl349/dannce/predict_train200_summary.json")
    with summary_path.open('w', encoding='utf-8') as handle:
        json.dump(summary, handle, indent=2)
    LOGGER.info('All done. Summary written to %s', summary_path)


if __name__ == "__main__":
    main()
