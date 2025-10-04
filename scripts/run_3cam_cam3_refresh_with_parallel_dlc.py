#!/usr/bin/env python3
"""
Full 3-camera pipeline with parallel DLC processing.

This script runs every phase of the DeepLabCut + Anipose workflow for a single
session. Unlike the earlier "cam3 refresh" variant, no stages are optional:
calibration, parallel DLC inference, cleanup, 2D/3D filtering, triangulation,
and projection are always executed in order.

The script assumes the session directory contains a ``calibration`` folder and
lives inside an experiment root that also houses the Anipose configuration
files (for example ``/work/.../DeepLabCut/CP(full)`` containing ``session1``).
All Anipose CLI commands are executed with ``cwd`` set to that experiment root
so the default configuration is discovered without appending extra arguments.
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Optional

# Ensure local modules are available when the script is invoked from elsewhere
script_dir = Path(__file__).parent.resolve()
if str(script_dir) not in sys.path:
    sys.path.insert(0, str(script_dir))

custom_calib_dir = script_dir / "custom_calib"
if custom_calib_dir.exists() and str(custom_calib_dir) not in sys.path:
    sys.path.insert(0, str(custom_calib_dir))


def run_calibration_phase(
    session_dir: Path,
    temp_dir: Optional[Path] = None,
    mode: str = "pairwise",
) -> bool:
    """Execute the selected calibration workflow."""

    mode = (mode or "pairwise").lower()
    headers = {
        "pairwise": "PHASE 1: PAIRWISE ANIPOSE CALIBRATION",
        "shared": "PHASE 1: SHARED-FRAME ANIPOSE CALIBRATION",
    }

    if mode not in headers:
        print(f"[ERROR] Unknown calibration mode: {mode}")
        return False

    print(f"\n== {headers[mode]} ==")

    try:
        if mode == "shared":
            from three_camera_anipose_calibration import ThreeCameraAniposeCalibrator

            calibrator = (
                ThreeCameraAniposeCalibrator(temp_base_dir=temp_dir)
                if temp_dir
                else ThreeCameraAniposeCalibrator()
            )
            success = calibrator.run_joint_calibration(session_dir, cleanup=False)
        else:
            from pairwise_anipose_calibration import PairwiseAniposeCalibrator

            calibrator = (
                PairwiseAniposeCalibrator(temp_base_dir=temp_dir)
                if temp_dir
                else PairwiseAniposeCalibrator()
            )
            success = calibrator.run_pairwise_calibration(session_dir, cleanup=False)

        if success and hasattr(calibrator, "temp_base_dir"):
            print(f"[INFO] Calibration temporary data preserved at {calibrator.temp_base_dir}")

        if success:
            print("[OK] Calibration phase completed.")
            return True

        print("[ERROR] Calibration phase failed.")
        return False

    except ImportError as exc:
        print(f"[ERROR] Could not import calibration module: {exc}")
        return False
    except Exception as exc:  # pragma: no cover - defensive catch for CLI use
        print(f"[ERROR] Calibration failed: {exc}")
        return False


def submit_parallel_dlc_jobs(session_dir: Path, videos_per_job: int = 1) -> bool:
    """Submit parallel DLC jobs for all available videos."""

    print("\n== PHASE 2: PARALLEL DLC JOB SUBMISSION ==")

    try:
        from submit_parallel_dlc_jobs import ParallelDLCJobManager

        manager = ParallelDLCJobManager(session_dir)
        job_ids = manager.submit_parallel_dlc_jobs(videos_per_job)

        if job_ids:
            print(f"[OK] Submitted {len(job_ids)} DLC job(s): {', '.join(job_ids)}")
            return True

        print("[ERROR] No DLC jobs were submitted.")
        return False

    except Exception as exc:  # pragma: no cover - defensive catch for CLI use
        print(f"[ERROR] DLC job submission failed: {exc}")
        return False


def monitor_dlc_jobs(session_dir: Path, timeout_minutes: int = 1440) -> bool:
    """Monitor DLC jobs until they succeed or timeout."""

    print("\n== PHASE 3: MONITORING DLC JOBS ==")

    from submit_parallel_dlc_jobs import ParallelDLCJobManager

    manager = ParallelDLCJobManager(session_dir)
    start_time = time.time()
    check_interval = 60  # seconds

    while True:
        status_dict = manager.check_job_status()

        if not status_dict:
            tracked_ids = getattr(manager, "job_ids", [])
            if tracked_ids:
                status_dict = {job_id: "PENDING" for job_id in tracked_ids}
                print("[WARN] sacct returned no status; treating jobs as pending.")
            else:
                print("[ERROR] No jobs found to monitor.")
                return False

        completed = sum(1 for state in status_dict.values() if state in {"COMPLETED", "COMPLETING"})
        failed = sum(1 for state in status_dict.values() if state in {"FAILED", "CANCELLED", "TIMEOUT"})
        running = sum(1 for state in status_dict.values() if state in {"RUNNING", "PENDING"})

        print(f"[INFO] Job status -> completed: {completed}, failed: {failed}, active: {running}")

        if running == 0:
            if failed == 0:
                print("[OK] All DLC jobs completed successfully.")
                return True

            print(f"[ERROR] {failed} job(s) failed. Check SLURM logs under the session directory.")
            return False

        elapsed_minutes = (time.time() - start_time) / 60
        if elapsed_minutes > timeout_minutes:
            print(f"[ERROR] Timeout reached after {timeout_minutes} minutes. Some jobs are still active.")
            return False

        print(f"[INFO] Waiting {check_interval} seconds before the next check (elapsed {elapsed_minutes:.1f} minutes).")
        time.sleep(check_interval)


def cleanup_dlc_h5_files(session_dir: Path) -> bool:
    """Normalize DLC output filenames in pose-2d."""

    print("\n== PHASE 4: DLC H5 FILE CLEANUP ==")

    try:
        from cleanup_dlc_h5_files import DLCFileCleanup

        cleaner = DLCFileCleanup(session_dir, dry_run=False)
        if cleaner.run_cleanup():
            print("[OK] pose-2d cleanup completed.")
            return True

        print("[ERROR] pose-2d cleanup failed.")
        return False

    except ImportError as exc:
        print(f"[ERROR] Could not import cleanup module: {exc}")
        return False
    except Exception as exc:  # pragma: no cover - defensive catch for CLI use
        print(f"[ERROR] pose-2d cleanup failed: {exc}")
        return False


def _ensure_filters_disabled(config_path: Path) -> None:
    """Force [filter] and [filter3d] enabled flags to false."""

    if not config_path.exists():
        return

    original = config_path.read_text()
    lines = original.splitlines()
    in_filter = False
    in_filter3d = False
    modified = False

    for index, line in enumerate(lines):
        stripped = line.strip()
        lower = stripped.lower()

        if lower.startswith("[") and lower.endswith("]"):
            in_filter = lower == "[filter]"
            in_filter3d = lower == "[filter3d]"
            continue

        if (in_filter or in_filter3d) and lower.startswith("enabled"):
            indent = line[: len(line) - len(line.lstrip())]
            if "false" not in lower:
                lines[index] = f"{indent}enabled = false"
                modified = True

    if modified:
        newline = "\n" if original.endswith("\n") else ""
        config_path.write_text("\n".join(lines) + newline)
        print(f"[INFO] Disabled filters in {config_path}.")


def run_2d_filtering_phase(session_dir: Path, experiment_root: Path) -> bool:
    """Invoke anipose 2D filtering from the experiment root."""

    print("\n== PHASE 5: ANIPOSE 2D FILTERING ==")

    try:
        import subprocess

        config_path = experiment_root / "config.toml"
        if not config_path.exists():
            print(f"[WARN] config.toml not found in {experiment_root}; writing a minimal default.")
            basic_config = """
nesting = 1
video_extension = 'mp4'

[filter]
enabled = false
type = "viterbi"
score_threshold = 0.2
medfilt = 13
offset_threshold = 25
spline = true

[triangulation]
triangulate = true
cam_regex = '-cam([0-9]+)'
cam_align = "1"
ransac = false
optim = true

[filter3d]
enabled = false
medfilt = 7
offset_threshold = 40
n_back_track = 5
score_threshold = 0.1
spline = true
"""
            config_path.write_text(basic_config.strip())
        _ensure_filters_disabled(config_path)

        print(f"[INFO] Running 'anipose filter' in {experiment_root}.")
        subprocess.run(["anipose", "filter"], cwd=experiment_root, check=True)
        print("[OK] 2D filtering completed.")
        return True

    except subprocess.CalledProcessError as exc:
        print(f"[ERROR] anipose filter failed: {exc}")
        return False
    except Exception as exc:  # pragma: no cover - defensive catch for CLI use
        print(f"[ERROR] 2D filtering failed: {exc}")
        return False


def cleanup_filtered_h5_files(session_dir: Path) -> bool:
    """Normalize filenames inside pose-2d-filtered."""

    print("\n== PHASE 5.5: FILTERED H5 FILE CLEANUP ==")

    try:
        from cleanup_dlc_h5_files import DLCFileCleanup

        cleaner = DLCFileCleanup(session_dir, dry_run=False)
        cleaner.pose_2d_dir = session_dir / "pose-2d-filtered"
        cleaner.archive_dir = session_dir / "pose-2d-filtered-archive"

        if not cleaner.pose_2d_dir.exists():
            print("[WARN] pose-2d-filtered directory not found; nothing to clean.")
            return True

        if cleaner.run_cleanup():
            print("[OK] pose-2d-filtered cleanup completed.")
            return True

        print("[ERROR] pose-2d-filtered cleanup failed.")
        return False

    except ImportError as exc:
        print(f"[ERROR] Could not import cleanup module: {exc}")
        return False
    except Exception as exc:  # pragma: no cover - defensive catch for CLI use
        print(f"[ERROR] pose-2d-filtered cleanup failed: {exc}")
        return False


def run_post_filter_calibration(experiment_root: Path) -> bool:
    """Run an additional anipose calibration after 2D filtering."""

    print("\n== PHASE 5.7: POST-FILTER CALIBRATION ==")

    try:
        import subprocess

        print(f"[INFO] Running 'anipose calibrate' in {experiment_root}.")
        subprocess.run(["anipose", "calibrate"], cwd=experiment_root, check=True)
        print("[OK] Post-filter calibration completed.")
        return True

    except subprocess.CalledProcessError as exc:
        print(f"[ERROR] anipose calibrate failed: {exc}")
        return False
    except Exception as exc:  # pragma: no cover - defensive catch for CLI use
        print(f"[ERROR] Post-filter calibration failed: {exc}")
        return False


def run_triangulation_phase(session_dir: Path, experiment_root: Path) -> bool:
    """Perform 3D triangulation and smoothing."""

    print("\n== PHASE 6: 3D TRIANGULATION AND FILTERING ==")

    try:
        import subprocess

        print(f"[INFO] Running 'anipose triangulate' in {experiment_root}.")
        subprocess.run(["anipose", "triangulate"], cwd=experiment_root, check=True)

        print(f"[INFO] Running 'anipose filter-3d' in {experiment_root}.")
        subprocess.run(["anipose", "filter-3d"], cwd=experiment_root, check=True)

        print("[OK] Triangulation and 3D filtering completed.")
        return True

    except subprocess.CalledProcessError as exc:
        print(f"[ERROR] Triangulation phase failed: {exc}")
        return False
    except Exception as exc:  # pragma: no cover - defensive catch for CLI use
        print(f"[ERROR] Triangulation failed: {exc}")
        return False


def run_projection_phase(session_dir: Path, experiment_root: Path) -> bool:
    """Project 3D results back to 2D views and label."""

    print("\n== PHASE 7: 2D PROJECTION AND LABELING ==")

    try:
        import subprocess

        print(f"[INFO] Running 'anipose project-2d' in {experiment_root}.")
        subprocess.run(["anipose", "project-2d"], cwd=experiment_root, check=True)

        print(f"[INFO] Running 'anipose label-2d-proj' in {experiment_root}.")
        subprocess.run(["anipose", "label-2d-proj"], cwd=experiment_root, check=True)

        print("[OK] 2D projection and labeling completed.")
        return True

    except subprocess.CalledProcessError as exc:
        print(f"[ERROR] Projection phase failed: {exc}")
        return False
    except Exception as exc:  # pragma: no cover - defensive catch for CLI use
        print(f"[ERROR] Projection failed: {exc}")
        return False


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the full 3-camera DLC + Anipose pipeline with parallel DLC jobs."
    )
    parser.add_argument(
        "session_dir",
        help="Path to the session directory (for example 'session1')."
    )
    parser.add_argument(
        "--videos-per-job",
        type=int,
        default=1,
        help="Number of videos to bundle per DLC job (default: 1)."
    )
    parser.add_argument(
        "--calibration-mode",
        choices=["pairwise", "shared"],
        default="pairwise",
        help="Calibration strategy to run before DLC (default: pairwise)."
    )
    parser.add_argument(
        "--dlc-timeout",
        type=int,
        default=1440,
        help="Timeout for DLC jobs in minutes before giving up (default: 1440)."
    )
    parser.add_argument(
        "--temp-dir",
        type=str,
        default=None,
        help="Optional directory for temporary calibration projects."
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable additional logging within helper modules."
    )

    args = parser.parse_args()

    session_dir = Path(args.session_dir).expanduser().resolve()
    if not session_dir.exists():
        print(f"[ERROR] Session directory not found: {session_dir}")
        return 1

    calibration_dir = session_dir / "calibration"
    if not calibration_dir.exists():
        print(f"[WARN] Expected calibration directory not found at {calibration_dir}.")

    experiment_root = session_dir.parent
    expected_root = calibration_dir.parent.parent if calibration_dir.exists() else experiment_root
    if expected_root != experiment_root:
        print(
            "[WARN] Session layout suggests experiment root %s, but using %s." %
            (expected_root, experiment_root)
        )

    print("\n=== 3-CAMERA FULL PIPELINE (PARALLEL DLC) ===")
    print(f"Session directory : {session_dir}")
    print(f"Experiment root    : {experiment_root}")
    print(f"Videos per job     : {args.videos_per_job}")
    print(f"Calibration mode   : {args.calibration_mode}")
    print(f"DLC timeout (min)  : {args.dlc_timeout}")
    if args.temp_dir:
        print(f"Calibration temp   : {Path(args.temp_dir).expanduser().resolve()}")
    print("============================================")

    temp_dir = Path(args.temp_dir).expanduser().resolve() if args.temp_dir else None
    if temp_dir:
        temp_dir.mkdir(parents=True, exist_ok=True)

    if not run_calibration_phase(session_dir, temp_dir, mode=args.calibration_mode):
        return 1

    if not submit_parallel_dlc_jobs(session_dir, args.videos_per_job):
        return 1

    if not monitor_dlc_jobs(session_dir, args.dlc_timeout):
        return 1

    if not cleanup_dlc_h5_files(session_dir):
        return 1

    if not run_2d_filtering_phase(session_dir, experiment_root):
        return 1

    if not cleanup_filtered_h5_files(session_dir):
        return 1

    if not run_post_filter_calibration(experiment_root):
        return 1

    if not run_triangulation_phase(session_dir, experiment_root):
        return 1

    if not run_projection_phase(session_dir, experiment_root):
        return 1

    print("\n=== PIPELINE COMPLETED SUCCESSFULLY ===")
    print(f"Results available in: {session_dir}")
    print(f"Pose 2D outputs   : {session_dir / 'pose-2d'}")
    print(f"Pose 3D outputs   : {session_dir / 'pose-3d'}")
    print("All stages executed without skipping.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
