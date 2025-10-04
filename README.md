# Behavior Analysis

This repository aggregates MATLAB and Python tooling that supports the spontaneous pain study pipeline from camera calibration through behavior analysis. The sources were exported from the Duke DCC environment to make iteration and collaboration easier.

## Integrated Pipeline

1. **Calibration** – `scripts/run_3cam_cam3_refresh_with_parallel_dlc.py` orchestrates three-camera calibration via the helpers in `scripts/custom_calib/` (pairwise or shared-frame Anipose strategies).
2. **DeepLabCut** – The same driver submits parallel DLC jobs through `scripts/submit_parallel_dlc_jobs.py`, applying camera-specific crops defined in `scripts/dlc_cropping.py`.
3. **Setup DANNCE** – `scripts/setsdanncemouse14_wt_cp.py` and `scripts/setsdanncemouse19_wt.py` convert multi-session DLC outputs into s-DANNCE-ready layouts for 14- and 19-joint skeletons.
4. **Train & Predict DANNCE** – `scripts/run_train200_predictions.py` automates train200 checkpoint inference on WT and CP datasets, while `scripts/predict_and_package_train200.py` reprojections and packages results for downstream consumers.
5. **Behavior Analysis** – Output from the above stages feeds the MATLAB and Python utilities in `MNN/` and `SocialMapper-main/` for embedding, clustering, and visualization.

## Directory Overview

- `scripts/`
  - `run_3cam_cam3_refresh_with_parallel_dlc.py`: end-to-end calibration + DLC + Anipose pipeline driver.
  - `run_train200_predictions.py`: SLURM-aware automation for DANNCE predictions and overlays.
  - `predict_and_package_train200.py`: packages prediction artifacts, generates reprojections, and writes summaries.
  - `setsdanncemouse14_wt_cp.py`, `setsdanncemouse19_wt.py`: convert DLC results into standardized s-DANNCE datasets.
  - `cleanup_dlc_h5_files.py`, `submit_parallel_dlc_jobs.py`, `dlc_cropping.py`: shared helpers for file hygiene and job management.
  - `custom_calib/`: calibration modules (`pairwise_anipose_calibration.py`, `three_camera_anipose_calibration.py`, etc.) plus configuration templates.
- `MNN/mouse19_pipeline/`
  - `embed_mouse19_aligned.m`: MATLAB entry point for aligning embeddings in the mouse19 dataset.
  - `generate_videos_from_mouse19_aligned.py`: Generates behavioral videos from aligned embeddings.
  - `render_overlay_imageio.py`: Renders overlay videos for qualitative inspection.
  - `reproject_mouse19_predictions.py`: Reprojects predictions into the video coordinate frame.
- `SocialMapper-main/`
  - `Animator-master/`: Modular MATLAB animators and dependencies for visualizing time series, keypoints, and volumetric data.
  - `MotionMapperUtilities/`: MATLAB utilities supporting MotionMapper-style embeddings, including wavelet transforms, t-SNE routines, and signal processing helpers.
  - `Scripts/`: Higher-level scripts for social feature extraction and re-embedding workflows.

## Usage

1. Install MATLAB with the Signal Processing, Statistics and Machine Learning, and Image Processing toolboxes for the MATLAB components.
2. Set up Python 3.8+ environments for DLC/Anipose and DANNCE with packages such as `numpy`, `scipy`, `pandas`, `opencv-python`, `yaml`, and `tomli`.
3. Update hard-coded paths inside the scripts to match your compute environment (SLURM partitions, micromamba locations, dataset roots).
4. Run the pipeline scripts in the order listed above, confirming outputs at each stage before progressing to behavior analysis.

## Provenance

Files originate from:
- `/work/rl349/scripts` (pipeline orchestration and calibration helpers).
- `/work/rl349/spontaneous pain/MNN` (mouse19 behavioral pipeline).
- `/work/rl349/spontaneous pain/SocialMapper-main` (downstream embedding utilities).

## Contributing

Open pull requests for modifications. When adding new utilities, document required data layout and verify MATLAB/Python scripts locally before submission.
