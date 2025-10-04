# Behavior Analysis

This repository aggregates MATLAB and Python tools used for behavioral analysis workflows in the spontaneous pain project. Code is organized into two primary modules sourced from the original workspace:

- `MNN/`: MATLAB scripts and Python utilities for the mouse19 pipeline, including projection, overlay rendering, and video generation helpers.
- `SocialMapper-main/`: Downstream analysis and visualization utilities derived from the SocialMapper project, such as Animator components, MotionMapper utilities, and related scripts for feature extraction and embedding.

## Directory Overview

- `MNN/mouse19_pipeline/`
  - `embed_mouse19_aligned.m`: MATLAB entry point for aligning embeddings in the mouse19 dataset.
  - `generate_videos_from_mouse19_aligned.py`: Generates behavioral videos from aligned embeddings.
  - `render_overlay_imageio.py`: Renders overlay videos for qualitative inspection.
  - `reproject_mouse19_predictions.py`: Reprojects predictions into the video coordinate frame.
- `SocialMapper-main/Animator-master/`: Modular MATLAB animators and dependencies for visualizing time series, keypoints, and volumetric data.
- `SocialMapper-main/MotionMapperUtilities/`: MATLAB utilities supporting MotionMapper-style embeddings, including wavelet transforms, t-SNE routines, and signal processing helpers.
- `SocialMapper-main/Scripts/`: Higher-level scripts for social feature extraction and re-embedding workflows.

## Usage

These sources assume the surrounding project structure (pose estimation outputs, embeddings, and metadata) that exists in the spontaneous pain workspace. When pulling them into a new environment:

1. Install MATLAB with required toolboxes (Signal Processing, Statistics and Machine Learning, Image Processing) for the MATLAB components.
2. Ensure Python 3.8+ with packages such as `numpy`, `scipy`, `matplotlib`, and `imageio` for the Python utilities.
3. Update file paths within scripts to match your dataset layout before running.

## Provenance

Files were exported from `/work/rl349/spontaneous pain/MNN` and `/work/rl349/spontaneous pain/SocialMapper-main` on the Duke DCC environment to facilitate version control and collaboration.

## Contributing

Please open pull requests for modifications. Verify MATLAB scripts locally and include notes about the expected dataset structure when adding new utilities.
