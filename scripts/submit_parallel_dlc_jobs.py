#!/usr/bin/env python3
"""
Parallel DLC Job Submission System

This script creates and submits individual SLURM jobs for DeepLabCut processing,
allowing multiple videos to be processed simultaneously on different nodes.

Usage:
    python submit_parallel_dlc_jobs.py --session-dir /path/to/session
    python submit_parallel_dlc_jobs.py --session-dir /path/to/session --videos-per-job 2
    python submit_parallel_dlc_jobs.py --session-dir /path/to/session --check-status

Author: Parallel DLC Jobs
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path
from typing import List, Dict, Optional
import logging
from datetime import datetime
import json

from dlc_cropping import CAMERA_CROPPING

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ParallelDLCJobManager:
    """Manager for parallel DLC job submission"""
    
    def __init__(self, session_dir: Path, job_base_name: str = "dlc_parallel"):
        self.session_dir = session_dir
        self.job_base_name = job_base_name
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.job_dir = session_dir / f"dlc_jobs_{self.timestamp}"
        self.job_ids = []
        
    def create_dlc_job_script(self, job_name: str, video_files: List[Path], 
                             job_id: int, pose_2d_dir: Path) -> Path:
        """
        Create SLURM job script for DLC processing
        
        Args:
            job_name: Name for this job
            video_files: List of video files to process in this job
            job_id: Unique job ID
            pose_2d_dir: Output directory for pose-2d results
            
        Returns:
            Path to created job script
        """
        script_path = self.job_dir / f"{job_name}_{job_id}.sh"
        
        # Convert video paths to absolute paths for script
        video_paths = [str(v.resolve()) for v in video_files]  # Use .resolve() for absolute paths
        video_list = '", "'.join(video_paths)
        
        cropping_json = json.dumps(CAMERA_CROPPING)

        script_content = f'''#!/bin/bash
#SBATCH --job-name={job_name}_{job_id}
#SBATCH --partition=scavenger-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=80G
#SBATCH --gres=gpu:5000_ada:1
#SBATCH --time=7-00:00:00
#SBATCH --output={self.job_dir}/{job_name}_{job_id}_%j.out
#SBATCH --error={self.job_dir}/{job_name}_{job_id}_%j.err

# ============ Software setup ============
eval "$($HOME/.local/bin/micromamba shell hook -s bash)"
micromamba activate dlc-anipose

# Print diagnostics
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "CUDA devices: $CUDA_VISIBLE_DEVICES"
echo "Processing {len(video_files)} videos..."
nvidia-smi

# Set Python path for custom calibration
export PYTHONPATH="/work/rl349/scripts/custom_calib:$PYTHONPATH"

# ============ DLC Processing ============
cd "{self.session_dir}"

python3 << 'EOF'
import sys
import os
import json
from pathlib import Path
import deeplabcut
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Video files to process (already absolute paths)
video_files = ["{video_list}"]
pose_2d_dir = "{pose_2d_dir}"

CAMERA_CROPPING = json.loads('{cropping_json}')


def parse_camera_from_name(path_str: str):
    stem = Path(path_str).stem.lower()
    normalized = stem.replace('_', '-')
    for part in normalized.split('-'):
        part = part.strip()
        if part.startswith('cam') and part[3:].isdigit():
            return part
    return None


def get_crop_for_video(path_str: str):
    camera = parse_camera_from_name(path_str)
    if camera and camera in CAMERA_CROPPING:
        return camera, CAMERA_CROPPING[camera]
    return camera, None


COMMON_KWARGS = dict(
    superanimal_name="superanimal_quadruped",
    model_name="hrnet_w32",
    detector_name="fasterrcnn_resnet50_fpn_v2",
    videotype='.mp4',
    video_adapt=True,
    scale_list=[],
    max_individuals=1,
    dest_folder=pose_2d_dir,
    batch_size=8,
    detector_batch_size=8,
    video_adapt_batch_size=4,
)

logger.info(f"Processing {{len(video_files)}} videos in job {job_id}")
for i, video_file in enumerate(video_files):
    logger.info(f"  {{i+1}}/{{len(video_files)}}: {{Path(video_file).name}}")

try:
    # Run DLC inference per video so we can customize cam-specific settings
    for video_file in video_files:
        kwargs = dict(COMMON_KWARGS)
        video_name = Path(video_file).name

        camera, crop = get_crop_for_video(video_file)
        if crop:
            kwargs["cropping"] = crop
            logger.info("  Applying %s cropping %s to %s", camera, crop, video_name)
        else:
            if camera:
                logger.info("  Camera %s does not require cropping for %s", camera, video_name)
            else:
                logger.info("  No cropping applied to %s", video_name)

        deeplabcut.video_inference_superanimal([video_file], **kwargs)
    
    logger.info(f"✅ Job {job_id} completed successfully")
    
    # Create completion marker (use absolute path)
    import os
    job_dir = os.path.abspath("{self.job_dir}")
    os.makedirs(job_dir, exist_ok=True)
    
    with open(os.path.join(job_dir, f"job_{job_id}_completed.txt"), "w") as f:
        f.write(f"Job {job_id} completed at {{datetime.now()}}\\n")
        f.write(f"Videos processed: {{len(video_files)}}\\n")
        for video in video_files:
            f.write(f"  - {{Path(video).name}}\\n")
            
except Exception as e:
    logger.error(f"❌ Job {job_id} failed: {{e}}")
    
    # Create error marker (use absolute path)
    import os
    job_dir = os.path.abspath("{self.job_dir}")
    os.makedirs(job_dir, exist_ok=True)
    
    with open(os.path.join(job_dir, f"job_{job_id}_failed.txt"), "w") as f:
        f.write(f"Job {job_id} failed at {{datetime.now()}}\\n")
        f.write(f"Error: {{str(e)}}\\n")
    
    sys.exit(1)

EOF

echo "Job {job_id} script completed"
'''
        
        # Write script file
        self.job_dir.mkdir(parents=True, exist_ok=True)
        with open(script_path, 'w') as f:
            f.write(script_content)
        
        # Make executable
        os.chmod(script_path, 0o755)
        
        logger.info(f"Created job script: {script_path}")
        return script_path
    
    def group_videos_for_jobs(self, video_files: List[Path], videos_per_job: int = 1) -> List[List[Path]]:
        """
        Group videos into batches for parallel processing
        
        Args:
            video_files: List of all video files
            videos_per_job: Number of videos per job
            
        Returns:
            List of video groups
        """
        groups = []
        for i in range(0, len(video_files), videos_per_job):
            group = video_files[i:i + videos_per_job]
            groups.append(group)
        
        logger.info(f"Grouped {len(video_files)} videos into {len(groups)} jobs ({videos_per_job} videos/job)")
        return groups
    
    def submit_parallel_dlc_jobs(self, videos_per_job: int = 1) -> List[str]:
        """
        Submit parallel DLC jobs for all videos in session
        
        Args:
            videos_per_job: Number of videos to process per job
            
        Returns:
            List of submitted job IDs
        """
        logger.info("🚀 SUBMITTING PARALLEL DLC JOBS")
        logger.info("="*50)
        
        # Find video files
        videos_dir = self.session_dir / "videos-raw"
        if not videos_dir.exists():
            logger.error(f"Videos directory not found: {videos_dir}")
            return []
        
        video_files = list(videos_dir.glob("*.mp4"))
        if not video_files:
            logger.error("No MP4 files found in videos-raw directory")
            return []
        
        logger.info(f"Found {len(video_files)} videos to process")
        
        # Create pose-2d output directory
        pose_2d_dir = self.session_dir / "pose-2d"
        pose_2d_dir.mkdir(exist_ok=True)
        
        # Group videos for parallel processing
        video_groups = self.group_videos_for_jobs(video_files, videos_per_job)
        
        # Create and submit job scripts
        submitted_jobs = []
        
        for job_id, video_group in enumerate(video_groups, 1):
            job_name = f"{self.job_base_name}_{self.timestamp}"
            
            logger.info(f"Creating job {job_id}/{len(video_groups)}: {len(video_group)} videos")
            for video in video_group:
                logger.info(f"  - {video.name}")
            
            # Create job script (pass absolute pose_2d_dir)
            script_path = self.create_dlc_job_script(
                job_name, video_group, job_id, pose_2d_dir.resolve()
            )
            
            # Submit job
            try:
                result = subprocess.run([
                    'sbatch', str(script_path)
                ], capture_output=True, text=True, check=True)
                
                # Extract job ID from sbatch output
                job_output = result.stdout.strip()
                if "Submitted batch job" in job_output:
                    slurm_job_id = job_output.split()[-1]
                    submitted_jobs.append(slurm_job_id)
                    self.job_ids.append(slurm_job_id)
                    logger.info(f"  ✅ Submitted job {job_id}: SLURM ID {slurm_job_id}")
                else:
                    logger.error(f"  ❌ Unexpected sbatch output: {job_output}")
                
            except subprocess.CalledProcessError as e:
                logger.error(f"  ❌ Failed to submit job {job_id}: {e}")
                logger.error(f"     Stderr: {e.stderr}")
        
        # Save job tracking info
        self.save_job_tracking_info(submitted_jobs, video_groups)
        
        logger.info(f"🎉 Submitted {len(submitted_jobs)} parallel DLC jobs")
        logger.info(f"Job directory: {self.job_dir}")
        
        return submitted_jobs
    
    def save_job_tracking_info(self, job_ids: List[str], video_groups: List[List[Path]]):
        """Save job tracking information for monitoring"""
        tracking_info = {
            'timestamp': self.timestamp,
            'session_dir': str(self.session_dir),
            'job_dir': str(self.job_dir),
            'submitted_jobs': []
        }
        
        for i, (job_id, video_group) in enumerate(zip(job_ids, video_groups), 1):
            tracking_info['submitted_jobs'].append({
                'job_number': i,
                'slurm_job_id': job_id,
                'videos': [str(v) for v in video_group],
                'status': 'submitted'
            })
        
        tracking_file = self.job_dir / "job_tracking.json"
        with open(tracking_file, 'w') as f:
            json.dump(tracking_info, f, indent=2)
        
        logger.info(f"Job tracking saved: {tracking_file}")
    
    def check_job_status(self) -> Dict[str, str]:
        """
        Check status of submitted jobs
        
        Returns:
            Dict mapping job IDs to status
        """
        logger.info("📊 CHECKING PARALLEL DLC JOB STATUS")
        logger.info("="*50)
        
        if not self.job_ids:
            # Try to load from tracking file
            tracking_files = list(self.session_dir.glob("dlc_jobs_*/job_tracking.json"))
            if tracking_files:
                latest_tracking = max(tracking_files, key=lambda f: f.stat().st_mtime)
                with open(latest_tracking, 'r') as f:
                    tracking_info = json.load(f)
                
                self.job_ids = [job['slurm_job_id'] for job in tracking_info['submitted_jobs']]
                self.job_dir = Path(tracking_info['job_dir'])
                logger.info(f"Loaded {len(self.job_ids)} job IDs from {latest_tracking}")
        
        if not self.job_ids:
            logger.error("No job IDs found to check")
            return {}
        
        job_status = {}
        
        # Check SLURM queue status
        try:
            result = subprocess.run([
                'sacct', '-j', ','.join(self.job_ids), '--format=JobID,State,ExitCode', '--noheader'
            ], capture_output=True, text=True, check=True)
            
            for line in result.stdout.strip().split('\n'):
                if line.strip():
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        job_id = parts[0].split('.')[0]  # Remove step numbers
                        state = parts[1]
                        job_status[job_id] = state
            
        except subprocess.CalledProcessError as e:
            logger.warning(f"Could not get job status from sacct: {e}")
        
        # Check completion markers
        completed_jobs = []
        failed_jobs = []
        
        if self.job_dir.exists():
            completed_markers = list(self.job_dir.glob("job_*_completed.txt"))
            failed_markers = list(self.job_dir.glob("job_*_failed.txt"))
            
            completed_jobs = [f.stem.split('_')[1] for f in completed_markers]
            failed_jobs = [f.stem.split('_')[1] for f in failed_markers]
        
        # Print status summary
        logger.info("Job Status Summary:")
        for job_id in self.job_ids:
            slurm_status = job_status.get(job_id, "UNKNOWN")
            
            if job_id in completed_jobs:
                status = "✅ COMPLETED"
            elif job_id in failed_jobs:
                status = "❌ FAILED"
            else:
                status = f"🔄 {slurm_status}"
            
            logger.info(f"  Job {job_id}: {status}")
        
        # Summary counts
        total_jobs = len(self.job_ids)
        completed_count = len(completed_jobs)
        failed_count = len(failed_jobs)
        running_count = total_jobs - completed_count - failed_count
        
        logger.info(f"\nSummary: {completed_count} completed, {failed_count} failed, {running_count} running/pending")
        
        return job_status
    
    def create_summary_script(self) -> Path:
        """Create a summary script to run after all jobs complete"""
        summary_script = self.job_dir / "summarize_results.py"
        
        script_content = f'''#!/usr/bin/env python3
"""
Summary script for parallel DLC jobs
Runs after all jobs complete to organize results
"""

import os
import sys
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    session_dir = Path("{self.session_dir}")
    pose_2d_dir = session_dir / "pose-2d"
    
    logger.info("📊 SUMMARIZING PARALLEL DLC RESULTS")
    logger.info("="*50)
    
    # Count H5 files
    h5_files = list(pose_2d_dir.glob("*.h5"))
    logger.info(f"Found {{len(h5_files)}} H5 files in pose-2d")
    
    # Count video files that were processed
    videos_dir = session_dir / "videos-raw"
    video_files = list(videos_dir.glob("*.mp4"))
    
    logger.info(f"Expected {{len(video_files)}} H5 files (one per video)")
    
    if len(h5_files) >= len(video_files):
        logger.info("✅ All videos appear to have been processed")
    else:
        missing_count = len(video_files) - len(h5_files)
        logger.warning(f"⚠️  {{missing_count}} videos may not have been processed")
    
    # List any failed jobs
    job_dir = Path("{self.job_dir}")
    failed_markers = list(job_dir.glob("job_*_failed.txt"))
    
    if failed_markers:
        logger.warning(f"Found {{len(failed_markers)}} failed jobs:")
        for marker in failed_markers:
            logger.warning(f"  - {{marker.name}}")
    
    logger.info("📋 Next steps:")
    logger.info("  1. Check pose-2d directory for H5 files")
    logger.info("  2. Run H5 conversion if needed")
    logger.info("  3. Proceed with anipose triangulation")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
'''
        
        with open(summary_script, 'w') as f:
            f.write(script_content)
        
        os.chmod(summary_script, 0o755)
        logger.info(f"Created summary script: {summary_script}")
        return summary_script

def main():
    parser = argparse.ArgumentParser(description="Parallel DLC Job Submission")
    parser.add_argument("--session-dir", required=True, type=str,
                       help="Path to session directory")
    parser.add_argument("--videos-per-job", type=int, default=1,
                       help="Number of videos to process per job (default: 1)")
    parser.add_argument("--job-name", type=str, default="dlc_parallel",
                       help="Base name for SLURM jobs")
    parser.add_argument("--check-status", action="store_true",
                       help="Check status of previously submitted jobs")
    parser.add_argument("--cancel-jobs", action="store_true",
                       help="Cancel all submitted jobs")
    parser.add_argument("--max-jobs", type=int, default=10,
                       help="Maximum number of parallel jobs to submit")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Enable verbose logging")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Parse session directory
    session_dir = Path(args.session_dir)
    if not session_dir.exists():
        logger.error(f"Session directory not found: {session_dir}")
        return 1
    
    # Initialize job manager
    manager = ParallelDLCJobManager(session_dir, args.job_name)
    
    try:
        if args.check_status:
            # Check status of existing jobs
            manager.check_job_status()
            return 0
            
        elif args.cancel_jobs:
            # Cancel submitted jobs
            if manager.job_ids:
                logger.info(f"Canceling {len(manager.job_ids)} jobs...")
                for job_id in manager.job_ids:
                    try:
                        subprocess.run(['scancel', job_id], check=True)
                        logger.info(f"  ✅ Canceled job {job_id}")
                    except subprocess.CalledProcessError:
                        logger.warning(f"  ⚠️ Could not cancel job {job_id}")
            else:
                logger.warning("No job IDs found to cancel")
            return 0
        
        else:
            # Submit new parallel jobs
            job_ids = manager.submit_parallel_dlc_jobs(args.videos_per_job)
            
            if job_ids:
                print(f"\n🎉 SUBMITTED {len(job_ids)} PARALLEL DLC JOBS")
                print("="*50)
                print("Job IDs:")
                for i, job_id in enumerate(job_ids, 1):
                    print(f"  {i}. {job_id}")
                
                print(f"\n📁 Job directory: {manager.job_dir}")
                print("\nMonitoring commands:")
                print(f"  Check status: python3 {__file__} --session-dir {session_dir} --check-status")
                print(f"  Cancel jobs:  python3 {__file__} --session-dir {session_dir} --cancel-jobs")
                print(f"  SLURM queue:  squeue -u $USER")
                
                # Create summary script
                summary_script = manager.create_summary_script()
                print(f"\n📋 Run after completion: python3 {summary_script}")
                
                return 0
            else:
                print("\n❌ NO JOBS WERE SUBMITTED")
                return 1
    
    except KeyboardInterrupt:
        logger.info("Job submission interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Job submission failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
