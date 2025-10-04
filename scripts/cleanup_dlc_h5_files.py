#!/usr/bin/env python3
"""
DLC H5 File Cleanup Script

After DLC processing, this script:
1. Finds the latest H5 file for each video
2. Renames it to clean format (vid1-cam1.h5)
3. Archives other H5 files to backup directory
4. Removes "individuals" level for anipose compatibility

Usage:
    python cleanup_dlc_h5_files.py --session-dir /path/to/session
    python cleanup_dlc_h5_files.py --session-dir /path/to/session --dry-run

Author: DLC H5 Cleanup
"""

import os
import sys
import re
import shutil
import pandas as pd
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DLCFileCleanup:
    """Cleanup DLC H5 files to standard naming format"""
    
    def __init__(self, session_dir: Path, dry_run: bool = False):
        self.session_dir = session_dir
        self.pose_2d_dir = session_dir / "pose-2d"
        self.archive_dir = session_dir / "pose-2d-archive"
        self.dry_run = dry_run
        
    def find_video_h5_groups(self) -> Dict[str, List[Path]]:
        """
        Group H5 files by their source video
        
        Returns:
            Dict mapping video base name to list of H5 files
        """
        logger.info("🔍 FINDING H5 FILE GROUPS")
        logger.info("="*50)
        
        if not self.pose_2d_dir.exists():
            logger.error(f"pose-2d directory not found: {self.pose_2d_dir}")
            return {}
        
        h5_files = list(self.pose_2d_dir.glob("*.h5"))
        logger.info(f"Found {len(h5_files)} H5 files")
        
        # Group files by video base name
        video_groups = {}
        
        for h5_file in h5_files:
            # Extract base video name from DLC output filename
            base_name = self._extract_video_base_name(h5_file.name)
            
            if base_name:
                if base_name not in video_groups:
                    video_groups[base_name] = []
                video_groups[base_name].append(h5_file)
                logger.debug(f"  {h5_file.name} -> {base_name}")
            else:
                logger.warning(f"Could not extract base name from: {h5_file.name}")
        
        logger.info(f"Grouped into {len(video_groups)} video groups:")
        for base_name, files in video_groups.items():
            logger.info(f"  {base_name}: {len(files)} files")
        
        return video_groups
    
    def _extract_video_base_name(self, h5_filename: str) -> str:
        """
        Extract base video name from DLC H5 filename
        
        Examples:
            vid1-cam1_superanimal_quadruped_hrnet_w32_snapshot-1000000.h5 -> vid1-cam1
            vid2-cam2DLC_resnet50_iteration-0_shuffle-1.h5 -> vid2-cam2
        """
        # Remove .h5 extension
        name_without_ext = h5_filename.replace('.h5', '')
        
        # Common DLC patterns to remove
        patterns_to_remove = [
            r'_superanimal.*',
            r'DLC_.*',
            r'_resnet.*',
            r'_snapshot.*',
            r'_iteration.*',
            r'_hrnet.*',
            r'_fasterrcnn.*'
        ]
        
        base_name = name_without_ext
        for pattern in patterns_to_remove:
            base_name = re.split(pattern, base_name)[0]
        
        return base_name
    
    def find_latest_h5_file(self, h5_files: List[Path]) -> Path:
        """
        Find the latest H5 file (by modification time)
        
        Args:
            h5_files: List of H5 files for same video
            
        Returns:
            Path to latest H5 file
        """
        latest_file = max(h5_files, key=lambda f: f.stat().st_mtime)
        logger.debug(f"  Latest file: {latest_file.name}")
        return latest_file
    
    def remove_individuals_level(self, h5_file: Path) -> bool:
        """
        Remove 'individuals' level from H5 file for anipose compatibility
        
        Args:
            h5_file: Path to H5 file
            
        Returns:
            True if conversion successful
        """
        try:
            logger.debug(f"  Converting H5 structure: {h5_file.name}")
            
            # Load the H5 file
            df = pd.read_hdf(h5_file)
            
            # Check if it has multi-level columns with individuals
            if isinstance(df.columns, pd.MultiIndex):
                if df.columns.nlevels == 4 and 'individuals' in df.columns.names:
                    # Remove the 'individuals' level
                    individuals_level_idx = df.columns.names.index('individuals')
                    new_columns = df.columns.droplevel(individuals_level_idx)
                    df.columns = new_columns
                    
                    # Save back to same file
                    if not self.dry_run:
                        df.to_hdf(h5_file, key='df_with_missing', mode='w')
                    
                    logger.info(f"    ✅ Removed 'individuals' level from {h5_file.name}")
                    return True
                else:
                    logger.debug(f"    ⏭️ No individuals level to remove: {h5_file.name}")
                    return True
            else:
                logger.debug(f"    ⏭️ Not multi-level columns: {h5_file.name}")
                return True
                
        except Exception as e:
            logger.error(f"    ❌ Failed to convert {h5_file.name}: {e}")
            return False
    
    def cleanup_video_h5_files(self, video_groups: Dict[str, List[Path]]) -> Dict[str, Path]:
        """
        Clean up H5 files for each video group
        
        Args:
            video_groups: Dict mapping video base names to H5 file lists
            
        Returns:
            Dict mapping video base names to their final H5 files
        """
        logger.info("🧹 CLEANING UP H5 FILES")
        logger.info("="*50)
        
        final_files = {}
        
        # Create archive directory
        if not self.dry_run:
            self.archive_dir.mkdir(exist_ok=True)
        
        for base_name, h5_files in video_groups.items():
            logger.info(f"Processing {base_name} ({len(h5_files)} files)...")
            
            if len(h5_files) == 1:
                # Only one file - just rename it
                h5_file = h5_files[0]
                target_name = f"{base_name}.h5"
                target_path = self.pose_2d_dir / target_name
                
                if h5_file.name != target_name:
                    logger.info(f"  Renaming: {h5_file.name} -> {target_name}")
                    if not self.dry_run:
                        h5_file.rename(target_path)
                        final_files[base_name] = target_path
                else:
                    logger.info(f"  ✅ Already correct name: {target_name}")
                    final_files[base_name] = h5_file
                
                # Convert H5 structure
                self.remove_individuals_level(final_files[base_name])
                
            else:
                # Multiple files - keep latest, archive others
                latest_file = self.find_latest_h5_file(h5_files)
                target_name = f"{base_name}.h5"
                target_path = self.pose_2d_dir / target_name
                
                logger.info(f"  Latest: {latest_file.name} -> {target_name}")
                
                # Archive other files
                for h5_file in h5_files:
                    if h5_file != latest_file:
                        archive_path = self.archive_dir / h5_file.name
                        logger.info(f"  Archive: {h5_file.name} -> {archive_path.name}")
                        
                        if not self.dry_run:
                            shutil.move(str(h5_file), str(archive_path))
                
                # Rename latest file
                if not self.dry_run:
                    latest_file.rename(target_path)
                    final_files[base_name] = target_path
                else:
                    final_files[base_name] = target_path
                
                # Convert H5 structure
                self.remove_individuals_level(final_files[base_name])
        
        return final_files
    
    def verify_video_h5_mapping(self, final_files: Dict[str, Path]) -> bool:
        """
        Verify that we have H5 files for all videos
        
        Args:
            final_files: Dict of final H5 files
            
        Returns:
            True if all videos have corresponding H5 files
        """
        logger.info("🔍 VERIFYING VIDEO-H5 MAPPING")
        logger.info("="*50)
        
        # Find all videos
        videos_dir = self.session_dir / "videos-raw"
        if not videos_dir.exists():
            logger.warning("videos-raw directory not found")
            return True
        
        video_files = list(videos_dir.glob("*.mp4"))
        video_base_names = [f.stem for f in video_files]  # Remove .mp4
        
        logger.info(f"Found {len(video_files)} videos")
        logger.info(f"Found {len(final_files)} H5 files")
        
        # Check mapping
        missing_h5 = []
        extra_h5 = []
        
        for video_base in video_base_names:
            if video_base not in final_files:
                missing_h5.append(video_base)
        
        for h5_base in final_files.keys():
            if h5_base not in video_base_names:
                extra_h5.append(h5_base)
        
        if missing_h5:
            logger.warning(f"Videos without H5 files: {missing_h5}")
        
        if extra_h5:
            logger.info(f"H5 files without videos: {extra_h5}")
        
        success_rate = (len(video_base_names) - len(missing_h5)) / len(video_base_names) if video_base_names else 0
        logger.info(f"Success rate: {success_rate:.1%} ({len(video_base_names) - len(missing_h5)}/{len(video_base_names)})")
        
        return len(missing_h5) == 0
    
    def run_cleanup(self) -> bool:
        """
        Run complete H5 file cleanup process
        
        Returns:
            True if cleanup successful
        """
        logger.info("🧹 STARTING DLC H5 FILE CLEANUP")
        logger.info("="*60)
        logger.info(f"Session directory: {self.session_dir}")
        logger.info(f"Pose-2d directory: {self.pose_2d_dir}")
        logger.info(f"Archive directory: {self.archive_dir}")
        logger.info(f"Dry run: {self.dry_run}")
        
        try:
            # Find and group H5 files
            video_groups = self.find_video_h5_groups()
            
            if not video_groups:
                logger.error("No H5 files found to process")
                return False
            
            # Clean up files
            final_files = self.cleanup_video_h5_files(video_groups)
            
            # Verify results
            mapping_success = self.verify_video_h5_mapping(final_files)
            
            # Print summary
            print("\n" + "="*60)
            print("DLC H5 CLEANUP SUMMARY")
            print("="*60)
            print(f"Processed: {len(video_groups)} video groups")
            print(f"Final H5 files: {len(final_files)}")
            
            if not self.dry_run:
                print(f"Archived files: {self.archive_dir}")
                
                print("\nFinal H5 files:")
                for base_name, h5_path in final_files.items():
                    size_mb = h5_path.stat().st_size / (1024*1024)
                    print(f"  ✅ {h5_path.name} ({size_mb:.1f}MB)")
            else:
                print("DRY RUN - No files were actually moved/renamed")
            
            print("="*60)
            
            if mapping_success:
                logger.info("✅ All videos have corresponding H5 files")
            else:
                logger.warning("⚠️ Some videos missing H5 files")
            
            logger.info("🎉 DLC H5 cleanup completed!")
            return True
            
        except Exception as e:
            logger.error(f"Cleanup failed: {e}")
            import traceback
            traceback.print_exc()
            return False

def main():
    parser = argparse.ArgumentParser(description="DLC H5 File Cleanup")
    parser.add_argument("--session-dir", required=True, type=str,
                       help="Path to session directory")
    parser.add_argument("--dry-run", action="store_true",
                       help="Show what would be done without making changes")
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
    
    # Run cleanup
    cleaner = DLCFileCleanup(session_dir, args.dry_run)
    
    try:
        success = cleaner.run_cleanup()
        return 0 if success else 1
        
    except KeyboardInterrupt:
        logger.info("Cleanup interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Cleanup failed: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
