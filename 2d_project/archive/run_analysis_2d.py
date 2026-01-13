#!/usr/bin/env python3
"""
2D Gait Analysis System - Master Execution Script
Focuses on Pattern Stability using 2D MediaPipe Landmarks.
"""

import sys
import argparse
import time
from datetime import datetime
from pathlib import Path
import pandas as pd

# Import local 2D extractor
from sagittal_extractor_2d import MediaPipeSagittalExtractor

def get_video_files(data_dir):
    """Recursively find sagittal videos (ending with _side.mp4 or similar convention if known, or just all mp4s)."""
    # Assuming structure /data/gait/data/{subject_id}/*.mp4
    # We need to identify side views. Usually 'side' is in name.
    # Existing code used specific logic. Let's look for known patterns or simple .mp4
    videos = []
    base_path = Path(data_dir)
    for vid in base_path.rglob("*.mp4"):
        # simple heuristic: use all video files for now or filter
        if "side" in vid.name.lower() or "sagittal" in vid.name.lower() or True: # Accept all for now, user can filter
            videos.append(vid)
    return sorted(videos)

def run_analysis(subject_id=None, run_all=False):
    output_dir = Path("/data/gait/2d_project/results")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    data_dir = "/data/gait/data"
    all_videos = get_video_files(data_dir)
    
    targets = []
    if subject_id:
        targets = [v for v in all_videos if str(subject_id) in v.parts] # simple match
    elif run_all:
        targets = all_videos
    else:
        # Default: Test mode (first 3)
        targets = all_videos[:3]
        print("Defaulting to quick test (first 3 videos). Use --all for full run.")

    if not targets:
        print("No videos found.")
        return

    print(f"Found {len(targets)} videos to process.")
    
    extractor = MediaPipeSagittalExtractor()
    
    results_summary = []

    for vid_path in targets:
        try:
            print(f"\nProcessing: {vid_path.name}")
            # Determine frame limit
            frame_limit = None if run_all else 300
            if frame_limit:
                 print(f"Running in test mode (max_frames={frame_limit})")

            averaged, cycles, info = extractor.process_video(str(vid_path), side='right', max_frames=frame_limit)
            
            if averaged is not None:
                # Save individual result
                rel_name = vid_path.stem
                out_path = output_dir / f"{rel_name}_2d_results.csv"
                averaged.to_csv(out_path, index=False)
                print(f"Saved: {out_path}")
                
                results_summary.append({
                    'video': rel_name,
                    'fps': info['fps'],
                    'cycles': len(cycles),
                    'status': 'Success'
                })
            else:
                results_summary.append({
                    'video': vid_path.name,
                    'status': 'Failed (No cycles)'
                })
                
        except Exception as e:
            print(f"Error processing {vid_path.name}: {e}")
            results_summary.append({
                'video': vid_path.name,
                'status': f'Error: {str(e)}'
            })

    # Save summary
    summary_df = pd.DataFrame(results_summary)
    summary_path = output_dir / f"analysis_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"\nAnalysis Complete. Summary saved to {summary_path}")

def main():
    parser = argparse.ArgumentParser(description='2D Gait Analysis System')
    parser.add_argument('--all', action='store_true', help='Process all videos')
    parser.add_argument('--subject', type=str, help='Process specific subject ID')
    
    args = parser.parse_args()
    
    run_analysis(subject_id=args.subject, run_all=args.all)

if __name__ == "__main__":
    main()