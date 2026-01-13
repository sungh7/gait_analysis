#!/usr/bin/env python3
import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, resample
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
from pathlib import Path
from sagittal_extractor_2d import MediaPipeSagittalExtractor
from tqdm import tqdm
import traceback

# Configuration
DATA_DIR = "/data/gait/data"
OUTPUT_DIR = "/data/gait/2d_project/dtw_pattern_results"
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

# Recycle GT parsing logic
def load_gt_template(sid):
    """
    Load pre-calculated GT average cycle from processed_new.
    Returns: 1D array of knee flexion angles (101 points).
    """
    # Map sid '1' -> '01', '10' -> '10'
    sid_str = f"{int(sid):02d}"
    csv_path = Path(f"/data/gait/data/processed_new/S1_{sid_str}_gait_long.csv")
    
    if not csv_path.exists():
        print(f"GT file missing: {csv_path}")
        return None
        
    try:
        df = pd.read_csv(csv_path)
        # Filter: Right Knee, Sagittal
        # Joint names in file: r.an.angle? No, need to check if r.kn.angle consists.
        # User showed S1_16 having r.an.angle. Assuming r.kn.angle exists too.
        
        target_joint = 'r.kn.angle' # Right Knee
        
        subset = df[
            (df['joint'] == target_joint) & 
            (df['plane'] == 'sagittal')
        ].sort_values('gait_cycle')
        
        if subset.empty:
            print(f"Joint {target_joint} not found in {csv_path}")
            return None
            
        # condition1_avg is the mean
        # Flip sign? In V3D export, Y was Flexion.
        # In processed file (head output): -3.0 avg.
        # Standard: Flexion is positive? Or negative?
        # If mean is -3, maybe it's Extension-Min?
        # Let's check ranges later or assume pattern shape matters more.
        # For DTW, shape is key.
        
        template = subset['condition1_avg'].values
        return template
        
    except Exception as e:
        print(f"Error loading GT {csv_path}: {e}")
        return None

def find_best_match_dtw(mp_signal, template, step=10):
    """
    Sliding window DTW to find best match of template in mp_signal.
    """
    if len(mp_signal) < len(template):
        return None, None, float('inf')
        
    window_size = len(template)
    
    best_dist = float('inf')
    best_idx = -1
    best_segment = None
    
    # Coarse search
    print(f"Scanning {len(mp_signal)} frames (Window={window_size}, Step={step})...")
    for i in range(0, len(mp_signal) - window_size, step):
        segment = mp_signal[i : i+window_size]
        
        # radius=1 for speed
        dist, path = fastdtw(segment, template, radius=1, dist=lambda x, y: abs(x - y))
        norm_dist = dist / len(path)
        
        if norm_dist < best_dist:
            best_dist = norm_dist
            best_idx = i
            best_segment = segment
            
    return best_idx, best_segment, best_dist

def process_subject(sid, extractor):
    subject_dir = Path(DATA_DIR) / str(sid)
    video_path = subject_dir / f"{sid}-2.mp4"
    
    if not video_path.exists():
        print(f"Video missing: {video_path}")
        return None

    # Load GT Template (Clean)
    gt_avg_cycle = load_gt_template(sid)
    if gt_avg_cycle is None:
        return None
        
    # 1. Get MP Signal (Full)
    try:
        # Get raw landmarksdf
        landmarks_df, info = extractor.extract_pose_landmarks(str(video_path))
        angles_df = extractor.calculate_joint_angles(landmarks_df)
        mp_full = angles_df['right_knee_angle'].values
        # Fill NaNs
        mp_full = pd.Series(mp_full).interpolate().fillna(method='bfill').fillna(method='ffill').values
    except Exception as e:
        print(f"MP Error {sid}: {e}")
        return None
        
    # 2. Prepare Template
    # MP FPS ~30. GT Standard Cycle ~100 points.
    # We must downsample GT template to match approx MP Frame Rate.
    # Approx 35 frames per cycle.
    cycle_frames = 35 
    gt_template_single = resample(gt_avg_cycle, cycle_frames)
    
    # Repeat 3 times
    repeats = 3
    gt_template = np.tile(gt_template_single, repeats)
    
    # 3. Search
    print(f"Searching for pattern (len={len(gt_template)}) in MP (len={len(mp_full)})...")
    idx, segment, dist = find_best_match_dtw(mp_full, gt_template, step=5)
    
    if idx == -1: return None
    
    # 4. Plot
    plt.figure(figsize=(12, 6))
    plt.subplot(2,1,1)
    plt.plot(mp_full, 'b-', label='MP Full', alpha=0.5)
    # Highlight Match
    plt.plot(np.arange(idx, idx+len(segment)), segment, 'r-', linewidth=2, label='Best Match Segment')
    plt.title(f"Subject {sid}: Pattern Search (DTW Dist={dist:.2f})")
    plt.legend()
    
    plt.subplot(2,1,2)
    plt.plot(segment, 'r-', label='MP Segment')
    plt.plot(gt_template, 'k--', label='GT Template (3x)')
    plt.legend()
    plt.title("Pattern Alignment")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/subject_{sid}_pattern.png")
    plt.close()
    
    return {
        'subject': sid,
        'dtw_dist': dist,
        'match_start': idx,
        'match_len': len(segment)
    }

def main():
    print("Starting DTW Pattern Search (Repeated GT Template)...")
    extractor = MediaPipeSagittalExtractor()
    results = []
    
    subject_dirs = sorted(glob.glob(f"{DATA_DIR}/[0-9]*"))
    subject_ids = [Path(d).name for d in subject_dirs if Path(d).name.isdigit()]
    
    for sid in tqdm(subject_ids):
        # Skip if done
        if (Path(OUTPUT_DIR) / f"subject_{sid}_pattern.png").exists():
             continue
        try:
            res = process_subject(sid, extractor)
            if res:
                results.append(res)
        except Exception as e:
            print(f"Error {sid}: {e}")
            traceback.print_exc()

    if results:
        pd.DataFrame(results).to_csv(f"{OUTPUT_DIR}/pattern_results.csv", index=False)

if __name__ == "__main__":
    main()