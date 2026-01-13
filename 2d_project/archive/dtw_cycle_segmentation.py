#!/usr/bin/env python3
import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import resample, find_peaks
from fastdtw import fastdtw
from pathlib import Path
from sagittal_extractor_2d import MediaPipeSagittalExtractor
from tqdm import tqdm
import traceback

DATA_DIR = "/data/gait/data"
OUTPUT_DIR = "/data/gait/2d_project/dtw_segment_results"
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

def load_gt_template(sid):
    # Map sid '1' -> '01'
    sid_str = f"{int(sid):02d}"
    csv_path = Path(f"/data/gait/data/processed_new/S1_{sid_str}_gait_long.csv")
    if not csv_path.exists(): return None
    try:
        df = pd.read_csv(csv_path)
        subset = df[(df['joint'] == 'r.kn.angle') & (df['plane'] == 'sagittal')].sort_values('gait_cycle')
        if subset.empty: return None
        return subset['condition1_avg'].values # 101 points
    except: return None

def find_dtw_matches(mp_signal, gt_template):
    # Sliding window DTW
    # Window size = len(gt_template) (approx 35-40 frames usually, but template is 101 points resampled)
    # Be careful: GT template is 101 points (normalized 0-100%).
    # Real MP cycle is ~30-40 frames at 30fps.
    # We must RESAMPLE the MP window to 101 points to match GT for valid DTW?
    # OR we assume GT is 35 frames? 
    # NO. GT from 'processed_new' is likely 101 points (0-100%).
    
    # PROBLEM: We don't know the exact length of MP cycle in frames to slice.
    # Strategy: 
    # 1. Assume avg cycle is ~35 frames (approx 1.2s at 30fps).
    # 2. Slice window of size 35. Resample to 101. Compare.
    # 3. Slide by 2-3 frames.
    
    T_len = 35 # Avg frames per cycle assumption
    step = 4
    
    dtw_profile = []
    
    # Add padding to handle edges? Or just ignore ends.
    for i in range(0, len(mp_signal) - T_len, step):
        window = mp_signal[i : i+T_len]
        
        # Resample window to 101 to match GT length
        window_norm = resample(window, len(gt_template))
        
        # Euclidean Distance (Fast Search)
        # DTW is too slow for sliding window. Euclidean pattern matching is a good proxy for segmentation.
        dist = np.linalg.norm(window_norm - gt_template)
        dtw_profile.append(dist)
        
    dtw_profile = np.array(dtw_profile)
    
    # 2. Find Minima in DTW profile
    # Minima = Best matches
    # Use find_peaks on inverted profile
    inverted_dist = -dtw_profile
    # distance=35/step (roughly 1 cycle width in index space) to avoid double detection
    peaks, _ = find_peaks(inverted_dist, distance=int(T_len/step * 0.7))
    
    matched_indices = peaks * step # Convert back to frame index
    
    segments = []
    for idx in matched_indices:
        segments.append({
            'start': idx,
            'end': idx + T_len,
            'signal': mp_signal[idx : idx+T_len],
            'score': dtw_profile[idx//step] # Approx
        })
        
    return segments, dtw_profile

def visual_check(sid, mp_signal, segments, gt_template):
    num = len(segments)
    if num == 0: return

    # 1. Segmented Signal Plot (Colored)
    plt.figure(figsize=(15, 6))
    plt.plot(mp_signal, color='lightgray', linewidth=1, label='Full Signal')
    
    colors = plt.cm.jet(np.linspace(0, 1, num))
    
    for i, seg in enumerate(segments):
        start, end = seg['start'], seg['end']
        plt.plot(range(start, end), seg['signal'], color=colors[i], linewidth=2)
        # plt.text(start, mp_signal[start], str(i+1), fontsize=8)

    plt.title(f"Subject {sid}: Segmented Cycles (DTW-based)")
    # plt.legend()
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/subject_{sid}_colored_signal.png")
    plt.close()

    # 2. Grid Plot
    cols = 5
    rows = (num // cols) + (1 if num % cols > 0 else 0)
    fig, axes = plt.subplots(rows, cols, figsize=(20, 4*rows))
    if num == 0: return 
    if rows == 1 and cols == 1: axes = [axes] # Handle single case
    axes = np.array(axes).flatten()
    
    for i, seg in enumerate(segments):
        ax = axes[i]
        
        # Resample seg to 101
        sig_norm = resample(seg['signal'], 101)
        
        ax.plot(np.linspace(0,100,101), gt_template, 'k-', linewidth=2, label='GT')
        ax.plot(np.linspace(0,100,101), sig_norm, 'r-', linewidth=2, label='MP')
        ax.set_title(f"Cycle {i+1} (D:{seg['score']:.1f})")
        
        if i==0: ax.legend()
        
    for j in range(i+1, len(axes)): axes[j].axis('off')
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/subject_{sid}_grid_dtw.png")
    plt.close()

def main():
    print("Starting DTW-based Segmentation...")
    extractor = MediaPipeSagittalExtractor() 
    
    subject_dirs = sorted(glob.glob(f"{DATA_DIR}/[0-9]*"))
    subject_ids = [Path(d).name for d in subject_dirs if Path(d).name.isdigit()]
    
    for sid in tqdm(subject_ids):
        try:
            # Load Data
            gt_template = load_gt_template(sid)
            if gt_template is None: continue
            
            video_path = f"{DATA_DIR}/{sid}/{sid}-2.mp4"
            if not os.path.exists(video_path): continue
            
            # Extract MP (Fast Debug for Subject 1 first? User wants fast feedback. Let's do 500 frames for S1 first)
            # Actually user asked for batch but let's be safe.
            # I will run FULL for everyone.
            
            landmarks, _ = extractor.extract_pose_landmarks(video_path)
            angles = extractor.calculate_joint_angles(landmarks)
            mp_signal = angles['right_knee_angle'].fillna(method='bfill').fillna(method='ffill').values
            
            # DTW Match
            segments, _ = find_dtw_matches_euclidean(mp_signal, gt_template)
            
            # Identify
            print(f"Subject {sid}: Found {len(segments)} cycles.")
            
            # Visualize
            visual_check(sid, mp_signal, segments, gt_template)
            
        except Exception as e:
            print(f"Error {sid}: {e}")
            traceback.print_exc()

if __name__ == "__main__":
    main()
