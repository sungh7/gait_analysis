#!/usr/bin/env python3
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import glob
import os
from scipy.signal import resample
from sagittal_extractor_2d import MediaPipeSagittalExtractor
from self_driven_segmentation import find_dtw_matches_euclidean

DATA_DIR = "/data/gait/data"
GT_DIR = "/data/gait/data/processed_new"
OUTPUT_DIR = "/data/gait/2d_project/normal_template_test"
if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)

def load_grand_mean():
    all_cycles = []
    # Load 21 subjects
    gt_files = glob.glob(f"{GT_DIR}/S1_*_gait_long.csv")
    for f in gt_files:
        try:
            df = pd.read_csv(f)
            subset = df[(df['joint'] == 'r.kn.angle') & (df['plane'] == 'sagittal')].sort_values('gait_cycle')
            cycle = subset['condition1_avg'].values
            if len(cycle) > 10:
                cycle_norm = resample(cycle, 101)
                all_cycles.append(cycle_norm)
        except: pass
    
    if not all_cycles: return None
    return np.mean(all_cycles, axis=0)

def main():
    # 1. Load Grand Mean (The "Normal" Template)
    normal_template = load_grand_mean()
    if normal_template is None:
        print("Failed to load Normal Template.")
        return

    # 2. Process Excluded Subjects (S4, S5, S6, S7, S12)
    sids = ['04', '05', '06', '07', '12']
    
    for sid in sids:
        print(f"Processing {sid} with Normal Template...")
        extractor = MediaPipeSagittalExtractor()
        
        # Path Handling
        video_path = f"{DATA_DIR}/{sid}/{sid}-2.mp4"
        if not os.path.exists(video_path):
            sid_int = str(int(sid))
            video_path = f"{DATA_DIR}/{sid_int}/{sid_int}-2.mp4"
            
        if not os.path.exists(video_path): continue
        
        landmarks, _ = extractor.extract_pose_landmarks(video_path)
        angles = extractor.calculate_joint_angles(landmarks)
        mp_signal = angles['right_knee_angle'].fillna(method='bfill').fillna(method='ffill').values
        
        # 3. DTW Segmentation (Pattern Search using Normal Template)
        starts = find_dtw_matches_euclidean(mp_signal, normal_template)
        
        # 4. Plot
        plt.figure(figsize=(12, 6))
        plt.subplot(2, 1, 1)
        plt.plot(normal_template, 'k-', linewidth=3, label='Normal Template (Prior)')
        plt.title(f"Reference: Normal Grand Mean (N=21)")
        plt.legend()
        plt.grid()
        
        plt.subplot(2, 1, 2)
        plt.plot(mp_signal, 'gray', alpha=0.6, label='Raw Signal (S4)')
        for s in starts:
            plt.axvline(s, color='b', linestyle='--', alpha=0.8)
        plt.title(f"Segmentation via Normal Template (Prior) - Subject {sid}: {len(starts)} cycles")
        plt.tight_layout()
        plt.savefig(f"{OUTPUT_DIR}/segmentation_via_normal_{sid}.png")
        print(f"Saved {OUTPUT_DIR}/segmentation_via_normal_{sid}.png")

        # 5. Grid Plot for visual check
        plt.figure(figsize=(10, 10))
        for i, s in enumerate(starts[:25]): # Show first 25
            if i >= 25: break
            plt.subplot(5, 5, i+1)
            window = mp_signal[s:s+35] # Approx period
            if len(window) < 10: continue
            norm_cycle = resample(window, 101)
            plt.plot(normal_template, 'k--', alpha=0.3) # Reference
            plt.plot(norm_cycle, 'r-', linewidth=2)
            plt.xticks([])
            plt.yticks([])
        plt.tight_layout()
        plt.savefig(f"{OUTPUT_DIR}/grid_via_normal_{sid}.png")
        plt.close('all')

if __name__ == "__main__":
    main()
