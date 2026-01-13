#!/usr/bin/env python3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, correlate, resample
from fastdtw import fastdtw
from sagittal_extractor_2d import MediaPipeSagittalExtractor
import sys

VIDEO_PATH = "/data/gait/data/1/1-2.mp4"
GT_PATH = "/data/gait/data/processed_new/S1_01_gait_long.csv"
OUTPUT_DIR = "/data/gait/2d_project/gt_free_test"

import os
if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)

def load_gt_template():
    try:
        df = pd.read_csv(GT_PATH)
        subset = df[(df['joint'] == 'r.kn.angle') & (df['plane'] == 'sagittal')].sort_values('gait_cycle')
        return subset['condition1_avg'].values
    except: return None

def derive_self_template(signal, fps=30):
    # 1. Autocorrelation to find approximate period
    # Detrend
    sig_detrend = signal - np.mean(signal)
    corr = correlate(sig_detrend, sig_detrend, mode='full')
    corr = corr[len(corr)//2:] # Right half
    
    # Find peaks in correlation
    # First peak at 0. Second peak at T.
    peaks, _ = find_peaks(corr, distance=15) # min 0.5s
    
    if len(peaks) < 2:
        print("Autocorrelation failed. Using fallback period 35.")
        period = 35
    else:
        period = peaks[1] # First non-zero peak
    
    print(f"Estimated Period: {period} frames")
    
    # 2. Rough Segmentation using Extension Peaks (Minima of Flexion)
    # We use this just to gather candidate cycles
    inverted = -signal
    # distance = 0.7 * period
    rough_peaks, _ = find_peaks(inverted, distance=int(period*0.7), prominence=5)
    
    candidates = []
    for i in range(len(rough_peaks)-1):
        start, end = rough_peaks[i], rough_peaks[i+1]
        length = end - start
        # Variance filter (allow 20% deviation from period)
        if abs(length - period) < period * 0.4:
            seg = signal[start:end]
            # Normalize to 0-100% time
            seg_norm = resample(seg, 101)
            candidates.append(seg_norm)
            
    if not candidates:
        print("No candidates found. Returning None.")
        return None
        
    candidates = np.array(candidates)
    
    # 3. Create Template (Median of candidates to reject outliers)
    self_template = np.median(candidates, axis=0)
    
    # --- PHASE ALIGNMENT (New) ---
    # Align the template so it starts at the Minimum value (Max Extension ~ Heel Strike)
    # This ensures consistency with clinical Gait Cycle definition (0% = HS)
    min_idx = np.argmin(self_template)
    self_template = np.roll(self_template, -min_idx)
    
    return self_template, candidates

def find_dtw_matches_euclidean(signal, template):
    # Same logic as dtw_cycle_segmentation, but using self_template
    dtw_profile = []
    T_len = 35 # Use standard window or derived period?
    # Let's use T=len(template) but template is 101. 
    # We need to map 101 back to frames?
    # Actually, we should just assume window size 35 frames for the SCAN, 
    # and resample that window to 101 to compare with template.
    
    step = 4
    for i in range(0, len(signal)-35, step):
        window = signal[i:i+35]
        window_norm = resample(window, 101)
        dist = np.linalg.norm(window_norm - template)
        dtw_profile.append(dist)
        
    dtw_profile = np.array(dtw_profile)
    peaks, _ = find_peaks(-dtw_profile, distance=int(35/step * 0.7))
    return peaks * step

def main():
    extractor = MediaPipeSagittalExtractor()
    # Use full video for robust autocorrelation
    landmarks, _ = extractor.extract_pose_landmarks(VIDEO_PATH) 
    angles = extractor.calculate_joint_angles(landmarks)
    mp_signal = angles['right_knee_angle'].fillna(method='bfill').fillna(method='ffill').values
    
    gt_template = load_gt_template()
    
    # 1. Derive Self Template
    self_template, candidates = derive_self_template(mp_signal)
    
    # 2. Segment using Self Template
    starts_self = find_dtw_matches_euclidean(mp_signal, self_template)
    
    # 3. Segment using GT Template (Control)
    starts_gt = find_dtw_matches_euclidean(mp_signal, gt_template)
    
    print(f"Self-Template Segmentation: {len(starts_self)} cycles")
    print(f"GT-Template Segmentation: {len(starts_gt)} cycles")
    
    # Visualization
    plt.figure(figsize=(12, 10))
    
    # Template Comparison
    plt.subplot(3, 1, 1)
    plt.plot(gt_template, 'k-', linewidth=3, label='External GT (V3D)')
    plt.plot(self_template, 'r--', linewidth=3, label='Self-Derived (Auto)')
    plt.title("Template Comparison: Can we learn the pattern from the data itself?")
    plt.legend()
    plt.grid(True)
    
    # Segmentation Result (Self)
    plt.subplot(3, 1, 2)
    plt.plot(mp_signal, 'gray', alpha=0.5)
    for s in starts_self:
        plt.axvline(s, color='r', linestyle='--', alpha=0.8)
    plt.title(f"Segmentation w/ Auto-Template ({len(starts_self)} cycles)")
    
    # Segmentation Result (GT)
    plt.subplot(3, 1, 3)
    plt.plot(mp_signal, 'gray', alpha=0.5)
    for s in starts_gt:
        plt.axvline(s, color='k', linestyle='--', alpha=0.8)
    plt.title(f"Segmentation w/ GT-Template ({len(starts_gt)} cycles)")
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/gt_free_validation.png")
    print(f"Saved to {OUTPUT_DIR}/gt_free_validation.png")

if __name__ == "__main__":
    main()
