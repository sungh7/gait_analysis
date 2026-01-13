#!/usr/bin/env python3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, savgol_filter, resample
from fastdtw import fastdtw
from sagittal_extractor_2d import MediaPipeSagittalExtractor
import sys

# Parameters
VIDEO_PATH = "/data/gait/data/1/1-2.mp4"
GT_PATH = "/data/gait/data/processed_new/S1_01_gait_long.csv"
OUTPUT_DIR = "/data/gait/2d_project/segmentation_comparison"

import os
if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)

def load_gt_template():
    try:
        df = pd.read_csv(GT_PATH)
        subset = df[(df['joint'] == 'r.kn.angle') & (df['plane'] == 'sagittal')].sort_values('gait_cycle')
        return subset['condition1_avg'].values
    except: return None

def get_heel_peaks(landmarks_df):
    # Heel Y (vertical). 
    # MediaPipe: (0,0) is Top-Left. 
    # Ground is High Y. 
    # Heel Strike = Maxima in Y (Lowest physical point).
    
    heel_y = landmarks_df['RIGHT_HEEL_y'].values
    # Smooth a bit?
    heel_y_smooth = savgol_filter(heel_y, 11, 3) 
    
    # Find peaks (valleys in physical world? No, Maxima in MP Y coords)
    peaks, _ = find_peaks(heel_y_smooth, distance=30, prominence=0.01)
    return peaks, heel_y_smooth

def get_knee_peaks(angles_df):
    # Knee Angle. Heel Strike ~ Max Extension.
    # Flexion is usually ~60 deg. Extension ~0 deg.
    # MP extract angle... let's check sign.
    # Usually we treated Extension as Minima in Flexion plot.
    
    knee_angle = angles_df['right_knee_angle'].values
    # Invert to find maxima of extension? 
    # If angle is Flexion (0=Straight, 90=Bent), then Extension is Minima.
    # So finding peaks in (-knee_angle).
    
    knee_smooth = savgol_filter(knee_angle, 11, 3)
    peaks, _ = find_peaks(-knee_smooth, distance=30, prominence=5)
    return peaks, knee_smooth

def get_dtw_starts(knee_signal, gt_template):
    # From previous script (Euclidean sliding window)
    step = 4
    T_len = 35
    profile = []
    
    # Need to match lengths?
    # Simple Euclidean scan
    for i in range(0, len(knee_signal)-T_len, step):
        window = knee_signal[i:i+T_len]
        window_norm = resample(window, len(gt_template))
        dist = np.linalg.norm(window_norm - gt_template)
        profile.append(dist)
        
    profile = np.array(profile)
    # Minima in profile = Start
    peaks, _ = find_peaks(-profile, distance=int(T_len/step * 0.7))
    return peaks * step, profile

def main():
    extractor = MediaPipeSagittalExtractor()
    landmarks, _ = extractor.extract_pose_landmarks(VIDEO_PATH, max_frames=600)
    angles = extractor.calculate_joint_angles(landmarks)
    
    gt_template = load_gt_template()
    
    # 1. Knee Extension Peaks
    knee_peaks, knee_sig = get_knee_peaks(angles)
    
    # 2. Heel Y Peaks
    heel_peaks, heel_sig = get_heel_peaks(landmarks)
    
    # 3. DTW (Best Match) Starts
    dtw_starts, dtw_profile = get_dtw_starts(knee_sig, gt_template)
    
    print(f"Knee Peaks: {len(knee_peaks)}")
    print(f"Heel Peaks: {len(heel_peaks)}")
    print(f"DTW Starts: {len(dtw_starts)}")
    
    # Plotting
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 12), sharex=True)
    
    # Knee
    ax1.plot(knee_sig, 'b-', label='Knee Angle')
    ax1.plot(knee_peaks, knee_sig[knee_peaks], 'ro', label='Knee Peaks (Ext)')
    for d in dtw_starts: ax1.axvline(d, color='g', linestyle='--', alpha=0.5, label='DTW Start' if d==dtw_starts[0] else "")
    ax1.set_title("1. Knee Extension Peaks (Current Method)")
    ax1.legend()
    
    # Heel
    ax2.plot(heel_sig, 'm-', label='Heel Y')
    ax2.plot(heel_peaks, heel_sig[heel_peaks], 'ro', label='Heel Peaks (Contact)')
    for d in dtw_starts: ax2.axvline(d, color='g', linestyle='--', alpha=0.5)
    ax2.set_title("2. Heel Vertical Peaks (Proposed)")
    ax2.invert_yaxis() # MP Y is 0 at top. Wait. Max Y is bottom. 
    # If we find peaks in Y, we find lowest point. 
    # Let's keep normal y axis but remember high Y = low height.
    # Actually, let's just plot Y. Maxima = Heel Strike.
    ax2.legend()
    
    # Comparison
    # Normalize signals to 0-1 for overlay
    def norm(x): return (x - np.min(x)) / (np.max(x) - np.min(x))
    
    ax3.plot(norm(knee_sig), 'b-', alpha=0.5, label='Knee')
    ax3.plot(norm(heel_sig), 'm-', alpha=0.5, label='Heel Y')
    for k in knee_peaks: ax3.axvline(k, color='b', linestyle=':')
    for h in heel_peaks: ax3.axvline(h, color='m', linestyle=':')
    for d in dtw_starts: ax3.axvline(d, color='g', linewidth=2, label='DTW Truth' if d==dtw_starts[0] else "")
    ax3.set_title("Overlay Comparison (Which aligns with DTW?)")
    ax3.legend()
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/segmentation_comparison.png")
    print(f"Saved to {OUTPUT_DIR}/segmentation_comparison.png")

if __name__ == "__main__":
    main()
