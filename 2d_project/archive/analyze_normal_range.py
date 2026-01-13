#!/usr/bin/env python3
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import glob
import os
from scipy.signal import resample, correlate, find_peaks
from sagittal_extractor_2d import MediaPipeSagittalExtractor

DATA_DIR = "/data/gait/data"
GT_DIR = "/data/gait/data/processed_new"
OUTPUT_DIR = "/data/gait/2d_project/normal_range_analysis"
if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)

def load_all_gt_cycles():
    print("Loading all GT data to establish Normal Range...")
    all_cycles = []
    
    gt_files = glob.glob(f"{GT_DIR}/S1_*_gait_long.csv")
    for f in gt_files:
        try:
            df = pd.read_csv(f)
            subset = df[(df['joint'] == 'r.kn.angle') & (df['plane'] == 'sagittal')].sort_values('gait_cycle')
            cycle = subset['condition1_avg'].values
            if len(cycle) > 10:
                # Normalize to 101 points
                cycle_norm = resample(cycle, 101)
                all_cycles.append(cycle_norm)
        except Exception as e:
            print(f"Skipping {f}: {e}")
            
    if not all_cycles: return None, None
    
    all_cycles = np.array(all_cycles)
    grand_mean = np.mean(all_cycles, axis=0)
    grand_std = np.std(all_cycles, axis=0)
    
    print(f"Loaded {len(all_cycles)} GT templates.")
    return grand_mean, grand_std, all_cycles

def derive_s4_template(sid='04'):
    print(f"Deriving template for Subject {sid}...")
    extractor = MediaPipeSagittalExtractor()
    # Try both '04' and '4' formats
    video_path = f"{DATA_DIR}/{sid}/{sid}-2.mp4"
    if not os.path.exists(video_path):
        sid_int = str(int(sid))
        video_path = f"{DATA_DIR}/{sid_int}/{sid_int}-2.mp4"
    
    if not os.path.exists(video_path):
        print(f"Video not found: {video_path}")
        return None
    
    landmarks, _ = extractor.extract_pose_landmarks(video_path)
    angles = extractor.calculate_joint_angles(landmarks)
    mp_signal = angles['right_knee_angle'].fillna(method='bfill').fillna(method='ffill').values
    
    # Auto-Template Logic (Copied from self_driven_segmentation.py)
    sig_detrend = mp_signal - np.mean(mp_signal)
    corr = correlate(sig_detrend, sig_detrend, mode='full')
    corr = corr[len(corr)//2:]
    peaks, _ = find_peaks(corr, distance=15)
    period = peaks[1] if len(peaks) > 1 else 35
    
    inverted = -mp_signal # Find flexion peaks
    rough_peaks, _ = find_peaks(inverted, distance=int(period*0.7), prominence=5)
    
    candidates = []
    for i in range(len(rough_peaks)-1):
        start, end = rough_peaks[i], rough_peaks[i+1]
        if abs((end-start) - period) < period * 0.4:
            seg = resample(mp_signal[start:end], 101)
            candidates.append(seg)
            
    if not candidates: return None
    
    template = np.median(candidates, axis=0)
    
    # Phase Alignment (Max Extension at 0)
    min_idx = np.argmin(template)
    aligned_template = np.roll(template, -min_idx)
    
    return aligned_template

def main():
    # 1. Get Normal Range
    mean, std, raw_cycles = load_all_gt_cycles()
    if mean is None:
        print("Failed to load GT data.")
        return

    # 2. Get S4 Template
    s4_template = derive_s4_template('04')
    if s4_template is None:
        print("Failed to derive S4 template.")
        return
        
    # 3. Plot Comparison
    plt.figure(figsize=(10, 6))
    
    # Plot Normal Range
    plt.plot(mean, 'k-', linewidth=3, label='Normal Range (Mean)')
    plt.fill_between(range(101), mean - std, mean + std, color='gray', alpha=0.3, label='Normal +/- 1 STD')
    plt.fill_between(range(101), mean - 2*std, mean + 2*std, color='gray', alpha=0.1, label='Normal +/- 2 STD')
    
    # Plot individual GTs (thin lines)
    # for c in raw_cycles:
    #     plt.plot(c, 'k-', alpha=0.05)
    
    # Plot S4
    plt.plot(s4_template, 'r-', linewidth=3, label='Subject 04 (Auto-Aligned)')
    
    # Check Inversion? (Correlation)
    corr_check = np.corrcoef(mean, s4_template)[0, 1]
    
    plt.title(f"Waveform Validity Check: S4 vs Normal Population (N={len(raw_cycles)})\nS4 Correlation to Normal Mean: r={corr_check:.2f}")
    plt.xlabel("% Gait Cycle")
    plt.ylabel("Knee Flexion (Deg)")
    plt.legend()
    plt.grid(True)
    
    out_path = f"{OUTPUT_DIR}/s4_vs_normal.png"
    plt.savefig(out_path)
    print(f"Saved comparison plot to {out_path}")

if __name__ == "__main__":
    main()
