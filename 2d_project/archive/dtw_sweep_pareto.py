#!/usr/bin/env python3
"""
DTW Parameter Sweep for Pareto Plot
Sweep window sizes to show accuracy vs robustness tradeoff.
"""

import pandas as pd
import numpy as np
import glob
import os
import sys
from scipy.signal import resample, find_peaks, correlate
import matplotlib.pyplot as plt

sys.path.append("/data/gait/2d_project")

DATA_DIR = "/data/gait/data"
PROCESSED_DIR = "/data/gait/data/processed_new"
OUTPUT_DIR = "/data/gait/2d_project/rebuttal_experiments"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def derive_template_with_params(signal, fps=30):
    """Derive self-template from signal"""
    sig_detrend = signal - np.mean(signal)
    corr = correlate(sig_detrend, sig_detrend, mode='full')
    corr = corr[len(corr)//2:]
    
    peaks, _ = find_peaks(corr, distance=15)
    if len(peaks) < 2:
        period = 35
    else:
        period = peaks[1]
    
    inverted = -signal
    rough_peaks, _ = find_peaks(inverted, distance=int(period*0.7), prominence=5)
    
    candidates = []
    for i in range(len(rough_peaks)-1):
        start, end = rough_peaks[i], rough_peaks[i+1]
        length = end - start
        if abs(length - period) < period * 0.4:
            seg = signal[start:end]
            seg_norm = resample(seg, 101)
            candidates.append(seg_norm)
            
    if not candidates:
        return None, period
        
    candidates = np.array(candidates)
    self_template = np.median(candidates, axis=0)
    min_idx = np.argmin(self_template)
    self_template = np.roll(self_template, -min_idx)
    
    return self_template, period

def find_matches_with_window(signal, template, window_size, step=4):
    """Find cycle matches with specific window size"""
    dtw_profile = []
    
    for i in range(0, len(signal) - window_size, step):
        window = signal[i:i+window_size]
        window_norm = resample(window, 101)
        dist = np.linalg.norm(window_norm - template)
        dtw_profile.append(dist)
        
    dtw_profile = np.array(dtw_profile)
    peaks, _ = find_peaks(-dtw_profile, distance=int(window_size/step * 0.7))
    return peaks * step, dtw_profile

def get_gt_cycle_count(info_json_path):
    """Extract GT cycle count from Vicon info.json"""
    import json
    try:
        with open(info_json_path, 'r') as f:
            info = json.load(f)
        r_strides = info.get('demographics', {}).get('right_strides', 0)
        return r_strides
    except:
        return 0

def run_sweep():
    """Run DTW window size sweep"""
    # Window sizes to test (in frames)
    window_sizes = [25, 30, 35, 40, 45, 50]
    
    results = []
    
    # Load pre-extracted signals (use cached MP angles if available)
    # For speed, we'll use cached data from previous runs
    cached_csv = f"{OUTPUT_DIR}/gt_cycle_confusion_matrix.csv"
    if not os.path.exists(cached_csv):
        print("Run reanalyze_cycles_gt_based.py first!")
        return
    
    # Load subject list from cache
    df_cache = pd.read_csv(cached_csv)
    subjects = df_cache['Subject'].values
    
    from sagittal_extractor_2d import MediaPipeSagittalExtractor
    
    for window in window_sizes:
        print(f"\n=== Window Size: {window} frames ===")
        
        total_gt = 0
        total_tp = 0
        total_fn = 0
        total_fp = 0
        
        for sid in subjects[:5]:  # Limit to 5 subjects for speed
            try:
                # Get GT count
                info_path = f"{PROCESSED_DIR}/S1_{sid:02d}_info.json"
                gt_cycles = get_gt_cycle_count(info_path)
                if gt_cycles == 0:
                    continue
                
                # Load MP signal (need to extract or cache)
                video_path = f"{DATA_DIR}/{sid}/{sid}-2.mp4"
                if not os.path.exists(video_path):
                    continue
                
                # Use extractor
                extractor = MediaPipeSagittalExtractor()
                lm, _ = extractor.extract_pose_landmarks(video_path)
                angles = extractor.calculate_joint_angles(lm)
                sig = angles['right_knee_angle'].fillna(method='bfill').fillna(method='ffill').values
                
                # Derive template
                template, period = derive_template_with_params(sig)
                if template is None:
                    continue
                
                # Find matches with this window size
                detected, _ = find_matches_with_window(sig, template, window)
                n_detected = len(detected)
                
                # Simple matching
                tp = min(gt_cycles, n_detected)
                fn = max(0, gt_cycles - n_detected)
                fp = max(0, n_detected - gt_cycles)
                
                total_gt += gt_cycles
                total_tp += tp
                total_fn += fn
                total_fp += fp
                
                print(f"  S{sid}: GT={gt_cycles}, Detected={n_detected}")
                
            except Exception as e:
                print(f"  Error S{sid}: {e}")
        
        # Calculate metrics for this window size
        if (total_tp + total_fn) > 0:
            recall = total_tp / (total_tp + total_fn)
            failure_rate = total_fn / (total_tp + total_fn)
        else:
            recall = 0
            failure_rate = 1
        
        if (total_tp + total_fp) > 0:
            precision = total_tp / (total_tp + total_fp)
        else:
            precision = 0
        
        results.append({
            'Window': window,
            'GT_Total': total_gt,
            'TPc': total_tp,
            'FNc': total_fn,
            'FPc': total_fp,
            'Recall': recall,
            'Precision': precision,
            'FailureRate': failure_rate
        })
        
        print(f"  Recall: {recall:.1%}, Precision: {precision:.1%}, Failure Rate: {failure_rate:.1%}")
    
    # Save results
    df_results = pd.DataFrame(results)
    df_results.to_csv(f"{OUTPUT_DIR}/dtw_sweep_results.csv", index=False)
    
    # Generate Pareto plot
    plt.figure(figsize=(10, 6))
    
    for i, row in df_results.iterrows():
        plt.scatter(row['FailureRate'] * 100, row['FPc'] / max(row['TPc'], 1) * 100, 
                   s=100, label=f"W={row['Window']}")
        plt.annotate(f"W={row['Window']}", (row['FailureRate'] * 100 + 0.5, row['FPc'] / max(row['TPc'], 1) * 100))
    
    plt.xlabel('Cycle Failure Rate (%)', fontsize=12)
    plt.ylabel('Over-segmentation Rate (FPc/TPc %)', fontsize=12)
    plt.title('DTW Window Size: Accuracy vs Robustness Tradeoff', fontsize=14)
    plt.axhline(y=100, color='r', linestyle='--', alpha=0.5, label='1:1 Ratio')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/pareto_dtw_sweep.png", dpi=150)
    plt.close()
    
    print(f"\nSaved: {OUTPUT_DIR}/pareto_dtw_sweep.png")
    print(f"Saved: {OUTPUT_DIR}/dtw_sweep_results.csv")

if __name__ == "__main__":
    run_sweep()
