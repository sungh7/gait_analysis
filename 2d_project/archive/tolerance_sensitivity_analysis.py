#!/usr/bin/env python3
"""
Tolerance Sensitivity Analysis for Section 3.10
Calculate PPV at different tolerance levels: ±3, ±5, ±7, ±10 frames
Also calculate chance baseline for comparison.
"""

import numpy as np
import pandas as pd
import os
import sys
from scipy.signal import find_peaks, savgol_filter

sys.path.append("/data/gait/2d_project")
from sagittal_extractor_2d import MediaPipeSagittalExtractor
from self_driven_segmentation import derive_self_template, find_dtw_matches_euclidean

DATA_DIR = "/data/gait/data"
OUTPUT_DIR = "/data/gait/2d_project"


def detect_hs_from_knee_extension(knee_angle, fps=30, min_period=0.7):
    """Kinematic HS detection using knee extension peaks."""
    if len(knee_angle) < 15:
        return np.array([])
    knee_smooth = savgol_filter(knee_angle, window_length=11, polyorder=2)
    min_dist = int(fps * min_period)
    peaks, _ = find_peaks(-knee_smooth, distance=min_dist, prominence=3)
    return peaks


def calculate_ppv_at_tolerance(at_rtm, silver_gt, tolerance):
    """Calculate PPV at given tolerance."""
    if len(at_rtm) == 0:
        return 0.0
    matched = sum(1 for det in at_rtm if any(abs(det - gt) <= tolerance for gt in silver_gt))
    return matched / len(at_rtm) * 100


def calculate_chance_ppv(n_detections, n_silver_gt, signal_length, tolerance):
    """
    Calculate expected PPV by chance (random detection).
    If we randomly place n_detections points in signal_length frames,
    what's the probability of matching any of n_silver_gt events within tolerance?
    """
    # For each random detection, probability of matching any GT event
    # = n_silver_gt × (2 × tolerance + 1) / signal_length
    # But capped at 1.0
    match_prob_per_det = min(1.0, n_silver_gt * (2 * tolerance + 1) / signal_length)
    return match_prob_per_det * 100


def run_tolerance_analysis(sid, max_frames=800):
    """Run tolerance sensitivity analysis for one subject."""
    video_path = f"{DATA_DIR}/{sid}/{sid}-2.mp4"
    if not os.path.exists(video_path):
        return None
    
    extractor = MediaPipeSagittalExtractor()
    try:
        landmarks, meta = extractor.extract_pose_landmarks(video_path, max_frames=max_frames)
    except Exception as e:
        return None
    
    fps = meta.get('fps', 30) if meta else 30
    angles = extractor.calculate_joint_angles(landmarks)
    knee_angle = angles['right_knee_angle'].fillna(method='bfill').fillna(method='ffill').values
    
    # AT-RTM detection
    template, period = derive_self_template(knee_angle)
    if template is None:
        return None
    
    if isinstance(period, (np.ndarray, list)):
        period = float(np.array(period).flatten()[0]) if len(np.array(period).flatten()) > 0 else 32.0
    
    at_rtm = find_dtw_matches_euclidean(knee_angle, template)
    silver_gt = detect_hs_from_knee_extension(knee_angle, fps=fps)
    
    if len(at_rtm) == 0 or len(silver_gt) == 0:
        return None
    
    # Calculate PPV at different tolerances
    tolerances = [3, 5, 7, 10]
    results = {'sid': sid, 'n_atrtm': len(at_rtm), 'n_silver_gt': len(silver_gt)}
    
    for tol in tolerances:
        ppv = calculate_ppv_at_tolerance(at_rtm, silver_gt, tol)
        chance = calculate_chance_ppv(len(at_rtm), len(silver_gt), len(knee_angle), tol)
        results[f'ppv_{tol}'] = ppv
        results[f'chance_{tol}'] = chance
    
    return results


def main():
    test_sids = list(range(1, 27))
    all_results = []
    
    for sid in test_sids:
        print(f"Processing S{sid}...", end=" ", flush=True)
        result = run_tolerance_analysis(sid, max_frames=800)
        if result:
            all_results.append(result)
            print(f"PPV@±5={result['ppv_5']:.1f}%, PPV@±10={result['ppv_10']:.1f}%")
        else:
            print("Skipped")
    
    if all_results:
        df = pd.DataFrame(all_results)
        
        # Summary statistics
        print("\n" + "="*70)
        print("TOLERANCE SENSITIVITY ANALYSIS")
        print("="*70)
        
        tolerances = [3, 5, 7, 10]
        print(f"\n{'Tolerance':<12} {'Mean PPV':<12} {'Median PPV':<12} {'PPV≥50%':<12} {'Chance':<12}")
        print("-"*60)
        
        for tol in tolerances:
            mean_ppv = df[f'ppv_{tol}'].mean()
            median_ppv = df[f'ppv_{tol}'].median()
            pct_above_50 = (df[f'ppv_{tol}'] >= 50).mean() * 100
            mean_chance = df[f'chance_{tol}'].mean()
            print(f"±{tol} frames    {mean_ppv:.1f}%        {median_ppv:.1f}%        {pct_above_50:.1f}%        {mean_chance:.1f}%")
        
        print("\n" + "="*70)
        print("SUMMARY FOR PAPER")
        print("="*70)
        
        # PPV vs Chance comparison
        for tol in tolerances:
            mean_ppv = df[f'ppv_{tol}'].mean()
            mean_chance = df[f'chance_{tol}'].mean()
            ratio = mean_ppv / mean_chance if mean_chance > 0 else float('inf')
            print(f"±{tol} frames: PPV={mean_ppv:.1f}% vs Chance={mean_chance:.1f}% ({ratio:.1f}× better)")
        
        # Save results
        df.to_csv(f"{OUTPUT_DIR}/tolerance_sensitivity_analysis.csv", index=False)
        print(f"\nSaved: {OUTPUT_DIR}/tolerance_sensitivity_analysis.csv")


if __name__ == "__main__":
    main()
