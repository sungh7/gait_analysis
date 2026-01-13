#!/usr/bin/env python3
"""
Ablation Study: Real DTW (Constrained) vs Euclidean (Resampling)
Direct comparison of segmentation performance and speed.
"""

import numpy as np
import pandas as pd
import time
import os
import sys
import matplotlib.pyplot as plt
from scipy.signal import resample, find_peaks
try:
    from fastdtw import fastdtw
    from scipy.spatial.distance import euclidean
except ImportError:
    print("Error: fastdtw not installed. Please install it.")
    sys.exit(1)

sys.path.append("/data/gait/2d_project")
from sagittal_extractor_2d import MediaPipeSagittalExtractor
from self_driven_segmentation import derive_self_template, find_dtw_matches_euclidean

DATA_DIR = "/data/gait/data"
OUTPUT_DIR = "/data/gait/2d_project"


def find_matches_real_dtw(signal, template, radius=10, step=4):
    """
    Sliding window segmentation using FastDTW (O(N) approximation).
    
    Args:
        signal: 1D array, full signal
        template: 1D array, gait template
        radius: Sakoe-Chiba constraint radius (approx)
        step: sliding step size
        
    Returns:
        peaks: detected start indices
        profile: distance profile
        exec_time: execution time in seconds
    """
    start_time = time.time()
    
    dtw_profile = []
    T_len = len(template)
    
    # We iterate through the signal
    # Note: DTW allows length warping, but for sliding window we need a reference length.
    # We'll use a window of size roughly T_len.
    # Strict DTW would require checking multiple window sizes, but here we assume relatively constant period.
    # To be comparable to Euclidean method (which resamples), we use fixed window size T_len.
    
    indices = range(0, len(signal) - T_len, step)
    
    for i in indices:
        window = signal[i : i+T_len]
        
        # FastDTW
        # We do NOT resample the window here (DTW handles timing differences).
        # We just compare raw window vs template.
        # But template is likely resampled to 100 points, while window is raw frames (~30-40).
        # We must ensure they are on somewhat similar scale or let DTW handle it.
        # Ideally, template should be in "Frame Domain" for DTW, OR window resampled to "Template Domain"
        # If we resample window, we are doing "Resampled DTW", which is closer to Euclidean.
        # The reviewer asked for "Real DTW" which usually implies handling raw temporal distortions.
        # So we should use the template resampled back to 'average period length' frames.
        
        # For 1D scalar signal, use L2 distance (squared difference) to match AT-RTM
        distance, path = fastdtw(window, template, dist=lambda x, y: (x - y) ** 2, radius=radius)
        dtw_profile.append(distance)
        
    # Find minima
    dtw_profile = np.array(dtw_profile)
    # find_peaks finds MAXIMA. We want MINIMA.
    # Invert signal
    inverted = -dtw_profile
    
    # Min distance between peaks ~ 70% of cycle
    min_dist = int((T_len / step) * 0.7)
    
    peaks, properties = find_peaks(inverted, distance=max(1, min_dist))
    
    matched_indices = peaks * step
    
    exec_time = time.time() - start_time
    return matched_indices, dtw_profile, exec_time


def run_ablation(sid, max_frames=None):
    print(f"\nProcessing Subject {sid}...")
    
    # 1. Get Signal
    extractor = MediaPipeSagittalExtractor()
    video_path = f"{DATA_DIR}/{sid}/{sid}-2.mp4"
    
    if not os.path.exists(video_path):
        print("Video not found.")
        return None

    try:
        landmarks, _ = extractor.extract_pose_landmarks(video_path, max_frames=max_frames)
        angles = extractor.calculate_joint_angles(landmarks)
        signal = angles['right_knee_angle'].fillna(method='bfill').fillna(method='ffill').values
    except Exception as e:
        print(f"Extraction failed: {e}")
        return None
        
    # 2. Derive Template (Common for both)
    # Using existing method to get the template
    template_norm, period = derive_self_template(signal)
    if template_norm is None:
        print("Template derivation failed.")
        return None
        
    # Ensure period is scalar and valid
    if isinstance(period, (np.ndarray, list)):
        period = period.flatten()[0] if len(period) > 0 else 30
    if period < 15:
        print(f"  Warning: Estimated period {period} is too small. Defaulting to 32.")
        period = 32
    print(f"  Estimated Period: {period} frames")
    
    # For DTW, we need a template in "Frame Domain" (not 0-100%)
    # Resample template_norm (length 100) to 'period' (length ~32)
    template_frames = resample(template_norm, int(period))
    
    # 3. Method A: Euclidean (Resampling + L2) - "AT-RTM"
    print("  Running Method A: Euclidean (AT-RTM)...", end=" ", flush=True)
    start_time = time.time()
    matches_a = find_dtw_matches_euclidean(signal, template_norm) # It handles resampling internally
    time_a = time.time() - start_time
    print(f"Done ({time_a:.3f}s)")
    
    # 4. Method B: Real DTW (FastDTW)
    print("  Running Method B: FastDTW...", end=" ", flush=True)
    matches_b, profile_b, time_b = find_matches_real_dtw(signal, template_frames, radius=5)
    print(f"Done ({time_b:.3f}s)")
    
    # 5. Compare
    print(f"  Comparison:")
    print(f"    Method A (RTM): {len(matches_a)} cycles found")
    print(f"    Method B (DTW): {len(matches_b)} cycles found")
    print(f"    Speedup: A is {time_b / time_a:.1f}x faster")
    
    # Check Agreement
    # Count how many of A's matches are within tolerance of B's
    tolerance = 5
    agreement_count = 0
    for ma in matches_a:
        for mb in matches_b:
            if abs(ma - mb) <= tolerance:
                agreement_count += 1
                break
    
    agreement_pct = agreement_count / len(matches_a) * 100 if len(matches_a) > 0 else 0
    print(f"    Agreement: {agreement_count}/{len(matches_a)} ({agreement_pct:.1f}%) cycles match within {tolerance} frames")
    
    return {
        'sid': sid,
        'signal': signal,
        'matches_a': matches_a,
        'matches_b': matches_b,
        'time_a': time_a,
        'time_b': time_b,
        'agreement': agreement_pct
    }


def visualize_ablation(res, output_path):
    """Plot matches from both methods"""
    if res is None: return
    
    signal = res['signal']
    
    plt.figure(figsize=(14, 6))
    plt.plot(signal, 'k-', linewidth=0.8, alpha=0.6, label='Signal')
    
    # Method A (Top ticks)
    for i, m in enumerate(res['matches_a']):
        label = 'Euclidean (RTM)' if i == 0 else None
        plt.scatter(m, np.max(signal) + 5, marker='v', color='blue', s=40, label=label, zorder=3)
        plt.axvline(m, color='blue', linestyle='--', alpha=0.3)
        
    # Method B (Bottom ticks)
    for i, m in enumerate(res['matches_b']):
        label = 'FastDTW' if i == 0 else None
        plt.scatter(m, np.min(signal) - 5, marker='^', color='red', s=40, label=label, zorder=3)
        plt.axvline(m, color='red', linestyle='--', alpha=0.3)
        
    plt.title(f"Ablation: Euclidean (RTM) vs FastDTW - S{res['sid']}\n"
              f"Agreement: {res['agreement']:.1f}% | RTM Time: {res['time_a']:.2f}s | FastDTW Time: {res['time_b']:.2f}s")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Saved: {output_path}")


def main():
    sids = [1, 3] # Compare on a few subjects
    
    results = []
    
    for sid in sids:
        res = run_ablation(sid, max_frames=800) # Limit frames for speed
        if res:
            results.append(res)
            visualize_ablation(res, f"{OUTPUT_DIR}/ablation_dtw_S{sid}.png")
            
    # Summary
    if results:
        print("\n=== SUMMARY ===")
        print(f"{'Subject':<5} | {'RTM Time':<10} | {'DTW Time':<10} | {'Ratio':<8} | {'Agreement':<10}")
        for r in results:
            ratio = r['time_b']/r['time_a'] if r['time_a']>0 else 0
            print(f"S{r['sid']:<4} | {r['time_a']:.3f}s     | {r['time_b']:.3f}s     | {ratio:.1f}x     | {r['agreement']:.1f}%")

if __name__ == "__main__":
    main()
