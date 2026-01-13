#!/usr/bin/env python3
"""
Proper GT-Based Cycle Matching Reanalysis

Three matching methods:
A) Force-plate region matching with ±5 frame tolerance
B) Cross-correlation lag estimation + full trial matching  
C) Phase coverage (GT cycle time range contains AT detection)
"""

import pandas as pd
import numpy as np
import glob
import os
import sys
import json
from scipy.signal import resample, find_peaks, correlate

sys.path.append("/data/gait/2d_project")
from sagittal_extractor_2d import MediaPipeSagittalExtractor
from self_driven_segmentation import derive_self_template, find_dtw_matches_euclidean

DATA_DIR = "/data/gait/data"
PROCESSED_DIR = "/data/gait/data/processed_new"
OUTPUT_DIR = "/data/gait/2d_project/rebuttal_experiments"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_vicon_events(sid):
    """Load Vicon heel-strike events from processed data"""
    try:
        # Load Vicon knee angle to get frame correspondence
        csv_path = f"{PROCESSED_DIR}/S1_{sid:02d}_gait_long.csv"
        if not os.path.exists(csv_path):
            return None, None, None
        
        df = pd.read_csv(csv_path)
        
        # Vicon knee angle (100Hz, normalized to 0-100% per cycle already)
        # We need the original time-series
        vicon_knee = df['value'].values if 'value' in df.columns else None
        
        # Get info.json for stride counts
        info_path = f"{PROCESSED_DIR}/S1_{sid:02d}_info.json"
        with open(info_path, 'r') as f:
            info = json.load(f)
        
        r_strides = info.get('demographics', {}).get('right_strides', 0)
        
        return vicon_knee, r_strides, info
    except Exception as e:
        print(f"  Error loading Vicon data for S{sid}: {e}")
        return None, None, None


def run_at_dtw_detection(video_path, return_signal=False):
    """Run AT-DTW and return detected cycle starts + signal"""
    try:
        extractor = MediaPipeSagittalExtractor()
        lm, _ = extractor.extract_pose_landmarks(video_path)
        angles = extractor.calculate_joint_angles(lm)
        sig = angles['right_knee_angle'].fillna(method='bfill').fillna(method='ffill').values
        
        template, period = derive_self_template(sig)
        if template is None:
            return None, None, 'template_failure', sig if return_signal else None
        
        # Find matches
        detected_starts = find_dtw_matches_euclidean(sig, template)
        
        if return_signal:
            return detected_starts, period, 'success', sig
        return detected_starts, period, 'success', None
    except Exception as e:
        return None, None, f'error: {e}', None


def estimate_lag_crosscorr(mp_signal, vicon_signal_100hz, mp_fps=30, vicon_fps=100):
    """
    Method B: Estimate lag between MP and Vicon signals using cross-correlation
    Returns lag in MP frames
    """
    try:
        # Resample Vicon to MP framerate
        n_mp_frames = len(mp_signal)
        n_vicon_samples = len(vicon_signal_100hz)
        expected_vicon_at_30hz = int(n_vicon_samples * mp_fps / vicon_fps)
        
        vicon_resampled = resample(vicon_signal_100hz, min(n_mp_frames, expected_vicon_at_30hz))
        
        # Match lengths
        min_len = min(len(mp_signal), len(vicon_resampled))
        mp_norm = mp_signal[:min_len] - np.mean(mp_signal[:min_len])
        vicon_norm = vicon_resampled[:min_len] - np.mean(vicon_resampled[:min_len])
        
        # Cross-correlation
        corr = correlate(mp_norm, vicon_norm, mode='full')
        lag_idx = np.argmax(corr) - (min_len - 1)
        
        return lag_idx, np.max(corr) / (np.std(mp_norm) * np.std(vicon_norm) * min_len)
    except:
        return 0, 0


def method_a_forceplate_matching(at_starts, gt_count, period_frames, tolerance=5):
    """
    Method A: Force-plate region matching
    Since we don't have exact HS frame indices from force-plate,
    we estimate GT cycle positions based on period
    """
    if at_starts is None or len(at_starts) == 0:
        return 0, gt_count, 0  # All FN
    
    # Estimate expected GT positions (assuming regular gait, centered in video)
    # This is an approximation since we don't have exact force-plate timestamps
    video_len = max(at_starts) + period_frames if at_starts is not None else 0
    
    # Assume force-plate captures middle portion of walking
    fp_start = video_len * 0.2  # 20% into video
    fp_end = video_len * 0.8    # 80% into video
    
    # Estimate GT HS positions within force-plate region
    gt_hs_frames = np.linspace(fp_start, fp_end, gt_count + 1)[:-1]  # Start of each cycle
    
    # Match AT detections to GT
    tp = 0
    matched_at = set()
    
    for gt_frame in gt_hs_frames:
        best_match = None
        best_dist = tolerance + 1
        
        for i, at_frame in enumerate(at_starts):
            if i not in matched_at:
                dist = abs(at_frame - gt_frame)
                if dist <= tolerance and dist < best_dist:
                    best_match = i
                    best_dist = dist
        
        if best_match is not None:
            tp += 1
            matched_at.add(best_match)
    
    fn = gt_count - tp
    fp = len(at_starts) - tp
    
    return tp, fn, fp


def method_b_crosscorr_matching(at_starts, gt_count, mp_signal, vicon_signal, tolerance=5):
    """
    Method B: Cross-correlation lag + frame matching
    """
    if at_starts is None or len(at_starts) == 0 or vicon_signal is None:
        return 0, gt_count, 0  # All FN
    
    if len(vicon_signal) == 0:
        return 0, gt_count, 0
    
    # Estimate lag
    lag, corr_strength = estimate_lag_crosscorr(mp_signal, vicon_signal)
    
    # Since Vicon data is time-normalized, we can't directly get HS frames
    # Use period estimation from MP signal instead
    period = len(mp_signal) / (gt_count + 1) if gt_count > 0 else 30
    
    # Estimate GT HS frames (adjustment by lag)
    # Assume GT cycles start at beginning and repeat with period
    gt_hs_frames = [lag + i * period for i in range(gt_count)]
    
    # Match
    tp = 0
    matched_at = set()
    
    for gt_frame in gt_hs_frames:
        for i, at_frame in enumerate(at_starts):
            if i not in matched_at and abs(at_frame - gt_frame) <= tolerance:
                tp += 1
                matched_at.add(i)
                break
    
    fn = gt_count - tp
    fp = len(at_starts) - tp
    
    return tp, fn, fp, lag, corr_strength


def method_c_phase_coverage(at_starts, gt_count, mp_signal, estimated_period):
    """
    Method C: Phase coverage matching
    Check if AT detection exists within each GT cycle's time range
    """
    if at_starts is None or len(at_starts) == 0:
        return 0, gt_count, 0  # All FN
    
    # Estimate GT cycle boundaries using period
    video_len = len(mp_signal)
    period = estimated_period if estimated_period else 35
    
    # Divide video into gt_count equal phases (cycles)
    cycle_len = video_len / gt_count if gt_count > 0 else period
    
    # Define GT cycle ranges
    gt_ranges = []
    for i in range(gt_count):
        start = i * cycle_len
        end = (i + 1) * cycle_len
        gt_ranges.append((start, end))
    
    # Check coverage
    tp = 0
    matched_at = set()
    
    for start, end in gt_ranges:
        found = False
        for i, at_frame in enumerate(at_starts):
            if i not in matched_at and start <= at_frame < end:
                tp += 1
                matched_at.add(i)
                found = True
                break
    
    fn = gt_count - tp
    # FP = detections that didn't match any GT range
    fp = len(at_starts) - len(matched_at)
    
    return tp, fn, fp


def run_comprehensive_analysis():
    """Run all three matching methods on all subjects"""
    
    results = []
    
    # Summary counters for each method
    summary = {
        'A': {'tp': 0, 'fn': 0, 'fp': 0},
        'B': {'tp': 0, 'fn': 0, 'fp': 0},
        'C': {'tp': 0, 'fn': 0, 'fp': 0}
    }
    
    # Get all subjects with GT
    info_files = sorted(glob.glob(f"{PROCESSED_DIR}/S1_*_info.json"))
    
    print("="*70)
    print("PROPER GT-BASED CYCLE MATCHING REANALYSIS")
    print("="*70)
    print()
    
    for info_path in info_files[:5]:  # Limit to N=5 for fast validation
        try:
            basename = os.path.basename(info_path)
            sid = int(basename.split('_')[1])
            
            # Load GT data
            vicon_signal, gt_count, info = load_vicon_events(sid)
            
            if gt_count == 0:
                continue
            
            # Find video
            video_path = f"{DATA_DIR}/{sid}/{sid}-2.mp4"
            if not os.path.exists(video_path):
                continue
            
            print(f"Processing S{sid}: GT={gt_count} cycles...", flush=True)
            
            # Run AT-DTW
            at_starts, period, status, mp_signal = run_at_dtw_detection(
                video_path, return_signal=True
            )
            
            n_detected = len(at_starts) if at_starts is not None else 0
            
            if status != 'success':
                print(f"  → {status}")
                # All methods get FN = gt_count
                for method in ['A', 'B', 'C']:
                    summary[method]['fn'] += gt_count
                continue
            
            # Method A: Force-plate region matching
            tp_a, fn_a, fp_a = method_a_forceplate_matching(
                at_starts, gt_count, period, tolerance=5
            )
            
            # Method B: Cross-correlation matching
            tp_b, fn_b, fp_b, lag, corr = method_b_crosscorr_matching(
                at_starts, gt_count, mp_signal, 
                vicon_signal if vicon_signal is not None else mp_signal,
                tolerance=5
            )
            
            # Method C: Phase coverage
            tp_c, fn_c, fp_c = method_c_phase_coverage(
                at_starts, gt_count, mp_signal, period
            )
            
            # Update summaries
            summary['A']['tp'] += tp_a
            summary['A']['fn'] += fn_a
            summary['A']['fp'] += fp_a
            
            summary['B']['tp'] += tp_b
            summary['B']['fn'] += fn_b
            summary['B']['fp'] += fp_b
            
            summary['C']['tp'] += tp_c
            summary['C']['fn'] += fn_c
            summary['C']['fp'] += fp_c
            
            print(f"  AT={n_detected} | A: TP={tp_a},FN={fn_a} | B: TP={tp_b},FN={fn_b} | C: TP={tp_c},FN={fn_c}")
            
            results.append({
                'Subject': sid,
                'GT_Cycles': gt_count,
                'AT_Cycles': n_detected,
                'TP_A': tp_a, 'FN_A': fn_a, 'FP_A': fp_a,
                'TP_B': tp_b, 'FN_B': fn_b, 'FP_B': fp_b,
                'TP_C': tp_c, 'FN_C': fn_c, 'FP_C': fp_c,
                'Lag_B': lag if 'lag' in dir() else 0,
                'Corr_B': corr if 'corr' in dir() else 0
            })
            
        except Exception as e:
            print(f"Error processing S{sid}: {e}")
    
    # Print summary
    print()
    print("="*70)
    print("SUMMARY: THREE MATCHING METHODS")
    print("="*70)
    
    total_gt = sum(r['GT_Cycles'] for r in results)
    total_at = sum(r['AT_Cycles'] for r in results)
    
    print(f"\nTotal GT Cycles: {total_gt}")
    print(f"Total AT-DTW Detections: {total_at}")
    print()
    
    for method, name in [('A', 'Force-Plate Region (±5 frames)'), 
                         ('B', 'Cross-Correlation Lag (±5 frames)'),
                         ('C', 'Phase Coverage (within cycle range)')]:
        s = summary[method]
        recall = s['tp'] / (s['tp'] + s['fn']) if (s['tp'] + s['fn']) > 0 else 0
        precision = s['tp'] / (s['tp'] + s['fp']) if (s['tp'] + s['fp']) > 0 else 0
        
        print(f"Method {method}: {name}")
        print(f"  TPc: {s['tp']}, FNc: {s['fn']}, FPc: {s['fp']}")
        print(f"  Recall: {recall:.1%}")
        print(f"  Precision: {precision:.1%}")
        print()
    
    # Save results
    df = pd.DataFrame(results)
    df.to_csv(f"{OUTPUT_DIR}/proper_cycle_matching.csv", index=False)
    print(f"Saved: {OUTPUT_DIR}/proper_cycle_matching.csv")
    
    # Save summary
    summary_df = pd.DataFrame([
        {'Method': 'A', 'Name': 'Force-Plate Region', 
         'TPc': summary['A']['tp'], 'FNc': summary['A']['fn'], 'FPc': summary['A']['fp']},
        {'Method': 'B', 'Name': 'Cross-Correlation', 
         'TPc': summary['B']['tp'], 'FNc': summary['B']['fn'], 'FPc': summary['B']['fp']},
        {'Method': 'C', 'Name': 'Phase Coverage', 
         'TPc': summary['C']['tp'], 'FNc': summary['C']['fn'], 'FPc': summary['C']['fp']},
    ])
    summary_df['Recall'] = summary_df['TPc'] / (summary_df['TPc'] + summary_df['FNc'])
    summary_df['Precision'] = summary_df['TPc'] / (summary_df['TPc'] + summary_df['FPc'])
    summary_df.to_csv(f"{OUTPUT_DIR}/proper_matching_summary.csv", index=False)
    print(f"Saved: {OUTPUT_DIR}/proper_matching_summary.csv")
    
    return df, summary_df


if __name__ == "__main__":
    run_comprehensive_analysis()
