#!/usr/bin/env python3
"""
Off-Plate Validation: Generate Silver GT using kinematic event detection
Then calculate PPV for off-plate AT-RTM detections.

Method: Use MediaPipe heel landmark Y-coordinate for HS detection (Zeni-inspired)
        - HS = local maximum in heel Y (heel at highest point before foot contact)
        - Compare with AT-RTM detections to calculate PPV
"""

import numpy as np
import pandas as pd
import os
import sys
import json
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, savgol_filter

sys.path.append("/data/gait/2d_project")
from sagittal_extractor_2d import MediaPipeSagittalExtractor
from self_driven_segmentation import derive_self_template, find_dtw_matches_euclidean

DATA_DIR = "/data/gait/data"
OUTPUT_DIR = "/data/gait/2d_project"
PROCESSED_DIR = "/data/gait/data/processed_new"


def load_force_plate_events(sid):
    """Load force-plate validated HS events from info.json"""
    try:
        info_path = f"{PROCESSED_DIR}/S1_{sid:02d}_info.json"
        if not os.path.exists(info_path):
            return None
        with open(info_path, 'r') as f:
            info = json.load(f)
        r_strides = info.get('demographics', {}).get('right_strides', 0)
        return r_strides
    except:
        return None


def detect_hs_from_heel_y(landmarks_df, fps=30, min_period=0.7, max_period=1.8):
    """
    Kinematic HS detection using heel Y coordinate.
    
    HS occurs when the heel is at its lowest vertical position (peak in Y for image coords)
    OR when heel velocity changes from anterior to posterior.
    
    For sagittal view: heel Y is vertical position in image (higher Y = lower in real world)
    HS = local MAXIMUM in heel Y (foot at lowest point = ground contact)
    """
    # Get right heel Y coordinate (landmark 30 in MediaPipe)
    # In 2D sagittal: Y increases downward (image coordinates)
    heel_y = landmarks_df['right_heel_y'].values if 'right_heel_y' in landmarks_df.columns else None
    
    if heel_y is None:
        # Construct from landmark 30
        if '30_y' in landmarks_df.columns:
            heel_y = landmarks_df['30_y'].values
        elif 'RIGHT_HEEL_y' in landmarks_df.columns:
            heel_y = landmarks_df['RIGHT_HEEL_y'].values
        else:
            return None
    
    # Smooth the signal
    if len(heel_y) < 15:
        return None
    heel_y_smooth = savgol_filter(heel_y, window_length=11, polyorder=2)
    
    # Find peaks (HS = maximum heel Y = foot on ground)
    min_dist = int(fps * min_period)  # minimum distance between HS
    
    # Note: heel_y is normalized (0-1), so prominence should be low
    peaks, properties = find_peaks(heel_y_smooth, distance=min_dist, prominence=0.01)
    
    return peaks


def extract_landmarks_for_heel(video_path, max_frames=None):
    """Extract MediaPipe landmarks and return DataFrame with heel coordinates"""
    extractor = MediaPipeSagittalExtractor()
    landmarks, _ = extractor.extract_pose_landmarks(video_path, max_frames=max_frames)
    
    # landmarks is a DataFrame with columns like 'frame', '0_x', '0_y', '0_z', etc.
    # Right heel is landmark 30, Left heel is landmark 29
    if landmarks is None:
        return None
    
    # Create simplified dataframe
    result = pd.DataFrame()
    result['frame'] = landmarks['frame'] if 'frame' in landmarks.columns else range(len(landmarks))
    
    # Right heel
    if '30_y' in landmarks.columns:
        result['right_heel_y'] = landmarks['30_y']
        result['right_heel_x'] = landmarks['30_x']
    elif 'RIGHT_HEEL_y' in landmarks.columns:
        result['right_heel_y'] = landmarks['RIGHT_HEEL_y']
        result['right_heel_x'] = landmarks['RIGHT_HEEL_x']
    
    # Left heel
    if '29_y' in landmarks.columns:
        result['left_heel_y'] = landmarks['29_y']
        result['left_heel_x'] = landmarks['29_x']
    elif 'LEFT_HEEL_y' in landmarks.columns:
        result['left_heel_y'] = landmarks['LEFT_HEEL_y']
        result['left_heel_x'] = landmarks['LEFT_HEEL_x']
    
    return result, landmarks


def run_validation(sid, max_frames=None, tolerance=5):
    """
    Run off-plate validation for one subject.
    
    1. Run AT-RTM detection
    2. Generate Silver GT from kinematic HS detection
    3. Calculate PPV by matching
    """
    print(f"\n=== Processing Subject {sid} ===")
    
    video_path = f"{DATA_DIR}/{sid}/{sid}-2.mp4"
    if not os.path.exists(video_path):
        print(f"Video not found: {video_path}")
        return None
    
    # 1. Get AT-RTM detections (same as main validation)
    extractor = MediaPipeSagittalExtractor()
    try:
        landmarks, _ = extractor.extract_pose_landmarks(video_path, max_frames=max_frames)
    except Exception as e:
        print(f"Extraction failed: {e}")
        return None
    
    angles = extractor.calculate_joint_angles(landmarks)
    signal = angles['right_knee_angle'].fillna(method='bfill').fillna(method='ffill').values
    
    template, period = derive_self_template(signal)
    if template is None:
        print("Template derivation failed")
        return None
    
    at_rtm_detections = find_dtw_matches_euclidean(signal, template)
    print(f"  AT-RTM detected: {len(at_rtm_detections)} cycles")
    
    # 2. Get Silver GT from kinematic HS detection
    # Extract heel Y coordinate
    heel_df, full_lm = extract_landmarks_for_heel(video_path, max_frames=max_frames)
    
    if heel_df is None or 'right_heel_y' not in heel_df.columns:
        print("Could not extract heel landmarks")
        return None
    
    silver_gt = detect_hs_from_heel_y(heel_df)
    if silver_gt is None or len(silver_gt) == 0:
        print("Kinematic HS detection failed")
        return None
    print(f"  Silver GT (kinematic HS): {len(silver_gt)} events detected")
    
    # 3. Load Force-Plate GT count for reference
    fp_strides = load_force_plate_events(sid)
    print(f"  Force-Plate GT: {fp_strides} strides (reference)")
    
    # 4. Match AT-RTM detections to Silver GT
    # For each AT-RTM detection, check if there's a Silver GT within tolerance
    matched_atrtm = 0
    matched_silver = set()
    
    for det in at_rtm_detections:
        for i, gt in enumerate(silver_gt):
            if abs(det - gt) <= tolerance:
                matched_atrtm += 1
                matched_silver.add(i)
                break
    
    # PPV = matched / total AT-RTM detections
    ppv = matched_atrtm / len(at_rtm_detections) * 100 if len(at_rtm_detections) > 0 else 0
    
    # Recall = matched / total Silver GT
    recall = len(matched_silver) / len(silver_gt) * 100 if len(silver_gt) > 0 else 0
    
    # FP count (AT-RTM detections not matching any Silver GT)
    fp_count = len(at_rtm_detections) - matched_atrtm
    
    # Calculate FP rate per minute
    video_duration_min = len(signal) / 30 / 60  # assuming 30fps
    fp_per_min = fp_count / video_duration_min if video_duration_min > 0 else 0
    
    print(f"  Results:")
    print(f"    PPV (AT-RTM → Silver GT): {ppv:.1f}%")
    print(f"    Recall (Silver GT matched): {recall:.1f}%")
    print(f"    FP count: {fp_count}")
    print(f"    FP rate: {fp_per_min:.1f}/min")
    
    return {
        'sid': sid,
        'at_rtm_count': len(at_rtm_detections),
        'silver_gt_count': len(silver_gt),
        'fp_gt_count': fp_strides,
        'matched': matched_atrtm,
        'ppv': ppv,
        'recall': recall,
        'fp_count': fp_count,
        'fp_per_min': fp_per_min,
        'at_rtm_detections': at_rtm_detections,
        'silver_gt': silver_gt,
        'signal': signal,
        'heel_y': heel_df['right_heel_y'].values
    }


def visualize_validation(result, output_path):
    """Visualize AT-RTM vs Silver GT matching"""
    if result is None:
        return
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    
    # Top: Knee angle with AT-RTM detections
    ax1 = axes[0]
    ax1.plot(result['signal'], 'b-', linewidth=0.8, alpha=0.7, label='Knee Angle')
    for i, det in enumerate(result['at_rtm_detections']):
        label = 'AT-RTM Detection' if i == 0 else None
        ax1.axvline(det, color='blue', linestyle='--', alpha=0.5, label=label)
    ax1.set_ylabel('Knee Angle (°)')
    ax1.set_title(f"Subject {result['sid']}: AT-RTM Detections (N={result['at_rtm_count']})")
    ax1.legend()
    
    # Bottom: Heel Y with Silver GT
    ax2 = axes[1]
    ax2.plot(result['heel_y'], 'r-', linewidth=0.8, alpha=0.7, label='Heel Y (image)')
    for i, gt in enumerate(result['silver_gt']):
        label = 'Silver GT (Kinematic HS)' if i == 0 else None
        ax2.axvline(gt, color='green', linestyle='-', alpha=0.5, label=label)
    
    # Mark matched vs unmatched
    for det in result['at_rtm_detections']:
        matched = any(abs(det - gt) <= 5 for gt in result['silver_gt'])
        color = 'lime' if matched else 'red'
        ax2.scatter(det, result['heel_y'][min(det, len(result['heel_y'])-1)], 
                    c=color, s=50, zorder=5, marker='o')
    
    ax2.set_xlabel('Frame')
    ax2.set_ylabel('Heel Y Position')
    ax2.set_title(f"Silver GT (N={result['silver_gt_count']}) | PPV={result['ppv']:.1f}% | FP={result['fp_count']}")
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved: {output_path}")


def main():
    # Run on multiple subjects (limited for speed)
    test_sids = [1, 3, 8]  # Sample of subjects
    
    all_results = []
    
    for sid in test_sids:
        result = run_validation(sid, max_frames=800, tolerance=5)  # Limit frames
        if result:
            all_results.append(result)
            visualize_validation(result, f"{OUTPUT_DIR}/offplate_validation_S{sid}.png")
    
    # Summary
    if all_results:
        print("\n" + "="*60)
        print("OFF-PLATE VALIDATION SUMMARY (Silver GT)")
        print("="*60)
        
        total_atrtm = sum(r['at_rtm_count'] for r in all_results)
        total_matched = sum(r['matched'] for r in all_results)
        total_silver_gt = sum(r['silver_gt_count'] for r in all_results)
        total_fp = sum(r['fp_count'] for r in all_results)
        
        overall_ppv = total_matched / total_atrtm * 100 if total_atrtm > 0 else 0
        overall_recall = sum(1 for r in all_results for gt in r['silver_gt'] 
                            if any(abs(gt - det) <= 5 for det in r['at_rtm_detections']))
        
        print(f"\n{'Subject':<8} {'AT-RTM':<10} {'Silver GT':<12} {'Matched':<10} {'PPV':<8} {'FP/min':<8}")
        print("-"*60)
        for r in all_results:
            print(f"S{r['sid']:<7} {r['at_rtm_count']:<10} {r['silver_gt_count']:<12} "
                  f"{r['matched']:<10} {r['ppv']:.1f}%    {r['fp_per_min']:.1f}")
        
        print("-"*60)
        print(f"{'TOTAL':<8} {total_atrtm:<10} {total_silver_gt:<12} {total_matched:<10} "
              f"{overall_ppv:.1f}%    -")
        
        # Save summary CSV
        df = pd.DataFrame(all_results)
        df = df[['sid', 'at_rtm_count', 'silver_gt_count', 'matched', 'ppv', 'recall', 'fp_count', 'fp_per_min']]
        df.to_csv(f"{OUTPUT_DIR}/offplate_validation_summary.csv", index=False)
        print(f"\nSaved: {OUTPUT_DIR}/offplate_validation_summary.csv")


if __name__ == "__main__":
    main()
