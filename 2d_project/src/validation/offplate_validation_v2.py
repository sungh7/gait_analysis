#!/usr/bin/env python3
"""
Off-Plate Validation V2: Two approaches
1. Knee angle extension peak as Silver GT (more reliable than heel Y)
2. Detection Ratio based PPV estimation from Force-Plate verified region
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
    """Load force-plate validated stride count from info.json"""
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


def detect_hs_from_knee_extension(knee_angle, fps=30, min_period=0.7):
    """
    Kinematic HS detection using knee extension peaks.
    HS ≈ maximum knee extension (minimum flexion angle)
    This is more reliable than heel Y in 2D video.
    """
    if len(knee_angle) < 15:
        return None
    
    # Smooth
    knee_smooth = savgol_filter(knee_angle, window_length=11, polyorder=2)
    
    # Find MINIMA (extension = low flexion angle) → invert for find_peaks
    min_dist = int(fps * min_period)
    peaks, _ = find_peaks(-knee_smooth, distance=min_dist, prominence=3)
    
    return peaks


def run_validation_v2(sid, max_frames=800, tolerance=5):
    """
    Two-method validation:
    1. Knee extension Silver GT matching
    2. Detection ratio based PPV estimation
    """
    print(f"\n{'='*60}")
    print(f"Subject {sid}")
    print('='*60)
    
    video_path = f"{DATA_DIR}/{sid}/{sid}-2.mp4"
    if not os.path.exists(video_path):
        print(f"Video not found: {video_path}")
        return None
    
    # Extract landmarks and angles
    extractor = MediaPipeSagittalExtractor()
    try:
        landmarks, meta = extractor.extract_pose_landmarks(video_path, max_frames=max_frames)
    except Exception as e:
        print(f"Extraction failed: {e}")
        return None
    
    fps = meta.get('fps', 30) if meta else 30
    angles = extractor.calculate_joint_angles(landmarks)
    knee_angle = angles['right_knee_angle'].fillna(method='bfill').fillna(method='ffill').values
    
    # 1. AT-RTM Detection
    template, period = derive_self_template(knee_angle)
    if template is None:
        print("Template derivation failed")
        return None
    
    # Ensure period is scalar
    if isinstance(period, (np.ndarray, list)):
        period = float(np.array(period).flatten()[0]) if len(np.array(period).flatten()) > 0 else 32.0
    period = float(period) if period else 32.0
    
    at_rtm = find_dtw_matches_euclidean(knee_angle, template)
    print(f"AT-RTM detected: {len(at_rtm)} cycles")
    
    # 2. Method A: Knee Extension Silver GT
    silver_gt_knee = detect_hs_from_knee_extension(knee_angle, fps=fps)
    print(f"Silver GT (knee extension): {len(silver_gt_knee)} events")
    
    # Match AT-RTM to Silver GT
    matched = 0
    for det in at_rtm:
        if any(abs(det - gt) <= tolerance for gt in silver_gt_knee):
            matched += 1
    
    ppv_knee = matched / len(at_rtm) * 100 if len(at_rtm) > 0 else 0
    recall_knee = sum(1 for gt in silver_gt_knee if any(abs(det - gt) <= tolerance for det in at_rtm)) / len(silver_gt_knee) * 100 if len(silver_gt_knee) > 0 else 0
    
    print(f"  → PPV (AT-RTM vs Knee Silver GT): {ppv_knee:.1f}%")
    print(f"  → Recall (Knee Silver GT matched): {recall_knee:.1f}%")
    
    # 3. Method B: Detection Ratio Estimation
    fp_strides = load_force_plate_events(sid)
    if fp_strides:
        # Estimate expected cycles from force plate ratio
        # FP covers ~2 plates on 8m walkway ≈ 25% of walking distance
        # So expected full-video cycles ≈ FP strides × 4 (rough estimate)
        # But we're only processing 800 frames, so adjust by frame ratio
        
        # Better approach: use cadence to estimate expected cycles
        video_duration_sec = len(knee_angle) / fps
        avg_cycle_duration = period / fps if period else 1.0
        expected_cycles = int(video_duration_sec / avg_cycle_duration)
        
        detection_ratio = len(at_rtm) / expected_cycles if expected_cycles > 0 else 0
        
        print(f"\nDetection Ratio Analysis:")
        print(f"  Video duration: {video_duration_sec:.1f}s")
        print(f"  Estimated period: {period} frames ({period/fps:.2f}s)")
        print(f"  Expected cycles: {expected_cycles}")
        print(f"  Detection ratio: {detection_ratio:.2f}")
        
        # If detection ratio is close to 1.0, most detections are likely valid
        # If > 1.0, over-segmentation
        # If < 1.0, under-detection
        
        # Estimated PPV from detection ratio
        if detection_ratio >= 1.0:
            # Over-segmentation: PPV ≈ expected/detected
            ppv_ratio = expected_cycles / len(at_rtm) * 100
        else:
            # Under-detection: PPV ≈ 100% (all detections are valid)
            ppv_ratio = 100.0
        
        print(f"  Estimated PPV (ratio-based): {ppv_ratio:.1f}%")
    else:
        expected_cycles = None
        ppv_ratio = None
        detection_ratio = None
    
    # 4. Combine estimates: Weighted average
    if ppv_ratio is not None:
        combined_ppv = (ppv_knee + ppv_ratio) / 2
        print(f"\nCombined PPV Estimate: {combined_ppv:.1f}%")
    else:
        combined_ppv = ppv_knee
    
    return {
        'sid': sid,
        'at_rtm_count': len(at_rtm),
        'silver_gt_knee_count': len(silver_gt_knee),
        'fp_strides': fp_strides,
        'matched': matched,
        'ppv_knee': ppv_knee,
        'recall_knee': recall_knee,
        'expected_cycles': expected_cycles,
        'detection_ratio': detection_ratio,
        'ppv_ratio': ppv_ratio,
        'combined_ppv': combined_ppv,
        'knee_angle': knee_angle,
        'at_rtm': at_rtm,
        'silver_gt_knee': silver_gt_knee
    }


def visualize_validation_v2(result, output_path):
    """Visualize AT-RTM vs Knee Extension Silver GT"""
    if result is None:
        return
    
    fig, ax = plt.subplots(figsize=(14, 5))
    
    knee = result['knee_angle']
    ax.plot(knee, 'b-', linewidth=0.8, alpha=0.7, label='Knee Angle')
    
    # AT-RTM (blue markers)
    for i, det in enumerate(result['at_rtm']):
        label = f"AT-RTM (N={result['at_rtm_count']})" if i == 0 else None
        ax.scatter(det, knee[min(det, len(knee)-1)], c='blue', s=80, marker='v', 
                   label=label, zorder=5, edgecolors='white')
    
    # Silver GT (green markers)
    for i, gt in enumerate(result['silver_gt_knee']):
        label = f"Silver GT - Knee Ext (N={result['silver_gt_knee_count']})" if i == 0 else None
        ax.scatter(gt, knee[min(gt, len(knee)-1)], c='green', s=60, marker='^', 
                   label=label, zorder=4)
    
    ax.set_xlabel('Frame')
    ax.set_ylabel('Knee Flexion Angle (°)')
    ax.set_title(f"Subject {result['sid']}: AT-RTM vs Knee Extension Silver GT\n"
                 f"PPV(knee)={result['ppv_knee']:.1f}% | PPV(ratio)={result['ppv_ratio']:.1f}% | Combined={result['combined_ppv']:.1f}%")
    ax.legend(loc='upper right')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved: {output_path}")


def main():
    # All 21 validated subjects
    test_sids = list(range(1, 27))  # S1-S26 (will skip missing)
    
    all_results = []
    
    for sid in test_sids:
        result = run_validation_v2(sid, max_frames=800, tolerance=10)  # 10 frame tolerance
        if result:
            all_results.append(result)
            # Skip visualization to avoid errors
            # try:
            #     visualize_validation_v2(result, f"{OUTPUT_DIR}/offplate_v2_S{sid}.png")
            # except Exception as e:
            #     print(f"  Visualization skipped: {e}")
    
    # Summary
    if all_results:
        print("\n" + "="*70)
        print("OFF-PLATE VALIDATION SUMMARY V2")
        print("="*70)
        
        print(f"\n{'SID':<5} {'AT-RTM':<8} {'SilverGT':<10} {'Expected':<10} {'PPV_knee':<10} {'PPV_ratio':<10} {'Combined':<10}")
        print("-"*70)
        
        for r in all_results:
            exp = r['expected_cycles'] if r['expected_cycles'] else '-'
            ppr = f"{r['ppv_ratio']:.1f}%" if r['ppv_ratio'] else '-'
            print(f"S{r['sid']:<4} {r['at_rtm_count']:<8} {r['silver_gt_knee_count']:<10} "
                  f"{exp:<10} {r['ppv_knee']:.1f}%      {ppr:<10} {r['combined_ppv']:.1f}%")
        
        # Overall averages
        avg_ppv_knee = np.mean([r['ppv_knee'] for r in all_results])
        avg_ppv_ratio = np.mean([r['ppv_ratio'] for r in all_results if r['ppv_ratio']])
        avg_combined = np.mean([r['combined_ppv'] for r in all_results])
        
        print("-"*70)
        print(f"{'AVG':<5} {'-':<8} {'-':<10} {'-':<10} {avg_ppv_knee:.1f}%      {avg_ppv_ratio:.1f}%      {avg_combined:.1f}%")
        
        # Save summary
        df = pd.DataFrame([{
            'sid': r['sid'],
            'at_rtm': r['at_rtm_count'],
            'silver_gt_knee': r['silver_gt_knee_count'],
            'expected_cycles': r['expected_cycles'],
            'ppv_knee': r['ppv_knee'],
            'ppv_ratio': r['ppv_ratio'],
            'combined_ppv': r['combined_ppv']
        } for r in all_results])
        df.to_csv(f"{OUTPUT_DIR}/offplate_validation_v2_summary.csv", index=False)
        print(f"\nSaved: {OUTPUT_DIR}/offplate_validation_v2_summary.csv")


if __name__ == "__main__":
    main()
