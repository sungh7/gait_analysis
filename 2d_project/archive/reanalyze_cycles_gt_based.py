#!/usr/bin/env python3
"""
GT-Based Cycle Reanalysis
Calculate TPc/FNc/FPc confusion matrix using GT-cycle as denominator.
"""

import pandas as pd
import numpy as np
import glob
import os
import sys
from scipy.signal import resample, find_peaks, correlate

sys.path.append("/data/gait/2d_project")
from sagittal_extractor_2d import MediaPipeSagittalExtractor
from self_driven_segmentation import derive_self_template, find_dtw_matches_euclidean

DATA_DIR = "/data/gait/data"
PROCESSED_DIR = "/data/gait/data/processed_new"
OUTPUT_DIR = "/data/gait/2d_project/rebuttal_experiments"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def get_gt_cycle_count(info_json_path):
    """Extract GT cycle count from Vicon info.json"""
    import json
    try:
        with open(info_json_path, 'r') as f:
            info = json.load(f)
        # Right strides from demographics
        r_strides = info.get('demographics', {}).get('right_strides', 0)
        l_strides = info.get('demographics', {}).get('left_strides', 0)
        return r_strides, l_strides
    except:
        return 0, 0

def run_at_dtw_detection(video_path):
    """Run AT-DTW and return detected cycle starts"""
    try:
        extractor = MediaPipeSagittalExtractor()
        lm, _ = extractor.extract_pose_landmarks(video_path)
        angles = extractor.calculate_joint_angles(lm)
        sig = angles['right_knee_angle'].fillna(method='bfill').fillna(method='ffill').values
        
        template, candidates = derive_self_template(sig)
        if template is None:
            return None, 'template_failure', 0
        
        # Find matches
        detected_starts = find_dtw_matches_euclidean(sig, template)
        n_detected = len(detected_starts)
        
        return detected_starts, 'success', n_detected
    except Exception as e:
        return None, f'error: {e}', 0

def calculate_confusion_matrix():
    """Calculate TPc, FNc, FPc for all subjects"""
    results = []
    
    # Iterate through all subjects with GT
    info_files = sorted(glob.glob(f"{PROCESSED_DIR}/S1_*_info.json"))
    
    total_gt_cycles = 0
    total_at_cycles = 0
    total_tp = 0
    total_fn = 0
    total_fp = 0
    template_failures = 0
    
    for info_path in info_files:
        try:
            # Extract subject ID
            basename = os.path.basename(info_path)
            sid = int(basename.split('_')[1])
            
            # Get GT cycle count
            r_strides, l_strides = get_gt_cycle_count(info_path)
            gt_cycles = r_strides  # Focus on right side
            
            if gt_cycles == 0:
                continue
            
            # Find corresponding video
            video_path = f"{DATA_DIR}/{sid}/{sid}-2.mp4"
            if not os.path.exists(video_path):
                continue
            
            # Run AT-DTW
            print(f"Processing S{sid}: GT={gt_cycles} cycles...")
            detected, status, n_detected = run_at_dtw_detection(video_path)
            
            if status == 'template_failure':
                template_failures += 1
                fn = gt_cycles
                tp = 0
                fp = 0
            elif detected is None:
                fn = gt_cycles
                tp = 0
                fp = 0
            else:
                # Simple matching: assume 1:1 if counts are close
                # More sophisticated matching would require temporal alignment
                tp = min(gt_cycles, n_detected)
                fn = max(0, gt_cycles - n_detected)
                fp = max(0, n_detected - gt_cycles)
            
            total_gt_cycles += gt_cycles
            total_at_cycles += n_detected
            total_tp += tp
            total_fn += fn
            total_fp += fp
            
            results.append({
                'Subject': sid,
                'GT_Cycles': gt_cycles,
                'AT_Cycles': n_detected,
                'TPc': tp,
                'FNc': fn,
                'FPc': fp,
                'Status': status
            })
            
        except Exception as e:
            print(f"Error processing {info_path}: {e}")
    
    # Summary
    print("\n" + "="*60)
    print("GT-BASED CYCLE CONFUSION MATRIX")
    print("="*60)
    print(f"Total Subjects Analyzed: {len(results)}")
    print(f"Template Failures: {template_failures}")
    print()
    print(f"Total GT Cycles (Denominator): {total_gt_cycles}")
    print(f"Total AT-DTW Cycles: {total_at_cycles}")
    print()
    print(f"TPc (True Positive Cycles): {total_tp}")
    print(f"FNc (False Negative / Missed): {total_fn}")
    print(f"FPc (False Positive / Phantom): {total_fp}")
    print()
    
    if (total_tp + total_fn) > 0:
        recall = total_tp / (total_tp + total_fn)
        print(f"Cycle Recall: {recall:.1%}")
    
    if (total_tp + total_fp) > 0:
        precision = total_tp / (total_tp + total_fp)
        print(f"Cycle Precision: {precision:.1%}")
    
    # Save results
    df = pd.DataFrame(results)
    df.to_csv(f"{OUTPUT_DIR}/gt_cycle_confusion_matrix.csv", index=False)
    print(f"\nSaved to: {OUTPUT_DIR}/gt_cycle_confusion_matrix.csv")
    
    return df

if __name__ == "__main__":
    calculate_confusion_matrix()
