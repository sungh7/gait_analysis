#!/usr/bin/env python3
"""
Left Knee AT-DTW Analysis
Compare left vs right knee angle for gait cycle segmentation
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
import json
from scipy.signal import resample, find_peaks, correlate

sys.path.append("/data/gait/2d_project")
from sagittal_extractor_2d import MediaPipeSagittalExtractor
from self_driven_segmentation import derive_self_template, find_dtw_matches_euclidean

DATA_DIR = "/data/gait/data"
PROCESSED_DIR = "/data/gait/data/processed_new"
OUTPUT_DIR = "/data/gait/2d_project"


def load_gt_info(sid):
    """Load GT stride counts"""
    info_path = f"{PROCESSED_DIR}/S1_{sid:02d}_info.json"
    if not os.path.exists(info_path):
        return None, None
    try:
        with open(info_path, 'r') as f:
            info = json.load(f)
        dem = info.get('demographics', {})
        return dem.get('right_strides', 0), dem.get('left_strides', 0)
    except:
        return None, None


def analyze_subject(sid, extractor):
    """Analyze both left and right knee for a single subject"""
    video_path = f"{DATA_DIR}/{sid}/{sid}-2.mp4"
    if not os.path.exists(video_path):
        return None
    
    print(f"  Extracting poses from video...", end=" ", flush=True)
    
    try:
        # Limit frames for faster processing
        landmarks, _ = extractor.extract_pose_landmarks(video_path, max_frames=500)
        angles = extractor.calculate_joint_angles(landmarks)
    except Exception as e:
        print(f"Error: {e}")
        return None
    
    print("Done")
    
    results = {}
    
    for side in ['right', 'left']:
        col_name = f'{side}_knee_angle'
        if col_name not in angles.columns:
            continue
        
        signal = angles[col_name].fillna(method='bfill').fillna(method='ffill').values
        
        # Derive self-template
        template_result = derive_self_template(signal)
        if template_result is None:
            results[side] = {
                'signal': signal,
                'template': None,
                'starts': [],
                'n_cycles': 0,
                'status': 'template_fail'
            }
            continue
        
        template, _ = template_result
        
        # Find matches
        starts = find_dtw_matches_euclidean(signal, template)
        
        results[side] = {
            'signal': signal,
            'template': template,
            'starts': starts,
            'n_cycles': len(starts),
            'status': 'success'
        }
    
    return results


def compare_left_right(results, sid, gt_right, gt_left, output_path):
    """Create comparison visualization"""
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    
    for idx, side in enumerate(['right', 'left']):
        if side not in results:
            continue
        
        data = results[side]
        gt_count = gt_right if side == 'right' else gt_left
        gt_count = gt_count if gt_count else 0
        
        # Signal with detected cycles
        ax1 = axes[idx, 0]
        signal = data['signal']
        ax1.plot(signal, 'gray', linewidth=1, alpha=0.8)
        
        for i, s in enumerate(data['starts']):
            color = 'red' if side == 'right' else 'blue'
            ax1.axvline(s, color=color, linewidth=1.5, alpha=0.6)
        
        side_label = 'Right (오른쪽)' if side == 'right' else 'Left (왼쪽)'
        ax1.set_title(f'{side_label} Knee: {data["n_cycles"]} cycles detected (GT≈{gt_count})', fontsize=12)
        ax1.set_xlabel('Frame')
        ax1.set_ylabel('Knee Flexion (°)')
        ax1.grid(alpha=0.3)
        
        # Template
        ax2 = axes[idx, 1]
        if data['template'] is not None:
            color = 'red' if side == 'right' else 'blue'
            ax2.plot(np.linspace(0, 100, len(data['template'])), data['template'], 
                    color=color, linewidth=2)
            ax2.set_title(f'{side_label} Auto-Template', fontsize=12)
            ax2.set_xlabel('Gait Cycle (%)')
            ax2.set_ylabel('Knee Flexion (°)')
            ax2.axhline(np.min(data['template']), color='green', linestyle='--', 
                       alpha=0.5, label='HS (0%)')
            ax2.legend()
        else:
            ax2.text(0.5, 0.5, 'Template Failed', ha='center', va='center', fontsize=14)
        ax2.grid(alpha=0.3)
    
    plt.suptitle(f'Subject {sid}: Left vs Right Knee Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def main():
    print("=" * 60)
    print("LEFT vs RIGHT KNEE AT-DTW ANALYSIS")
    print("=" * 60)
    
    extractor = MediaPipeSagittalExtractor()
    
    # Test subjects - single subject for speed
    test_sids = [1]
    
    all_results = []
    
    for sid in test_sids:
        print(f"\nSubject {sid}:")
        
        # Load GT info
        gt_right, gt_left = load_gt_info(sid)
        print(f"  GT strides - Right: {gt_right}, Left: {gt_left}")
        
        # Analyze
        results = analyze_subject(sid, extractor)
        
        if results is None:
            continue
        
        # Print summary
        for side in ['right', 'left']:
            if side in results:
                print(f"  {side.capitalize()} Knee: {results[side]['n_cycles']} cycles, "
                      f"Status: {results[side]['status']}")
        
        # Store for summary
        all_results.append({
            'Subject': sid,
            'Right_Cycles': results.get('right', {}).get('n_cycles', 0),
            'Left_Cycles': results.get('left', {}).get('n_cycles', 0),
            'GT_Right': gt_right or 0,
            'GT_Left': gt_left or 0
        })
        
        # Visualization
        output_path = f"{OUTPUT_DIR}/left_vs_right_S{sid}.png"
        compare_left_right(results, sid, gt_right, gt_left, output_path)
    
    # Summary table
    if all_results:
        df = pd.DataFrame(all_results)
        print("\n" + "=" * 60)
        print("SUMMARY: LEFT vs RIGHT KNEE DETECTION")
        print("=" * 60)
        print(df.to_string(index=False))
        
        # Save
        df.to_csv(f"{OUTPUT_DIR}/left_vs_right_comparison.csv", index=False)
        print(f"\nSaved: {OUTPUT_DIR}/left_vs_right_comparison.csv")
    
    print("\n✅ Analysis complete!")


if __name__ == "__main__":
    main()
