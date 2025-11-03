"""
GT Label Verification Tool

For problematic subjects (S1_27, S1_11, S1_16), this tool:
1. Loads original video
2. Loads MediaPipe pose data
3. Visualizes left/right foot predictions
4. Compares with GT labels
5. Allows manual verification

Goal: Determine if GT labels are correct or swapped
"""

import cv2
import pandas as pd
import numpy as np
import json
from pathlib import Path
import matplotlib.pyplot as plt


def load_pose_data(subject_id: str, data_root: Path) -> pd.DataFrame:
    """Load MediaPipe pose data for a subject"""

    # Find the pose CSV file
    subject_num = subject_id.replace('S1_', '')
    pose_file = data_root / subject_num / f"{subject_num}-2_side_pose_fps30.csv"

    if not pose_file.exists():
        # Try alternative naming
        pose_file = data_root / subject_num / f"{subject_num}-2_side_pose.csv"

    if not pose_file.exists():
        raise FileNotFoundError(f"Pose file not found for {subject_id}")

    # Load CSV (assuming long format)
    df = pd.read_csv(pose_file)

    return df


def load_gt_data(subject_id: str, processed_root: Path) -> dict:
    """Load ground truth data"""

    subject_num = subject_id.replace('S1_', '')
    info_file = processed_root / subject_num / "info.json"

    if not info_file.exists():
        raise FileNotFoundError(f"GT file not found for {subject_id}")

    with open(info_file, 'r') as f:
        info = json.load(f)

    return info


def analyze_foot_positions(df: pd.DataFrame) -> dict:
    """
    Analyze foot positions from MediaPipe data.

    In sagittal (side) view:
    - X axis: forward/backward
    - Y axis: up/down
    - Z axis: depth (distance from camera)

    The foot CLOSER to camera should have smaller Z value.
    """

    # Get heel positions
    left_heel_data = df[df['position'] == 'left_heel']
    right_heel_data = df[df['position'] == 'right_heel']

    if len(left_heel_data) == 0 or len(right_heel_data) == 0:
        return None

    # Average Z (depth) values
    left_z_mean = left_heel_data['z'].mean()
    right_z_mean = right_heel_data['z'].mean()

    # Average X (forward) range
    left_x_range = left_heel_data['x'].max() - left_heel_data['x'].min()
    right_x_range = right_heel_data['x'].max() - right_heel_data['x'].min()

    # Visibility
    left_vis_mean = left_heel_data['visibility'].mean()
    right_vis_mean = right_heel_data['visibility'].mean()

    analysis = {
        'left_heel': {
            'z_mean': left_z_mean,
            'x_range': left_x_range,
            'visibility': left_vis_mean,
            'n_frames': len(left_heel_data)
        },
        'right_heel': {
            'z_mean': right_z_mean,
            'x_range': right_x_range,
            'visibility': right_vis_mean,
            'n_frames': len(right_heel_data)
        },
        'interpretation': {
            'closer_to_camera': 'left' if left_z_mean < right_z_mean else 'right',
            'more_visible': 'left' if left_vis_mean > right_vis_mean else 'right',
            'z_difference': abs(left_z_mean - right_z_mean)
        }
    }

    return analysis


def verify_subject(subject_id: str,
                   data_root: Path = Path("/data/gait/data"),
                   processed_root: Path = Path("/data/gait/data/processed_new")):
    """
    Main verification function for a subject.
    """

    print("="*80)
    print(f"GT VERIFICATION: {subject_id}")
    print("="*80)

    try:
        # Load data
        print("\n[1/3] Loading pose data...")
        df_pose = load_pose_data(subject_id, data_root)
        print(f"  ✓ Loaded {len(df_pose)} pose records")

        print("\n[2/3] Loading ground truth...")
        gt_info = load_gt_data(subject_id, processed_root)

        gt_left_step = gt_info['patient']['left']['step_length_cm']
        gt_right_step = gt_info['patient']['right']['step_length_cm']
        gt_left_stride = gt_info['patient']['left']['stride_length_cm']
        gt_right_stride = gt_info['patient']['right']['stride_length_cm']

        print(f"  Ground Truth:")
        print(f"    Left  step: {gt_left_step:.2f} cm, stride: {gt_left_stride:.2f} cm")
        print(f"    Right step: {gt_right_step:.2f} cm, stride: {gt_right_stride:.2f} cm")
        print(f"    Bilateral symmetry: {abs(gt_left_step - gt_right_step):.2f} cm difference ({abs(gt_left_step - gt_right_step) / ((gt_left_step + gt_right_step)/2) * 100:.1f}%)")

        print("\n[3/3] Analyzing MediaPipe foot positions...")
        foot_analysis = analyze_foot_positions(df_pose)

        if foot_analysis:
            print(f"\n  MediaPipe Analysis:")
            print(f"    Left heel:")
            print(f"      Z (depth):     {foot_analysis['left_heel']['z_mean']:.3f}")
            print(f"      X range:       {foot_analysis['left_heel']['x_range']:.3f}")
            print(f"      Visibility:    {foot_analysis['left_heel']['visibility']:.2f}")

            print(f"    Right heel:")
            print(f"      Z (depth):     {foot_analysis['right_heel']['z_mean']:.3f}")
            print(f"      X range:       {foot_analysis['right_heel']['x_range']:.3f}")
            print(f"      Visibility:    {foot_analysis['right_heel']['visibility']:.2f}")

            print(f"\n  Interpretation:")
            print(f"    Closer to camera:  {foot_analysis['interpretation']['closer_to_camera'].upper()}")
            print(f"    Z difference:      {foot_analysis['interpretation']['z_difference']:.3f}")
            print(f"    More visible:      {foot_analysis['interpretation']['more_visible'].upper()}")

        # Load V5.3.3 predictions for comparison
        print("\n[4/3] Loading V5.3.3 predictions...")
        with open('/data/gait/tiered_evaluation_report_v533.json', 'r') as f:
            v533 = json.load(f)

        if subject_id in v533['subjects']:
            subj = v533['subjects'][subject_id]
            pred = subj['temporal']['prediction']

            pred_left = pred['step_length_cm']['left']
            pred_right = pred['step_length_cm']['right']

            print(f"  V5.3.3 Predictions:")
            print(f"    Left:  {pred_left:.2f} cm (Error: {abs(pred_left - gt_left_step):.2f} cm, {abs(pred_left - gt_left_step)/gt_left_step*100:.1f}%)")
            print(f"    Right: {pred_right:.2f} cm (Error: {abs(pred_right - gt_right_step):.2f} cm, {abs(pred_right - gt_right_step)/gt_right_step*100:.1f}%)")

            # Check if swapping would improve
            swap_left_error = abs(pred_left - gt_right_step)
            swap_right_error = abs(pred_right - gt_left_step)

            current_total_error = abs(pred_left - gt_left_step) + abs(pred_right - gt_right_step)
            swap_total_error = swap_left_error + swap_right_error

            print(f"\n  Swap Analysis:")
            print(f"    Current total error:  {current_total_error:.2f} cm")
            print(f"    After swap error:     {swap_total_error:.2f} cm")

            if swap_total_error < current_total_error * 0.5:
                print(f"    ⚠️  SWAP WOULD REDUCE ERROR BY {(1 - swap_total_error/current_total_error)*100:.0f}%!")
                print(f"    Recommendation: GT LABELS LIKELY SWAPPED")
            elif swap_total_error < current_total_error:
                print(f"    → Swap would reduce error by {(1 - swap_total_error/current_total_error)*100:.0f}%")
                print(f"    Recommendation: Possible GT label issue")
            else:
                print(f"    ✓ Current labels are correct")

        print(f"\n{'='*80}")
        print(f"VERDICT FOR {subject_id}")
        print(f"{'='*80}")

        # Synthesize all evidence
        evidence = []

        # Evidence 1: Bilateral GT symmetry
        gt_asymmetry = abs(gt_left_step - gt_right_step) / ((gt_left_step + gt_right_step)/2)
        if gt_asymmetry < 0.05:  # <5% difference
            evidence.append("✓ GT shows bilateral symmetry (healthy gait)")
        else:
            evidence.append(f"⚠️ GT shows {gt_asymmetry*100:.1f}% asymmetry (unusual for healthy)")

        # Evidence 2: Unilateral accuracy
        left_error_pct = abs(pred_left - gt_left_step) / gt_left_step
        right_error_pct = abs(pred_right - gt_right_step) / gt_right_step

        if left_error_pct < 0.05 and right_error_pct > 0.2:
            evidence.append(f"⚠️ Left prediction perfect ({left_error_pct*100:.1f}%), Right terrible ({right_error_pct*100:.1f}%)")
            evidence.append("   → Suggests GT label swap")
        elif right_error_pct < 0.05 and left_error_pct > 0.2:
            evidence.append(f"⚠️ Right prediction perfect ({right_error_pct*100:.1f}%), Left terrible ({left_error_pct*100:.1f}%)")
            evidence.append("   → Suggests GT label swap")

        # Evidence 3: Swap improvement
        if swap_total_error < current_total_error * 0.5:
            evidence.append(f"⚠️ Swapping labels would reduce error by {(1-swap_total_error/current_total_error)*100:.0f}%")
            evidence.append("   → Strong evidence for GT label swap")

        print("\nEvidence:")
        for e in evidence:
            print(f"  {e}")

        # Final recommendation
        print(f"\nRecommendation:")
        if len([e for e in evidence if 'GT label swap' in e]) >= 2:
            print(f"  🚨 HIGH CONFIDENCE: GT labels are likely SWAPPED")
            print(f"  Action: Manually verify video and correct GT if needed")
            return "SWAP_LIKELY"
        elif len([e for e in evidence if 'GT label swap' in e]) >= 1:
            print(f"  ⚠️  MEDIUM CONFIDENCE: GT labels might be swapped")
            print(f"  Action: Manual verification recommended")
            return "SWAP_POSSIBLE"
        else:
            print(f"  ✓ Labels appear correct")
            return "LABELS_OK"

    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return "ERROR"


def verify_all_problems():
    """Verify all problematic subjects"""

    problem_subjects = ['S1_27', 'S1_11', 'S1_16']

    results = {}

    for subject_id in problem_subjects:
        result = verify_subject(subject_id)
        results[subject_id] = result
        print("\n" + "="*80 + "\n")

    # Summary
    print("="*80)
    print("VERIFICATION SUMMARY")
    print("="*80)

    for subject_id, result in results.items():
        print(f"  {subject_id}: {result}")

    swap_likely = sum(1 for r in results.values() if r == 'SWAP_LIKELY')
    swap_possible = sum(1 for r in results.values() if r == 'SWAP_POSSIBLE')

    print(f"\n  High confidence swap: {swap_likely}/3")
    print(f"  Medium confidence swap: {swap_possible}/3")
    print(f"  Total requiring attention: {swap_likely + swap_possible}/3")

    print("\n" + "="*80)


if __name__ == "__main__":
    verify_all_problems()
