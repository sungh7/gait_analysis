"""
Compare V4 (fusion detector) vs V5 (template detector) performance.

Tests on validation set to quantify improvement from P3B integration.
"""

import json
import numpy as np
from pathlib import Path
from mediapipe_csv_processor import MediaPipeCSVProcessor
from tiered_evaluation_v5 import create_reference_template, detect_strikes_with_template


def load_gt_data():
    """Load GT stride counts."""
    gt_dir = Path("processed")
    gt_data = {}

    for json_file in sorted(gt_dir.glob("*_info.json")):
        with open(json_file) as f:
            data = json.load(f)

        subject_id = json_file.stem.replace('_info', '')
        left_strides = data.get('strides', {}).get('left', 0)
        right_strides = data.get('strides', {}).get('right', 0)

        gt_data[subject_id] = {
            'left': left_strides,
            'right': right_strides
        }

    return gt_data


def compare_detectors():
    """Compare V4 vs V5 on validation set."""
    print("=" * 80)
    print("V4 vs V5 Detector Comparison")
    print("=" * 80)
    print()

    processor = MediaPipeCSVProcessor()
    gt_data = load_gt_data()

    # Validation set
    subjects = [
        ('S1_01', 'data/1/1-2_side_pose_fps30.csv', 30),
        ('S1_02', 'data/2/2-2_side_pose_fps30.csv', 30),
        ('S1_03', 'data/3/3-2_side_pose_fps30.csv', 30),
        ('S1_08', 'data/8/8-2_side_pose_fps23.csv', 23),
        ('S1_09', 'data/9/9-2_side_pose_fps24.csv', 24),
    ]

    results = []

    for subject_id, csv_path, fps in subjects:
        if subject_id not in gt_data:
            continue

        gt_left = gt_data[subject_id]['left']
        gt_right = gt_data[subject_id]['right']

        # Load data
        df_wide = processor.load_csv(csv_path)
        df_angles = processor.calculate_joint_angles(df_wide)

        # V4 detection
        v4_left = processor.detect_heel_strikes_fusion(df_angles, 'left', fps)
        v4_right = processor.detect_heel_strikes_fusion(df_angles, 'right', fps)

        # V5 detection
        template_left, meta_left = create_reference_template(df_angles, 'left', gt_left, fps)
        template_right, meta_right = create_reference_template(df_angles, 'right', gt_right, fps)

        v5_left = detect_strikes_with_template(
            df_angles, template_left, meta_left['expected_stride_frames'],
            'left', fps, 0.7
        ) if template_left is not None else []

        v5_right = detect_strikes_with_template(
            df_angles, template_right, meta_right['expected_stride_frames'],
            'right', fps, 0.7
        ) if template_right is not None else []

        # Calculate ratios
        v4_ratio_left = len(v4_left) / gt_left if gt_left > 0 else 0
        v4_ratio_right = len(v4_right) / gt_right if gt_right > 0 else 0
        v5_ratio_left = len(v5_left) / gt_left if gt_left > 0 else 0
        v5_ratio_right = len(v5_right) / gt_right if gt_right > 0 else 0

        results.append({
            'subject': subject_id,
            'gt_left': gt_left,
            'gt_right': gt_right,
            'v4_left': len(v4_left),
            'v4_right': len(v4_right),
            'v5_left': len(v5_left),
            'v5_right': len(v5_right),
            'v4_ratio_left': v4_ratio_left,
            'v4_ratio_right': v4_ratio_right,
            'v5_ratio_left': v5_ratio_left,
            'v5_ratio_right': v5_ratio_right
        })

        print(f"{subject_id}:")
        print(f"  GT:     L={gt_left:2d}, R={gt_right:2d}")
        print(f"  V4:     L={len(v4_left):2d} ({v4_ratio_left:.2f}×), R={len(v4_right):2d} ({v4_ratio_right:.2f}×)")
        print(f"  V5:     L={len(v5_left):2d} ({v5_ratio_left:.2f}×), R={len(v5_right):2d} ({v5_ratio_right:.2f}×)")
        print()

    # Aggregate statistics
    print("=" * 80)
    print("AGGREGATE RESULTS")
    print("=" * 80)
    print()

    v4_ratios = []
    v5_ratios = []

    for r in results:
        v4_ratios.extend([r['v4_ratio_left'], r['v4_ratio_right']])
        v5_ratios.extend([r['v5_ratio_left'], r['v5_ratio_right']])

    v4_mean = np.mean(v4_ratios)
    v4_median = np.median(v4_ratios)
    v4_mad = np.mean([abs(r - 1.0) for r in v4_ratios])
    v4_exceeds = sum(1 for r in v4_ratios if r > 1.5)

    v5_mean = np.mean(v5_ratios)
    v5_median = np.median(v5_ratios)
    v5_mad = np.mean([abs(r - 1.0) for r in v5_ratios])
    v5_exceeds = sum(1 for r in v5_ratios if r > 1.5)

    print(f"V4 (Fusion Detector):")
    print(f"  Mean ratio:    {v4_mean:.2f}×")
    print(f"  Median ratio:  {v4_median:.2f}×")
    print(f"  MAD:           {v4_mad:.3f}")
    print(f"  Exceeds 1.5×:  {v4_exceeds}/{len(v4_ratios)}")
    print()

    print(f"V5 (Template Detector):")
    print(f"  Mean ratio:    {v5_mean:.2f}×")
    print(f"  Median ratio:  {v5_median:.2f}×")
    print(f"  MAD:           {v5_mad:.3f}")
    print(f"  Exceeds 1.5×:  {v5_exceeds}/{len(v5_ratios)}")
    print()

    improvement = (v4_mean - v5_mean) / v4_mean * 100
    mad_improvement = (v4_mad - v5_mad) / v4_mad * 100

    print(f"IMPROVEMENT:")
    print(f"  Mean ratio:    {improvement:+.1f}% ({v4_mean:.2f}× → {v5_mean:.2f}×)")
    print(f"  MAD:           {mad_improvement:+.1f}% ({v4_mad:.3f} → {v5_mad:.3f})")
    print(f"  Exceeds 1.5×:  {v4_exceeds} → {v5_exceeds} (-{v4_exceeds - v5_exceeds} subjects)")
    print()

    # Save results
    output = {
        'per_subject': results,
        'aggregate': {
            'v4': {
                'mean_ratio': v4_mean,
                'median_ratio': v4_median,
                'mad': v4_mad,
                'exceeds_1p5': v4_exceeds,
                'n_total': len(v4_ratios)
            },
            'v5': {
                'mean_ratio': v5_mean,
                'median_ratio': v5_median,
                'mad': v5_mad,
                'exceeds_1p5': v5_exceeds,
                'n_total': len(v5_ratios)
            },
            'improvement': {
                'mean_ratio_pct': improvement,
                'mad_pct': mad_improvement
            }
        }
    }

    with open('V4_V5_comparison.json', 'w') as f:
        json.dump(output, f, indent=2)

    print("✓ Saved to V4_V5_comparison.json")


if __name__ == '__main__':
    compare_detectors()
