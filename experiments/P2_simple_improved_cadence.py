"""
P2: Simplified Improved Cadence Estimator

Analysis: RANSAC over-estimates because it picks consensus from short intervals
          (caused by strike over-detection)

New approach: Median + physiological constraints + outlier removal
"""

import numpy as np
from typing import List, Tuple, Dict
import json
from pathlib import Path


def estimate_cadence_simple_improved(
    strikes: List[int],
    fps: float,
    min_stride_time: float = 0.7,  # ~86 steps/min max
    max_stride_time: float = 2.0,   # ~60 steps/min min
    outlier_std_threshold: float = 2.0
) -> Tuple[float, Dict]:
    """
    Simple improved cadence estimator with physiological constraints.

    Key improvements over V4:
    1. Stricter minimum stride time (0.7s vs no constraint)
    2. Z-score based outlier removal (more principled than percentile)
    3. Uses median (robust to remaining outliers)

    Args:
        strikes: Heel strike frame indices
        fps: Frames per second
        min_stride_time: Minimum valid stride time (physiological)
        max_stride_time: Maximum valid stride time
        outlier_std_threshold: Z-score threshold for outlier removal

    Returns:
        cadence: steps/min
        diagnostics: dict
    """
    if fps <= 0 or len(strikes) < 3:
        return 0.0, {'error': 'insufficient_strikes', 'n_strikes': len(strikes)}

    # Calculate inter-strike intervals
    intervals = np.diff(strikes) / fps  # seconds

    # Step 1: Apply physiological constraints
    valid_mask = (intervals >= min_stride_time) & (intervals <= max_stride_time)
    intervals_valid = intervals[valid_mask]

    n_rejected_physiological = len(intervals) - len(intervals_valid)

    if len(intervals_valid) < 2:
        return 0.0, {
            'error': 'insufficient_valid_intervals',
            'n_total': len(intervals),
            'n_valid': len(intervals_valid)
        }

    # Step 2: Z-score outlier removal
    mean_interval = np.mean(intervals_valid)
    std_interval = np.std(intervals_valid)

    if std_interval > 0:
        z_scores = np.abs((intervals_valid - mean_interval) / std_interval)
        inlier_mask = z_scores < outlier_std_threshold
        intervals_clean = intervals_valid[inlier_mask]
    else:
        intervals_clean = intervals_valid

    n_rejected_outliers = len(intervals_valid) - len(intervals_clean)

    if len(intervals_clean) == 0:
        return 0.0, {'error': 'all_removed_as_outliers'}

    # Step 3: Robust estimation (median)
    stride_time = float(np.median(intervals_clean))
    cadence = 120.0 / stride_time

    # Diagnostics
    diagnostics = {
        'n_strikes': len(strikes),
        'n_intervals_total': len(intervals),
        'n_intervals_valid': len(intervals_valid),
        'n_intervals_clean': len(intervals_clean),
        'rejected_physiological': n_rejected_physiological,
        'rejected_outliers': n_rejected_outliers,
        'stride_time_median': stride_time,
        'stride_time_mean': float(np.mean(intervals_clean)),
        'stride_time_std': float(np.std(intervals_clean)),
        'cadence': cadence,
        'method': 'simple_improved'
    }

    return cadence, diagnostics


def test_comparison(subject_id: str = 'S1_01'):
    """Compare V4, RANSAC, and Simple Improved"""
    from mediapipe_csv_processor import MediaPipeCSVProcessor
    from P2_ransac_cadence import estimate_cadence_ransac, estimate_cadence_percentile

    print(f"Testing Cadence Methods on {subject_id}")
    print("="*80)

    # Load GT
    processed_root = Path('/data/gait/data/processed_new')
    with open(processed_root / f'{subject_id}_info.json') as f:
        info = json.load(f)

    patient = info.get('patient', {})
    gt_cad_left = patient.get('left', {}).get('cadence_steps_min')
    gt_cad_right = patient.get('right', {}).get('cadence_steps_min')
    gt_cad_avg = (gt_cad_left + gt_cad_right) / 2

    # Load MediaPipe
    subject_num = subject_id.split('_')[-1]
    data_root = Path('/data/gait/data')
    mp_csv = None
    for pattern in [f"{int(subject_num)}/*_side_pose_fps*.csv"]:
        csv_files = list(data_root.glob(pattern))
        if csv_files:
            mp_csv = csv_files[0]
            break

    processor = MediaPipeCSVProcessor()
    df_wide = processor.load_csv(mp_csv)
    df_angles = processor.calculate_joint_angles(df_wide)

    left_strikes = processor.detect_heel_strikes_fusion(df_angles, side='left', fps=30.0)
    right_strikes = processor.detect_heel_strikes_fusion(df_angles, side='right', fps=30.0)

    print(f"GT Cadence: {gt_cad_avg:.1f} steps/min")
    print()

    # Method 1: V4 Percentile
    v4_left, _ = estimate_cadence_percentile(left_strikes, 30.0)
    v4_right, _ = estimate_cadence_percentile(right_strikes, 30.0)
    v4_avg = (v4_left + v4_right) / 2
    v4_error = v4_avg - gt_cad_avg

    # Method 2: RANSAC
    ransac_left, _ = estimate_cadence_ransac(left_strikes, 30.0)
    ransac_right, _ = estimate_cadence_ransac(right_strikes, 30.0)
    ransac_avg = (ransac_left + ransac_right) / 2
    ransac_error = ransac_avg - gt_cad_avg

    # Method 3: Simple Improved
    simple_left, diag_l = estimate_cadence_simple_improved(left_strikes, 30.0)
    simple_right, diag_r = estimate_cadence_simple_improved(right_strikes, 30.0)
    simple_avg = (simple_left + simple_right) / 2
    simple_error = simple_avg - gt_cad_avg

    print(f"{'Method':<20s} {'Cadence':>10s} {'Error':>10s} {'|Error|':>10s}")
    print("-" * 55)
    print(f"{'V4 Percentile':<20s} {v4_avg:>10.1f} {v4_error:>+10.1f} {abs(v4_error):>10.1f}")
    print(f"{'RANSAC':<20s} {ransac_avg:>10.1f} {ransac_error:>+10.1f} {abs(ransac_error):>10.1f}")
    print(f"{'Simple Improved':<20s} {simple_avg:>10.1f} {simple_error:>+10.1f} {abs(simple_error):>10.1f}")
    print()

    # Winner
    errors = {
        'V4': abs(v4_error),
        'RANSAC': abs(ransac_error),
        'Simple': abs(simple_error)
    }
    winner = min(errors, key=errors.get)
    print(f"Best method: {winner} (|error| = {errors[winner]:.1f})")
    print()

    return {
        'subject_id': subject_id,
        'gt': gt_cad_avg,
        'v4': v4_avg,
        'ransac': ransac_avg,
        'simple': simple_avg,
        'v4_error': v4_error,
        'ransac_error': ransac_error,
        'simple_error': simple_error
    }


if __name__ == '__main__':
    print("="*80)
    print("P2: SIMPLE IMPROVED CADENCE ESTIMATOR")
    print("="*80)
    print()

    test_subjects = ['S1_01', 'S1_02', 'S1_03', 'S1_08', 'S1_09']
    results = []

    for subj_id in test_subjects:
        try:
            result = test_comparison(subj_id)
            results.append(result)
        except Exception as e:
            print(f"Error on {subj_id}: {e}\n")

    # Summary
    print("="*80)
    print("SUMMARY (n=5)")
    print("="*80)
    print()

    v4_errors = [r['v4_error'] for r in results]
    ransac_errors = [r['ransac_error'] for r in results]
    simple_errors = [r['simple_error'] for r in results]

    print(f"{'Subject':<10s} {'GT':>10s} {'V4':>10s} {'RANSAC':>10s} {'Simple':>10s}")
    print("-" * 55)
    for r in results:
        print(f"{r['subject_id']:<10s} {r['gt']:>10.1f} {r['v4']:>10.1f} {r['ransac']:>10.1f} {r['simple']:>10.1f}")

    print("-" * 55)
    print(f"{'MAE':<10s} {'':>10s} {np.mean(np.abs(v4_errors)):>10.1f} {np.mean(np.abs(ransac_errors)):>10.1f} {np.mean(np.abs(simple_errors)):>10.1f}")
    print()

    # Statistical test
    from scipy.stats import ttest_rel

    # V4 vs Simple
    t_v4_simple, p_v4_simple = ttest_rel(np.abs(v4_errors), np.abs(simple_errors))
    print(f"V4 vs Simple: t={t_v4_simple:.3f}, p={p_v4_simple:.4f} {'***' if p_v4_simple < 0.001 else '**' if p_v4_simple < 0.01 else '*' if p_v4_simple < 0.05 else 'n.s.'}")

    # RANSAC vs Simple
    t_ransac_simple, p_ransac_simple = ttest_rel(np.abs(ransac_errors), np.abs(simple_errors))
    print(f"RANSAC vs Simple: t={t_ransac_simple:.3f}, p={p_ransac_simple:.4f}")
    print()

    # Winner
    mae_v4 = np.mean(np.abs(v4_errors))
    mae_ransac = np.mean(np.abs(ransac_errors))
    mae_simple = np.mean(np.abs(simple_errors))

    if mae_simple < mae_v4 and mae_simple < mae_ransac:
        print("🏆 Winner: Simple Improved")
    elif mae_v4 < mae_simple and mae_v4 < mae_ransac:
        print("🏆 Winner: V4 Percentile (keep current)")
    else:
        print("🏆 Winner: RANSAC")

    # Save
    output = {
        'test_date': '2025-10-10',
        'n_subjects': len(results),
        'results': results,
        'summary': {
            'v4_mae': float(mae_v4),
            'ransac_mae': float(mae_ransac),
            'simple_mae': float(mae_simple)
        }
    }

    with open('P2_method_comparison_results.json', 'w') as f:
        json.dump(output, f, indent=2)

    print(f"✓ Results saved to P2_method_comparison_results.json")
