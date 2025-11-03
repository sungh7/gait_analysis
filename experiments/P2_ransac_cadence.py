"""
Phase 2: RANSAC-based Cadence Estimator

Replaces heuristic percentile-trimming with robust RANSAC consensus.
"""

import numpy as np
from typing import List, Tuple, Dict, Optional


def estimate_cadence_ransac(
    strikes: List[int],
    fps: float,
    min_stride_time: float = 0.6,
    max_stride_time: float = 2.5,
    n_iterations: int = 500,
    inlier_threshold: float = 0.3
) -> Tuple[float, Dict]:
    """
    Estimate cadence using RANSAC to find consensus stride time.

    Args:
        strikes: List of heel strike frame indices
        fps: Frames per second
        min_stride_time: Minimum physiologically valid stride time (sec)
        max_stride_time: Maximum physiologically valid stride time (sec)
        n_iterations: Number of RANSAC iterations
        inlier_threshold: Inlier threshold (seconds)

    Returns:
        cadence: Estimated cadence in steps/min
        diagnostics: Dict with intermediate values
    """
    if fps <= 0 or len(strikes) < 3:
        return 0.0, {'error': 'insufficient_data', 'n_strikes': len(strikes)}

    # Calculate inter-strike intervals (stride times)
    intervals = np.diff(strikes) / fps  # Convert frames to seconds

    # Filter out invalid intervals
    valid_mask = (intervals >= min_stride_time) & (intervals <= max_stride_time)
    intervals = intervals[valid_mask]

    if len(intervals) < 2:
        return 0.0, {'error': 'no_valid_intervals', 'n_intervals': len(intervals)}

    # RANSAC: Find consensus stride time
    best_inliers = []
    best_model = None

    for _ in range(n_iterations):
        # Randomly sample one interval as hypothesis
        sample_idx = np.random.randint(0, len(intervals))
        sample_value = intervals[sample_idx]

        # Find inliers (intervals close to hypothesis)
        inliers_mask = np.abs(intervals - sample_value) < inlier_threshold
        inliers = intervals[inliers_mask]

        # Update best model if more inliers found
        if len(inliers) > len(best_inliers):
            best_inliers = inliers
            best_model = sample_value

    if len(best_inliers) == 0:
        return 0.0, {'error': 'no_consensus', 'n_iterations': n_iterations}

    # Estimate stride time from inliers (use median for robustness)
    stride_time = float(np.median(best_inliers))

    # Convert to cadence (steps/min)
    # Note: One stride = two steps (left + right)
    cadence = 120.0 / stride_time

    # Diagnostics
    diagnostics = {
        'n_strikes': len(strikes),
        'n_intervals_total': len(np.diff(strikes)),
        'n_intervals_valid': len(intervals),
        'n_inliers': len(best_inliers),
        'inlier_ratio': len(best_inliers) / len(intervals) if len(intervals) > 0 else 0,
        'stride_time_median': stride_time,
        'stride_time_mean': float(np.mean(best_inliers)),
        'stride_time_std': float(np.std(best_inliers)),
        'cadence': cadence,
        'method': 'ransac',
        'outliers_rejected': len(intervals) - len(best_inliers)
    }

    return cadence, diagnostics


def estimate_cadence_percentile(
    strikes: List[int],
    fps: float
) -> Tuple[float, Dict]:
    """
    Baseline method (V3/V4 style) for comparison.
    """
    if fps <= 0 or len(strikes) < 2:
        return 0.0, {'error': 'insufficient_strikes'}

    intervals = np.diff(strikes) / fps
    intervals = intervals[intervals > 0]

    if intervals.size == 0:
        return 0.0, {'error': 'no_valid_intervals'}

    # Percentile trimming
    median = np.median(intervals)
    lower = median * 0.5
    upper = median * 1.75
    trimmed = intervals[(intervals >= lower) & (intervals <= upper)]

    if trimmed.size >= 4:
        intervals = trimmed

    stride_time = float(np.percentile(intervals, 70))
    cadence = 120.0 / stride_time if stride_time > 0 else 0.0

    diagnostics = {
        'n_strikes': len(strikes),
        'n_intervals': len(intervals),
        'stride_time_p70': stride_time,
        'cadence': cadence,
        'method': 'percentile'
    }

    return cadence, diagnostics


def test_on_subject(subject_id: str = 'S1_01'):
    """Test RANSAC vs Percentile on a single subject"""
    from pathlib import Path
    import json
    import pandas as pd
    from mediapipe_csv_processor import MediaPipeCSVProcessor

    print(f"Testing P2 Cadence Methods on {subject_id}")
    print("="*80)

    # Load data
    processed_root = Path('/data/gait/data/processed_new')
    info_path = processed_root / f'{subject_id}_info.json'

    with open(info_path) as f:
        info = json.load(f)

    # Find MediaPipe CSV
    subject_num = subject_id.split('_')[-1]
    data_root = Path('/data/gait/data')

    mp_csv = None
    for pattern in [f"{int(subject_num)}/*_side_pose_fps*.csv", f"{subject_num}/*_side_pose_fps*.csv"]:
        csv_files = list(data_root.glob(pattern))
        if csv_files:
            mp_csv = csv_files[0]
            break

    if not mp_csv:
        raise FileNotFoundError(f"No CSV found for {subject_id}")

    # Process MediaPipe data
    processor = MediaPipeCSVProcessor()
    df_wide = processor.load_csv(mp_csv)
    df_angles = processor.calculate_joint_angles(df_wide)

    # Detect heel strikes
    left_strikes = processor.detect_heel_strikes_fusion(df_angles, side='left', fps=30.0)
    right_strikes = processor.detect_heel_strikes_fusion(df_angles, side='right', fps=30.0)

    print(f"Detected strikes: Left={len(left_strikes)}, Right={len(right_strikes)}")
    print()

    # Ground truth
    patient = info.get('patient', {})
    gt_cadence_left = patient.get('left', {}).get('cadence_steps_min')
    gt_cadence_right = patient.get('right', {}).get('cadence_steps_min')
    gt_cadence_avg = (gt_cadence_left + gt_cadence_right) / 2 if gt_cadence_left and gt_cadence_right else None

    print(f"Ground Truth Cadence: Left={gt_cadence_left:.1f}, Right={gt_cadence_right:.1f}, Avg={gt_cadence_avg:.1f} steps/min")
    print()

    # Method 1: RANSAC
    print("-" * 80)
    print("METHOD 1: RANSAC")
    print("-" * 80)

    ransac_left, diag_l = estimate_cadence_ransac(left_strikes, fps=30.0)
    ransac_right, diag_r = estimate_cadence_ransac(right_strikes, fps=30.0)
    ransac_avg = (ransac_left + ransac_right) / 2

    print(f"Left:  {ransac_left:.1f} steps/min (inliers: {diag_l['n_inliers']}/{diag_l['n_intervals_valid']}, {diag_l['inlier_ratio']:.1%})")
    print(f"Right: {ransac_right:.1f} steps/min (inliers: {diag_r['n_inliers']}/{diag_r['n_intervals_valid']}, {diag_r['inlier_ratio']:.1%})")
    print(f"Avg:   {ransac_avg:.1f} steps/min")
    print(f"Error: {ransac_avg - gt_cadence_avg:+.1f} steps/min ({(ransac_avg - gt_cadence_avg)/gt_cadence_avg*100:+.1f}%)")
    print()

    # Method 2: Percentile (baseline)
    print("-" * 80)
    print("METHOD 2: PERCENTILE (Baseline)")
    print("-" * 80)

    pct_left, pct_diag_l = estimate_cadence_percentile(left_strikes, fps=30.0)
    pct_right, pct_diag_r = estimate_cadence_percentile(right_strikes, fps=30.0)
    pct_avg = (pct_left + pct_right) / 2

    print(f"Left:  {pct_left:.1f} steps/min")
    print(f"Right: {pct_right:.1f} steps/min")
    print(f"Avg:   {pct_avg:.1f} steps/min")
    print(f"Error: {pct_avg - gt_cadence_avg:+.1f} steps/min ({(pct_avg - gt_cadence_avg)/gt_cadence_avg*100:+.1f}%)")
    print()

    # Comparison
    print("="*80)
    print("COMPARISON")
    print("="*80)
    ransac_abs_error = abs(ransac_avg - gt_cadence_avg)
    pct_abs_error = abs(pct_avg - gt_cadence_avg)
    improvement = pct_abs_error - ransac_abs_error

    print(f"RANSAC Error:     {ransac_abs_error:.1f} steps/min")
    print(f"Percentile Error: {pct_abs_error:.1f} steps/min")
    print(f"Improvement:      {improvement:+.1f} steps/min ({improvement/pct_abs_error*100:+.1f}%)")

    if improvement > 0:
        print("✅ RANSAC is better")
    elif improvement < 0:
        print("❌ Percentile is better")
    else:
        print("➖ Tie")

    print()

    return {
        'subject_id': subject_id,
        'gt_cadence_avg': gt_cadence_avg,
        'ransac_avg': ransac_avg,
        'pct_avg': pct_avg,
        'ransac_error': ransac_avg - gt_cadence_avg,
        'pct_error': pct_avg - gt_cadence_avg,
        'improvement': improvement
    }


if __name__ == '__main__':
    # Test on single subject
    result = test_on_subject('S1_01')
    print()

    # Test on multiple subjects
    print("="*80)
    print("TESTING ON MULTIPLE SUBJECTS")
    print("="*80)
    print()

    test_subjects = ['S1_01', 'S1_02', 'S1_03', 'S1_08', 'S1_09']
    results = []

    for subj_id in test_subjects:
        try:
            result = test_on_subject(subj_id)
            results.append(result)
            print()
        except Exception as e:
            print(f"Error on {subj_id}: {e}\n")

    # Summary
    if results:
        print("="*80)
        print("SUMMARY (n=5)")
        print("="*80)
        print()

        ransac_errors = [r['ransac_error'] for r in results]
        pct_errors = [r['pct_error'] for r in results]
        improvements = [r['improvement'] for r in results]

        print(f"{'Subject':<10s} {'GT':>10s} {'RANSAC':>10s} {'Percentile':>10s} {'Improvement':>12s}")
        print("-" * 60)

        for r in results:
            print(f"{r['subject_id']:<10s} {r['gt_cadence_avg']:>10.1f} {r['ransac_avg']:>10.1f} {r['pct_avg']:>10.1f} {r['improvement']:>+12.1f}")

        print("-" * 60)
        print(f"{'Mean Error':<10s} {'':>10s} {np.mean(np.abs(ransac_errors)):>10.1f} {np.mean(np.abs(pct_errors)):>10.1f} {np.mean(improvements):>+12.1f}")
        print()

        # Statistical test
        from scipy.stats import ttest_rel
        t_stat, p_value = ttest_rel(np.abs(pct_errors), np.abs(ransac_errors))
        print(f"Paired t-test: t={t_stat:.3f}, p={p_value:.6f} {'***' if p_value < 0.001 else '**' if p_value < 0.01 else '*' if p_value < 0.05 else 'n.s.'}")

        # Save results
        output = {
            'test_date': '2025-10-10',
            'n_subjects': len(results),
            'results': results,
            'summary': {
                'ransac_mae': float(np.mean(np.abs(ransac_errors))),
                'percentile_mae': float(np.mean(np.abs(pct_errors))),
                'mean_improvement': float(np.mean(improvements)),
                't_statistic': float(t_stat),
                'p_value': float(p_value)
            }
        }

        import json
        with open('P2_ransac_test_results.json', 'w') as f:
            json.dump(output, f, indent=2)

        print("✓ Results saved to P2_ransac_test_results.json")
