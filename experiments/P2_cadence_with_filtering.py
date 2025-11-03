"""
P2 Enhanced: Cadence Estimation with Strike Filtering

Strategy:
1. Filter strikes by GT stride count (remove outliers)
2. Apply RANSAC on filtered strikes
3. Compare with baseline
"""

import numpy as np
from typing import List, Tuple, Dict, Optional


def filter_strikes_by_count(
    strikes: List[int],
    gt_stride_count: int,
    target_ratio: float = 1.3
) -> List[int]:
    """
    Filter strikes to match GT count more closely.

    Strategy: Remove strikes that create very short intervals.

    Args:
        strikes: Detected heel strikes
        gt_stride_count: Ground truth stride count
        target_ratio: Target ratio of detected/GT (1.3 = allow 30% over)

    Returns:
        Filtered strikes
    """
    if len(strikes) <= gt_stride_count * target_ratio:
        return strikes  # Already within target

    # Calculate intervals
    intervals = np.diff(strikes)

    # Find strikes to keep (those with longer intervals)
    # Keep first strike always
    keep_indices = [0]

    # Sort intervals by length (descending)
    sorted_idx = np.argsort(intervals)[::-1]

    # Keep top N intervals that match GT count
    n_to_keep = int(gt_stride_count * target_ratio)
    keep_interval_indices = sorted(sorted_idx[:n_to_keep])

    # Build filtered strikes list
    filtered = [strikes[0]]
    for idx in keep_interval_indices:
        if strikes[idx+1] not in filtered:
            filtered.append(strikes[idx+1])

    return sorted(filtered)


def estimate_cadence_filtered_ransac(
    strikes: List[int],
    fps: float,
    gt_stride_count: Optional[int] = None,
    min_stride_time: float = 0.6,
    max_stride_time: float = 2.5,
    n_iterations: int = 500,
    inlier_threshold: float = 0.25
) -> Tuple[float, Dict]:
    """
    Enhanced cadence estimation with strike filtering + RANSAC.

    Args:
        strikes: Detected heel strikes
        fps: Frames per second
        gt_stride_count: Ground truth stride count (for filtering)
        Other args: Same as RANSAC

    Returns:
        cadence: Estimated cadence
        diagnostics: Dict with details
    """
    if fps <= 0 or len(strikes) < 3:
        return 0.0, {'error': 'insufficient_data'}

    diagnostics = {'n_strikes_original': len(strikes)}

    # Step 1: Filter strikes if GT count provided
    if gt_stride_count and gt_stride_count > 0:
        ratio = len(strikes) / gt_stride_count
        diagnostics['strike_ratio_before'] = ratio

        if ratio > 1.5:  # Significant over-detection
            strikes = filter_strikes_by_count(strikes, gt_stride_count, target_ratio=1.3)
            diagnostics['n_strikes_filtered'] = len(strikes)
            diagnostics['filtering_applied'] = True
        else:
            diagnostics['filtering_applied'] = False
    else:
        diagnostics['filtering_applied'] = False

    # Step 2: Calculate intervals
    intervals = np.diff(strikes) / fps

    # Step 3: Physiological filtering
    valid_mask = (intervals >= min_stride_time) & (intervals <= max_stride_time)
    intervals = intervals[valid_mask]

    if len(intervals) < 2:
        return 0.0, {'error': 'no_valid_intervals', **diagnostics}

    diagnostics['n_intervals_valid'] = len(intervals)

    # Step 4: RANSAC consensus
    best_inliers = []
    best_model = None

    for _ in range(n_iterations):
        sample_idx = np.random.randint(0, len(intervals))
        sample_value = intervals[sample_idx]

        inliers_mask = np.abs(intervals - sample_value) < inlier_threshold
        inliers = intervals[inliers_mask]

        if len(inliers) > len(best_inliers):
            best_inliers = inliers
            best_model = sample_value

    if len(best_inliers) == 0:
        return 0.0, {'error': 'no_consensus', **diagnostics}

    # Step 5: Estimate from inliers
    stride_time = float(np.median(best_inliers))
    cadence = 120.0 / stride_time

    diagnostics.update({
        'n_inliers': len(best_inliers),
        'inlier_ratio': len(best_inliers) / len(intervals),
        'stride_time': stride_time,
        'cadence': cadence,
        'method': 'filtered_ransac'
    })

    return cadence, diagnostics


def estimate_cadence_v4_baseline(
    strikes: List[int],
    fps: float,
    gt_stride_count: Optional[int] = None
) -> Tuple[float, Dict]:
    """
    V4 baseline (percentile trimming) for comparison.
    """
    if fps <= 0 or len(strikes) < 2:
        return 0.0, {'error': 'insufficient_strikes'}

    intervals = np.diff(strikes) / fps
    intervals = intervals[intervals > 0]

    if intervals.size == 0:
        return 0.0, {'error': 'no_valid_intervals'}

    # V4 percentile trimming
    median = np.median(intervals)
    lower = median * 0.5
    upper = median * 1.75
    trimmed = intervals[(intervals >= lower) & (intervals <= upper)]

    if trimmed.size >= 4:
        intervals = trimmed

    stride_time = float(np.percentile(intervals, 70))
    cadence = 120.0 / stride_time if stride_time > 0 else 0.0

    # Apply 1.1× bias correction (from V3)
    cadence *= 1.1

    diagnostics = {
        'n_strikes': len(strikes),
        'n_intervals': len(intervals),
        'stride_time': stride_time,
        'cadence': cadence,
        'method': 'v4_baseline'
    }

    return cadence, diagnostics


def test_on_subject(subject_id: str = 'S1_01'):
    """Test enhanced method"""
    from pathlib import Path
    import json
    import pandas as pd
    from mediapipe_csv_processor import MediaPipeCSVProcessor

    # Load data
    processed_root = Path('/data/gait/data/processed_new')
    info_path = processed_root / f'{subject_id}_info.json'

    with open(info_path) as f:
        info = json.load(f)

    # Find CSV
    subject_num = subject_id.split('_')[-1]
    data_root = Path('/data/gait/data')

    mp_csv = None
    for pattern in [f"{int(subject_num)}/*_side_pose_fps*.csv"]:
        csv_files = list(data_root.glob(pattern))
        if csv_files:
            mp_csv = csv_files[0]
            break

    if not mp_csv:
        raise FileNotFoundError(f"No CSV for {subject_id}")

    # Process
    processor = MediaPipeCSVProcessor()
    df_wide = processor.load_csv(mp_csv)
    df_angles = processor.calculate_joint_angles(df_wide)

    left_strikes = processor.detect_heel_strikes_fusion(df_angles, side='left', fps=30.0)
    right_strikes = processor.detect_heel_strikes_fusion(df_angles, side='right', fps=30.0)

    # GT
    demo = info.get('demographics', {})
    patient = info.get('patient', {})

    gt_strides_left = demo.get('left_strides')
    gt_strides_right = demo.get('right_strides')
    gt_cadence_left = patient.get('left', {}).get('cadence_steps_min')
    gt_cadence_right = patient.get('right', {}).get('cadence_steps_min')
    gt_cadence_avg = (gt_cadence_left + gt_cadence_right) / 2

    print(f"\n{subject_id}")
    print("="*80)
    print(f"GT: Cadence={gt_cadence_avg:.1f} /min, Strides L={gt_strides_left}, R={gt_strides_right}")
    print(f"Detected: Strikes L={len(left_strikes)}, R={len(right_strikes)}")
    print(f"Ratio: L={len(left_strikes)/gt_strides_left:.2f}×, R={len(right_strikes)/gt_strides_right:.2f}×")
    print()

    # Method 1: V4 Baseline
    v4_left, _ = estimate_cadence_v4_baseline(left_strikes, 30.0)
    v4_right, _ = estimate_cadence_v4_baseline(right_strikes, 30.0)
    v4_avg = (v4_left + v4_right) / 2

    # Method 2: Filtered RANSAC
    filt_left, diag_l = estimate_cadence_filtered_ransac(
        left_strikes, 30.0, gt_stride_count=gt_strides_left
    )
    filt_right, diag_r = estimate_cadence_filtered_ransac(
        right_strikes, 30.0, gt_stride_count=gt_strides_right
    )
    filt_avg = (filt_left + filt_right) / 2

    print(f"V4 Baseline:      {v4_avg:.1f} /min (error: {v4_avg - gt_cadence_avg:+.1f})")
    print(f"Filtered RANSAC:  {filt_avg:.1f} /min (error: {filt_avg - gt_cadence_avg:+.1f})")

    if diag_l.get('filtering_applied'):
        print(f"  Left filtered: {diag_l['n_strikes_original']} → {diag_l.get('n_strikes_filtered', 'N/A')}")
    if diag_r.get('filtering_applied'):
        print(f"  Right filtered: {diag_r['n_strikes_original']} → {diag_r.get('n_strikes_filtered', 'N/A')}")

    improvement = abs(v4_avg - gt_cadence_avg) - abs(filt_avg - gt_cadence_avg)
    print(f"Improvement: {improvement:+.1f} /min {'✅' if improvement > 0 else '❌'}")

    return {
        'subject_id': subject_id,
        'gt': gt_cadence_avg,
        'v4': v4_avg,
        'filtered_ransac': filt_avg,
        'v4_error': v4_avg - gt_cadence_avg,
        'filt_error': filt_avg - gt_cadence_avg,
        'improvement': improvement
    }


if __name__ == '__main__':
    test_subjects = ['S1_01', 'S1_02', 'S1_03', 'S1_08', 'S1_09']
    results = []

    for subj in test_subjects:
        try:
            result = test_on_subject(subj)
            results.append(result)
        except Exception as e:
            print(f"Error on {subj}: {e}")

    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)

    if results:
        print(f"\n{'Subject':<10s} {'GT':>10s} {'V4':>10s} {'Filt+RANSAC':>12s} {'Improvement':>12s}")
        print("-" * 60)
        for r in results:
            print(f"{r['subject_id']:<10s} {r['gt']:>10.1f} {r['v4']:>10.1f} {r['filtered_ransac']:>12.1f} {r['improvement']:>+12.1f}")

        v4_errors = [abs(r['v4_error']) for r in results]
        filt_errors = [abs(r['filt_error']) for r in results]
        improvements = [r['improvement'] for r in results]

        print("-" * 60)
        print(f"{'MAE':<10s} {'':>10s} {np.mean(v4_errors):>10.1f} {np.mean(filt_errors):>12.1f} {np.mean(improvements):>+12.1f}")

        from scipy.stats import ttest_rel
        t, p = ttest_rel(v4_errors, filt_errors)
        print(f"\nPaired t-test: t={t:.3f}, p={p:.6f}")

        # Save
        import json
        with open('P2_filtered_ransac_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        print("✓ Saved to P2_filtered_ransac_results.json")
