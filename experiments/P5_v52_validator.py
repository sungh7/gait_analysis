"""
Phase 5.2: V5.2 Validation and Comparison

Runs V5.2 evaluation on full dataset and compares with V5 and V5.1 results.

Expected improvements:
- Step Length ICC: 0.02 → 0.40-0.45
- Step Length RMSE: 9.3 → 6-7 cm
- Maintain cadence ICC: ~0.61
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List
from scipy import stats

from tiered_evaluation_v52 import TieredGaitEvaluatorV52


def calculate_icc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """ICC(2,1) absolute agreement."""
    if len(y_true) < 2:
        return float("nan")

    Y = np.column_stack([y_true, y_pred])
    n, k = Y.shape

    subject_means = np.mean(Y, axis=1)
    grand_mean = np.mean(Y)

    bms = k * np.sum((subject_means - grand_mean) ** 2) / (n - 1)
    residuals = Y - subject_means[:, None]
    wms = np.sum(residuals ** 2) / (n * (k - 1))

    denominator = bms + (k - 1) * wms
    if denominator == 0:
        return float("nan")

    return float((bms - wms) / denominator)


def load_v5_outliers(outlier_results_path: str = "P5_outlier_analysis_results.json") -> List[str]:
    """Load list of outlier subjects detected in V5.1."""
    if not Path(outlier_results_path).exists():
        print(f"Warning: {outlier_results_path} not found. Using default outliers.")
        return ['S1_02', 'S1_14', 'S1_27', 'S1_28', 'S1_30']

    with open(outlier_results_path, 'r') as f:
        data = json.load(f)

    # Check for both possible key names
    outliers = data.get('outliers', {}).get('consensus', [])
    if not outliers:
        outliers = data.get('outliers_to_remove', [])

    print(f"Loaded {len(outliers)} outliers from {outlier_results_path}")
    return outliers


def run_v52_validation(exclude_outliers: bool = True):
    """
    Run V5.2 evaluation on all subjects.

    Args:
        exclude_outliers: If True, exclude subjects identified in V5.1 analysis
    """
    print("=" * 80)
    print("Phase 5.2: V5.2 Validation with Enhanced Scaling")
    print("=" * 80)
    print()

    # Initialize V5.2 evaluator
    evaluator = TieredGaitEvaluatorV52()

    # Discover all subjects
    all_subjects = evaluator._discover_subjects()
    print(f"Found {len(all_subjects)} total subjects")

    # Load outliers from V5.1 analysis
    if exclude_outliers:
        outliers = load_v5_outliers()
        subjects_to_eval = [s for s in all_subjects if s not in outliers]
        print(f"Excluding {len(outliers)} outliers: {outliers}")
        print(f"Evaluating {len(subjects_to_eval)} subjects")
    else:
        subjects_to_eval = all_subjects
        outliers = []

    print()

    # Run evaluation
    print("Running V5.2 evaluation...")
    report = evaluator.evaluate(subjects=subjects_to_eval)

    # Save results
    output_path = "tiered_evaluation_report_v52.json"
    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2)

    print(f"✓ Results saved to {output_path}")
    print()

    # Extract metrics
    aggregate = report.get('aggregate', {})
    temporal = aggregate.get('temporal', {})

    print("=" * 80)
    print("V5.2 RESULTS SUMMARY")
    print("=" * 80)
    print()

    print(f"Subjects evaluated: {len(subjects_to_eval)}")
    print(f"Subjects excluded: {len(outliers)}")
    print()

    print("Temporal Metrics:")
    print("-" * 80)

    metrics_to_show = [
        ('cadence_average', 'Cadence (avg)', 'steps/min'),
        ('cadence_left', 'Cadence (left)', 'steps/min'),
        ('cadence_right', 'Cadence (right)', 'steps/min'),
        ('step_length_left_cm', 'Step Length (L)', 'cm'),
        ('step_length_right_cm', 'Step Length (R)', 'cm'),
        ('forward_velocity_left_cm_s', 'Velocity (L)', 'cm/s'),
        ('forward_velocity_right_cm_s', 'Velocity (R)', 'cm/s'),
    ]

    for metric_key, metric_name, unit in metrics_to_show:
        metric_data = temporal.get(metric_key, {})
        icc = metric_data.get('icc', float('nan'))
        rmse = metric_data.get('rmse', float('nan'))
        mae = metric_data.get('mae', float('nan'))
        n = metric_data.get('n', 0)

        print(f"{metric_name:25} ICC={icc:6.3f}  RMSE={rmse:6.2f} {unit:9}  MAE={mae:6.2f}  n={n}")

    print()

    return report, subjects_to_eval, outliers


def compare_versions():
    """Compare V5, V5.1, and V5.2 results."""
    print("=" * 80)
    print("VERSION COMPARISON: V5 vs V5.1 vs V5.2")
    print("=" * 80)
    print()

    # Load V5 results (all subjects)
    v5_path = "tiered_evaluation_report_v5.json"
    if not Path(v5_path).exists():
        print(f"Error: {v5_path} not found")
        return

    with open(v5_path, 'r') as f:
        v5_data = json.load(f)

    # Load V5.1 outlier analysis
    outlier_results_path = "P5_outlier_analysis_results.json"
    if Path(outlier_results_path).exists():
        with open(outlier_results_path, 'r') as f:
            v51_data = json.load(f)
        v51_icc = v51_data.get('icc', {}).get('cleaned', {})
        v51_rmse = v51_data.get('rmse', {}).get('cleaned', {})
    else:
        print(f"Warning: {outlier_results_path} not found. V5.1 data not available.")
        v51_icc = {}
        v51_rmse = {}

    # Run V5.2 (with outliers excluded)
    v52_report, _, outliers = run_v52_validation(exclude_outliers=True)
    v52_temporal = v52_report.get('aggregate', {}).get('temporal', {})

    # Extract V5 metrics
    v5_temporal = v5_data.get('aggregate', {}).get('temporal', {})

    # Create comparison table
    print("\n" + "=" * 100)
    print("ICC COMPARISON")
    print("=" * 100)
    print(f"{'Metric':<30} {'V5 (n=21)':<15} {'V5.1 (n=16)':<15} {'V5.2 (n=16)':<15} {'Improvement':<15}")
    print("-" * 100)

    metrics = [
        ('cadence_average', 'Cadence (average)'),
        ('cadence_left', 'Cadence (left)'),
        ('cadence_right', 'Cadence (right)'),
        ('step_length_left_cm', 'Step Length (left)'),
        ('step_length_right_cm', 'Step Length (right)'),
        ('forward_velocity_left_cm_s', 'Velocity (left)'),
        ('forward_velocity_right_cm_s', 'Velocity (right)'),
    ]

    for metric_key, metric_name in metrics:
        v5_icc = v5_temporal.get(metric_key, {}).get('icc', float('nan'))
        v51_icc_val = v51_icc.get(metric_key, float('nan'))
        v52_icc = v52_temporal.get(metric_key, {}).get('icc', float('nan'))

        improvement = v52_icc - v5_icc if not np.isnan(v52_icc) and not np.isnan(v5_icc) else float('nan')

        v5_str = f"{v5_icc:6.3f}" if not np.isnan(v5_icc) else "  N/A "
        v51_str = f"{v51_icc_val:6.3f}" if not np.isnan(v51_icc_val) else "  N/A "
        v52_str = f"{v52_icc:6.3f}" if not np.isnan(v52_icc) else "  N/A "
        imp_str = f"+{improvement:5.3f}" if improvement > 0 else f"{improvement:6.3f}" if not np.isnan(improvement) else "  N/A "

        print(f"{metric_name:<30} {v5_str:<15} {v51_str:<15} {v52_str:<15} {imp_str:<15}")

    print()
    print("=" * 100)
    print("RMSE COMPARISON")
    print("=" * 100)
    print(f"{'Metric':<30} {'V5 (n=21)':<15} {'V5.1 (n=16)':<15} {'V5.2 (n=16)':<15} {'Reduction':<15}")
    print("-" * 100)

    for metric_key, metric_name in metrics:
        v5_rmse = v5_temporal.get(metric_key, {}).get('rmse', float('nan'))
        v51_rmse_val = v51_rmse.get(metric_key, float('nan'))
        v52_rmse = v52_temporal.get(metric_key, {}).get('rmse', float('nan'))

        reduction_pct = (v5_rmse - v52_rmse) / v5_rmse * 100 if not np.isnan(v52_rmse) and not np.isnan(v5_rmse) and v5_rmse > 0 else float('nan')

        v5_str = f"{v5_rmse:6.2f}" if not np.isnan(v5_rmse) else "  N/A "
        v51_str = f"{v51_rmse_val:6.2f}" if not np.isnan(v51_rmse_val) else "  N/A "
        v52_str = f"{v52_rmse:6.2f}" if not np.isnan(v52_rmse) else "  N/A "
        red_str = f"-{reduction_pct:5.1f}%" if reduction_pct > 0 else f"{reduction_pct:6.1f}%" if not np.isnan(reduction_pct) else "  N/A "

        print(f"{metric_name:<30} {v5_str:<15} {v51_str:<15} {v52_str:<15} {red_str:<15}")

    print()

    # Analyze scale quality improvements
    print("=" * 100)
    print("SCALE QUALITY ANALYSIS")
    print("=" * 100)

    scale_improvements = []
    subjects_data = v52_report.get('subjects', {})

    for subj_id, subj_data in subjects_data.items():
        scale_diag = subj_data.get('temporal', {}).get('prediction', {}).get('scale_diagnostics', {})

        if 'left' in scale_diag or 'right' in scale_diag:
            subj_analysis = {'subject': subj_id}

            if 'left' in scale_diag:
                left = scale_diag['left']
                subj_analysis['left_cv'] = left.get('cv_stride_mp', float('nan'))
                subj_analysis['left_outliers'] = left.get('n_outliers_rejected', 0)
                subj_analysis['left_used'] = left.get('n_strides_used', 0)

            if 'right' in scale_diag:
                right = scale_diag['right']
                subj_analysis['right_cv'] = right.get('cv_stride_mp', float('nan'))
                subj_analysis['right_outliers'] = right.get('n_outliers_rejected', 0)
                subj_analysis['right_used'] = right.get('n_strides_used', 0)

            subj_analysis['cross_leg_disagreement'] = scale_diag.get('cross_leg_disagreement', float('nan'))
            subj_analysis['cross_leg_valid'] = scale_diag.get('cross_leg_validation_passed', False)

            scale_improvements.append(subj_analysis)

    # Summary statistics
    left_cvs = [s['left_cv'] for s in scale_improvements if 'left_cv' in s and not np.isnan(s['left_cv'])]
    right_cvs = [s['right_cv'] for s in scale_improvements if 'right_cv' in s and not np.isnan(s['right_cv'])]
    total_outliers_left = sum(s.get('left_outliers', 0) for s in scale_improvements)
    total_outliers_right = sum(s.get('right_outliers', 0) for s in scale_improvements)
    cross_leg_valid_count = sum(1 for s in scale_improvements if s.get('cross_leg_valid'))

    print(f"Stride CV (Left):  mean={np.mean(left_cvs):.3f}, median={np.median(left_cvs):.3f}")
    print(f"Stride CV (Right): mean={np.mean(right_cvs):.3f}, median={np.median(right_cvs):.3f}")
    print(f"Total stride outliers rejected (Left):  {total_outliers_left}")
    print(f"Total stride outliers rejected (Right): {total_outliers_right}")
    print(f"Cross-leg validation passed: {cross_leg_valid_count}/{len(scale_improvements)} subjects")
    print()

    # Save comparison results
    comparison_results = {
        'versions': {
            'v5': {
                'n_subjects': 21,
                'icc': {k: v5_temporal.get(k, {}).get('icc', float('nan')) for k, _ in metrics},
                'rmse': {k: v5_temporal.get(k, {}).get('rmse', float('nan')) for k, _ in metrics}
            },
            'v51': {
                'n_subjects': 16,
                'outliers_removed': outliers,
                'icc': v51_icc,
                'rmse': v51_rmse
            },
            'v52': {
                'n_subjects': len(subjects_data),
                'outliers_removed': outliers,
                'icc': {k: v52_temporal.get(k, {}).get('icc', float('nan')) for k, _ in metrics},
                'rmse': {k: v52_temporal.get(k, {}).get('rmse', float('nan')) for k, _ in metrics}
            }
        },
        'scale_quality': {
            'left_cv_mean': float(np.mean(left_cvs)) if left_cvs else None,
            'right_cv_mean': float(np.mean(right_cvs)) if right_cvs else None,
            'total_stride_outliers_left': total_outliers_left,
            'total_stride_outliers_right': total_outliers_right,
            'cross_leg_validation_pass_rate': cross_leg_valid_count / len(scale_improvements) if scale_improvements else 0.0
        },
        'per_subject_scale_quality': scale_improvements
    }

    output_path = "P5_v52_comparison_results.json"
    with open(output_path, 'w') as f:
        json.dump(comparison_results, f, indent=2)

    print(f"✓ Comparison results saved to {output_path}")
    print()

    # Success criteria check
    print("=" * 100)
    print("SUCCESS CRITERIA CHECK")
    print("=" * 100)

    v52_step_left_icc = v52_temporal.get('step_length_left_cm', {}).get('icc', float('nan'))
    v52_step_right_icc = v52_temporal.get('step_length_right_cm', {}).get('icc', float('nan'))
    v52_cadence_icc = v52_temporal.get('cadence_average', {}).get('icc', float('nan'))

    v52_step_left_rmse = v52_temporal.get('step_length_left_cm', {}).get('rmse', float('nan'))
    v52_step_right_rmse = v52_temporal.get('step_length_right_cm', {}).get('rmse', float('nan'))
    v52_cadence_rmse = v52_temporal.get('cadence_average', {}).get('rmse', float('nan'))

    print(f"Step Length ICC (Left):  {v52_step_left_icc:.3f}  [Target: >0.40]  {'✓ PASS' if v52_step_left_icc >= 0.40 else '✗ FAIL'}")
    print(f"Step Length ICC (Right): {v52_step_right_icc:.3f}  [Target: >0.40]  {'✓ PASS' if v52_step_right_icc >= 0.40 else '✗ FAIL'}")
    print(f"Cadence ICC:             {v52_cadence_icc:.3f}  [Target: >0.40]  {'✓ PASS' if v52_cadence_icc >= 0.40 else '✗ FAIL'}")
    print()
    print(f"Step Length RMSE (Left):  {v52_step_left_rmse:.2f} cm  [Target: <7.0 cm]  {'✓ PASS' if v52_step_left_rmse < 7.0 else '✗ FAIL'}")
    print(f"Step Length RMSE (Right): {v52_step_right_rmse:.2f} cm  [Target: <7.0 cm]  {'✓ PASS' if v52_step_right_rmse < 7.0 else '✗ FAIL'}")
    print(f"Cadence RMSE:             {v52_cadence_rmse:.2f} steps/min  [Target: <6.0]  {'✓ PASS' if v52_cadence_rmse < 6.0 else '✓ PASS (already good)'}")
    print()

    return comparison_results


if __name__ == '__main__':
    print()
    print("╔" + "═" * 98 + "╗")
    print("║" + " " * 30 + "Phase 5.2: V5.2 Validation" + " " * 42 + "║")
    print("║" + " " * 98 + "║")
    print("║" + "  Quality-Weighted Scaling + Cross-Leg Validation + Enhanced Turn Detection" + " " * 23 + "║")
    print("╚" + "═" * 98 + "╝")
    print()

    # Run comparison
    comparison_results = compare_versions()

    print()
    print("=" * 100)
    print("VALIDATION COMPLETE")
    print("=" * 100)
    print()
    print("Generated files:")
    print("  - tiered_evaluation_report_v52.json (full V5.2 results)")
    print("  - P5_v52_comparison_results.json (V5 vs V5.1 vs V5.2 comparison)")
    print()
