"""
P0: Baseline Audit - Comprehensive diagnostic script

Captures current V3 metrics and analyzes key issues:
1. Cadence variability and bias
2. Strike count vs GT stride ratios
3. Spatial metric errors (step length, velocity)
4. Per-subject breakdowns
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd


def load_v3_report():
    """Load tiered_evaluation_report_v3.json"""
    with open('/data/gait/tiered_evaluation_report_v3.json') as f:
        return json.load(f)


def analyze_aggregate_metrics(data):
    """Extract and summarize aggregate metrics"""
    agg = data.get('aggregate', {})
    temporal = agg.get('temporal', {})

    metrics = {
        'cadence_average': temporal.get('cadence_average', {}),
        'cadence_left': temporal.get('cadence_left', {}),
        'cadence_right': temporal.get('cadence_right', {}),
        'step_length_left_cm': temporal.get('step_length_left_cm', {}),
        'step_length_right_cm': temporal.get('step_length_right_cm', {}),
        'stride_length_left_cm': temporal.get('stride_length_left_cm', {}),
        'stride_length_right_cm': temporal.get('stride_length_right_cm', {}),
        'forward_velocity_left_cm_s': temporal.get('forward_velocity_left_cm_s', {}),
        'forward_velocity_right_cm_s': temporal.get('forward_velocity_right_cm_s', {}),
        'strides_left': temporal.get('strides_left', {}),
        'strides_right': temporal.get('strides_right', {}),
    }

    return metrics


def analyze_per_subject(data):
    """Analyze per-subject metrics"""
    subjects = data.get('subjects', {})

    results = []
    for subj_id, subj_data in sorted(subjects.items()):
        temporal = subj_data.get('temporal', {})
        gt = temporal.get('ground_truth', {})
        pred = temporal.get('prediction', {})

        # Cadence
        gt_cad = gt.get('cadence_steps_min', {})
        pred_cad = pred.get('cadence_steps_min', {})
        gt_cad_left = gt_cad.get('left')
        gt_cad_right = gt_cad.get('right')
        gt_cad_avg = gt_cad.get('average')
        pred_cad_left = pred_cad.get('left')
        pred_cad_right = pred_cad.get('right')
        pred_cad_avg = pred_cad.get('average')

        # Strides
        gt_strides = gt.get('strides', {})
        pred_strides = pred.get('strides', {})
        gt_strides_left = gt_strides.get('left')
        gt_strides_right = gt_strides.get('right')
        pred_strides_left = pred_strides.get('left')
        pred_strides_right = pred_strides.get('right')

        # Strike ratios
        ratio_left = pred_strides_left / gt_strides_left if gt_strides_left and gt_strides_left > 0 else None
        ratio_right = pred_strides_right / gt_strides_right if gt_strides_right and gt_strides_right > 0 else None

        # Step length
        gt_step = gt.get('step_length_cm', {})
        pred_step = pred.get('step_length_cm', {})
        gt_step_left = gt_step.get('left')
        pred_step_left = pred_step.get('left')
        step_error_left = pred_step_left - gt_step_left if (gt_step_left and pred_step_left) else None

        # Velocity
        gt_vel = gt.get('forward_velocity_cm_s', {})
        pred_vel = pred.get('forward_velocity_cm_s', {})
        gt_vel_left = gt_vel.get('left')
        pred_vel_left = pred_vel.get('left')
        vel_error_left = pred_vel_left - gt_vel_left if (gt_vel_left and pred_vel_left) else None

        results.append({
            'subject_id': subj_id,
            'gt_cadence_avg': gt_cad_avg,
            'pred_cadence_avg': pred_cad_avg,
            'cadence_error': pred_cad_avg - gt_cad_avg if (gt_cad_avg and pred_cad_avg) else None,
            'gt_strides_left': gt_strides_left,
            'pred_strides_left': pred_strides_left,
            'strike_ratio_left': ratio_left,
            'gt_strides_right': gt_strides_right,
            'pred_strides_right': pred_strides_right,
            'strike_ratio_right': ratio_right,
            'gt_step_length_left': gt_step_left,
            'pred_step_length_left': pred_step_left,
            'step_length_error_left': step_error_left,
            'gt_velocity_left': gt_vel_left,
            'pred_velocity_left': pred_vel_left,
            'velocity_error_left': vel_error_left,
        })

    return results


def compute_statistics(results):
    """Compute summary statistics across subjects"""
    # Filter valid values
    cadence_errors = [r['cadence_error'] for r in results if r['cadence_error'] is not None]
    strike_ratios_left = [r['strike_ratio_left'] for r in results if r['strike_ratio_left'] is not None]
    strike_ratios_right = [r['strike_ratio_right'] for r in results if r['strike_ratio_right'] is not None]
    step_errors = [r['step_length_error_left'] for r in results if r['step_length_error_left'] is not None]
    vel_errors = [r['velocity_error_left'] for r in results if r['velocity_error_left'] is not None]

    stats = {
        'n_subjects': len(results),
        'cadence_error_mean': float(np.mean(cadence_errors)) if cadence_errors else None,
        'cadence_error_std': float(np.std(cadence_errors)) if cadence_errors else None,
        'cadence_error_median': float(np.median(cadence_errors)) if cadence_errors else None,
        'strike_ratio_left_mean': float(np.mean(strike_ratios_left)) if strike_ratios_left else None,
        'strike_ratio_left_median': float(np.median(strike_ratios_left)) if strike_ratios_left else None,
        'strike_ratio_right_mean': float(np.mean(strike_ratios_right)) if strike_ratios_right else None,
        'strike_ratio_right_median': float(np.median(strike_ratios_right)) if strike_ratios_right else None,
        'subjects_over_1_5x_left': sum(1 for r in strike_ratios_left if r > 1.5),
        'subjects_over_1_5x_right': sum(1 for r in strike_ratios_right if r > 1.5),
        'step_length_error_mean': float(np.mean(step_errors)) if step_errors else None,
        'step_length_error_std': float(np.std(step_errors)) if step_errors else None,
        'velocity_error_mean': float(np.mean(vel_errors)) if vel_errors else None,
        'velocity_error_std': float(np.std(vel_errors)) if vel_errors else None,
    }

    return stats


def main():
    print("="*80)
    print("P0: BASELINE AUDIT")
    print("="*80)
    print()

    # Load V3 report
    print("Loading tiered_evaluation_report_v3.json...")
    data = load_v3_report()
    print("✓ Loaded")
    print()

    # Aggregate metrics
    print("="*80)
    print("AGGREGATE METRICS")
    print("="*80)
    metrics = analyze_aggregate_metrics(data)

    print("\n--- Cadence Metrics ---")
    for name in ['cadence_average', 'cadence_left', 'cadence_right']:
        m = metrics[name]
        print(f"{name:25s}: ICC={m.get('icc', float('nan')):.3f}, RMSE={m.get('rmse', float('nan')):.2f}, n={m.get('n', 0)}")

    print("\n--- Spatial Metrics ---")
    for name in ['step_length_left_cm', 'step_length_right_cm',
                 'stride_length_left_cm', 'stride_length_right_cm']:
        m = metrics[name]
        print(f"{name:30s}: ICC={m.get('icc', float('nan')):.3f}, RMSE={m.get('rmse', float('nan')):.2f}")

    print("\n--- Velocity Metrics ---")
    for name in ['forward_velocity_left_cm_s', 'forward_velocity_right_cm_s']:
        m = metrics[name]
        print(f"{name:30s}: ICC={m.get('icc', float('nan')):.3f}, RMSE={m.get('rmse', float('nan')):.2f}")

    print("\n--- Stride Count ---")
    for name in ['strides_left', 'strides_right']:
        m = metrics[name]
        print(f"{name:25s}: ICC={m.get('icc', float('nan')):.3f}, RMSE={m.get('rmse', float('nan')):.2f}")

    # Per-subject analysis
    print("\n" + "="*80)
    print("PER-SUBJECT ANALYSIS")
    print("="*80)

    results = analyze_per_subject(data)

    print("\n--- Cadence Comparison (steps/min) ---")
    print(f"{'Subject':10s} {'GT Avg':>10s} {'Pred Avg':>10s} {'Error':>10s} {'Error %':>10s}")
    print("-" * 55)
    for r in results:
        gt = r['gt_cadence_avg']
        pred = r['pred_cadence_avg']
        err = r['cadence_error']
        err_pct = (err / gt * 100) if (gt and gt > 0 and err) else None

        gt_str = f"{gt:.1f}" if gt else "N/A"
        pred_str = f"{pred:.1f}" if pred else "N/A"
        err_str = f"{err:+.1f}" if err else "N/A"
        err_pct_str = f"{err_pct:+.1f}%" if err_pct else "N/A"

        print(f"{r['subject_id']:10s} {gt_str:>10s} {pred_str:>10s} {err_str:>10s} {err_pct_str:>10s}")

    print("\n--- Strike Count Ratios (Detected / GT Strides) ---")
    print(f"{'Subject':10s} {'GT L':>8s} {'Pred L':>8s} {'Ratio L':>8s} {'GT R':>8s} {'Pred R':>8s} {'Ratio R':>8s} {'Flag':>6s}")
    print("-" * 75)
    for r in results:
        gt_l = r['gt_strides_left']
        pred_l = r['pred_strides_left']
        ratio_l = r['strike_ratio_left']
        gt_r = r['gt_strides_right']
        pred_r = r['pred_strides_right']
        ratio_r = r['strike_ratio_right']

        flag = ""
        if ratio_l and ratio_l > 1.5:
            flag += "L!"
        if ratio_r and ratio_r > 1.5:
            flag += "R!"

        gt_l_str = f"{gt_l}" if gt_l else "N/A"
        pred_l_str = f"{pred_l}" if pred_l else "N/A"
        ratio_l_str = f"{ratio_l:.2f}" if ratio_l else "N/A"
        gt_r_str = f"{gt_r}" if gt_r else "N/A"
        pred_r_str = f"{pred_r}" if pred_r else "N/A"
        ratio_r_str = f"{ratio_r:.2f}" if ratio_r else "N/A"

        print(f"{r['subject_id']:10s} {gt_l_str:>8s} {pred_l_str:>8s} {ratio_l_str:>8s} {gt_r_str:>8s} {pred_r_str:>8s} {ratio_r_str:>8s} {flag:>6s}")

    print("\n--- Step Length (cm) ---")
    print(f"{'Subject':10s} {'GT':>10s} {'Pred':>10s} {'Error':>10s}")
    print("-" * 45)
    for r in results:
        gt = r['gt_step_length_left']
        pred = r['pred_step_length_left']
        err = r['step_length_error_left']

        gt_str = f"{gt:.1f}" if gt else "N/A"
        pred_str = f"{pred:.1f}" if pred else "N/A"
        err_str = f"{err:+.1f}" if err else "N/A"

        print(f"{r['subject_id']:10s} {gt_str:>10s} {pred_str:>10s} {err_str:>10s}")

    # Summary statistics
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)

    stats = compute_statistics(results)

    print(f"\nNumber of subjects: {stats['n_subjects']}")
    print(f"\nCadence Error (steps/min):")
    print(f"  Mean:   {stats['cadence_error_mean']:+.2f}" if stats['cadence_error_mean'] else "  N/A")
    print(f"  Median: {stats['cadence_error_median']:+.2f}" if stats['cadence_error_median'] else "  N/A")
    print(f"  Std:    {stats['cadence_error_std']:.2f}" if stats['cadence_error_std'] else "  N/A")

    print(f"\nStrike Ratio (Detected/GT):")
    print(f"  Left  - Mean: {stats['strike_ratio_left_mean']:.2f}, Median: {stats['strike_ratio_left_median']:.2f}" if stats['strike_ratio_left_mean'] else "  N/A")
    print(f"  Right - Mean: {stats['strike_ratio_right_mean']:.2f}, Median: {stats['strike_ratio_right_median']:.2f}" if stats['strike_ratio_right_mean'] else "  N/A")
    print(f"  Subjects with >1.5x left:  {stats['subjects_over_1_5x_left']}")
    print(f"  Subjects with >1.5x right: {stats['subjects_over_1_5x_right']}")

    print(f"\nStep Length Error (cm):")
    print(f"  Mean: {stats['step_length_error_mean']:+.2f}" if stats['step_length_error_mean'] else "  N/A")
    print(f"  Std:  {stats['step_length_error_std']:.2f}" if stats['step_length_error_std'] else "  N/A")

    print(f"\nVelocity Error (cm/s):")
    print(f"  Mean: {stats['velocity_error_mean']:+.2f}" if stats['velocity_error_mean'] else "  N/A")
    print(f"  Std:  {stats['velocity_error_std']:.2f}" if stats['velocity_error_std'] else "  N/A")

    # Key findings
    print("\n" + "="*80)
    print("KEY FINDINGS")
    print("="*80)

    print("\n1. CADENCE:")
    print(f"   - ICC: {metrics['cadence_average'].get('icc', float('nan')):.3f} (Poor, should be >0.5)")
    print(f"   - RMSE: {metrics['cadence_average'].get('rmse', float('nan')):.1f} steps/min")
    print(f"   - Mean bias: {stats['cadence_error_mean']:+.1f} steps/min" if stats['cadence_error_mean'] else "   - Mean bias: N/A")

    print("\n2. STRIKE DETECTION:")
    print(f"   - Mean ratio: {stats['strike_ratio_left_mean']:.2f}x (should be ~1.0)" if stats['strike_ratio_left_mean'] else "   - N/A")
    print(f"   - Over-detection: {stats['subjects_over_1_5x_left']} subjects >1.5x threshold")

    print("\n3. SPATIAL METRICS:")
    print(f"   - Step length ICC: {metrics['step_length_left_cm'].get('icc', float('nan')):.3f}")
    print(f"   - Step length RMSE: {metrics['step_length_left_cm'].get('rmse', float('nan')):.1f} cm")
    print(f"   - Mean error: {stats['step_length_error_mean']:+.1f} cm" if stats['step_length_error_mean'] else "   - Mean error: N/A")

    print("\n4. VELOCITY:")
    print(f"   - ICC: {metrics['forward_velocity_left_cm_s'].get('icc', float('nan')):.3f}")
    print(f"   - RMSE: {metrics['forward_velocity_left_cm_s'].get('rmse', float('nan')):.1f} cm/s")

    # Save full report
    output = {
        'timestamp': pd.Timestamp.now().isoformat(),
        'phase': 'P0_baseline',
        'aggregate_metrics': metrics,
        'per_subject_results': results,
        'summary_statistics': stats
    }

    output_path = '/data/gait/P0_baseline_audit_results.json'
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\n✓ Full report saved to {output_path}")


if __name__ == '__main__':
    main()
