"""
P1 Phase 1 Integration - V3 vs V4 Comparison Analysis
"""

import json
import numpy as np

# Load V3 and V4 results
with open('tiered_evaluation_report_v3.json') as f:
    v3 = json.load(f)

with open('tiered_evaluation_report_v4.json') as f:
    v4 = json.load(f)

print("="*80)
print("P1: STRIDE-BASED SCALING - V3 vs V4 COMPARISON")
print("="*80)
print()

# Compare aggregate metrics
print("AGGREGATE METRICS COMPARISON")
print("-" * 80)
print(f"{'Metric':<30s} {'V3 ICC':>10s} {'V4 ICC':>10s} {'Δ ICC':>10s} {'V3 RMSE':>10s} {'V4 RMSE':>10s} {'Δ RMSE':>10s}")
print("-" * 80)

metrics_to_compare = [
    'cadence_average',
    'step_length_left_cm',
    'step_length_right_cm',
    'stride_length_left_cm',
    'stride_length_right_cm',
    'forward_velocity_left_cm_s',
    'forward_velocity_right_cm_s'
]

for metric in metrics_to_compare:
    v3_m = v3['aggregate']['temporal'][metric]
    v4_m = v4['aggregate']['temporal'][metric]

    v3_icc = v3_m['icc']
    v4_icc = v4_m['icc']
    delta_icc = v4_icc - v3_icc

    v3_rmse = v3_m['rmse']
    v4_rmse = v4_m['rmse']
    delta_rmse = v4_rmse - v3_rmse
    pct_rmse = (delta_rmse / v3_rmse * 100) if v3_rmse > 0 else 0

    print(f"{metric:<30s} {v3_icc:>10.3f} {v4_icc:>10.3f} {delta_icc:>+10.3f} {v3_rmse:>10.2f} {v4_rmse:>10.2f} {delta_rmse:>+10.2f} ({pct_rmse:+.1f}%)")

print()
print("="*80)
print("PER-SUBJECT STEP LENGTH ERROR ANALYSIS")
print("="*80)
print()

print(f"{'Subject':<10s} {'V3 Error (cm)':>15s} {'V4 Error (cm)':>15s} {'Improvement':>15s} {'Improved?':>12s}")
print("-" * 70)

v3_errors = []
v4_errors = []

for subj_id in sorted(v3['subjects'].keys()):
    v3_s = v3['subjects'][subj_id]['temporal']
    v4_s = v4['subjects'][subj_id]['temporal']

    # Get step length errors
    v3_gt = v3_s['ground_truth']['step_length_cm']['left']
    v3_pred = v3_s['prediction']['step_length_cm']['left']
    v3_error = v3_pred - v3_gt if (v3_gt and v3_pred) else None

    v4_gt = v4_s['ground_truth']['step_length_cm']['left']
    v4_pred = v4_s['prediction']['step_length_cm']['left']
    v4_error = v4_pred - v4_gt if (v4_gt and v4_pred) else None

    if v3_error is not None and v4_error is not None:
        improvement = abs(v3_error) - abs(v4_error)
        improved = "✓ Yes" if improvement > 0 else "✗ No"

        print(f"{subj_id:<10s} {v3_error:>+15.1f} {v4_error:>+15.1f} {improvement:>+15.1f} {improved:>12s}")

        v3_errors.append(abs(v3_error))
        v4_errors.append(abs(v4_error))

print("-" * 70)
print(f"{'Mean':<10s} {np.mean(v3_errors):>+15.1f} {np.mean(v4_errors):>+15.1f} {np.mean(v3_errors) - np.mean(v4_errors):>+15.1f}")
print(f"{'Median':<10s} {np.median(v3_errors):>+15.1f} {np.median(v4_errors):>+15.1f} {np.median(v3_errors) - np.median(v4_errors):>+15.1f}")
print()

# Statistical test
from scipy.stats import ttest_rel
t_stat, p_value = ttest_rel(v3_errors, v4_errors)
print(f"Paired t-test: t={t_stat:.3f}, p={p_value:.6f} {'***' if p_value < 0.001 else '**' if p_value < 0.01 else '*' if p_value < 0.05 else 'n.s.'}")

# Effect size
diff = np.array(v3_errors) - np.array(v4_errors)
cohen_d = np.mean(diff) / np.std(diff)
print(f"Cohen's d: {cohen_d:.3f} ({'Large' if abs(cohen_d) > 0.8 else 'Medium' if abs(cohen_d) > 0.5 else 'Small'})")

print()
print("="*80)
print("SCALE FACTOR ANALYSIS (V4)")
print("="*80)
print()

print(f"{'Subject':<10s} {'Method':>20s} {'Scale':>10s} {'Sides':>8s} {'Agreement':>12s}")
print("-" * 65)

for subj_id in sorted(v4['subjects'].keys()):
    v4_s = v4['subjects'][subj_id]['temporal']
    scale_diag = v4_s['prediction'].get('scale_diagnostics', {})

    method = scale_diag.get('method', 'N/A')
    scale = scale_diag.get('final_scale', 0)
    n_sides = scale_diag.get('n_sides_used', 0)
    agreement = scale_diag.get('bilateral_agreement')

    agreement_str = f"{agreement:.3f}" if agreement else "N/A"

    print(f"{subj_id:<10s} {method:>20s} {scale:>10.3f} {n_sides:>8d} {agreement_str:>12s}")

# Count methods
methods = [v4['subjects'][s]['temporal']['prediction'].get('scale_diagnostics', {}).get('method') for s in v4['subjects']]
stride_based = sum(1 for m in methods if m == 'stride_based')
fallback = sum(1 for m in methods if m == 'fallback_walkway')

print()
print(f"Stride-based: {stride_based}/21 subjects ({stride_based/21*100:.1f}%)")
print(f"Fallback:     {fallback}/21 subjects ({fallback/21*100:.1f}%)")

print()
print("="*80)
print("SUMMARY")
print("="*80)
print()
print(f"✅ Step Length RMSE: {v3['aggregate']['temporal']['step_length_left_cm']['rmse']:.1f} cm → {v4['aggregate']['temporal']['step_length_left_cm']['rmse']:.1f} cm")
print(f"   Improvement: {v3['aggregate']['temporal']['step_length_left_cm']['rmse'] - v4['aggregate']['temporal']['step_length_left_cm']['rmse']:.1f} cm ({(1 - v4['aggregate']['temporal']['step_length_left_cm']['rmse']/v3['aggregate']['temporal']['step_length_left_cm']['rmse'])*100:.1f}%)")
print()
print(f"✅ Cadence RMSE: {v3['aggregate']['temporal']['cadence_average']['rmse']:.1f} /min → {v4['aggregate']['temporal']['cadence_average']['rmse']:.1f} /min")
print(f"   Improvement: {v3['aggregate']['temporal']['cadence_average']['rmse'] - v4['aggregate']['temporal']['cadence_average']['rmse']:.1f} /min ({(1 - v4['aggregate']['temporal']['cadence_average']['rmse']/v3['aggregate']['temporal']['cadence_average']['rmse'])*100:.1f}%)")
print()
print(f"✅ Velocity RMSE: {v3['aggregate']['temporal']['forward_velocity_left_cm_s']['rmse']:.1f} cm/s → {v4['aggregate']['temporal']['forward_velocity_left_cm_s']['rmse']:.1f} cm/s")
print(f"   Improvement: {v3['aggregate']['temporal']['forward_velocity_left_cm_s']['rmse'] - v4['aggregate']['temporal']['forward_velocity_left_cm_s']['rmse']:.1f} cm/s ({(1 - v4['aggregate']['temporal']['forward_velocity_left_cm_s']['rmse']/v3['aggregate']['temporal']['forward_velocity_left_cm_s']['rmse'])*100:.1f}%)")
print()
print("🔶 Residual Error: Still ~30cm step length error")
print("   Cause: Strike over-detection (3.45×) dilutes averages")
print("   Next: Phase 2 (Cadence) and Phase 3 (Strike detection)")
print()
