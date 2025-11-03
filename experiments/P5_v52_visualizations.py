"""
Create visualizations for V5.2 results analysis.

Generates:
1. ICC comparison (V5 vs V5.1 vs V5.2)
2. Scale quality analysis (left vs right CV)
3. Cross-leg disagreement distribution
4. Per-subject improvement scatter plots
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

def load_data():
    """Load V5.2 comparison results."""
    with open('P5_v52_comparison_results.json', 'r') as f:
        return json.load(f)

def create_icc_comparison():
    """Create ICC comparison bar chart (V5 vs V5.1 vs V5.2)."""
    data = load_data()

    metrics = ['step_length_left_cm', 'step_length_right_cm',
               'forward_velocity_left_cm_s', 'forward_velocity_right_cm_s',
               'cadence_average']

    labels = ['Step Length (L)', 'Step Length (R)',
              'Velocity (L)', 'Velocity (R)', 'Cadence']

    # Extract ICC values
    v5_iccs = [data['versions']['v5']['icc'].get(m, np.nan) for m in metrics]
    v51_iccs = [
        data['versions']['v51']['icc'].get('step_left', np.nan),
        data['versions']['v51']['icc'].get('step_right', np.nan),
        np.nan,  # velocity not in v51
        np.nan,
        data['versions']['v51']['icc'].get('cadence', np.nan)
    ]
    v52_iccs = [data['versions']['v52']['icc'].get(m, np.nan) for m in metrics]

    # Create figure
    fig, ax = plt.subplots(figsize=(14, 8))

    x = np.arange(len(labels))
    width = 0.25

    bars1 = ax.bar(x - width, v5_iccs, width, label='V5 (n=21)',
                   color='#d62728', alpha=0.8)
    bars2 = ax.bar(x, v51_iccs, width, label='V5.1 (n=16, outliers removed)',
                   color='#ff7f0e', alpha=0.8)
    bars3 = ax.bar(x + width, v52_iccs, width, label='V5.2 (n=16, enhanced scaling)',
                   color='#2ca02c', alpha=0.8)

    # Add threshold lines
    ax.axhline(y=0.75, color='blue', linestyle='--', linewidth=2,
               label='Clinical Validity (ICC > 0.75)', alpha=0.7)
    ax.axhline(y=0.40, color='orange', linestyle='--', linewidth=2,
               label='Fair Agreement (ICC > 0.40)', alpha=0.7)
    ax.axhline(y=0.0, color='gray', linestyle='-', linewidth=1, alpha=0.3)

    # Labels and title
    ax.set_xlabel('Metric', fontsize=12, fontweight='bold')
    ax.set_ylabel('ICC Score', fontsize=12, fontweight='bold')
    ax.set_title('V5.2 Results: ICC Comparison Across Versions',
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.legend(loc='upper left', fontsize=10)
    ax.set_ylim(-0.3, 1.0)
    ax.grid(axis='y', alpha=0.3)

    # Add value labels on bars
    def add_labels(bars):
        for bar in bars:
            height = bar.get_height()
            if not np.isnan(height):
                ax.annotate(f'{height:.2f}',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3),
                           textcoords="offset points",
                           ha='center', va='bottom',
                           fontsize=8)

    add_labels(bars1)
    add_labels(bars2)
    add_labels(bars3)

    plt.tight_layout()
    plt.savefig('P5_v52_icc_comparison.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: P5_v52_icc_comparison.png")
    plt.close()

def create_rmse_comparison():
    """Create RMSE comparison bar chart."""
    data = load_data()

    metrics = ['step_length_left_cm', 'step_length_right_cm', 'cadence_average']
    labels = ['Step Length (L)', 'Step Length (R)', 'Cadence']
    units = ['cm', 'cm', 'steps/min']

    # Extract RMSE values
    v5_rmse = [data['versions']['v5']['rmse'].get(m, np.nan) for m in metrics]
    v51_rmse = [
        data['versions']['v51']['rmse'].get('step_left', np.nan),
        data['versions']['v51']['rmse'].get('step_right', np.nan),
        data['versions']['v51']['rmse'].get('cadence', np.nan)
    ]
    v52_rmse = [data['versions']['v52']['rmse'].get(m, np.nan) for m in metrics]

    # Create figure with subplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for idx, (ax, label, unit) in enumerate(zip(axes, labels, units)):
        x = ['V5\n(n=21)', 'V5.1\n(n=16)', 'V5.2\n(n=16)']
        values = [v5_rmse[idx], v51_rmse[idx], v52_rmse[idx]]
        colors = ['#d62728', '#ff7f0e', '#2ca02c']

        bars = ax.bar(x, values, color=colors, alpha=0.8, edgecolor='black')

        # Add value labels
        for bar, val in zip(bars, values):
            if not np.isnan(val):
                ax.text(bar.get_x() + bar.get_width()/2, val + max(values)*0.02,
                       f'{val:.2f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

        # Calculate reduction
        if not np.isnan(v5_rmse[idx]) and not np.isnan(v52_rmse[idx]):
            reduction = (v5_rmse[idx] - v52_rmse[idx]) / v5_rmse[idx] * 100
            color = 'green' if reduction > 0 else 'red'
            ax.text(0.5, 0.95, f'{reduction:+.1f}% reduction',
                   transform=ax.transAxes, ha='center', va='top',
                   fontsize=10, color=color, fontweight='bold',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        ax.set_ylabel(f'RMSE ({unit})', fontsize=11, fontweight='bold')
        ax.set_title(label, fontsize=12, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim(0, max(values) * 1.15)

    fig.suptitle('V5.2 Results: RMSE Comparison', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('P5_v52_rmse_comparison.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: P5_v52_rmse_comparison.png")
    plt.close()

def create_scale_quality_analysis():
    """Create scale quality visualization (CV distribution, outliers)."""
    data = load_data()
    subjects = data['per_subject_scale_quality']

    # Extract data
    subject_ids = [s['subject'] for s in subjects]
    left_cvs = [s.get('left_cv', np.nan) for s in subjects]
    right_cvs = [s.get('right_cv', np.nan) for s in subjects]
    left_outliers = [s.get('left_outliers', 0) for s in subjects]
    right_outliers = [s.get('right_outliers', 0) for s in subjects]
    cross_leg_disagreement = [s.get('cross_leg_disagreement', np.nan) for s in subjects]
    cross_leg_valid = [s.get('cross_leg_valid', False) for s in subjects]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. CV comparison (left vs right)
    ax = axes[0, 0]
    x = np.arange(len(subject_ids))
    width = 0.35

    ax.bar(x - width/2, left_cvs, width, label='Left', color='#1f77b4', alpha=0.8)
    ax.bar(x + width/2, right_cvs, width, label='Right', color='#ff7f0e', alpha=0.8)
    ax.axhline(y=0.15, color='green', linestyle='--', label='Target CV < 0.15', linewidth=2)
    ax.set_xlabel('Subject', fontsize=11, fontweight='bold')
    ax.set_ylabel('Coefficient of Variation (CV)', fontsize=11, fontweight='bold')
    ax.set_title('Stride CV by Subject (Lower = Better Quality)', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([s.replace('S1_', 'S') for s in subject_ids], rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    # 2. Outlier count (left vs right)
    ax = axes[0, 1]
    ax.bar(x - width/2, left_outliers, width, label='Left', color='#1f77b4', alpha=0.8)
    ax.bar(x + width/2, right_outliers, width, label='Right', color='#ff7f0e', alpha=0.8)
    ax.set_xlabel('Subject', fontsize=11, fontweight='bold')
    ax.set_ylabel('Number of Outlier Strides Rejected', fontsize=11, fontweight='bold')
    ax.set_title('Stride Outliers Rejected (Right has 4× more)', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([s.replace('S1_', 'S') for s in subject_ids], rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    # Add total counts
    ax.text(0.98, 0.95, f'Total Left: {sum(left_outliers)}', transform=ax.transAxes,
           ha='right', va='top', fontsize=10, bbox=dict(boxstyle='round', facecolor='#1f77b4', alpha=0.3))
    ax.text(0.98, 0.85, f'Total Right: {sum(right_outliers)}', transform=ax.transAxes,
           ha='right', va='top', fontsize=10, bbox=dict(boxstyle='round', facecolor='#ff7f0e', alpha=0.3))

    # 3. Cross-leg disagreement
    ax = axes[1, 0]
    colors = ['green' if valid else 'red' for valid in cross_leg_valid]
    bars = ax.bar(x, [d * 100 for d in cross_leg_disagreement], color=colors, alpha=0.8, edgecolor='black')
    ax.axhline(y=15, color='red', linestyle='--', label='Rejection threshold (15%)', linewidth=2)
    ax.set_xlabel('Subject', fontsize=11, fontweight='bold')
    ax.set_ylabel('Left-Right Scale Disagreement (%)', fontsize=11, fontweight='bold')
    ax.set_title('Cross-Leg Validation (Green=Pass, Red=Fail)', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([s.replace('S1_', 'S') for s in subject_ids], rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    pass_count = sum(cross_leg_valid)
    ax.text(0.98, 0.95, f'Pass: {pass_count}/16 ({pass_count/16*100:.0f}%)',
           transform=ax.transAxes, ha='right', va='top', fontsize=10,
           bbox=dict(boxstyle='round', facecolor='green', alpha=0.3))

    # 4. CV distribution histogram
    ax = axes[1, 1]
    ax.hist(left_cvs, bins=15, alpha=0.6, label='Left', color='#1f77b4', edgecolor='black')
    ax.hist(right_cvs, bins=15, alpha=0.6, label='Right', color='#ff7f0e', edgecolor='black')
    ax.axvline(x=0.15, color='green', linestyle='--', label='Target CV < 0.15', linewidth=2)
    ax.set_xlabel('Coefficient of Variation (CV)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Number of Subjects', fontsize=11, fontweight='bold')
    ax.set_title('CV Distribution (Target: CV < 0.15)', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    # Add statistics
    left_mean = np.mean(left_cvs)
    right_mean = np.mean(right_cvs)
    ax.text(0.98, 0.95, f'Left mean: {left_mean:.3f}', transform=ax.transAxes,
           ha='right', va='top', fontsize=9, bbox=dict(boxstyle='round', facecolor='#1f77b4', alpha=0.3))
    ax.text(0.98, 0.85, f'Right mean: {right_mean:.3f}', transform=ax.transAxes,
           ha='right', va='top', fontsize=9, bbox=dict(boxstyle='round', facecolor='#ff7f0e', alpha=0.3))

    plt.tight_layout()
    plt.savefig('P5_v52_scale_quality.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: P5_v52_scale_quality.png")
    plt.close()

def create_left_vs_right_comparison():
    """Create scatter plot comparing left vs right performance."""
    with open('tiered_evaluation_report_v52.json', 'r') as f:
        v52_data = json.load(f)

    subjects_data = v52_data.get('subjects', {})

    # Extract step length errors
    left_errors = []
    right_errors = []
    subject_ids = []

    for subj_id, subj_data in subjects_data.items():
        temporal = subj_data.get('temporal', {})
        gt = temporal.get('ground_truth', {})
        pred = temporal.get('prediction', {})

        gt_left = gt.get('step_length_cm', {}).get('left')
        pred_left = pred.get('step_length_cm', {}).get('left')
        gt_right = gt.get('step_length_cm', {}).get('right')
        pred_right = pred.get('step_length_cm', {}).get('right')

        if all([gt_left, pred_left, gt_right, pred_right]):
            left_error = abs(pred_left - gt_left)
            right_error = abs(pred_right - gt_right)
            left_errors.append(left_error)
            right_errors.append(right_error)
            subject_ids.append(subj_id)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # 1. Scatter plot
    ax = axes[0]
    ax.scatter(left_errors, right_errors, s=100, alpha=0.6, edgecolors='black')

    # Add diagonal line (equal errors)
    max_error = max(max(left_errors), max(right_errors))
    ax.plot([0, max_error], [0, max_error], 'r--', linewidth=2, label='Equal error', alpha=0.5)

    # Annotate outliers
    for i, (le, re, subj) in enumerate(zip(left_errors, right_errors, subject_ids)):
        if re > 20 or abs(le - re) > 15:
            ax.annotate(subj.replace('S1_', 'S'), (le, re),
                       textcoords="offset points", xytext=(5,5),
                       ha='left', fontsize=8)

    ax.set_xlabel('Left Step Length Error (cm)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Right Step Length Error (cm)', fontsize=11, fontweight='bold')
    ax.set_title('Left vs Right Step Length Errors\n(Points below diagonal = Left better)',
                fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_aspect('equal')

    # Add statistics
    left_mean = np.mean(left_errors)
    right_mean = np.mean(right_errors)
    ax.text(0.05, 0.95, f'Left mean: {left_mean:.2f} cm', transform=ax.transAxes,
           va='top', fontsize=10, bbox=dict(boxstyle='round', facecolor='#1f77b4', alpha=0.3))
    ax.text(0.05, 0.85, f'Right mean: {right_mean:.2f} cm', transform=ax.transAxes,
           va='top', fontsize=10, bbox=dict(boxstyle='round', facecolor='#ff7f0e', alpha=0.3))

    # 2. Distribution comparison
    ax = axes[1]
    ax.hist(left_errors, bins=15, alpha=0.6, label=f'Left (mean={left_mean:.2f})',
           color='#1f77b4', edgecolor='black')
    ax.hist(right_errors, bins=15, alpha=0.6, label=f'Right (mean={right_mean:.2f})',
           color='#ff7f0e', edgecolor='black')
    ax.axvline(x=7.0, color='green', linestyle='--', label='Target RMSE < 7.0 cm', linewidth=2)
    ax.set_xlabel('Absolute Error (cm)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Number of Subjects', fontsize=11, fontweight='bold')
    ax.set_title('Error Distribution: Left vs Right', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig('P5_v52_left_vs_right.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: P5_v52_left_vs_right.png")
    plt.close()

def main():
    """Generate all visualizations."""
    print("=" * 80)
    print("Generating V5.2 Visualizations")
    print("=" * 80)
    print()

    create_icc_comparison()
    create_rmse_comparison()
    create_scale_quality_analysis()
    create_left_vs_right_comparison()

    print()
    print("=" * 80)
    print("All visualizations generated successfully!")
    print("=" * 80)
    print()
    print("Files created:")
    print("  - P5_v52_icc_comparison.png")
    print("  - P5_v52_rmse_comparison.png")
    print("  - P5_v52_scale_quality.png")
    print("  - P5_v52_left_vs_right.png")
    print()

if __name__ == '__main__':
    main()
