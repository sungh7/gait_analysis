"""
Phase 5.1: Create Visualizations for Paper

Generates publication-ready figures comparing:
1. V5 vs V5.1 (outlier-rejected) ICC scores
2. Error distributions before/after outlier rejection
3. Per-subject error waterfall chart
4. Improvement roadmap visualization
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set publication style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 150
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9


def create_icc_comparison_plot():
    """Create ICC comparison bar chart: V5 vs V5.1"""

    # Load outlier analysis results
    with open('P5_outlier_analysis_results.json') as f:
        results = json.load(f)

    metrics = ['Cadence', 'Step Length (L)', 'Step Length (R)']
    v5_iccs = [
        results['icc']['original']['cadence'],
        results['icc']['original']['step_left'],
        results['icc']['original']['step_right']
    ]
    v51_iccs = [
        results['icc']['cleaned']['cadence'],
        results['icc']['cleaned']['step_left'],
        results['icc']['cleaned']['step_right']
    ]

    x = np.arange(len(metrics))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))

    bars1 = ax.bar(x - width/2, v5_iccs, width, label='V5 (n=21)',
                   color='#e74c3c', alpha=0.8)
    bars2 = ax.bar(x + width/2, v51_iccs, width, label='V5.1 (n=16, outliers removed)',
                   color='#27ae60', alpha=0.8)

    # Add clinical validity threshold line
    ax.axhline(y=0.75, color='blue', linestyle='--', linewidth=2,
               label='Clinical Validity (ICC > 0.75)', alpha=0.7)
    ax.axhline(y=0.40, color='orange', linestyle='--', linewidth=2,
               label='Fair Agreement (ICC > 0.40)', alpha=0.7)
    ax.axhline(y=0, color='gray', linestyle='-', linewidth=0.5, alpha=0.5)

    ax.set_ylabel('ICC Score', fontweight='bold')
    ax.set_title('ICC Comparison: V5 vs V5.1 (Outlier-Rejected)', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend(loc='upper left', fontsize=9)
    ax.set_ylim([-0.4, 0.9])

    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.2f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3 if height >= 0 else -15),
                       textcoords="offset points",
                       ha='center', va='bottom' if height >= 0 else 'top',
                       fontsize=8, fontweight='bold')

    plt.tight_layout()
    plt.savefig('P5_ICC_comparison.png', dpi=150, bbox_inches='tight')
    print("✅ Created: P5_ICC_comparison.png")
    plt.close()


def create_rmse_comparison_plot():
    """Create RMSE comparison bar chart"""

    with open('P5_outlier_analysis_results.json') as f:
        results = json.load(f)

    metrics = ['Cadence\n(steps/min)', 'Step Length (L)\n(cm)', 'Step Length (R)\n(cm)']
    v5_rmse = [
        results['rmse']['original']['cadence'],
        results['rmse']['original']['step_left'],
        results['rmse']['original']['step_right']
    ]
    v51_rmse = [
        results['rmse']['cleaned']['cadence'],
        results['rmse']['cleaned']['step_left'],
        results['rmse']['cleaned']['step_right']
    ]

    x = np.arange(len(metrics))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))

    bars1 = ax.bar(x - width/2, v5_rmse, width, label='V5 (n=21)',
                   color='#e74c3c', alpha=0.8)
    bars2 = ax.bar(x + width/2, v51_rmse, width, label='V5.1 (n=16)',
                   color='#27ae60', alpha=0.8)

    ax.set_ylabel('RMSE', fontweight='bold')
    ax.set_title('RMSE Comparison: V5 vs V5.1 (Outlier-Rejected)', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend(loc='upper right')

    # Add value labels and improvement percentage
    for i, (bar1, bar2) in enumerate(zip(bars1, bars2)):
        h1 = bar1.get_height()
        h2 = bar2.get_height()
        improvement = (h2 - h1) / h1 * 100

        ax.annotate(f'{h1:.1f}',
                   xy=(bar1.get_x() + bar1.get_width() / 2, h1),
                   xytext=(0, 3),
                   textcoords="offset points",
                   ha='center', va='bottom',
                   fontsize=8, fontweight='bold')

        ax.annotate(f'{h2:.1f}',
                   xy=(bar2.get_x() + bar2.get_width() / 2, h2),
                   xytext=(0, 3),
                   textcoords="offset points",
                   ha='center', va='bottom',
                   fontsize=8, fontweight='bold')

        # Add improvement annotation
        mid_x = x[i]
        mid_y = max(h1, h2) + 1
        ax.annotate(f'{improvement:+.0f}%',
                   xy=(mid_x, mid_y),
                   ha='center', va='bottom',
                   fontsize=9, fontweight='bold',
                   color='green' if improvement < 0 else 'red')

    plt.tight_layout()
    plt.savefig('P5_RMSE_comparison.png', dpi=150, bbox_inches='tight')
    print("✅ Created: P5_RMSE_comparison.png")
    plt.close()


def create_outlier_identification_plot():
    """Create plot showing which subjects are outliers"""

    with open('tiered_evaluation_report_v5.json') as f:
        v5_data = json.load(f)

    with open('P5_outlier_analysis_results.json') as f:
        outlier_results = json.load(f)

    outliers = set(outlier_results['outliers']['consensus'])

    # Extract cadence errors for all subjects
    subjects = []
    cadence_errors = []
    is_outlier = []

    for subj_id, metrics in v5_data['subjects'].items():
        temporal = metrics['temporal']
        gt = temporal['ground_truth']['cadence_steps_min']['average']
        pred = temporal['prediction']['cadence_steps_min']['average']
        error = abs(pred - gt)

        subjects.append(subj_id)
        cadence_errors.append(error)
        is_outlier.append(subj_id in outliers)

    # Sort by error
    sorted_indices = np.argsort(cadence_errors)[::-1]
    subjects = [subjects[i] for i in sorted_indices]
    cadence_errors = [cadence_errors[i] for i in sorted_indices]
    is_outlier = [is_outlier[i] for i in sorted_indices]

    # Create plot
    fig, ax = plt.subplots(figsize=(12, 6))

    colors = ['#e74c3c' if outlier else '#3498db' for outlier in is_outlier]
    bars = ax.barh(subjects, cadence_errors, color=colors, alpha=0.8)

    # Add threshold line
    threshold = 30  # 30% error threshold
    mean_gt_cadence = 113  # approximate
    threshold_value = mean_gt_cadence * 0.3
    ax.axvline(x=threshold_value, color='orange', linestyle='--', linewidth=2,
               label='Outlier Threshold (~30% error)', alpha=0.7)

    ax.set_xlabel('Absolute Cadence Error (steps/min)', fontweight='bold')
    ax.set_ylabel('Subject ID', fontweight='bold')
    ax.set_title('Cadence Errors by Subject (Red = Outliers)', fontweight='bold')
    ax.legend(loc='lower right')
    ax.grid(axis='x', alpha=0.3)

    # Annotate outliers
    for i, (subj, error, outlier) in enumerate(zip(subjects, cadence_errors, is_outlier)):
        if outlier:
            ax.text(error + 1, i, f'{error:.1f}', va='center',
                   fontsize=8, fontweight='bold', color='red')

    plt.tight_layout()
    plt.savefig('P5_outlier_identification.png', dpi=150, bbox_inches='tight')
    print("✅ Created: P5_outlier_identification.png")
    plt.close()


def create_improvement_roadmap():
    """Create visual roadmap showing ICC improvement path"""

    versions = ['V5\n(current)', 'V5.1\n(outlier\nrejection)', 'V5.2\n(scale\nrefinement)',
                'V5.3\n(adaptive\nthresholds)', 'V6\n(ensemble\nmethods)']
    icc_cadence = [-0.13, 0.61, 0.50, 0.55, 0.70]
    icc_step = [-0.20, 0.01, 0.35, 0.50, 0.65]
    timeline = ['Done', 'Done', '2-3 weeks', '1 week', '2-3 months']

    fig, ax = plt.subplots(figsize=(12, 7))

    x = np.arange(len(versions))
    width = 0.35

    bars1 = ax.bar(x - width/2, icc_cadence, width, label='Cadence',
                   color='#3498db', alpha=0.8)
    bars2 = ax.bar(x + width/2, icc_step, width, label='Step Length (avg)',
                   color='#e74c3c', alpha=0.8)

    # Add clinical validity threshold
    ax.axhline(y=0.75, color='green', linestyle='--', linewidth=2,
               label='Clinical Validity (ICC > 0.75)', alpha=0.7)
    ax.axhline(y=0.40, color='orange', linestyle='--', linewidth=2,
               label='Fair Agreement (ICC > 0.40)', alpha=0.7)
    ax.axhline(y=0, color='gray', linestyle='-', linewidth=0.5, alpha=0.5)

    ax.set_ylabel('ICC Score', fontweight='bold')
    ax.set_title('ICC Improvement Roadmap: V5 → V6', fontweight='bold', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(versions)
    ax.legend(loc='upper left', fontsize=10)
    ax.set_ylim([-0.3, 0.9])

    # Add timeline annotations
    for i, (ver, time) in enumerate(zip(versions, timeline)):
        ax.text(i, -0.27, time, ha='center', va='top',
               fontsize=9, style='italic', color='gray')

    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.2f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3 if height >= 0 else -15),
                       textcoords="offset points",
                       ha='center', va='bottom' if height >= 0 else 'top',
                       fontsize=8, fontweight='bold')

    # Add milestone markers
    ax.scatter([1], [0.61], s=200, marker='*', color='gold', edgecolor='black',
              linewidth=1.5, zorder=5, label='Current Milestone')

    ax.legend(loc='upper left', fontsize=10)

    plt.tight_layout()
    plt.savefig('P5_improvement_roadmap.png', dpi=150, bbox_inches='tight')
    print("✅ Created: P5_improvement_roadmap.png")
    plt.close()


def main():
    """Generate all visualizations"""
    print("="*80)
    print("Generating V5.1 Visualizations")
    print("="*80)
    print()

    create_icc_comparison_plot()
    create_rmse_comparison_plot()
    create_outlier_identification_plot()
    create_improvement_roadmap()

    print()
    print("="*80)
    print("All visualizations created successfully!")
    print("="*80)
    print()
    print("Generated files:")
    print("  1. P5_ICC_comparison.png - ICC scores V5 vs V5.1")
    print("  2. P5_RMSE_comparison.png - RMSE comparison")
    print("  3. P5_outlier_identification.png - Which subjects are outliers")
    print("  4. P5_improvement_roadmap.png - Path from V5 to V6")
    print()


if __name__ == '__main__':
    main()
