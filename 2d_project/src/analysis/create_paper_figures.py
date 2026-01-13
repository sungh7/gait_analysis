#!/usr/bin/env python3
"""
Generate publication-ready figures for research paper.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import ast

# Set publication style
plt.style.use('seaborn-v0_8-paper')
sns.set_palette("husl")

print("="*80)
print(" GENERATING PUBLICATION-READY FIGURES")
print("="*80)

# Load results
df = pd.read_csv('/data/gait/2d_project/batch_results/batch_results_summary.csv')

# Parse joint metrics
knee_data = []
hip_data = []

for idx, row in df.iterrows():
    if row['status'] != 'SUCCESS':
        continue

    metrics = ast.literal_eval(str(row['joint_metrics']))
    subject = row['subject']

    knee = metrics.get('right_knee_angle', {})
    hip = metrics.get('right_hip_angle', {})

    if knee:
        knee_data.append({
            'subject': subject,
            'jerk_reduction': knee.get('jerk_reduction_pct', 0),
            'quality': knee.get('quality_score', 0),
            'baseline_jerk': knee.get('baseline_jerk', 0),
            'improved_jerk': knee.get('final_jerk', 0)
        })

    if hip:
        hip_data.append({
            'subject': subject,
            'jerk_reduction': hip.get('jerk_reduction_pct', 0),
            'quality': hip.get('quality_score', 0),
            'baseline_jerk': hip.get('baseline_jerk', 0),
            'improved_jerk': hip.get('final_jerk', 0)
        })

knee_df = pd.DataFrame(knee_data)
hip_df = pd.DataFrame(hip_data)

# ============================================================================
# FIGURE 1: Improvement Distribution (Publication Quality)
# ============================================================================
print("\n[1/3] Creating Figure 1: Improvement Distribution...")

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Knee
ax1 = axes[0]
ax1.hist(knee_df['jerk_reduction'], bins=15, color='#2E86AB',
         alpha=0.7, edgecolor='black', linewidth=1.5)
ax1.axvline(knee_df['jerk_reduction'].mean(), color='#C1121F',
            linestyle='--', linewidth=2.5,
            label=f'Mean: {knee_df["jerk_reduction"].mean():.1f}%')
ax1.set_xlabel('Jerk Reduction (%)', fontsize=12, fontweight='bold')
ax1.set_ylabel('Number of Subjects', fontsize=12, fontweight='bold')
ax1.set_title('A. Knee Joint', fontsize=14, fontweight='bold')
ax1.legend(fontsize=11, frameon=True, shadow=True)
ax1.grid(True, alpha=0.3, linestyle='--')
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

# Hip
ax2 = axes[1]
ax2.hist(hip_df['jerk_reduction'], bins=15, color='#A23B72',
         alpha=0.7, edgecolor='black', linewidth=1.5)
ax2.axvline(hip_df['jerk_reduction'].mean(), color='#C1121F',
            linestyle='--', linewidth=2.5,
            label=f'Mean: {hip_df["jerk_reduction"].mean():.1f}%')
ax2.set_xlabel('Jerk Reduction (%)', fontsize=12, fontweight='bold')
ax2.set_ylabel('Number of Subjects', fontsize=12, fontweight='bold')
ax2.set_title('B. Hip Joint', fontsize=14, fontweight='bold')
ax2.legend(fontsize=11, frameon=True, shadow=True)
ax2.grid(True, alpha=0.3, linestyle='--')
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig('/data/gait/2d_project/batch_results/Figure_Improvement_Distribution.png',
            dpi=300, bbox_inches='tight')
print("  ✓ Saved: Figure_Improvement_Distribution.png (300 DPI)")

# ============================================================================
# FIGURE 2: Before/After Comparison (Violin Plot)
# ============================================================================
print("\n[2/3] Creating Figure 2: Before/After Comparison...")

fig, axes = plt.subplots(1, 2, figsize=(10, 5))

# Prepare data
knee_before_after = pd.DataFrame({
    'Jerk': list(knee_df['baseline_jerk']) + list(knee_df['improved_jerk']),
    'Condition': ['Baseline']*len(knee_df) + ['Improved']*len(knee_df),
    'Joint': ['Knee']*(2*len(knee_df))
})

hip_before_after = pd.DataFrame({
    'Jerk': list(hip_df['baseline_jerk']) + list(hip_df['improved_jerk']),
    'Condition': ['Baseline']*len(hip_df) + ['Improved']*len(hip_df),
    'Joint': ['Hip']*(2*len(hip_df))
})

# Knee violin plot
ax1 = axes[0]
parts = ax1.violinplot([knee_df['baseline_jerk'], knee_df['improved_jerk']],
                        positions=[1, 2], showmeans=True, showmedians=True)
for pc in parts['bodies']:
    pc.set_facecolor('#2E86AB')
    pc.set_alpha(0.7)
ax1.set_xticks([1, 2])
ax1.set_xticklabels(['Baseline', 'Improved'], fontsize=11, fontweight='bold')
ax1.set_ylabel('Jerk (arbitrary units)', fontsize=12, fontweight='bold')
ax1.set_title('A. Knee Joint', fontsize=14, fontweight='bold')
ax1.grid(True, alpha=0.3, axis='y', linestyle='--')
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

# Add statistical annotation
mean_baseline_knee = knee_df['baseline_jerk'].mean()
mean_improved_knee = knee_df['improved_jerk'].mean()
reduction_knee = (mean_baseline_knee - mean_improved_knee) / mean_baseline_knee * 100
ax1.text(1.5, ax1.get_ylim()[1]*0.95, f'{reduction_knee:.1f}% reduction',
         ha='center', fontsize=11, fontweight='bold',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# Hip violin plot
ax2 = axes[1]
parts = ax2.violinplot([hip_df['baseline_jerk'], hip_df['improved_jerk']],
                        positions=[1, 2], showmeans=True, showmedians=True)
for pc in parts['bodies']:
    pc.set_facecolor('#A23B72')
    pc.set_alpha(0.7)
ax2.set_xticks([1, 2])
ax2.set_xticklabels(['Baseline', 'Improved'], fontsize=11, fontweight='bold')
ax2.set_ylabel('Jerk (arbitrary units)', fontsize=12, fontweight='bold')
ax2.set_title('B. Hip Joint', fontsize=14, fontweight='bold')
ax2.grid(True, alpha=0.3, axis='y', linestyle='--')
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

# Add statistical annotation
mean_baseline_hip = hip_df['baseline_jerk'].mean()
mean_improved_hip = hip_df['improved_jerk'].mean()
reduction_hip = (mean_baseline_hip - mean_improved_hip) / mean_baseline_hip * 100
ax2.text(1.5, ax2.get_ylim()[1]*0.95, f'{reduction_hip:.1f}% reduction',
         ha='center', fontsize=11, fontweight='bold',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig('/data/gait/2d_project/batch_results/Figure_BeforeAfter_Comparison.png',
            dpi=300, bbox_inches='tight')
print("  ✓ Saved: Figure_BeforeAfter_Comparison.png (300 DPI)")

# ============================================================================
# FIGURE 3: Subject-by-Subject Improvement
# ============================================================================
print("\n[3/3] Creating Figure 3: Subject-by-Subject Improvement...")

fig, ax = plt.subplots(1, 1, figsize=(14, 6))

# Sort by knee improvement
knee_df_sorted = knee_df.sort_values('jerk_reduction', ascending=False)
x_pos = np.arange(len(knee_df_sorted))

# Bar plot
bars = ax.bar(x_pos, knee_df_sorted['jerk_reduction'],
              color=['#2E86AB' if x > 20 else '#6C757D' for x in knee_df_sorted['jerk_reduction']],
              alpha=0.8, edgecolor='black', linewidth=1)

# Add threshold line
ax.axhline(20, color='#C1121F', linestyle='--', linewidth=2,
           label='20% Threshold', alpha=0.7)
ax.axhline(knee_df['jerk_reduction'].mean(), color='#FCA311', linestyle='-',
           linewidth=2.5, label=f'Mean: {knee_df["jerk_reduction"].mean():.1f}%')

ax.set_xticks(x_pos)
ax.set_xticklabels(knee_df_sorted['subject'], rotation=45, ha='right', fontsize=9)
ax.set_xlabel('Subject ID', fontsize=12, fontweight='bold')
ax.set_ylabel('Jerk Reduction (%)', fontsize=12, fontweight='bold')
ax.set_title('Subject-by-Subject Knee Joint Improvement (N=22)',
             fontsize=14, fontweight='bold')
ax.legend(fontsize=11, frameon=True, shadow=True, loc='upper right')
ax.grid(True, alpha=0.3, axis='y', linestyle='--')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Annotate top 3
for i in range(min(3, len(x_pos))):
    height = knee_df_sorted['jerk_reduction'].iloc[i]
    ax.text(i, height + 1, f'{height:.1f}%', ha='center', va='bottom',
            fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig('/data/gait/2d_project/batch_results/Figure_Subject_Improvements.png',
            dpi=300, bbox_inches='tight')
print("  ✓ Saved: Figure_Subject_Improvements.png (300 DPI)")

# ============================================================================
# Summary Statistics Table (CSV for paper)
# ============================================================================
print("\n[Bonus] Creating summary statistics table...")

summary_stats = pd.DataFrame({
    'Joint': ['Knee', 'Hip'],
    'N': [len(knee_df), len(hip_df)],
    'Mean_Jerk_Reduction_%': [
        f"{knee_df['jerk_reduction'].mean():.1f}",
        f"{hip_df['jerk_reduction'].mean():.1f}"
    ],
    'SD': [
        f"{knee_df['jerk_reduction'].std():.1f}",
        f"{hip_df['jerk_reduction'].std():.1f}"
    ],
    'Range': [
        f"{knee_df['jerk_reduction'].min():.1f} - {knee_df['jerk_reduction'].max():.1f}",
        f"{hip_df['jerk_reduction'].min():.1f} - {hip_df['jerk_reduction'].max():.1f}"
    ],
    'Subjects_>20%': [
        f"{(knee_df['jerk_reduction'] > 20).sum()}/{len(knee_df)}",
        f"{(hip_df['jerk_reduction'] > 20).sum()}/{len(hip_df)}"
    ],
    'Mean_Quality_Score': [
        f"{knee_df['quality'].mean():.2f}",
        f"{hip_df['quality'].mean():.2f}"
    ]
})

summary_stats.to_csv('/data/gait/2d_project/batch_results/Table_Summary_Statistics.csv',
                     index=False)
print("  ✓ Saved: Table_Summary_Statistics.csv")

print("\n" + "="*80)
print(" PUBLICATION FIGURES COMPLETE")
print("="*80)
print("\nGenerated files:")
print("  1. Figure_Improvement_Distribution.png (300 DPI)")
print("  2. Figure_BeforeAfter_Comparison.png (300 DPI)")
print("  3. Figure_Subject_Improvements.png (300 DPI)")
print("  4. Table_Summary_Statistics.csv")
print("\nAll files saved to: /data/gait/2d_project/batch_results/")
print("="*80)
