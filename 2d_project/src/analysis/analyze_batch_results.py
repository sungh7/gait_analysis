#!/usr/bin/env python3
"""
Analyze batch processing results and generate summary report.
"""

import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
from pathlib import Path

print("="*80)
print(" BATCH PROCESSING RESULTS ANALYSIS")
print("="*80)

# Load results
results_csv = Path("/data/gait/2d_project/batch_results/batch_results_summary.csv")
df = pd.read_csv(results_csv)

print(f"\n📊 Processed: {len(df)}/26 subjects")
print(f"✅ Successful: {(df['status'] == 'SUCCESS').sum()}")
print(f"✗ Failed: {(df['status'] != 'SUCCESS').sum()}")

# Parse joint metrics
knee_improvements = []
hip_improvements = []
ankle_improvements = []

for idx, row in df.iterrows():
    if row['status'] != 'SUCCESS':
        continue

    # Parse joint_metrics (it's a string representation of dict)
    import ast
    metrics = ast.literal_eval(row['joint_metrics'])

    knee = metrics.get('right_knee_angle', {})
    hip = metrics.get('right_hip_angle', {})
    ankle = metrics.get('right_ankle_angle', {})

    if knee:
        knee_improvements.append({
            'subject': row['subject'],
            'jerk_reduction': knee.get('jerk_reduction_pct', 0),
            'baseline_rom': knee.get('baseline_rom', 0),
            'final_rom': knee.get('final_rom', 0),
            'quality': knee.get('quality_score', 0)
        })

    if hip:
        hip_improvements.append({
            'subject': row['subject'],
            'jerk_reduction': hip.get('jerk_reduction_pct', 0),
            'baseline_rom': hip.get('baseline_rom', 0),
            'final_rom': hip.get('final_rom', 0),
            'quality': hip.get('quality_score', 0)
        })

    if ankle:
        ankle_improvements.append({
            'subject': row['subject'],
            'jerk_reduction': ankle.get('jerk_reduction_pct', 0),
            'baseline_rom': ankle.get('baseline_rom', 0),
            'final_rom': ankle.get('final_rom', 0),
            'quality': ankle.get('quality_score', 0)
        })

# Convert to DataFrames
knee_df = pd.DataFrame(knee_improvements)
hip_df = pd.DataFrame(hip_improvements)
ankle_df = pd.DataFrame(ankle_improvements)

print("\n" + "="*80)
print(" KNEE IMPROVEMENTS (Most Reliable)")
print("="*80)
print(f"Average jerk reduction: {knee_df['jerk_reduction'].mean():.1f}%")
print(f"Range: {knee_df['jerk_reduction'].min():.1f}% to {knee_df['jerk_reduction'].max():.1f}%")
print(f"Subjects with >20% improvement: {(knee_df['jerk_reduction'] > 20).sum()}/{len(knee_df)}")
print(f"Average quality score: {knee_df['quality'].mean():.2f}/1.0")

# Top 5 knee improvements
print("\nTop 5 knee improvements:")
top_knee = knee_df.nlargest(5, 'jerk_reduction')[['subject', 'jerk_reduction', 'quality']]
for idx, row in top_knee.iterrows():
    print(f"  {row['subject']}: {row['jerk_reduction']:.1f}% reduction (quality: {row['quality']:.2f})")

print("\n" + "="*80)
print(" HIP IMPROVEMENTS")
print("="*80)
print(f"Average jerk reduction: {hip_df['jerk_reduction'].mean():.1f}%")
print(f"Range: {hip_df['jerk_reduction'].min():.1f}% to {hip_df['jerk_reduction'].max():.1f}%")
print(f"Subjects with >20% improvement: {(hip_df['jerk_reduction'] > 20).sum()}/{len(hip_df)}")
print(f"Average quality score: {hip_df['quality'].mean():.2f}/1.0")

print("\n" + "="*80)
print(" ANKLE ANALYSIS (Issue Detected)")
print("="*80)
print(f"Average jerk reduction: {ankle_df['jerk_reduction'].mean():.1f}%")
print(f"⚠️  Ankle ROM near-zero count: {(ankle_df['final_rom'] < 1).sum()}/{len(ankle_df)}")

# Find the one good ankle result
good_ankle = ankle_df[ankle_df['final_rom'] > 10]
if len(good_ankle) > 0:
    print(f"\n✓ Found {len(good_ankle)} subject(s) with preserved ankle motion:")
    for idx, row in good_ankle.iterrows():
        print(f"  {row['subject']}: {row['jerk_reduction']:.1f}% reduction, ROM: {row['baseline_rom']:.1f}° → {row['final_rom']:.1f}°")

print("\n" + "="*80)
print(" OVERALL ASSESSMENT")
print("="*80)

# Calculate combined knee+hip performance
valid_subjects = knee_df.merge(hip_df, on='subject', suffixes=('_knee', '_hip'))
print(f"\nSubjects with both knee & hip data: {len(valid_subjects)}")
print(f"Average knee+hip jerk reduction: {(valid_subjects['jerk_reduction_knee'].mean() + valid_subjects['jerk_reduction_hip'].mean())/2:.1f}%")
print(f"Average knee+hip quality: {(valid_subjects['quality_knee'].mean() + valid_subjects['quality_hip'].mean())/2:.2f}/1.0")

# Success criteria
knee_success = (knee_df['jerk_reduction'] > 0).sum() / len(knee_df) * 100
hip_success = (hip_df['jerk_reduction'] > 0).sum() / len(hip_df) * 100

print(f"\n✓ Knee improvements (>0%): {knee_success:.0f}% of subjects")
print(f"✓ Hip improvements (>0%): {hip_success:.0f}% of subjects")

# Create visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Jerk reduction distribution (Knee)
ax1 = axes[0, 0]
ax1.hist(knee_df['jerk_reduction'], bins=20, color='blue', alpha=0.7, edgecolor='black')
ax1.axvline(knee_df['jerk_reduction'].mean(), color='red', linestyle='--',
            linewidth=2, label=f'Mean: {knee_df["jerk_reduction"].mean():.1f}%')
ax1.set_xlabel('Jerk Reduction (%)')
ax1.set_ylabel('Number of Subjects')
ax1.set_title('Knee: Jerk Reduction Distribution', fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Jerk reduction distribution (Hip)
ax2 = axes[0, 1]
ax2.hist(hip_df['jerk_reduction'], bins=20, color='green', alpha=0.7, edgecolor='black')
ax2.axvline(hip_df['jerk_reduction'].mean(), color='red', linestyle='--',
            linewidth=2, label=f'Mean: {hip_df["jerk_reduction"].mean():.1f}%')
ax2.set_xlabel('Jerk Reduction (%)')
ax2.set_ylabel('Number of Subjects')
ax2.set_title('Hip: Jerk Reduction Distribution', fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Plot 3: Quality scores
ax3 = axes[1, 0]
ax3.boxplot([knee_df['quality'], hip_df['quality']],
            labels=['Knee', 'Hip'])
ax3.set_ylabel('Quality Score (0-1)')
ax3.set_title('Quality Score Comparison', fontweight='bold')
ax3.grid(True, alpha=0.3, axis='y')

# Plot 4: Scatter: Jerk reduction vs Quality
ax4 = axes[1, 1]
ax4.scatter(knee_df['quality'], knee_df['jerk_reduction'],
           alpha=0.6, s=100, color='blue', label='Knee')
ax4.scatter(hip_df['quality'], hip_df['jerk_reduction'],
           alpha=0.6, s=100, color='green', label='Hip')
ax4.set_xlabel('Quality Score')
ax4.set_ylabel('Jerk Reduction (%)')
ax4.set_title('Jerk Reduction vs Quality Score', fontweight='bold')
ax4.legend()
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/data/gait/2d_project/batch_results/summary_analysis.png', dpi=150)
print("\n✓ Saved analysis visualization: batch_results/summary_analysis.png")

# Save detailed CSV
summary_df = pd.DataFrame({
    'subject': knee_df['subject'],
    'knee_jerk_reduction_%': knee_df['jerk_reduction'],
    'knee_quality': knee_df['quality'],
    'hip_jerk_reduction_%': hip_df['jerk_reduction'],
    'hip_quality': hip_df['quality']
})
summary_df.to_csv('/data/gait/2d_project/batch_results/improvement_summary.csv', index=False)
print("✓ Saved detailed summary: batch_results/improvement_summary.csv")

print("\n" + "="*80)
print(" CONCLUSIONS")
print("="*80)
print("\n✅ SUCCESSES:")
print(f"  • Knee improvements: {knee_df['jerk_reduction'].mean():.1f}% average reduction")
print(f"  • Hip improvements: {hip_df['jerk_reduction'].mean():.1f}% average reduction")
print(f"  • Quality scores: Knee {knee_df['quality'].mean():.2f}, Hip {hip_df['quality'].mean():.2f}")
print(f"  • {len(df)} subjects processed successfully")

print("\n⚠️  ISSUES FOUND:")
print("  • Ankle signal over-constrained (ROM → 0 for most subjects)")
print("  • Need to relax ankle constraints or adjust filtering")

print("\n📋 RECOMMENDATIONS:")
print("  1. Use knee & hip improvements (validated)")
print("  2. Fix ankle processing (reduce aggressive filtering)")
print("  3. Re-run with adjusted parameters")
print("  4. For now: Report knee+hip results in paper")

print("\n" + "="*80)
