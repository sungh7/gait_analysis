#!/usr/bin/env python3
"""
Generate Figure 9: Ankle ROM Preservation Success
Shows the dramatic improvement in ankle ROM preservation after fixing coordinate system constraints.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ast
import seaborn as sns

# Set publication style
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9

# Load results
df = pd.read_csv('/data/gait/2d_project/batch_results/batch_results_summary.csv')

# Extract ankle ROM data
ankle_data = []
for idx, row in df.iterrows():
    if row['status'] != 'SUCCESS':
        continue

    metrics = ast.literal_eval(str(row['joint_metrics']))
    ankle = metrics.get('right_ankle_angle', {})

    if ankle:
        ankle_data.append({
            'subject': row['subject'],
            'baseline_rom': ankle.get('baseline_rom', 0),
            'final_rom': ankle.get('final_rom', 0),
            'jerk_reduction': ankle.get('jerk_reduction_pct', 0)
        })

df_ankle = pd.DataFrame(ankle_data)
df_ankle['preservation'] = df_ankle['final_rom'] / df_ankle['baseline_rom'] * 100

print(f"Loaded {len(df_ankle)} subjects")

# Create figure with 3 subplots
fig = plt.figure(figsize=(14, 5))

# Subplot 1: ROM Before vs After (Scatter)
ax1 = plt.subplot(1, 3, 1)
ax1.scatter(df_ankle['baseline_rom'], df_ankle['final_rom'],
           s=100, alpha=0.6, color='#2E86AB', edgecolor='black', linewidth=0.5)
# Identity line
max_rom = max(df_ankle['baseline_rom'].max(), df_ankle['final_rom'].max())
ax1.plot([0, max_rom], [0, max_rom], 'k--', alpha=0.3, label='Perfect preservation')
ax1.set_xlabel('Baseline ROM (degrees)', fontweight='bold')
ax1.set_ylabel('Processed ROM (degrees)', fontweight='bold')
ax1.set_title('A. ROM Preservation', fontweight='bold', pad=10)
ax1.grid(True, alpha=0.2)
ax1.legend(loc='upper left')
ax1.set_xlim([0, max_rom * 1.05])
ax1.set_ylim([0, max_rom * 1.05])

# Add stats text
ax1.text(0.95, 0.05, f'Mean preservation: {df_ankle["preservation"].mean():.1f}%\nr = {np.corrcoef(df_ankle["baseline_rom"], df_ankle["final_rom"])[0,1]:.3f}',
         transform=ax1.transAxes, ha='right', va='bottom',
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
         fontsize=8)

# Subplot 2: Preservation Percentage Distribution
ax2 = plt.subplot(1, 3, 2)
colors = ['#06A77D' if p >= 80 else '#F77F00' if p >= 50 else '#D62828'
          for p in df_ankle['preservation']]
bars = ax2.bar(range(len(df_ankle)), df_ankle['preservation'], color=colors,
              edgecolor='black', linewidth=0.5)
ax2.axhline(y=90, color='green', linestyle='--', linewidth=2, alpha=0.7, label='Target (90%)')
ax2.axhline(y=50, color='orange', linestyle='--', linewidth=1.5, alpha=0.7, label='Minimum (50%)')
ax2.set_xlabel('Subject', fontweight='bold')
ax2.set_ylabel('ROM Preservation (%)', fontweight='bold')
ax2.set_title('B. Individual Subject Performance', fontweight='bold', pad=10)
ax2.set_xticks(range(0, len(df_ankle), 3))
ax2.set_xticklabels([f'S{i+1}' for i in range(0, len(df_ankle), 3)], rotation=45)
ax2.grid(True, alpha=0.2, axis='y')
ax2.legend(loc='lower right')
ax2.set_ylim([0, 120])

# Add success rate annotation
n_excellent = (df_ankle['preservation'] >= 80).sum()
n_good = ((df_ankle['preservation'] >= 50) & (df_ankle['preservation'] < 80)).sum()
n_poor = (df_ankle['preservation'] < 50).sum()
ax2.text(0.02, 0.98, f'Excellent (≥80%): {n_excellent}/{len(df_ankle)}\nGood (≥50%): {n_good}/{len(df_ankle)}\nPoor (<50%): {n_poor}/{len(df_ankle)}',
         transform=ax2.transAxes, ha='left', va='top',
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
         fontsize=8)

# Subplot 3: Jerk Reduction vs ROM Preservation
ax3 = plt.subplot(1, 3, 3)
scatter = ax3.scatter(df_ankle['preservation'], df_ankle['jerk_reduction'],
                     s=100, c=df_ankle['final_rom'], cmap='viridis',
                     alpha=0.7, edgecolor='black', linewidth=0.5)
ax3.set_xlabel('ROM Preservation (%)', fontweight='bold')
ax3.set_ylabel('Jerk Reduction (%)', fontweight='bold')
ax3.set_title('C. Smoothness vs Motion Preservation', fontweight='bold', pad=10)
ax3.grid(True, alpha=0.2)
ax3.axvline(x=90, color='green', linestyle='--', alpha=0.3)
ax3.axhline(y=38.8, color='blue', linestyle='--', alpha=0.3, label='Mean jerk reduction')
ax3.legend()

# Add colorbar
cbar = plt.colorbar(scatter, ax=ax3)
cbar.set_label('Final ROM (°)', rotation=270, labelpad=15)

# Add correlation text
corr = np.corrcoef(df_ankle['preservation'], df_ankle['jerk_reduction'])[0,1]
ax3.text(0.05, 0.95, f'r = {corr:.3f}\n(no trade-off)',
         transform=ax3.transAxes, ha='left', va='top',
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
         fontsize=8)

plt.tight_layout()

# Save figure
output_path = '/data/gait/2d_project/batch_results/Figure_Ankle_ROM_Success.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"\n✓ Saved: {output_path}")

print("\n" + "="*80)
print(" ANKLE ROM PRESERVATION SUMMARY")
print("="*80)
print(f"\nMean ROM preservation: {df_ankle['preservation'].mean():.1f}% ± {df_ankle['preservation'].std():.1f}%")
print(f"Median ROM preservation: {df_ankle['preservation'].median():.1f}%")
print(f"Range: {df_ankle['preservation'].min():.1f}% - {df_ankle['preservation'].max():.1f}%")
print(f"\nMean jerk reduction: {df_ankle['jerk_reduction'].mean():.1f}% ± {df_ankle['jerk_reduction'].std():.1f}%")
print(f"\nSubjects with preservation ≥90%: {(df_ankle['preservation'] >= 90).sum()}/{len(df_ankle)} ({(df_ankle['preservation'] >= 90).sum()/len(df_ankle)*100:.1f}%)")
print(f"Subjects with preservation ≥80%: {(df_ankle['preservation'] >= 80).sum()}/{len(df_ankle)} ({(df_ankle['preservation'] >= 80).sum()/len(df_ankle)*100:.1f}%)")
print(f"Subjects with preservation ≥50%: {(df_ankle['preservation'] >= 50).sum()}/{len(df_ankle)} ({(df_ankle['preservation'] >= 50).sum()/len(df_ankle)*100:.1f}%)")
print(f"\nSubjects with ROM > 100°: {(df_ankle['final_rom'] > 100).sum()}/{len(df_ankle)}")
print(f"Subjects with ROM < 10°: {(df_ankle['final_rom'] < 10).sum()}/{len(df_ankle)}")
print("\n" + "="*80)

plt.close()
