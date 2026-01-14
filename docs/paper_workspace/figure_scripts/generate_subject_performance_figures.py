import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def generate_subject_performance_figures():
    # Load data
    df = pd.read_csv("/data/gait/comprehensive_analysis_results.csv")
    
    # Set style
    sns.set_theme(style="whitegrid")
    
    # 1. Correlation Heatmap (Subjects x Joints)
    # We aggregate left/right by taking the mean
    df_agg = df.groupby(['subject_id', 'joint'])[['corr_after', 'rmse_after']].mean().reset_index()
    
    # Pivot for heatmap
    heatmap_data = df_agg.pivot(index='subject_id', columns='joint', values='corr_after')
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(heatmap_data, annot=True, fmt=".2f", cmap="RdYlGn", vmin=0.5, vmax=1.0, cbar_kws={'label': 'Correlation'})
    plt.title('Gait Pattern Similarity (Correlation) by Subject and Joint', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig("/data/gait/subject_performance_heatmap.png", dpi=300)
    print("Saved subject_performance_heatmap.png")
    
    # 2. RMSE Bar Chart (Grouped by Joint)
    plt.figure(figsize=(14, 6))
    sns.barplot(data=df, x='subject_id', y='rmse_after', hue='joint', errorbar=None, palette="viridis")
    plt.title('RMSE by Subject and Joint (Lower is Better)', fontsize=16, fontweight='bold')
    plt.ylabel('RMSE (degrees)', fontsize=12)
    plt.xlabel('Subject ID', fontsize=12)
    plt.xticks(rotation=45)
    plt.legend(title='Joint')
    plt.tight_layout()
    plt.savefig("/data/gait/subject_performance_rmse.png", dpi=300)
    print("Saved subject_performance_rmse.png")
    
    # 3. Correlation Bar Chart (Grouped by Joint)
    plt.figure(figsize=(14, 6))
    sns.barplot(data=df, x='subject_id', y='corr_after', hue='joint', errorbar=None, palette="magma")
    plt.title('Correlation by Subject and Joint (Higher is Better)', fontsize=16, fontweight='bold')
    plt.ylabel('Correlation Coefficient', fontsize=12)
    plt.xlabel('Subject ID', fontsize=12)
    plt.ylim(0, 1.05)
    plt.axhline(0.9, color='green', linestyle='--', alpha=0.5, label='Excellent (>0.9)')
    plt.axhline(0.7, color='orange', linestyle='--', alpha=0.5, label='Good (>0.7)')
    plt.xticks(rotation=45)
    plt.legend(title='Joint', loc='lower right')
    plt.tight_layout()
    plt.savefig("/data/gait/subject_performance_corr.png", dpi=300)
    print("Saved subject_performance_corr.png")

if __name__ == "__main__":
    generate_subject_performance_figures()
