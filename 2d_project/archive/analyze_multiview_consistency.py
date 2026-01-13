import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
import numpy as np

def main():
    # Load Data
    try:
        sag = pd.read_csv('/data/gait/2d_project/research_metrics/final_benchmarks.csv')
        front = pd.read_csv('/data/gait/2d_project/frontal_self_driven_results.csv')
    except Exception as e:
        print(f"File Load Error: {e}")
        return

    # Merge
    merged = pd.merge(sag, front, left_on='subject', right_on='Subject', how='inner')
    
    if len(merged) < 3:
        print("Not enough overlapping subjects.")
        return
        
    # Analysis
    r_sag = merged['correlation']
    r_front = merged['R']
    
    # Correlation between accuracies
    corr_acc, p_val = pearsonr(r_sag, r_front)
    
    print(f"Sagittal-Frontal Performance Correlation: r={corr_acc:.4f} (p={p_val:.4f})")
    
    # Scatter Plot
    plt.figure(figsize=(8, 6))
    
    # Quadrants
    plt.axhline(0.5, color='gray', linestyle='--')
    plt.axvline(0.5, color='gray', linestyle='--')
    
    sns.scatterplot(data=merged, x='correlation', y='R', s=100)
    
    # Label Points
    for i, row in merged.iterrows():
        plt.text(row['correlation']+0.02, row['R'], str(int(row['subject'])), fontsize=9)
        
    plt.xlabel('Sagittal Correlation (Knee Flexion)')
    plt.ylabel('Frontal Correlation (Valgus Proxy)')
    plt.title(f'Multi-View Consistency (r={corr_acc:.2f})\nTop-Right: Good in Both Views')
    plt.grid(True, alpha=0.3)
    
    plt.savefig('multiview_consistency.png')
    print("Saved multiview_consistency.png")
    
    # Identify "Star Subjects" (Good in both)
    stars = merged[(merged['correlation'] > 0.7) & (merged['R'] > 0.7)]
    print("\n=== Star Subjects (High Reliability in Both) ===")
    print(stars[['subject', 'correlation', 'R']])

if __name__ == "__main__":
    main()
