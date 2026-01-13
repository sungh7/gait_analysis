
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

CSV_PATH = "/data/gait/2d_project/research_metrics/final_benchmarks.csv"
OUTPUT_DIR = "/data/gait/2d_project/research_metrics"

def analyze_pattern_validity():
    df = pd.read_csv(CSV_PATH)
    
    # Metrics
    df['scale_error'] = abs(df['gt_rom'] - df['mp_rom'])
    df['shape_quality'] = df['correlation']
    
    # Classify
    conditions = [
        (df['shape_quality'] > 0.8) & (df['scale_error'] < 10), # Perfect
        (df['shape_quality'] > 0.8) & (df['scale_error'] >= 10), # Scaled (User's Point)
        (df['shape_quality'] <= 0.8) # Garbage/Noise
    ]
    choices = ['Perfect (Accurate)', 'Scaled (Good Pattern, Wrong Size)', 'Garbage (Bad Pattern)']
    df['category'] = np.select(conditions, choices, default='Garbage')
    
    print("\n--- Pattern Validity Analysis ---")
    print(df['category'].value_counts())
    
    # Specific Examples
    print("\n[Scaled Group Examples (Support User's Idea)]")
    print(df[df['category'] == 'Scaled (Good Pattern, Wrong Size)'][['subject', 'correlation', 'mp_rom', 'gt_rom']])

    print("\n[Garbage Group (Warning)]")
    print(df[df['category'].str.contains('Garbage')][['subject', 'correlation', 'mp_rom', 'gt_rom']].head())

    # Plot
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x='scale_error', y='correlation', hue='category', data=df, s=100)
    plt.axhline(0.8, color='r', linestyle='--', label='Shape Cutoff')
    plt.axvline(10, color='k', linestyle='--', label='Scale Error Cutoff')
    plt.title("Is Pattern Preserved when Scale Fails?")
    plt.xlabel("ROM Error (deg)")
    plt.ylabel("Shape Correlation (r)")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{OUTPUT_DIR}/pattern_vs_scale.png")
    print(f"Saved plot to {OUTPUT_DIR}/pattern_vs_scale.png")

if __name__ == "__main__":
    analyze_pattern_validity()
