
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns
import os
import json
import glob

CSV_PATH = "/data/gait/2d_project/research_metrics/final_benchmarks.csv"
DATA_DIR = "/data/gait/data"
OUTPUT_DIR = "/data/gait/2d_project/research_metrics"

def analyze_multivariate():
    if not os.path.exists(CSV_PATH):
        print("CSV not found.")
        return

    # 1. Load Benchmarks
    df = pd.read_csv(CSV_PATH)
    
    # 2. Augment with Metadata (Height, Weight)
    heights = []
    weights = []
    ages = []
    
    print("Loading Subject Metadata...")
    for sid in df['subject']:
        sid_str = f"{int(sid):02d}"
        json_path = f"{DATA_DIR}/processed_new/S1_{sid_str}_info.json"
        
        h, w, a = np.nan, np.nan, np.nan
        if os.path.exists(json_path):
            try:
                with open(json_path, 'r') as f:
                    info = json.load(f)
                    # Try accessing demographics
                    # Based on previous inspection, it is info['demographics']
                    demo = info.get('demographics', {})
                    h = demo.get('height_cm', np.nan)
                    w = demo.get('weight_cm', np.nan) # Wait, is it weight_kg?
                    if pd.isna(w): w = demo.get('weight_kg', np.nan)
                    a = demo.get('age', np.nan)
            except:
                pass
        heights.append(h)
        weights.append(w)
        ages.append(a)
    
    df['height'] = heights
    df['weight'] = weights
    df['age'] = ages
    df['bmi'] = df['weight'] / ((df['height'] / 100) ** 2)

    # Clean missing metadata
    df_meta = df.dropna(subset=['height', 'weight']).copy()
    print(f"Subjects with valid metadata: {len(df_meta)}")
    
    # 3. Analyze Scale Factor Correlations
    # Scale Factor k = GT / MP
    # If k correlates with Height, we can build a model.
    df_meta['scale_factor'] = df_meta['gt_rom'] / df_meta['mp_rom']
    
    print("\n--- Correlation with Scale Factor (GT/MP) ---")
    corrs = df_meta[['scale_factor', 'height', 'weight', 'bmi', 'age', 'mp_rom']].corr()['scale_factor']
    print(corrs)
    print(f"\n*** MP_ROM vs Scale Factor Correlation: {corrs['mp_rom']:.4f} ***")
    
    # 4. Global Multivariate Regression
    # Model: GT_ROM ~ MP_ROM + Height + Weight
    try:
        import statsmodels.api as sm
        
        X = df_meta[['mp_rom', 'height', 'weight']]
        X = sm.add_constant(X)
        y = df_meta['gt_rom']
        
        model = sm.OLS(y, X).fit()
        print("\n--- Multivariate OLS Results (GT ~ MP + H + W) ---")
        print(model.summary())
        
        # Check if R2 improved significantly > 0.01
    except ImportError:
        print("statsmodels not installed, using simple print")
    
    # Plotting Scale Factor vs Potential Confounders
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    sns.scatterplot(x='height', y='scale_factor', data=df_meta)
    plt.title(f"Height vs Scale Factor (r={corrs['height']:.2f})")
    
    plt.subplot(1, 2, 2)
    sns.scatterplot(x='mp_rom', y='scale_factor', data=df_meta)
    plt.title(f"MP ROM vs Scale Factor (r={corrs['mp_rom']:.2f})")
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/multivariate_analysis.png")
    print("\nSaved plots to multivariate_analysis.png")

if __name__ == "__main__":
    analyze_multivariate()
