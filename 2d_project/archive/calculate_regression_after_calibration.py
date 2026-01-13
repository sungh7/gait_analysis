
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns
import os

CSV_PATH = "/data/gait/2d_project/research_metrics/final_benchmarks.csv"
OUTPUT_DIR = "/data/gait/2d_project/research_metrics"

def analyze_calibrated_regression():
    if not os.path.exists(CSV_PATH):
        print("CSV not found!")
        return

    df = pd.read_csv(CSV_PATH)
    print(f"Total Subjects: {len(df)}")
    
    # --- 1. Filter Valid Subjects ---
    # We should only calibrate subjects where tracking is valid, otherwise we are fitting noise.
    # Criterion: Correlation > 0.5
    df_clean = df[df['correlation'] > 0.5].copy()
    print(f"Valid Subjects for Calibration (r > 0.5): {len(df_clean)}")
    
    # --- 2. Apply Individual Calibration (Simulation) ---
    # Calibrated = (MP - Mean_MP) * Scale + Mean_GT
    # Since we only have ROM scalars here, we can simulate ROM calibration directly:
    # MP_Calib_ROM = MP_ROM * (GT_ROM / MP_ROM)
    
    # Calculate scale factor for each subject
    df_clean['scale_factor'] = df_clean['gt_rom'] / df_clean['mp_rom']
    df_clean['mp_rom_calibrated'] = df_clean['mp_rom'] * df_clean['scale_factor']
    
    # --- 3. Regression Analysis (Post-Calibration) ---
    slope, intercept, r_value, p_value, std_err = stats.linregress(df_clean['mp_rom_calibrated'], df_clean['gt_rom'])
    
    print("\n--- Regression After Individual Calibration ---")
    print(f"Equation: GT_ROM = {slope:.4f} * MP_Calib + {intercept:.4f}")
    print(f"R-squared: {r_value**2:.4f}")
    
    # --- 4. Plotting (Before vs After) ---
    plt.figure(figsize=(12, 6))
    
    # Subplot 1: Before
    plt.subplot(1, 2, 1)
    sns.regplot(x='mp_rom', y='gt_rom', data=df_clean, color='red', label='Before')
    plt.title(f"Before Calibration\n$R^2$ = {df_clean[['mp_rom','gt_rom']].corr().iloc[0,1]**2:.3f}")
    plt.xlabel("MediaPipe ROM (deg)")
    plt.ylabel("Vicon ROM (deg)")
    plt.grid(True)
    
    # Subplot 2: After
    plt.subplot(1, 2, 2)
    # Ideally this is y=x
    sns.regplot(x='mp_rom_calibrated', y='gt_rom', data=df_clean, color='blue', label='After')
    plt.plot([20, 80], [20, 80], 'k--', alpha=0.5, label='Ideal y=x')
    plt.title(f"After Individual Calibration\n$R^2$ = {r_value**2:.3f} (Ideal)")
    plt.xlabel("Calibrated MP ROM (deg)")
    plt.ylabel("Vicon ROM (deg)")
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    save_path = f"{OUTPUT_DIR}/regression_post_calibration.png"
    plt.savefig(save_path)
    print(f"Plot saved to {save_path}")

if __name__ == "__main__":
    analyze_calibrated_regression()
