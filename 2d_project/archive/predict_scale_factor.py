
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import seaborn as sns
import os

CSV_PATH = "/data/gait/2d_project/research_metrics/final_benchmarks.csv"
OUTPUT_DIR = "/data/gait/2d_project/research_metrics"

def model_func(x, a, b):
    # Model: Scale = a * (1/x) + b
    # This implies GT = a + b*x (Wait, if Scale = GT/x, then GT/x = a/x + b => GT = a + b*x)
    # This is just linear regression again? 
    # Let's check the scatter plot shape.
    return a / x + b

def analyze_scale_prediction():
    if not os.path.exists(CSV_PATH): return

    df = pd.read_csv(CSV_PATH)
    # Filter valid
    df = df[df['correlation'] > 0.5].copy()
    if len(df) < 3: return
    
    df['scale_factor'] = df['gt_rom'] / df['mp_rom']
    
    # Fit Model to Scale Factor
    popt, pcov = curve_fit(model_func, df['mp_rom'], df['scale_factor'])
    
    # Predict Scale
    df['pred_scale'] = model_func(df['mp_rom'], *popt)
    df['pred_gt_calib'] = df['mp_rom'] * df['pred_scale']
    
    # Calculate R2 for Predicted GT
    ss_res = np.sum((df['gt_rom'] - df['pred_gt_calib'])**2)
    ss_tot = np.sum((df['gt_rom'] - np.mean(df['gt_rom']))**2)
    r2 = 1 - (ss_res / ss_tot)
    
    print("\n--- System Calibration Model (Implicit Linear) ---")
    print(f"Model: Scale = {popt[0]:.2f}/MP + {popt[1]:.4f}")
    print(f"Implied GT Equation: GT = {popt[0]:.2f} + {popt[1]:.4f} * MP")
    print(f"New R-squared: {r2:.4f}")
    
    # Visual check
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    sns.scatterplot(x='mp_rom', y='scale_factor', data=df)
    x_range = np.linspace(df['mp_rom'].min(), df['mp_rom'].max(), 100)
    plt.plot(x_range, model_func(x_range, *popt), 'r-', label='Fit')
    plt.title(f"Scale Factor Prediction ($r={df[['mp_rom','scale_factor']].corr().iloc[0,1]:.2f}$)")
    
    plt.subplot(1, 2, 2)
    sns.scatterplot(x='pred_gt_calib', y='gt_rom', data=df)
    plt.plot([20, 80], [20, 80], 'k--')
    plt.title(f"Predicted GT vs Actual GT ($R^2={r2:.2f}$)")
    
    plt.savefig(f"{OUTPUT_DIR}/system_calibration_check.png")
    print("Saved plot.")

if __name__ == "__main__":
    analyze_scale_prediction()
