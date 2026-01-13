#!/usr/bin/env python3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os

INPUT_CSV = "/data/gait/2d_project/research_metrics/final_benchmarks.csv"
OUTPUT_DIR = "/data/gait/2d_project/research_metrics"

def calculate_icc(data, c_name):
    # Simplified ICC(2,1) for single observer vs GT
    # Need ANOVA table logic or use pingouin if available.
    # Since we can't pip install, we implement basic ICC formula.
    # ICC(2,1) = (BMS - EMS) / (BMS + (k-1)EMS + k(RMS - EMS)/n)
    # Actually, simpler: reliability between MP and GT.
    # We treat MP and GT as two raters. k=2.
    
    # Data columns: [subject, rater, value]
    # Restructure: Subject | GT | MP
    n = len(data)
    k = 2
    
    # Grand Mean
    grand_mean = (data['gt_' + c_name].mean() + data['mp_' + c_name].mean()) / 2
    
    # SST (Total Sum of Squares)
    sst = ((data['gt_' + c_name] - grand_mean)**2).sum() + ((data['mp_' + c_name] - grand_mean)**2).sum()
    
    # BMS (Between-subjects Mean Square)
    subject_means = (data['gt_' + c_name] + data['mp_' + c_name]) / 2
    ssb = k * ((subject_means - grand_mean)**2).sum()
    df_b = n - 1
    bms = ssb / df_b
    
    # WMS (Within-subjects Mean Square) - Error
    # For each subject, variance between MP and GT
    ssw = ((data['gt_' + c_name] - subject_means)**2).sum() + ((data['mp_' + c_name] - subject_means)**2).sum()
    # RMS (Between-raters Mean Square) - Systematic Error
    rater_means_gt = data['gt_' + c_name].mean()
    rater_means_mp = data['mp_' + c_name].mean()
    ssr = n * ((rater_means_gt - grand_mean)**2 + (rater_means_mp - grand_mean)**2)
    df_r = k - 1
    rms = ssr / df_r
    
    # EMS (Error Mean Square) = Residual
    sse = ssw - ssr
    df_e = (n - 1) * (k - 1)
    ems = sse / df_e
    
    # ICC(2,1) Formula
    # (BMS - EMS) / (BMS + (k-1)EMS + k(RMS - EMS)/n)
    icc = (bms - ems) / (bms + (k-1)*ems + k*(rms - ems)/n)
    
    return icc

def bland_altman_plot(data, metric, save_name):
    gt = data['gt_' + metric]
    mp = data['mp_' + metric]
    
    mean = (gt + mp) / 2
    diff = mp - gt
    md = np.mean(diff)
    sd = np.std(diff, axis=0)
    
    loa_upper = md + 1.96 * sd
    loa_lower = md - 1.96 * sd
    
    plt.figure(figsize=(8, 6))
    plt.scatter(mean, diff, alpha=0.5)
    plt.axhline(md, color='red', linestyle='--', label=f'Mean Diff ({md:.2f})')
    plt.axhline(loa_upper, color='green', linestyle=':', label=f'Upper LoA ({loa_upper:.2f})')
    plt.axhline(loa_lower, color='green', linestyle=':', label=f'Lower LoA ({loa_lower:.2f})')
    
    plt.title(f"Bland-Altman: {metric.upper()}")
    plt.xlabel("Mean of Methods")
    plt.ylabel("Difference (MP - GT)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/{save_name}")
    print(f"Saved {save_name}")
    
    return md, loa_lower, loa_upper

def main():
    if not os.path.exists(INPUT_CSV):
        print("CSV not found yet.")
        return

    df = pd.read_csv(INPUT_CSV)
    print(f"Loaded {len(df)} subjects.")
    
    if len(df) < 2:
        print("Not enough data for statistical analysis (N < 2). Skipping.")
        return
    
    # 1. ICC for ROM
    icc_rom = calculate_icc(df, 'rom')
    print(f"ICC(2,1) for ROM: {icc_rom:.4f}")
    
    # 2. Bland-Altman for ROM
    md, lower, upper = bland_altman_plot(df, 'rom', 'bland_altman_rom.png')
    
    # 3. Correlation for ROM (Pearson)
    r_val, p_val = stats.pearsonr(df['gt_rom'], df['mp_rom'])
    print(f"Pearson r for ROM: {r_val:.4f} (p={p_val:.4e})")
    
    # 4. Count Agreement (Clean)
    if 'mp_count_clean' in df.columns:
        md_c, lower_c, upper_c = bland_altman_plot(df, 'count_clean', 'bland_altman_count.png')
    else:
        # Fallback
        md_c, lower_c, upper_c = bland_altman_plot(df, 'count', 'bland_altman_count.png')
    
    # Save Stats Text
    with open(f"{OUTPUT_DIR}/stats_summary.txt", "w") as f:
        f.write("=== Research Statistical Report ===\n")
        f.write(f"N = {len(df)}\n")
        f.write(f"ROM ICC(2,1): {icc_rom:.4f}\n")
        f.write(f"ROM Pearson r: {r_val:.4f}\n")
        f.write(f"ROM Bland-Altman: Mean Diff={md:.2f}, LoA=[{lower:.2f}, {upper:.2f}]\n")
        f.write(f"Count Bland-Altman (Clean): Mean Diff={md_c:.2f}\n")

if __name__ == "__main__":
    main()
