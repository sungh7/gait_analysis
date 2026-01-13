#!/usr/bin/env python3
"""
Vicon Ground Truth Correlation Analysis
Computes correlations between MediaPipe (improved) and Vicon angles.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
from scipy.interpolate import interp1d
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print(" VICON GROUND TRUTH CORRELATION ANALYSIS")
print("="*80)

# Subjects with available Vicon data
VICON_DIR = Path("/data/gait/data/processed_new")
IMPROVED_DIR = Path("/data/gait/2d_project/batch_results")

# Find subjects with both Vicon and improved data
vicon_files = list(VICON_DIR.glob("S1_*_gait_long.csv"))
vicon_subjects = [f.stem.split('_gait')[0] for f in vicon_files]

print(f"\nFound {len(vicon_subjects)} subjects with Vicon ground truth")
print(f"Subjects: {', '.join(sorted(vicon_subjects[:10]))}...")

# Joint mapping
joint_map = {
    'knee': ('right_knee_angle', 'RKneeAngles_Flexion'),
    'hip': ('right_hip_angle', 'RHipAngles_Flexion'),
    'ankle': ('right_ankle_angle', 'RAnkleAngles_Plantarflexion')
}

results = []

for subject in sorted(vicon_subjects):
    print(f"\n{'='*80}")
    print(f"Processing {subject}")
    print('='*80)

    # Load Vicon data
    vicon_file = VICON_DIR / f"{subject}_gait_long.csv"
    try:
        vicon_df = pd.read_csv(vicon_file)
    except Exception as e:
        print(f"  ⚠️  Failed to load Vicon: {e}")
        continue

    # Load improved MediaPipe data
    improved_file = IMPROVED_DIR / f"{subject}_improved_angles.csv"
    if not improved_file.exists():
        print(f"  ⚠️  No improved data found")
        continue

    try:
        mp_df = pd.read_csv(improved_file)
    except Exception as e:
        print(f"  ⚠️  Failed to load MediaPipe: {e}")
        continue

    subject_results = {'subject': subject}

    # Process each joint
    for joint_name, (mp_col, vicon_col) in joint_map.items():
        # Check columns exist
        if mp_col not in mp_df.columns:
            print(f"  ⚠️  {joint_name}: MediaPipe column '{mp_col}' not found")
            continue

        if vicon_col not in vicon_df.columns:
            print(f"  ⚠️  {joint_name}: Vicon column '{vicon_col}' not found")
            continue

        # Extract signals
        mp_signal = mp_df[mp_col].values
        vicon_signal = vicon_df[vicon_col].values

        # Remove NaNs
        mp_valid = ~np.isnan(mp_signal)
        vicon_valid = ~np.isnan(vicon_signal)

        if mp_valid.sum() < 10 or vicon_valid.sum() < 10:
            print(f"  ⚠️  {joint_name}: Insufficient valid data")
            continue

        # Time-normalize both signals to 0-100%
        # Assume Vicon is 0-100% already, resample MediaPipe to match length
        n_vicon = len(vicon_signal)
        n_mp = len(mp_signal)

        # Resample MediaPipe to match Vicon length
        mp_time = np.linspace(0, 100, n_mp)
        vicon_time = np.linspace(0, 100, n_vicon)

        try:
            interp_func = interp1d(mp_time, mp_signal, kind='cubic', fill_value='extrapolate')
            mp_resampled = interp_func(vicon_time)
        except Exception as e:
            print(f"  ⚠️  {joint_name}: Interpolation failed - {e}")
            continue

        # Remove NaNs after resampling
        valid_idx = ~np.isnan(mp_resampled) & ~np.isnan(vicon_signal)
        mp_clean = mp_resampled[valid_idx]
        vicon_clean = vicon_signal[valid_idx]

        if len(mp_clean) < 10:
            print(f"  ⚠️  {joint_name}: Too few valid points after cleaning")
            continue

        # Compute correlation
        r, p = stats.pearsonr(mp_clean, vicon_clean)

        # Compute RMSE
        rmse = np.sqrt(np.mean((mp_clean - vicon_clean)**2))

        # Compute MAE
        mae = np.mean(np.abs(mp_clean - vicon_clean))

        # ROM comparison
        vicon_rom = np.ptp(vicon_clean)
        mp_rom = np.ptp(mp_clean)
        rom_ratio = mp_rom / vicon_rom if vicon_rom > 0 else np.nan

        print(f"  {joint_name.capitalize():5s}: r={r:6.3f}, p={p:.4f}, RMSE={rmse:6.2f}°, MAE={mae:5.2f}°, ROM={mp_rom:.1f}°/{vicon_rom:.1f}° ({rom_ratio:.2f})")

        # Store results
        subject_results[f'{joint_name}_r'] = r
        subject_results[f'{joint_name}_p'] = p
        subject_results[f'{joint_name}_rmse'] = rmse
        subject_results[f'{joint_name}_mae'] = mae
        subject_results[f'{joint_name}_vicon_rom'] = vicon_rom
        subject_results[f'{joint_name}_mp_rom'] = mp_rom
        subject_results[f'{joint_name}_rom_ratio'] = rom_ratio

    results.append(subject_results)

# Convert to DataFrame
df_results = pd.DataFrame(results)

print("\n" + "="*80)
print(" SUMMARY STATISTICS")
print("="*80)

for joint_name in ['knee', 'hip', 'ankle']:
    r_col = f'{joint_name}_r'
    rmse_col = f'{joint_name}_rmse'
    mae_col = f'{joint_name}_mae'

    if r_col not in df_results.columns:
        continue

    r_vals = df_results[r_col].dropna()
    rmse_vals = df_results[rmse_col].dropna()
    mae_vals = df_results[mae_col].dropna()

    n = len(r_vals)

    if n == 0:
        print(f"\n{joint_name.capitalize()}: No valid data")
        continue

    print(f"\n{joint_name.capitalize()} (N={n}):")
    print(f"  Correlation (r):")
    print(f"    Mean:   {r_vals.mean():.3f} ± {r_vals.std():.3f}")
    print(f"    Median: {r_vals.median():.3f}")
    print(f"    Range:  [{r_vals.min():.3f}, {r_vals.max():.3f}]")
    print(f"  RMSE:")
    print(f"    Mean:   {rmse_vals.mean():.2f}° ± {rmse_vals.std():.2f}°")
    print(f"    Median: {rmse_vals.median():.2f}°")
    print(f"  MAE:")
    print(f"    Mean:   {mae_vals.mean():.2f}° ± {mae_vals.std():.2f}°")
    print(f"    Median: {mae_vals.median():.2f}°")

    # Count subjects with good correlation
    n_excellent = (r_vals >= 0.9).sum()
    n_good = ((r_vals >= 0.7) & (r_vals < 0.9)).sum()
    n_moderate = ((r_vals >= 0.5) & (r_vals < 0.7)).sum()
    n_poor = (r_vals < 0.5).sum()

    print(f"  Correlation quality:")
    print(f"    Excellent (r≥0.9): {n_excellent}/{n} ({n_excellent/n*100:.1f}%)")
    print(f"    Good (r≥0.7):      {n_good}/{n} ({n_good/n*100:.1f}%)")
    print(f"    Moderate (r≥0.5):  {n_moderate}/{n} ({n_moderate/n*100:.1f}%)")
    print(f"    Poor (r<0.5):      {n_poor}/{n} ({n_poor/n*100:.1f}%)")

# Save results
output_path = '/data/gait/2d_project/batch_results/vicon_correlation_results.csv'
df_results.to_csv(output_path, index=False)
print(f"\n✓ Saved detailed results: {output_path}")

print("\n" + "="*80)
print(" CORRELATION ANALYSIS COMPLETE")
print("="*80)
print(f"\nAnalyzed {len(df_results)} subjects with Vicon ground truth")
print("="*80)
