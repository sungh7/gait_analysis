#!/usr/bin/env python3
"""
Extract V7 Pure 3D Features for All Patterns
=============================================

This script extracts all 10 V7 features from GAVD patterns
and saves them for ML training.

Author: Gait Analysis System
Date: 2025-11-04
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.signal import find_peaks
from tqdm import tqdm


def load_3d_pose(csv_path: str) -> pd.DataFrame:
    """Load 3D pose CSV"""
    try:
        df = pd.read_csv(csv_path)
        return df
    except:
        return None


def extract_v7_features(pattern: dict) -> dict:
    """Extract all 10 V7 Pure 3D features"""

    csv_path = pattern.get('csv_file')
    if not csv_path:
        return None

    df = load_3d_pose(csv_path)
    if df is None or len(df) < 30:
        return None

    fps = pattern.get('fps', 30)
    dt = 1.0 / fps

    # Get 3D trajectories for heels
    left_heel_df = df[df['position'] == 'left_heel'].sort_values('frame')
    right_heel_df = df[df['position'] == 'right_heel'].sort_values('frame')

    if len(left_heel_df) < 10 or len(right_heel_df) < 10:
        return None

    # 3D coordinates
    left_heel_3d = left_heel_df[['x', 'y', 'z']].values
    right_heel_3d = right_heel_df[['x', 'y', 'z']].values

    # Hip and ankle for stride length
    left_hip_df = df[df['position'] == 'left_hip'].sort_values('frame')
    right_hip_df = df[df['position'] == 'right_hip'].sort_values('frame')

    # 1. 3D CADENCE
    left_y = left_heel_3d[:, 1]  # y = vertical
    right_y = right_heel_3d[:, 1]

    peaks_left, _ = find_peaks(left_y, height=np.mean(left_y), distance=5)
    peaks_right, _ = find_peaks(right_y, height=np.mean(right_y), distance=5)

    n_steps = len(peaks_left) + len(peaks_right)
    duration = len(left_y) / fps
    cadence_3d = (n_steps / duration) * 60 if duration > 0 else 0

    # 2. 3D STEP HEIGHT VARIABILITY
    if len(peaks_left) > 1:
        var_left = np.std(left_y[peaks_left]) / (np.mean(left_y[peaks_left]) + 1e-6)
    else:
        var_left = 0

    if len(peaks_right) > 1:
        var_right = np.std(right_y[peaks_right]) / (np.mean(right_y[peaks_right]) + 1e-6)
    else:
        var_right = 0

    step_height_variability = (var_left + var_right) / 2

    # 3. 3D GAIT IRREGULARITY
    if len(peaks_left) > 2:
        intervals_left = np.diff(peaks_left)
        irreg_left = np.std(intervals_left) / (np.mean(intervals_left) + 1e-6)
    else:
        irreg_left = 0

    if len(peaks_right) > 2:
        intervals_right = np.diff(peaks_right)
        irreg_right = np.std(intervals_right) / (np.mean(intervals_right) + 1e-6)
    else:
        irreg_right = 0

    gait_irregularity_3d = (irreg_left + irreg_right) / 2

    # 4. 3D VELOCITY
    vel_left_3d = np.diff(left_heel_3d, axis=0) / dt
    vel_right_3d = np.diff(right_heel_3d, axis=0) / dt

    vel_left_mag = np.sqrt(np.sum(vel_left_3d**2, axis=1))
    vel_right_mag = np.sqrt(np.sum(vel_right_3d**2, axis=1))

    velocity_3d = (np.mean(vel_left_mag) + np.mean(vel_right_mag)) / 2

    # 5. 3D JERKINESS
    accel_left_3d = np.diff(vel_left_3d, axis=0) / dt
    accel_right_3d = np.diff(vel_right_3d, axis=0) / dt

    accel_left_mag = np.sqrt(np.sum(accel_left_3d**2, axis=1))
    accel_right_mag = np.sqrt(np.sum(accel_right_3d**2, axis=1))

    jerkiness_3d = (np.mean(accel_left_mag) + np.mean(accel_right_mag)) / 2

    # 6. 3D CYCLE DURATION
    cycle_duration_3d = np.mean(np.diff(peaks_left) / fps) if len(peaks_left) > 1 else 0

    # 7. 3D STRIDE LENGTH (from pattern)
    stride_length_3d = pattern.get('stride_length_3d', 0.0)

    # 8. TRUNK SWAY (from pattern)
    trunk_sway = pattern.get('trunk_sway', 0.0)

    # 9. 3D PATH LENGTH
    path_left = np.sum(np.sqrt(np.sum(np.diff(left_heel_3d, axis=0)**2, axis=1)))
    path_right = np.sum(np.sqrt(np.sum(np.diff(right_heel_3d, axis=0)**2, axis=1)))
    path_length_3d = (path_left + path_right) / 2 / duration if duration > 0 else 0

    # 10. 3D STEP WIDTH
    if len(left_hip_df) > 0 and len(right_hip_df) > 0:
        left_hip_x = left_hip_df['x'].values
        right_hip_x = right_hip_df['x'].values
        min_len = min(len(left_hip_x), len(right_hip_x))
        step_width_3d = np.mean(np.abs(left_hip_x[:min_len] - right_hip_x[:min_len]))
    else:
        step_width_3d = 0.0

    return {
        'cadence_3d': float(cadence_3d),
        'step_height_variability': float(step_height_variability),
        'gait_irregularity_3d': float(gait_irregularity_3d),
        'velocity_3d': float(velocity_3d),
        'jerkiness_3d': float(jerkiness_3d),
        'cycle_duration_3d': float(cycle_duration_3d),
        'stride_length_3d': float(stride_length_3d),
        'trunk_sway': float(trunk_sway),
        'path_length_3d': float(path_length_3d),
        'step_width_3d': float(step_width_3d)
    }


def main():
    """Extract V7 features for all patterns"""

    print("="*80)
    print("Extracting V7 Pure 3D Features")
    print("="*80)
    print()

    # Load patterns
    with open('gavd_all_views_patterns.json', 'r') as f:
        patterns = json.load(f)

    print(f"Loaded {len(patterns)} patterns")
    print()

    # Extract features for each pattern
    valid_patterns = []
    failed_count = 0

    print("Extracting features...")
    for p in tqdm(patterns):
        features = extract_v7_features(p)

        if features is not None:
            # Add features to pattern
            p.update(features)
            valid_patterns.append(p)
        else:
            failed_count += 1

    print()
    print(f"Successfully extracted features: {len(valid_patterns)}")
    print(f"Failed extractions: {failed_count}")
    print()

    # Distribution
    normal = [p for p in valid_patterns if p['gait_class'] == 'normal']
    pathological = [p for p in valid_patterns if p['gait_class'] == 'pathological']

    print(f"Valid patterns with V7 features: {len(valid_patterns)}")
    print(f"  Normal: {len(normal)}")
    print(f"  Pathological: {len(pathological)}")
    print()

    # Save to new file
    output_file = 'gavd_patterns_with_v7_features.json'
    with open(output_file, 'w') as f:
        json.dump(valid_patterns, f, indent=2)

    print(f"✅ Features saved to {output_file}")
    print()

    # Display sample
    if valid_patterns:
        print("Sample pattern with features:")
        sample = valid_patterns[0]
        print(f"  Seq ID: {sample['seq_id']}")
        print(f"  Class: {sample['gait_class']}")
        print(f"  Pathology: {sample.get('gait_pathology', 'N/A')}")
        print(f"  Features:")
        for fname in ['cadence_3d', 'step_height_variability', 'gait_irregularity_3d',
                      'velocity_3d', 'jerkiness_3d', 'cycle_duration_3d',
                      'stride_length_3d', 'trunk_sway', 'path_length_3d', 'step_width_3d']:
            print(f"    {fname:30s}: {sample[fname]:.6f}")


if __name__ == "__main__":
    main()
