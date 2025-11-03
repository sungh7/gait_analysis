"""
Phase 3: Strike Detection Parameter Optimization

Grid search over detector parameters to minimize over-detection ratio.

Target: Mean strike ratio ≤ 1.2× ground truth (all 21 subjects)

Parameters to optimize:
1. Peak prominence (height_prominence multiplier)
2. Min peak distance (frames between strikes)
3. Velocity threshold (for fusion filtering)
"""

import numpy as np
import pandas as pd
import json
from pathlib import Path
from typing import Dict, List, Tuple
from scipy.signal import savgol_filter, find_peaks
from itertools import product

from mediapipe_csv_processor import MediaPipeCSVProcessor
import sys
from io import StringIO


def load_ground_truth_stride_counts():
    """Load GT stride counts from processed JSON files."""
    gt_dir = Path("processed")
    gt_data = {}

    for json_file in sorted(gt_dir.glob("*_info.json")):
        with open(json_file) as f:
            data = json.load(f)

        # Extract subject ID from filename (e.g., S1_01_info.json -> S1_01)
        subject_id = json_file.stem.replace('_info', '')

        left_strides = data.get('strides', {}).get('left', 0)
        right_strides = data.get('strides', {}).get('right', 0)

        gt_data[subject_id] = {
            'left_strides': left_strides,
            'right_strides': right_strides
        }

    return gt_data


def detect_heel_strikes_parameterized(
    df_angles,
    side: str,
    fps: float,
    prominence_multiplier: float = 0.3,
    min_distance_frames: int = 15,
    velocity_threshold_multiplier: float = 0.5
) -> np.ndarray:
    """
    Parameterized heel strike detection (fusion method).

    Args:
        df_angles: DataFrame with landmarks
        side: 'left' or 'right'
        fps: Frame rate
        prominence_multiplier: Multiplier for ground_std to set prominence
        min_distance_frames: Minimum frames between peaks
        velocity_threshold_multiplier: Multiplier for velocity_std to filter peaks

    Returns:
        Array of heel strike frame indices
    """
    heel_col = f'y_{side}_heel'
    ankle_col = f'y_{side}_ankle'
    toe_col = f'y_{side}_foot_index'

    if any(col not in df_angles.columns for col in [heel_col, ankle_col, toe_col]):
        return np.array([])

    heel_y = df_angles[heel_col].values
    ankle_y = df_angles[ankle_col].values
    toe_y = df_angles[toe_col].values

    valid_idx = ~(np.isnan(heel_y) | np.isnan(ankle_y) | np.isnan(toe_y))
    if valid_idx.sum() < fps:
        return np.array([])

    valid_frames = df_angles['frame'].values[valid_idx]
    heel_y = heel_y[valid_idx]
    ankle_y = ankle_y[valid_idx]
    toe_y = toe_y[valid_idx]

    # Fused ground signal
    ground_signal = 0.6 * heel_y + 0.3 * ankle_y + 0.1 * toe_y
    heel_velocity = np.gradient(heel_y, 1 / max(fps, 1))

    # Smoothing
    window_size = min(11, max(3, len(ground_signal) // 3))
    if window_size % 2 == 0:
        window_size += 1
    window_size = max(window_size, 3)

    if len(ground_signal) >= window_size:
        ground_smooth = savgol_filter(ground_signal, window_size, 2)
    else:
        ground_smooth = ground_signal

    if len(heel_velocity) >= window_size:
        velocity_smooth = savgol_filter(heel_velocity, window_size, 2)
    else:
        velocity_smooth = heel_velocity

    ground_std = float(np.std(ground_smooth)) if len(ground_smooth) else 0.0
    velocity_std = float(np.std(velocity_smooth)) if len(velocity_smooth) else 0.0

    # PARAMETERIZED: Peak detection
    prominence = ground_std * prominence_multiplier if ground_std > 0 else 0.01

    peaks_idx, _ = find_peaks(
        -ground_smooth,
        distance=min_distance_frames,
        prominence=prominence
    )

    # PARAMETERIZED: Velocity filtering
    threshold = velocity_std * velocity_threshold_multiplier if velocity_std > 0 else 0.01
    filtered_peaks = []
    for peak_idx in peaks_idx:
        window_start = max(0, peak_idx - 3)
        window_end = min(len(velocity_smooth), peak_idx + 4)
        velocity_window = velocity_smooth[window_start:window_end]
        if len(velocity_window) == 0:
            continue
        if np.min(np.abs(velocity_window)) < threshold:
            filtered_peaks.append(peak_idx)

    if len(filtered_peaks) == 0:
        filtered_peaks = peaks_idx.tolist()

    heel_strike_frames = valid_frames[filtered_peaks]
    return heel_strike_frames


def evaluate_parameters(
    subject_files: List[Tuple[str, str, float]],
    gt_data: Dict,
    prominence_mult: float,
    min_distance: int,
    velocity_mult: float
) -> Dict:
    """
    Evaluate detector parameters on all subjects.

    Args:
        subject_files: List of (subject_id, csv_path, fps) tuples
        gt_data: Ground truth stride counts
        prominence_mult: Prominence multiplier
        min_distance: Min distance in frames
        velocity_mult: Velocity threshold multiplier

    Returns:
        Dict with evaluation metrics
    """
    processor = MediaPipeCSVProcessor()

    ratios_left = []
    ratios_right = []

    for subject_id, csv_path, fps in subject_files:
        if subject_id not in gt_data:
            continue

        gt_left = gt_data[subject_id]['left_strides']
        gt_right = gt_data[subject_id]['right_strides']

        if gt_left == 0 or gt_right == 0:
            continue

        # Load and process (suppress print statements)
        try:
            old_stdout = sys.stdout
            sys.stdout = StringIO()
            df_wide = processor.load_csv(csv_path)
            df_angles = processor.calculate_joint_angles(df_wide)
            sys.stdout = old_stdout
        except Exception as e:
            sys.stdout = old_stdout
            print(f"  Error loading {subject_id}: {e}")
            continue

        # Detect with parameters
        left_strikes = detect_heel_strikes_parameterized(
            df_angles, 'left', fps,
            prominence_mult, min_distance, velocity_mult
        )
        right_strikes = detect_heel_strikes_parameterized(
            df_angles, 'right', fps,
            prominence_mult, min_distance, velocity_mult
        )

        ratio_left = len(left_strikes) / gt_left if gt_left > 0 else 0
        ratio_right = len(right_strikes) / gt_right if gt_right > 0 else 0

        ratios_left.append(ratio_left)
        ratios_right.append(ratio_right)

    all_ratios = ratios_left + ratios_right

    if len(all_ratios) == 0:
        return {
            'mean_abs_deviation': 999.0,
            'mean_ratio': 0.0,
            'median_ratio': 0.0,
            'std_ratio': 0.0,
            'max_ratio': 0.0,
            'n_exceeds_1p5': 0,
            'n_total': 0
        }

    # Objective: Minimize deviation from 1.0
    mean_abs_deviation = np.mean([abs(r - 1.0) for r in all_ratios])
    mean_ratio = np.mean(all_ratios)
    median_ratio = np.median(all_ratios)
    std_ratio = np.std(all_ratios)
    max_ratio = np.max(all_ratios)
    n_exceeds_1p5 = sum(1 for r in all_ratios if r > 1.5)

    return {
        'mean_abs_deviation': mean_abs_deviation,
        'mean_ratio': mean_ratio,
        'median_ratio': median_ratio,
        'std_ratio': std_ratio,
        'max_ratio': max_ratio,
        'n_exceeds_1p5': n_exceeds_1p5,
        'n_total': len(all_ratios)
    }


def grid_search_optimization():
    """
    Grid search over parameter space.

    Returns:
        Best parameters and results
    """
    # Load GT data
    gt_data = load_ground_truth_stride_counts()

    # Subject files (21 subjects) - with correct paths
    subject_files = [
        ('S1_01', 'data/1/1-2_side_pose_fps30.csv', 30),
        ('S1_02', 'data/2/2-2_side_pose_fps30.csv', 30),
        ('S1_03', 'data/3/3-2_side_pose_fps30.csv', 30),
        ('S1_05', 'data/5/5-2_side_pose_fps30.csv', 30),
        ('S1_06', 'data/6/6-2_side_pose_fps30.csv', 30),
        ('S1_07', 'data/7/7-2_side_pose_fps30.csv', 30),
        ('S1_08', 'data/8/8-2_side_pose_fps23.csv', 23),
        ('S1_09', 'data/9/9-2_side_pose_fps24.csv', 24),
        ('S1_10', 'data/10/10-2_side_pose_fps30.csv', 30),
        ('S1_11', 'data/11/11-2_side_pose_fps30.csv', 30),
        ('S1_13', 'data/13/13-2_side_pose_fps30.csv', 30),
        ('S1_14', 'data/14/14-2_side_pose_fps30.csv', 30),
        ('S1_15', 'data/15/15-2_side_pose_fps30.csv', 30),
        ('S1_16', 'data/16/16-2_side_pose_fps30.csv', 30),
        ('S1_17', 'data/17/17-2_side_pose_fps30.csv', 30),
        ('S1_21', 'data/21/21-2_side_pose_fps30.csv', 30),
        ('S1_22', 'data/22/22-2_side_pose_fps30.csv', 30),
        ('S1_26', 'data/26/26-2_side_pose_fps30.csv', 30),
        ('S1_28', 'data/28/28-2_side_pose_fps30.csv', 30),
        ('S1_29', 'data/29/29-2_side_pose_fps30.csv', 30),
        ('S1_30', 'data/30/30-2_side_pose_fps30.csv', 30),
    ]

    # Parameter grid
    prominence_range = [0.4, 0.5, 0.6, 0.7, 0.8]
    min_distance_range = [15, 18, 20, 22, 25]
    velocity_range = [0.3, 0.4, 0.5, 0.6, 0.7]

    print(f"Grid search: {len(prominence_range)} × {len(min_distance_range)} × {len(velocity_range)} = {len(prominence_range) * len(min_distance_range) * len(velocity_range)} combinations")
    print(f"Testing on {len(subject_files)} subjects")
    print()

    results = []
    best_score = float('inf')
    best_params = None

    total_combinations = len(list(product(prominence_range, min_distance_range, velocity_range)))

    for i, (prom, dist, vel) in enumerate(product(prominence_range, min_distance_range, velocity_range)):
        print(f"[{i+1}/{total_combinations}] Testing: prominence={prom:.1f}, min_dist={dist}, vel={vel:.1f}...", end=' ')

        metrics = evaluate_parameters(
            subject_files,
            gt_data,
            prominence_mult=prom,
            min_distance=dist,
            velocity_mult=vel
        )

        score = metrics['mean_abs_deviation']

        result = {
            'prominence_mult': prom,
            'min_distance': dist,
            'velocity_mult': vel,
            **metrics,
            'score': score
        }
        results.append(result)

        print(f"MAD={score:.3f}, mean_ratio={metrics['mean_ratio']:.2f}, n_exceeds_1.5={metrics['n_exceeds_1p5']}")

        if score < best_score:
            best_score = score
            best_params = result
            print(f"  ✅ NEW BEST!")

    # Sort results by score
    results.sort(key=lambda x: x['score'])

    return results, best_params


def main():
    """Run grid search and save results."""
    print("=" * 80)
    print("Phase 3: Strike Detection Parameter Optimization")
    print("=" * 80)
    print()

    results, best_params = grid_search_optimization()

    print()
    print("=" * 80)
    print("OPTIMIZATION COMPLETE")
    print("=" * 80)
    print()
    print("BEST PARAMETERS:")
    print(f"  Prominence multiplier: {best_params['prominence_mult']:.1f}")
    print(f"  Min distance (frames): {best_params['min_distance']}")
    print(f"  Velocity multiplier:   {best_params['velocity_mult']:.1f}")
    print()
    print("PERFORMANCE:")
    print(f"  Mean abs deviation:  {best_params['mean_abs_deviation']:.3f}")
    print(f"  Mean ratio:          {best_params['mean_ratio']:.2f}×")
    print(f"  Median ratio:        {best_params['median_ratio']:.2f}×")
    print(f"  Std ratio:           {best_params['std_ratio']:.2f}")
    print(f"  Max ratio:           {best_params['max_ratio']:.2f}×")
    print(f"  Subjects > 1.5×:     {best_params['n_exceeds_1p5']}/{best_params['n_total']}")
    print()

    # Save results
    output = {
        'best_parameters': best_params,
        'all_results': results[:10],  # Top 10
        'baseline_comparison': {
            'baseline_mean_ratio': 3.45,
            'optimized_mean_ratio': best_params['mean_ratio'],
            'improvement': 3.45 - best_params['mean_ratio']
        }
    }

    with open('P3_optimization_results.json', 'w') as f:
        json.dump(output, f, indent=2)

    print(f"✓ Saved to P3_optimization_results.json")
    print()

    # Compare top 5
    print("TOP 5 PARAMETER COMBINATIONS:")
    print()
    for i, result in enumerate(results[:5]):
        print(f"{i+1}. prom={result['prominence_mult']:.1f}, dist={result['min_distance']}, vel={result['velocity_mult']:.1f}")
        print(f"   MAD={result['score']:.3f}, mean={result['mean_ratio']:.2f}×, n>1.5={result['n_exceeds_1p5']}")
        print()


if __name__ == '__main__':
    main()
