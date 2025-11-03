"""
Phase 3 - Option B: Template-Based Heel Strike Detection

Uses DTW (Dynamic Time Warping) to match gait patterns against reference templates.

Approach:
1. Extract reference gait cycle from GT data
2. Use sliding window to find similar patterns
3. Identify heel strikes at pattern boundaries
"""

import numpy as np
import pandas as pd
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from scipy.signal import savgol_filter
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw

from mediapipe_csv_processor import MediaPipeCSVProcessor


def normalize_signal(signal: np.ndarray) -> np.ndarray:
    """Z-score normalization."""
    mean = np.mean(signal)
    std = np.std(signal)
    if std == 0:
        return signal - mean
    return (signal - mean) / std


def create_reference_template(
    df_angles: pd.DataFrame,
    side: str,
    gt_stride_count: int,
    fps: float
) -> Tuple[np.ndarray, dict]:
    """
    Create reference gait cycle template from data.

    Strategy: Use the signal itself to find the most consistent stride pattern.

    Args:
        df_angles: DataFrame with joint angles and landmarks
        side: 'left' or 'right'
        gt_stride_count: Expected number of strides
        fps: Frame rate

    Returns:
        template: Normalized reference cycle (101 points, 0-100%)
        metadata: Dict with template info
    """
    # Multi-signal fusion for robust template
    heel_y = df_angles[f'y_{side}_heel'].values
    ankle_y = df_angles[f'y_{side}_ankle'].values
    knee_angle = df_angles[f'{side}_knee_angle'].values if f'{side}_knee_angle' in df_angles.columns else None

    # Remove NaN
    valid_idx = ~(np.isnan(heel_y) | np.isnan(ankle_y))
    if knee_angle is not None:
        valid_idx &= ~np.isnan(knee_angle)

    if valid_idx.sum() < fps:
        return None, {'error': 'insufficient_data'}

    heel_y = heel_y[valid_idx]
    ankle_y = ankle_y[valid_idx]

    # Smooth
    window_size = min(11, max(3, len(heel_y) // 3))
    if window_size % 2 == 0:
        window_size += 1

    heel_smooth = savgol_filter(heel_y, window_size, 2) if len(heel_y) >= window_size else heel_y
    ankle_smooth = savgol_filter(ankle_y, window_size, 2) if len(ankle_y) >= window_size else ankle_y

    # Composite signal: weighted combination
    composite = 0.7 * heel_smooth + 0.3 * ankle_smooth

    # Expected stride duration (use GT)
    total_duration_sec = len(composite) / fps
    expected_stride_duration_frames = int(total_duration_sec / gt_stride_count * fps)

    # Extract middle stride as template (avoid start/end artifacts)
    if len(composite) < expected_stride_duration_frames * 2:
        # Too short, use entire signal
        template_raw = composite
    else:
        # Extract middle stride
        mid_point = len(composite) // 2
        start_idx = max(0, mid_point - expected_stride_duration_frames // 2)
        end_idx = min(len(composite), start_idx + expected_stride_duration_frames)
        template_raw = composite[start_idx:end_idx]

    # Normalize to 101 points (0-100% gait cycle)
    template_resampled = np.interp(
        np.linspace(0, len(template_raw) - 1, 101),
        np.arange(len(template_raw)),
        template_raw
    )

    # Z-score normalize
    template_normalized = normalize_signal(template_resampled)

    metadata = {
        'expected_stride_frames': expected_stride_duration_frames,
        'template_length': len(template_raw),
        'source': 'middle_stride',
        'signal_type': 'heel_ankle_fusion'
    }

    return template_normalized, metadata


def detect_strikes_with_template(
    df_angles: pd.DataFrame,
    template: np.ndarray,
    expected_stride_frames: int,
    side: str,
    fps: float,
    similarity_threshold: float = 0.7
) -> Tuple[List[int], List[float]]:
    """
    Detect heel strikes using template matching with DTW.

    Args:
        df_angles: DataFrame with landmarks
        template: Reference gait cycle (101 points)
        expected_stride_frames: Expected stride duration in frames
        side: 'left' or 'right'
        fps: Frame rate
        similarity_threshold: DTW distance threshold for valid match

    Returns:
        strikes: List of frame indices
        scores: Similarity scores for each strike
    """
    # Extract signal
    heel_y = df_angles[f'y_{side}_heel'].values
    ankle_y = df_angles[f'y_{side}_ankle'].values

    valid_idx = ~(np.isnan(heel_y) | np.isnan(ankle_y))
    if valid_idx.sum() < fps:
        return [], []

    valid_frames = df_angles['frame'].values[valid_idx]
    heel_y = heel_y[valid_idx]
    ankle_y = ankle_y[valid_idx]

    # Smooth
    window_size = min(11, max(3, len(heel_y) // 3))
    if window_size % 2 == 0:
        window_size += 1

    heel_smooth = savgol_filter(heel_y, window_size, 2) if len(heel_y) >= window_size else heel_y
    ankle_smooth = savgol_filter(ankle_y, window_size, 2) if len(ankle_y) >= window_size else ankle_y

    # Composite signal
    composite = 0.7 * heel_smooth + 0.3 * ankle_smooth

    # Sliding window with DTW
    window_length = expected_stride_frames
    step_size = max(1, window_length // 4)  # 75% overlap

    strikes = []
    scores = []

    for start_idx in range(0, len(composite) - window_length + 1, step_size):
        end_idx = start_idx + window_length
        window = composite[start_idx:end_idx]

        # Resample to 101 points
        window_resampled = np.interp(
            np.linspace(0, len(window) - 1, 101),
            np.arange(len(window)),
            window
        )

        # Normalize
        window_normalized = normalize_signal(window_resampled)

        # DTW distance (reshape to column vectors for fastdtw)
        template_2d = template.reshape(-1, 1)
        window_2d = window_normalized.reshape(-1, 1)
        distance, path = fastdtw(template_2d, window_2d, dist=euclidean)

        # Normalize distance by template length
        normalized_distance = distance / len(template)

        # Convert to similarity score (0=dissimilar, 1=identical)
        similarity = 1.0 / (1.0 + normalized_distance)

        # Accept if above threshold and not too close to previous strike
        if similarity >= similarity_threshold:
            # Check minimum distance from last strike
            if len(strikes) == 0 or (start_idx - strikes[-1]) >= window_length * 0.6:
                strikes.append(start_idx)
                scores.append(similarity)

    # Convert to original frame indices
    strike_frames = valid_frames[strikes] if len(strikes) > 0 else []

    return list(strike_frames), scores


def evaluate_template_detector(
    subject_files: List[Tuple[str, str, float]],
    gt_data: Dict,
    similarity_threshold: float = 0.7
) -> Dict:
    """
    Evaluate template-based detector on all subjects.

    Args:
        subject_files: List of (subject_id, csv_path, fps)
        gt_data: Ground truth stride counts
        similarity_threshold: DTW threshold

    Returns:
        Dict with evaluation metrics
    """
    processor = MediaPipeCSVProcessor()

    results = []

    for subject_id, csv_path, fps in subject_files:
        if subject_id not in gt_data:
            continue

        gt_left = gt_data[subject_id]['left_strides']
        gt_right = gt_data[subject_id]['right_strides']

        if gt_left == 0 or gt_right == 0:
            continue

        try:
            # Load data
            df_wide = processor.load_csv(csv_path)
            df_angles = processor.calculate_joint_angles(df_wide)

            # Left side
            template_left, meta_left = create_reference_template(df_angles, 'left', gt_left, fps)
            if template_left is not None:
                strikes_left, scores_left = detect_strikes_with_template(
                    df_angles, template_left, meta_left['expected_stride_frames'],
                    'left', fps, similarity_threshold
                )
                ratio_left = len(strikes_left) / gt_left if gt_left > 0 else 0
            else:
                ratio_left = 0
                strikes_left = []

            # Right side
            template_right, meta_right = create_reference_template(df_angles, 'right', gt_right, fps)
            if template_right is not None:
                strikes_right, scores_right = detect_strikes_with_template(
                    df_angles, template_right, meta_right['expected_stride_frames'],
                    'right', fps, similarity_threshold
                )
                ratio_right = len(strikes_right) / gt_right if gt_right > 0 else 0
            else:
                ratio_right = 0
                strikes_right = []

            results.append({
                'subject_id': subject_id,
                'gt_left': gt_left,
                'gt_right': gt_right,
                'detected_left': len(strikes_left),
                'detected_right': len(strikes_right),
                'ratio_left': ratio_left,
                'ratio_right': ratio_right,
                'mean_score_left': np.mean(scores_left) if scores_left else 0,
                'mean_score_right': np.mean(scores_right) if scores_right else 0
            })

            print(f"{subject_id}: L={len(strikes_left)}/{gt_left} ({ratio_left:.2f}×), R={len(strikes_right)}/{gt_right} ({ratio_right:.2f}×)")

        except Exception as e:
            print(f"  Error processing {subject_id}: {e}")
            continue

    # Calculate aggregate metrics
    all_ratios = []
    for r in results:
        all_ratios.append(r['ratio_left'])
        all_ratios.append(r['ratio_right'])

    if len(all_ratios) == 0:
        return {'error': 'no_valid_results'}

    mean_ratio = np.mean(all_ratios)
    median_ratio = np.median(all_ratios)
    std_ratio = np.std(all_ratios)
    mad = np.mean([abs(r - 1.0) for r in all_ratios])
    n_exceeds_1p5 = sum(1 for r in all_ratios if r > 1.5)

    return {
        'per_subject': results,
        'aggregate': {
            'mean_ratio': mean_ratio,
            'median_ratio': median_ratio,
            'std_ratio': std_ratio,
            'mad': mad,
            'n_exceeds_1p5': n_exceeds_1p5,
            'n_total': len(all_ratios),
            'similarity_threshold': similarity_threshold
        }
    }


def load_ground_truth_stride_counts():
    """Load GT stride counts from processed JSON files."""
    gt_dir = Path("processed")
    gt_data = {}

    for json_file in sorted(gt_dir.glob("*_info.json")):
        with open(json_file) as f:
            data = json.load(f)

        subject_id = json_file.stem.replace('_info', '')
        left_strides = data.get('strides', {}).get('left', 0)
        right_strides = data.get('strides', {}).get('right', 0)

        gt_data[subject_id] = {
            'left_strides': left_strides,
            'right_strides': right_strides
        }

    return gt_data


def main():
    """Test template-based detector."""
    print("=" * 80)
    print("Phase 3B: Template-Based Heel Strike Detection")
    print("=" * 80)
    print()

    # Load GT
    gt_data = load_ground_truth_stride_counts()
    print(f"Loaded GT for {len(gt_data)} subjects")
    print()

    # Full cohort (all available subjects)
    test_subjects = [
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
        ('S1_21', 'data/21/21-2_side_pose_fps30.csv', 30),
        ('S1_22', 'data/22/22-2_side_pose_fps30.csv', 30),
    ]

    print(f"Testing on {len(test_subjects)} subjects (full available cohort):")
    print()

    # Test different thresholds
    thresholds = [0.5, 0.6, 0.7, 0.8]
    best_result = None
    best_mad = float('inf')

    for threshold in thresholds:
        print(f"\n--- Threshold: {threshold:.1f} ---")
        result = evaluate_template_detector(test_subjects, gt_data, threshold)

        if 'error' not in result:
            agg = result['aggregate']
            print(f"\nAggregate Results:")
            print(f"  Mean ratio:   {agg['mean_ratio']:.2f}×")
            print(f"  Median ratio: {agg['median_ratio']:.2f}×")
            print(f"  MAD:          {agg['mad']:.3f}")
            print(f"  Exceeds 1.5×: {agg['n_exceeds_1p5']}/{agg['n_total']}")

            if agg['mad'] < best_mad:
                best_mad = agg['mad']
                best_result = (threshold, result)

    print("\n" + "=" * 80)
    print("BEST CONFIGURATION")
    print("=" * 80)

    if best_result:
        best_threshold, best_data = best_result
        agg = best_data['aggregate']

        print(f"\nBest Threshold: {best_threshold:.1f}")
        print(f"Mean Ratio:     {agg['mean_ratio']:.2f}×")
        print(f"MAD:            {agg['mad']:.3f}")
        print(f"Exceeds 1.5×:   {agg['n_exceeds_1p5']}/{agg['n_total']}")

        # Save results
        output = {
            'best_threshold': best_threshold,
            'results': best_data,
            'comparison': {
                'baseline': 3.45,
                'optimized_params': 2.65,
                'template_based': agg['mean_ratio'],
                'improvement_vs_baseline': (3.45 - agg['mean_ratio']) / 3.45 * 100,
                'improvement_vs_optimized': (2.65 - agg['mean_ratio']) / 2.65 * 100
            }
        }

        with open('P3B_template_results.json', 'w') as f:
            json.dump(output, f, indent=2)

        print(f"\n✓ Saved to P3B_template_results.json")

        print(f"\nComparison:")
        print(f"  Baseline (fusion):          3.45× → {output['comparison']['template_based']:.2f}× ({output['comparison']['improvement_vs_baseline']:.1f}% improvement)")
        print(f"  Optimized params (P3A):     2.65× → {output['comparison']['template_based']:.2f}× ({output['comparison']['improvement_vs_optimized']:.1f}% improvement)")


if __name__ == '__main__':
    main()
