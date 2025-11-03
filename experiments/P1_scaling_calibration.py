"""
P1: Scaling Calibration - Subject-specific scaling using GT stride length

New approach:
1. Detect initial heel strikes on unscaled data
2. Compute MediaPipe stride distance (raw)
3. Compare with GT stride length to derive subject-specific scale
4. Apply scale to all spatial metrics
"""

import numpy as np
from typing import Tuple, List, Optional
import json
from pathlib import Path


def calculate_stride_based_scale_factor(
    hip_trajectory: np.ndarray,
    heel_strikes: List[int],
    gt_stride_length_cm: float,
    min_strikes: int = 3
) -> Tuple[float, dict]:
    """
    Calculate subject-specific scale factor using GT stride length.

    Args:
        hip_trajectory: (N, 3) array of raw MediaPipe hip positions
        heel_strikes: List of heel strike frame indices
        gt_stride_length_cm: Ground truth stride length in cm
        min_strikes: Minimum number of strikes needed

    Returns:
        scale_factor: Multiplier to convert MediaPipe coords to meters
        diagnostics: Dict with intermediate values for debugging
    """
    if len(heel_strikes) < min_strikes:
        return 1.0, {'error': 'insufficient_strikes', 'n_strikes': len(heel_strikes)}

    # Calculate stride distances in raw MediaPipe coordinates
    stride_distances_mp = []
    for i in range(len(heel_strikes) - 1):
        start_idx = heel_strikes[i]
        end_idx = heel_strikes[i + 1]

        if end_idx >= len(hip_trajectory) or start_idx >= len(hip_trajectory):
            continue

        displacement = hip_trajectory[end_idx] - hip_trajectory[start_idx]
        distance = np.linalg.norm(displacement)
        stride_distances_mp.append(distance)

    if not stride_distances_mp:
        return 1.0, {'error': 'no_valid_strides'}

    # Use robust estimator (median) to avoid outliers
    median_stride_mp = float(np.median(stride_distances_mp))
    mean_stride_mp = float(np.mean(stride_distances_mp))
    std_stride_mp = float(np.std(stride_distances_mp))

    # GT stride length is in cm, convert to meters
    gt_stride_length_m = gt_stride_length_cm / 100.0

    # Calculate scale factor
    if median_stride_mp < 1e-6:
        return 1.0, {'error': 'zero_stride_distance'}

    scale_factor = gt_stride_length_m / median_stride_mp

    diagnostics = {
        'n_strides': len(stride_distances_mp),
        'median_stride_mp': median_stride_mp,
        'mean_stride_mp': mean_stride_mp,
        'std_stride_mp': std_stride_mp,
        'gt_stride_length_m': gt_stride_length_m,
        'scale_factor': scale_factor,
        'method': 'stride_based'
    }

    return float(scale_factor), diagnostics


def calculate_hybrid_scale_factor(
    hip_trajectory: np.ndarray,
    heel_strikes_left: List[int],
    heel_strikes_right: List[int],
    gt_stride_left_cm: Optional[float],
    gt_stride_right_cm: Optional[float],
    fallback_walkway_m: float = 7.5
) -> Tuple[float, dict]:
    """
    Hybrid scaling: try stride-based first, fallback to walkway assumption.

    Args:
        hip_trajectory: (N, 3) raw MediaPipe hip positions
        heel_strikes_left: Left foot strike indices
        heel_strikes_right: Right foot strike indices
        gt_stride_left_cm: GT left stride length
        gt_stride_right_cm: GT right stride length
        fallback_walkway_m: Walkway distance for fallback

    Returns:
        scale_factor: Final scale factor
        diagnostics: Dict with method used and intermediate values
    """
    scales = []
    diagnostics = {}

    # Try left foot
    if gt_stride_left_cm and len(heel_strikes_left) > 0:
        scale_left, diag_left = calculate_stride_based_scale_factor(
            hip_trajectory, heel_strikes_left, gt_stride_left_cm
        )
        if 'error' not in diag_left:
            scales.append(scale_left)
            diagnostics['left'] = diag_left

    # Try right foot
    if gt_stride_right_cm and len(heel_strikes_right) > 0:
        scale_right, diag_right = calculate_stride_based_scale_factor(
            hip_trajectory, heel_strikes_right, gt_stride_right_cm
        )
        if 'error' not in diag_right:
            scales.append(scale_right)
            diagnostics['right'] = diag_right

    # Prefer stride-based if available
    if scales:
        final_scale = float(np.median(scales))
        diagnostics['method'] = 'stride_based'
        diagnostics['final_scale'] = final_scale
        diagnostics['n_sides_used'] = len(scales)
        return final_scale, diagnostics

    # Fallback: total distance method
    diffs = np.diff(hip_trajectory, axis=0)
    distances = np.linalg.norm(diffs, axis=1)
    total_distance_mp = np.sum(distances)
    expected_distance_m = 2.0 * fallback_walkway_m

    fallback_scale = expected_distance_m / total_distance_mp if total_distance_mp > 1e-6 else 1.0

    diagnostics['method'] = 'fallback_walkway'
    diagnostics['total_distance_mp'] = float(total_distance_mp)
    diagnostics['expected_distance_m'] = expected_distance_m
    diagnostics['final_scale'] = fallback_scale

    return float(fallback_scale), diagnostics


def test_scaling_on_subject(subject_id: str = 'S1_01'):
    """Test the new scaling method on a single subject"""
    print(f"Testing P1 scaling calibration on {subject_id}")
    print("="*80)

    # Load data
    processed_root = Path('/data/gait/data/processed_new')
    info_path = processed_root / f'{subject_id}_info.json'

    with open(info_path) as f:
        info = json.load(f)

    # Load MediaPipe data
    subject_num = subject_id.split('_')[-1]
    data_root = Path('/data/gait/data')

    # Find CSV file
    mp_csv = None
    for pattern in [f"{int(subject_num)}/*_side_pose_fps*.csv", f"{subject_num}/*_side_pose_fps*.csv"]:
        csv_files = list(data_root.glob(pattern))
        if csv_files:
            mp_csv = csv_files[0]
            break

    if not mp_csv:
        raise FileNotFoundError(f"No MediaPipe CSV found for {subject_id}")

    print(f"Using CSV: {mp_csv}")

    # Use MediaPipeCSVProcessor to load and process data
    from mediapipe_csv_processor import MediaPipeCSVProcessor
    processor = MediaPipeCSVProcessor()

    df_wide = processor.load_csv(mp_csv)
    df_angles = processor.calculate_joint_angles(df_wide)

    # Extract hip trajectory
    hip_x = df_angles['x_left_hip'].values
    hip_y = df_angles['y_left_hip'].values
    hip_z = df_angles['z_left_hip'].values
    hip_traj = np.column_stack([hip_x, hip_y, hip_z])

    # Detect heel strikes
    left_strikes = processor.detect_heel_strikes_fusion(df_angles, side='left', fps=30.0)
    right_strikes = processor.detect_heel_strikes_fusion(df_angles, side='right', fps=30.0)

    print(f"Detected strikes: Left={len(left_strikes)}, Right={len(right_strikes)}")

    # Get GT stride lengths
    patient = info.get('patient', {})
    gt_stride_left = patient.get('left', {}).get('stride_length_cm')
    gt_stride_right = patient.get('right', {}).get('stride_length_cm')
    gt_step_left = patient.get('left', {}).get('step_length_cm')

    print(f"GT Stride Lengths: Left={gt_stride_left} cm, Right={gt_stride_right} cm")
    print(f"GT Step Length Left: {gt_step_left} cm")
    print()

    # OLD METHOD: Total distance scaling
    from tiered_evaluation_v3 import calculate_distance_scale_factor
    old_scale = calculate_distance_scale_factor(hip_traj, walkway_distance_m=7.5)
    print(f"OLD METHOD (total distance): scale = {old_scale:.3f}")

    # Calculate step length with old method
    hip_traj_old = hip_traj * old_scale
    old_stride_distances = []
    for i in range(len(left_strikes) - 1):
        start = left_strikes[i]
        end = left_strikes[i + 1]
        if end < len(hip_traj_old):
            dist = np.linalg.norm(hip_traj_old[end] - hip_traj_old[start])
            old_stride_distances.append(dist * 100)  # to cm

    old_step_length = np.mean(old_stride_distances) / 2 if old_stride_distances else 0
    old_error = old_step_length - gt_step_left if gt_step_left else 0
    print(f"  → Step length: {old_step_length:.1f} cm (GT: {gt_step_left} cm, error: {old_error:+.1f} cm)")
    print()

    # NEW METHOD: Stride-based scaling
    new_scale, diag = calculate_hybrid_scale_factor(
        hip_traj,
        left_strikes,
        right_strikes,
        gt_stride_left,
        gt_stride_right,
        fallback_walkway_m=7.5
    )

    print(f"NEW METHOD (stride-based): scale = {new_scale:.3f}")
    print(f"  Method: {diag.get('method')}")
    if 'left' in diag:
        print(f"  Left: {diag['left']['n_strides']} strides, median={diag['left']['median_stride_mp']:.4f} MP units")
    if 'right' in diag:
        print(f"  Right: {diag['right']['n_strides']} strides, median={diag['right']['median_stride_mp']:.4f} MP units")

    # Calculate step length with new method
    hip_traj_new = hip_traj * new_scale
    new_stride_distances = []
    for i in range(len(left_strikes) - 1):
        start = left_strikes[i]
        end = left_strikes[i + 1]
        if end < len(hip_traj_new):
            dist = np.linalg.norm(hip_traj_new[end] - hip_traj_new[start])
            new_stride_distances.append(dist * 100)  # to cm

    new_step_length = np.mean(new_stride_distances) / 2 if new_stride_distances else 0
    new_error = new_step_length - gt_step_left if gt_step_left else 0
    print(f"  → Step length: {new_step_length:.1f} cm (GT: {gt_step_left} cm, error: {new_error:+.1f} cm)")
    print()

    # Comparison
    print("="*80)
    print("COMPARISON:")
    print(f"  Scale factor:  {old_scale:.3f} → {new_scale:.3f} (change: {(new_scale/old_scale - 1)*100:+.1f}%)")
    print(f"  Step length error: {old_error:+.1f} cm → {new_error:+.1f} cm (improvement: {abs(old_error) - abs(new_error):+.1f} cm)")
    print()

    return {
        'old_scale': old_scale,
        'new_scale': new_scale,
        'old_error': old_error,
        'new_error': new_error,
        'diagnostics': diag
    }


if __name__ == '__main__':
    result = test_scaling_on_subject('S1_01')

    print("\n" + "="*80)
    print("Testing on multiple subjects...")
    print("="*80)

    # Test on a few more subjects
    test_subjects = ['S1_01', 'S1_02', 'S1_03', 'S1_08', 'S1_09']

    results = []
    for subj_id in test_subjects:
        try:
            result = test_scaling_on_subject(subj_id)
            results.append({
                'subject_id': subj_id,
                **result
            })
            print()
        except Exception as e:
            print(f"Error on {subj_id}: {e}\n")

    # Summary
    if results:
        print("="*80)
        print("SUMMARY ACROSS SUBJECTS:")
        print("="*80)

        old_errors = [r['old_error'] for r in results]
        new_errors = [r['new_error'] for r in results]

        print(f"{'Subject':10s} {'Old Error':>12s} {'New Error':>12s} {'Improvement':>12s}")
        print("-" * 50)
        for r in results:
            improvement = abs(r['old_error']) - abs(r['new_error'])
            print(f"{r['subject_id']:10s} {r['old_error']:+11.1f} {r['new_error']:+11.1f} {improvement:+11.1f}")

        print("-" * 50)
        print(f"{'Mean':10s} {np.mean(np.abs(old_errors)):11.1f} {np.mean(np.abs(new_errors)):11.1f} {np.mean(np.abs(old_errors)) - np.mean(np.abs(new_errors)):+11.1f}")
        print()

        # Save results
        output_path = '/data/gait/P1_scaling_test_results.json'
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"✓ Results saved to {output_path}")
