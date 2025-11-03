"""
P6: Label Swap Detection

GT-based cross-matching logic to detect if MediaPipe left/right labels
are swapped relative to ground truth measurements.

Author: Research Team
Date: 2025-10-22
"""

import pandas as pd
import numpy as np
from typing import Dict, List
from scipy.signal import find_peaks


def detect_label_swap(
    subject_id: str,
    df_angles: pd.DataFrame,
    gt_info: Dict,
    fps: float = 30.0
) -> Dict:
    """
    Detect if MediaPipe left/right labels are swapped relative to GT.

    Strategy:
    1. Sample a few strides from MediaPipe (both left and right)
    2. Compare median stride length with GT left and right
    3. Check both normal matching and cross matching
    4. If cross matching is significantly better → SWAP DETECTED

    Args:
        subject_id: Subject identifier
        df_angles: DataFrame with pose data
        gt_info: Ground truth info dict with patient.left/right.stride_length_cm
        fps: Frames per second

    Returns:
        Dict with swap detection results:
        {
            'swap_needed': bool,
            'confidence': float (0-100),
            'normal_score': float (matching error),
            'cross_score': float (matching error),
            'mp_left_median': float,
            'mp_right_median': float,
            'gt_left': float,
            'gt_right': float,
            'reason': str
        }
    """

    # Extract GT strides (convert cm to meters)
    try:
        gt_right_stride = gt_info['patient']['right']['stride_length_cm'] / 100.0
        gt_left_stride = gt_info['patient']['left']['stride_length_cm'] / 100.0
    except (KeyError, TypeError):
        return {
            'swap_needed': False,
            'confidence': 0.0,
            'reason': 'GT_DATA_MISSING',
            'error': 'Could not extract GT stride lengths'
        }

    # Sample MediaPipe strides from straight sections
    mp_left_strides = sample_straight_strides(df_angles, 'left', fps=fps, n_samples=5)
    mp_right_strides = sample_straight_strides(df_angles, 'right', fps=fps, n_samples=5)

    if len(mp_left_strides) == 0 or len(mp_right_strides) == 0:
        return {
            'swap_needed': False,
            'confidence': 0.0,
            'reason': 'INSUFFICIENT_MP_STRIDES',
            'mp_left_count': len(mp_left_strides),
            'mp_right_count': len(mp_right_strides)
        }

    # Compute median stride lengths
    mp_left_median = float(np.median(mp_left_strides))
    mp_right_median = float(np.median(mp_right_strides))

    mp_avg = np.mean([mp_left_median, mp_right_median])
    gt_avg = np.mean([gt_left_stride, gt_right_stride])
    scale = (gt_avg / mp_avg) if mp_avg and mp_avg > 0 else 1.0

    mp_left_scaled = mp_left_median * scale
    mp_right_scaled = mp_right_median * scale

    # Normal matching: MP left → GT left, MP right → GT right
    normal_left_error = abs(mp_left_scaled - gt_left_stride)
    normal_right_error = abs(mp_right_scaled - gt_right_stride)
    normal_score = max(normal_left_error, normal_right_error)  # Worst case

    # Cross matching: MP left → GT right, MP right → GT left
    cross_left_error = abs(mp_left_scaled - gt_right_stride)
    cross_right_error = abs(mp_right_scaled - gt_left_stride)
    cross_score = max(cross_left_error, cross_right_error)  # Worst case

    # Decision: if cross matching is significantly better
    # V5.3.1: 0.9 (10%)
    # V5.3.2: 0.95 (5%) - too sensitive, caused false positives
    # V5.4: 0.9 (10%) - back to conservative
    swap_needed = cross_score < normal_score * 0.9

    # Confidence: relative improvement
    if normal_score > 0:
        confidence = abs(normal_score - cross_score) / normal_score * 100
    else:
        confidence = 0.0

    result = {
        'swap_needed': swap_needed,
        'confidence': confidence,
        'normal_score': normal_score,
        'cross_score': cross_score,
        'mp_left_median': mp_left_scaled,
        'mp_right_median': mp_right_scaled,
        'mp_left_raw': mp_left_median,
        'mp_right_raw': mp_right_median,
        'scale_factor': scale,
        'gt_left': gt_left_stride,
        'gt_right': gt_right_stride,
        'reason': 'CROSS_MATCHING_BETTER' if swap_needed else 'NORMAL_MATCHING_BETTER',
        'normal_errors': {
            'left': normal_left_error,
            'right': normal_right_error
        },
        'cross_errors': {
            'left_to_right': cross_left_error,
            'right_to_left': cross_right_error
        }
    }

    return result


def sample_straight_strides(
    df_angles: pd.DataFrame,
    side: str,
    fps: float = 30.0,
    n_samples: int = 5
) -> np.ndarray:
    """
    Sample a few stride lengths from straight walking sections.

    Uses simple peak detection on heel height to find stride boundaries,
    then calculates 3D distance traveled by hip.

    Args:
        df_angles: DataFrame with pose coordinates
        side: 'left' or 'right'
        fps: Frames per second
        n_samples: Number of strides to sample

    Returns:
        Array of stride lengths (raw MediaPipe units)
    """

    # Get heel vertical position (y-axis, inverted for peaks)
    heel_y_col = f'y_{side}_heel'

    if heel_y_col not in df_angles.columns:
        return np.array([])

    heel_y = df_angles[heel_y_col].dropna().values

    if len(heel_y) < fps * 2:  # Need at least 2 seconds
        return np.array([])

    # Find peaks (heel strikes) - use -heel_y because strikes are minima
    min_distance = int(fps * 0.5)  # At least 0.5 sec between strikes
    peaks, _ = find_peaks(-heel_y, distance=min_distance)

    if len(peaks) < 2:
        return np.array([])

    # Calculate stride lengths
    strides = []

    for i in range(min(n_samples, len(peaks) - 1)):
        start_frame = peaks[i]
        end_frame = peaks[i + 1]

        hip_x_col = f'x_{side}_hip'
        hip_y_col = f'y_{side}_hip'
        hip_z_col = f'z_{side}_hip'

        if hip_x_col not in df_angles.columns:
            hip_x_col = 'x_left_hip'
        if hip_y_col not in df_angles.columns:
            hip_y_col = 'y_left_hip'
        if hip_z_col not in df_angles.columns:
            hip_z_col = 'z_left_hip'

        hip_x = df_angles[hip_x_col].iloc[start_frame:end_frame].values
        hip_y = df_angles[hip_y_col].iloc[start_frame:end_frame].values
        hip_z = df_angles[hip_z_col].iloc[start_frame:end_frame].values

        # Skip if too much missing data
        if np.isnan(hip_x).sum() > len(hip_x) * 0.3:
            continue

        # Fill NaN with interpolation
        hip_x = pd.Series(hip_x).interpolate(method='linear', limit_direction='both').values
        hip_y = pd.Series(hip_y).interpolate(method='linear', limit_direction='both').values
        hip_z = pd.Series(hip_z).interpolate(method='linear', limit_direction='both').values

        # Calculate 3D path length
        dx = np.diff(hip_x)
        dy = np.diff(hip_y)
        dz = np.diff(hip_z)

        stride_distance = np.sqrt(np.sum(dx**2 + dy**2 + dz**2))

        if np.isfinite(stride_distance) and stride_distance > 0.01:
            strides.append(float(stride_distance))

    return np.array(strides)


def print_swap_detection_result(subject_id: str, result: Dict):
    """Pretty print swap detection result."""

    print(f"\n{'='*70}")
    print(f"Label Swap Detection: {subject_id}")
    print(f"{'='*70}")

    if 'error' in result:
        print(f"\n❌ Error: {result['error']}")
        print(f"   Reason: {result['reason']}")
        return

    print(f"\nGround Truth:")
    print(f"  Left stride:  {result['gt_left']:.3f} m")
    print(f"  Right stride: {result['gt_right']:.3f} m")

    print(f"\nMediaPipe (detected):")
    print(f"  Left median:  {result['mp_left_median']:.3f} m")
    print(f"  Right median: {result['mp_right_median']:.3f} m")

    print(f"\nMatching Scores (lower = better):")
    print(f"  Normal matching: {result['normal_score']:.3f} m")
    print(f"    L→L error: {result['normal_errors']['left']:.3f} m")
    print(f"    R→R error: {result['normal_errors']['right']:.3f} m")

    print(f"  Cross matching:  {result['cross_score']:.3f} m")
    print(f"    L→R error: {result['cross_errors']['left_to_right']:.3f} m")
    print(f"    R→L error: {result['cross_errors']['right_to_left']:.3f} m")

    print(f"\nDecision:")
    if result['swap_needed']:
        print(f"  🔄 SWAP NEEDED (confidence: {result['confidence']:.0f}%)")
        print(f"  → MediaPipe left/right labels appear swapped")
    else:
        print(f"  ✓ Labels OK (confidence: {result['confidence']:.0f}%)")
        print(f"  → MediaPipe labels match GT")

    print(f"\nReason: {result['reason']}")


if __name__ == "__main__":
    print("Label swap detection module loaded.")
    print("Import detect_label_swap() function to use.")
