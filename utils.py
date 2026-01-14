"""
Utility functions for gait analysis data processing and visualization.
Includes both legacy functions and new Excel data processing utilities.
"""

import pandas as pd
import numpy as np
from typing import List, Optional, Tuple, Callable, Union
import json
from pathlib import Path


# ============================================================================
# Excel Data Processing Utilities
# ============================================================================

def load_subject_info(subject_id: str, processed_dir: str = "/data/gait/processed") -> dict:
    """
    Load subject information from JSON file.

    Args:
        subject_id: Subject identifier
        processed_dir: Directory containing processed files

    Returns:
        Dictionary with subject information
    """
    info_file = Path(processed_dir) / f"{subject_id}_info.json"
    with open(info_file, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_gait_data(subject_id: str, processed_dir: str = "/data/gait/processed") -> pd.DataFrame:
    """
    Load gait data for a specific subject.

    Args:
        subject_id: Subject identifier
        processed_dir: Directory containing processed files

    Returns:
        DataFrame with gait measurements
    """
    data_file = Path(processed_dir) / f"{subject_id}_gait_long.csv"
    return pd.read_csv(data_file)


def load_all_subjects(processed_dir: str = "/data/gait/processed") -> pd.DataFrame:
    """
    Load combined data for all subjects.

    Args:
        processed_dir: Directory containing processed files

    Returns:
        DataFrame with all subjects' data
    """
    combined_file = Path(processed_dir) / "all_subjects_combined.csv"
    return pd.read_csv(combined_file)


def filter_joint_plane(df: pd.DataFrame, joint: str, plane: str = 'y') -> pd.DataFrame:
    """
    Filter data for specific joint and plane.

    Args:
        df: Gait data DataFrame
        joint: Joint code (e.g., 'r.kn.angle', 'l.hi.angle')
        plane: Plane ('x' for frontal, 'y' for sagittal, 'z' for transverse)

    Returns:
        Filtered DataFrame
    """
    return df[(df['joint'] == joint) & (df['plane'] == plane)].copy()


def filter_joints(df: pd.DataFrame, joints: List[str]) -> pd.DataFrame:
    """
    Filter data for multiple joints.

    Args:
        df: Gait data DataFrame
        joints: List of joint codes

    Returns:
        Filtered DataFrame
    """
    return df[df['joint'].isin(joints)].copy()


def calculate_deviation(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate deviation from normal values.

    Adds columns:
    - deviation_from_normal: Absolute difference from normal average
    - deviation_normalized: Deviation normalized by normal SD
    - is_outside_normal_sd: Boolean indicating if outside 1 SD
    - is_outside_normal_2sd: Boolean indicating if outside 2 SD

    Args:
        df: Gait data DataFrame

    Returns:
        DataFrame with deviation columns added
    """
    df = df.copy()

    # Calculate deviations
    df['deviation_from_normal'] = df['condition1_avg'] - df['normal_avg']
    df['deviation_normalized'] = df['deviation_from_normal'] / df['normal_sd']

    # Flag outliers
    df['is_outside_normal_sd'] = np.abs(df['deviation_from_normal']) > df['normal_sd']
    df['is_outside_normal_2sd'] = np.abs(df['deviation_from_normal']) > (df['normal_sd'] * 2)

    return df


def get_outlier_summary(df: pd.DataFrame, by: str = 'joint') -> pd.DataFrame:
    """
    Summarize outliers by joint or subject.

    Args:
        df: Gait data DataFrame with deviation columns
        by: Group by 'joint' or 'subject_id'

    Returns:
        Summary DataFrame with outlier counts and percentages
    """
    if 'deviation_normalized' not in df.columns:
        df = calculate_deviation(df)

    summary = df.groupby(by).agg({
        'gait_cycle': 'count',
        'is_outside_normal_sd': 'sum',
        'is_outside_normal_2sd': 'sum',
        'deviation_normalized': ['mean', 'std', 'min', 'max']
    }).round(3)

    summary.columns = ['total_points', 'outside_1sd', 'outside_2sd',
                       'dev_mean', 'dev_std', 'dev_min', 'dev_max']

    # Calculate percentages
    summary['pct_outside_1sd'] = (summary['outside_1sd'] / summary['total_points'] * 100).round(2)
    summary['pct_outside_2sd'] = (summary['outside_2sd'] / summary['total_points'] * 100).round(2)

    return summary


def get_joint_comparison(subject_id: str, joint: str, plane: str = 'y',
                         processed_dir: str = "/data/gait/processed") -> pd.DataFrame:
    """
    Get comparison data for a specific joint between subject and normal values.

    Args:
        subject_id: Subject identifier
        joint: Joint code
        plane: Plane ('x', 'y', or 'z')
        processed_dir: Directory containing processed files

    Returns:
        DataFrame with gait cycle, subject values, and normal ranges
    """
    df = load_gait_data(subject_id, processed_dir)
    df = filter_joint_plane(df, joint, plane)
    df = calculate_deviation(df)

    # Select relevant columns for comparison
    comparison = df[['gait_cycle', 'condition1_avg', 'condition1_sd',
                     'normal_avg', 'normal_sd', 'normal_upper_sd', 'normal_lower_sd',
                     'deviation_from_normal', 'deviation_normalized',
                     'is_outside_normal_sd', 'is_outside_normal_2sd']].copy()

    return comparison.sort_values('gait_cycle')


def get_bilateral_comparison(subject_id: str, joint_type: str = 'kn',
                             plane: str = 'y', processed_dir: str = "/data/gait/processed") -> pd.DataFrame:
    """
    Compare left and right sides for a specific joint type.

    Args:
        subject_id: Subject identifier
        joint_type: Joint type code ('kn', 'hi', 'an', etc.)
        plane: Plane ('x', 'y', or 'z')
        processed_dir: Directory containing processed files

    Returns:
        DataFrame with bilateral comparison
    """
    df = load_gait_data(subject_id, processed_dir)

    right_joint = f'r.{joint_type}.angle'
    left_joint = f'l.{joint_type}.angle'

    right_df = filter_joint_plane(df, right_joint, plane)[['gait_cycle', 'condition1_avg']]
    left_df = filter_joint_plane(df, left_joint, plane)[['gait_cycle', 'condition1_avg']]

    comparison = right_df.merge(left_df, on='gait_cycle', suffixes=('_right', '_left'))
    comparison['difference'] = comparison['condition1_avg_right'] - comparison['condition1_avg_left']
    comparison['abs_difference'] = np.abs(comparison['difference'])

    return comparison.sort_values('gait_cycle')


def get_joint_name_mapping() -> dict:
    """
    Get human-readable names for joint codes.

    Returns:
        Dictionary mapping joint codes to readable names
    """
    return {
        'r.an.angle': 'Right Ankle',
        'r.kn.angle': 'Right Knee',
        'r.hi.angle': 'Right Hip',
        'r.ga.angle': 'Right Gait',
        'r.pe.angle': 'Right Pelvis',
        'r.to.angle': 'Right Torso',
        'l.an.angle': 'Left Ankle',
        'l.kn.angle': 'Left Knee',
        'l.hi.angle': 'Left Hip',
        'l.ga.angle': 'Left Gait',
        'l.pe.angle': 'Left Pelvis',
        'l.to.angle': 'Left Torso',
        'r.sh.angle': 'Right Shoulder',
        'r.el.angle': 'Right Elbow',
        'l.sh.angle': 'Left Shoulder',
        'l.el.angle': 'Left Elbow',
    }


def get_plane_name_mapping() -> dict:
    """
    Get human-readable names for anatomical planes.

    Returns:
        Dictionary mapping plane codes to readable names
    """
    return {
        'x': 'Frontal (Coronal)',
        'y': 'Sagittal',
        'z': 'Transverse (Horizontal)'
    }


def list_available_subjects(processed_dir: str = "/data/gait/processed") -> List[str]:
    """
    List all available subject IDs in the processed directory.

    Args:
        processed_dir: Directory containing processed files

    Returns:
        List of subject IDs
    """
    processed_path = Path(processed_dir)
    info_files = list(processed_path.glob("*_info.json"))
    subjects = [f.stem.replace('_info', '') for f in info_files]
    return sorted(subjects)


def get_conversion_summary(processed_dir: str = "/data/gait/processed") -> dict:
    """
    Load conversion summary report.

    Args:
        processed_dir: Directory containing processed files

    Returns:
        Dictionary with conversion summary
    """
    summary_file = Path(processed_dir) / "conversion_summary.json"
    if summary_file.exists():
        with open(summary_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}


def create_gait_cycle_pivot(df: pd.DataFrame, joint: str, plane: str = 'y',
                            value_col: str = 'condition1_avg') -> pd.DataFrame:
    """
    Create pivot table with gait cycles as rows and subjects as columns.

    Args:
        df: Gait data DataFrame (must include subject_id)
        joint: Joint code
        plane: Plane
        value_col: Column to pivot (default: 'condition1_avg')

    Returns:
        Pivot table DataFrame
    """
    filtered = filter_joint_plane(df, joint, plane)
    pivot = filtered.pivot(index='gait_cycle', columns='subject_id', values=value_col)
    return pivot.sort_index()


# ============================================================================
# Legacy Utility Functions (existing)
# ============================================================================

def nan_helper(y: np.ndarray) -> Tuple[np.ndarray, Callable]:
    """
    Helper function to handle NaN values in arrays.

    Args:
        y: Input array that may contain NaN values

    Returns:
        Tuple of (nan_mask, lambda function for non-zero indices)
    """
    return np.isnan(y), lambda z: z.nonzero()[0]


def stretched_data(
    y_data: np.ndarray,
    stretch_size: int = 100,
    reverse: bool = False
) -> np.ndarray:
    """
    Stretch/normalize time series data to a fixed length using interpolation.

    This is useful for normalizing gait cycles of different lengths to a
    common length for comparison and averaging.

    Args:
        y_data: Input time series data
        stretch_size: Target size after stretching (default: 100)
        reverse: If True, reverse the data before stretching

    Returns:
        Interpolated array of length stretch_size
    """
    y = np.array(y_data)
    if reverse:
        y = y[::-1]

    every = stretch_size / y.size
    stretched = np.zeros(stretch_size)

    # Place original values at calculated positions
    for hd in range(y.size):
        stretched[round(hd * every)] = y[hd]

    # Set last value
    stretched[-1] = y[-1]

    # Interpolate zero values (treated as NaN)
    stretched[np.where(stretched == 0)] = np.nan
    nans, x = nan_helper(stretched)
    stretched[nans] = np.interp(x(nans), x(~nans), stretched[~nans])

    return stretched


def find_continuous_sections(
    indices: np.ndarray,
    gap_threshold: int = 5,
    min_section_size: int = 10
) -> list:
    """
    Split array of indices into continuous sections based on gaps.

    Args:
        indices: Array of frame indices
        gap_threshold: Maximum gap between consecutive indices to be considered continuous
        min_section_size: Minimum size of a section to be included

    Returns:
        List of arrays, each representing a continuous section
    """
    if indices.size == 0:
        return []

    sections = []
    start_idx = 0

    for i in range(indices.size - 1):
        if abs(indices[i] - indices[i + 1]) > gap_threshold:
            sections.append(indices[start_idx:i + 1])
            start_idx = i + 1
        elif i + 1 == indices.size - 1:
            sections.append(indices[start_idx:])

    # Filter by minimum size
    sections = [s for s in sections if s.size >= min_section_size]

    return sections if sections else [indices]


def remove_close_peaks(
    peaks: np.ndarray,
    min_distance_quantile: float = 0.2
) -> np.ndarray:
    """
    Remove peaks that are too close to each other.

    Args:
        peaks: Array of peak indices
        min_distance_quantile: Quantile threshold for minimum distance between peaks

    Returns:
        Filtered array of peak indices
    """
    if len(peaks) < 2:
        return peaks

    # Calculate differences between consecutive peaks
    diffs = np.diff(peaks)
    min_distance = np.quantile(diffs, min_distance_quantile)

    # Find peaks to remove (those too close to the next peak)
    will_remove_index = []
    for ix, item in enumerate(peaks[:-1]):
        if abs(item - peaks[ix + 1]) < min_distance:
            will_remove_index.append(ix + 1)

    return np.delete(peaks, will_remove_index)


def calculate_width_features(df, left_pos, right_pos, frame_column='frame'):
    """
    Calculate width between left and right body positions across frames.

    Args:
        df: DataFrame containing position data
        left_pos: DataFrame with left position data
        right_pos: DataFrame with right position data
        frame_column: Name of the frame column

    Returns:
        Array of width values for each frame
    """
    widths = []
    for frame in df[frame_column].unique():
        left_frame = left_pos[left_pos[frame_column] == frame]
        right_frame = right_pos[right_pos[frame_column] == frame]

        if len(left_frame) > 0 and len(right_frame) > 0:
            width = abs(left_frame.iloc[0]['x'] - right_frame.iloc[0]['x'])
            widths.append(width)
        else:
            widths.append(np.nan)

    return np.array(widths)
