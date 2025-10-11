"""
MediaPipe CSV Processor
Loads pre-extracted MediaPipe 3D pose CSV files and calculates joint angles for gait analysis
Focuses on Knee, Hip, and Ankle angles in sagittal plane
"""

import pandas as pd
import numpy as np
from scipy.signal import savgol_filter, find_peaks
from typing import Optional
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from preprocessing_pipeline import PreprocessingPipeline, PreprocessingResult, ProcessingLogEntry

class MediaPipeCSVProcessor:
    """Process MediaPipe CSV files to extract joint angles for gait analysis."""

    def __init__(self, preprocessor: Optional[PreprocessingPipeline] = None, use_fusion_detection: bool = False):
        """Initialize processor."""
        self.preprocessor = preprocessor or PreprocessingPipeline()
        self.use_fusion_detection = use_fusion_detection
        self.last_preprocessing_result: Optional[PreprocessingResult] = None
        self.landmark_names = [
            'nose', 'left_eye_inner', 'left_eye', 'left_eye_outer',
            'right_eye_inner', 'right_eye', 'right_eye_outer',
            'left_ear', 'right_ear', 'mouth_left', 'mouth_right',
            'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
            'left_wrist', 'right_wrist', 'left_pinky', 'right_pinky',
            'left_index', 'right_index', 'left_thumb', 'right_thumb',
            'left_hip', 'right_hip', 'left_knee', 'right_knee',
            'left_ankle', 'right_ankle', 'left_heel', 'right_heel',
            'left_foot_index', 'right_foot_index'
        ]

    def load_csv(self, csv_path):
        """
        Load MediaPipe CSV file and convert to wide format.

        Args:
            csv_path: Path to CSV file (format: frame,position,x,y,z,visibility)

        Returns:
            DataFrame with columns: frame, {landmark}_x, {landmark}_y, {landmark}_z, {landmark}_visibility
        """
        df_long = pd.read_csv(csv_path)

        # Pivot to wide format
        df_wide = df_long.pivot(index='frame', columns='position', values=['x', 'y', 'z', 'visibility'])
        df_wide.columns = [f'{col}_{pos}' for col, pos in df_wide.columns]
        df_wide = df_wide.reset_index()

        print(f"Loaded {len(df_wide)} frames from {Path(csv_path).name}")
        return df_wide

    def calculate_angle_3d(self, p1, p2, p3):
        """
        Calculate angle between three 3D points.

        Args:
            p1, p2, p3: Points as (x, y, z) tuples/arrays
            p2 is the vertex of the angle

        Returns:
            Angle in degrees
        """
        v1 = np.array([p1[0] - p2[0], p1[1] - p2[1], p1[2] - p2[2]])
        v2 = np.array([p3[0] - p2[0], p3[1] - p2[1], p3[2] - p2[2]])

        v1_norm = np.linalg.norm(v1)
        v2_norm = np.linalg.norm(v2)

        if v1_norm == 0 or v2_norm == 0:
            return np.nan

        cos_angle = np.dot(v1, v2) / (v1_norm * v2_norm)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angle = np.arccos(cos_angle)

        return np.degrees(angle)

    def calculate_joint_angles(self, df_wide):
        """
        Calculate knee, hip, and ankle angles from wide-format landmarks.

        Args:
            df_wide: DataFrame with landmark coordinates

        Returns:
            DataFrame with joint angles added
        """
        df = df_wide.copy()

        for side in ['left', 'right']:
            # Get 3D coordinates
            hip_x = df[f'x_{side}_hip'].values
            hip_y = df[f'y_{side}_hip'].values
            hip_z = df[f'z_{side}_hip'].values

            knee_x = df[f'x_{side}_knee'].values
            knee_y = df[f'y_{side}_knee'].values
            knee_z = df[f'z_{side}_knee'].values

            ankle_x = df[f'x_{side}_ankle'].values
            ankle_y = df[f'y_{side}_ankle'].values
            ankle_z = df[f'z_{side}_ankle'].values

            foot_x = df[f'x_{side}_foot_index'].values
            foot_y = df[f'y_{side}_foot_index'].values
            foot_z = df[f'z_{side}_foot_index'].values

            heel_x = df[f'x_{side}_heel'].values
            heel_y = df[f'y_{side}_heel'].values
            heel_z = df[f'z_{side}_heel'].values

            # Knee angle (hip-knee-ankle)
            knee_angles = []
            for i in range(len(df)):
                if pd.notna(hip_x[i]) and pd.notna(knee_x[i]) and pd.notna(ankle_x[i]):
                    angle = self.calculate_angle_3d(
                        (hip_x[i], hip_y[i], hip_z[i]),
                        (knee_x[i], knee_y[i], knee_z[i]),
                        (ankle_x[i], ankle_y[i], ankle_z[i])
                    )
                    knee_angles.append(angle)
                else:
                    knee_angles.append(np.nan)

            df[f'{side}_knee_angle'] = knee_angles

            # Ankle angle (knee-ankle-foot)
            ankle_angles = []
            for i in range(len(df)):
                if pd.notna(knee_x[i]) and pd.notna(ankle_x[i]) and pd.notna(foot_x[i]):
                    angle = self.calculate_angle_3d(
                        (knee_x[i], knee_y[i], knee_z[i]),
                        (ankle_x[i], ankle_y[i], ankle_z[i]),
                        (foot_x[i], foot_y[i], foot_z[i])
                    )
                    ankle_angles.append(angle)
                else:
                    ankle_angles.append(np.nan)

            df[f'{side}_ankle_angle'] = ankle_angles

            # Hip angle: Thigh angle relative to vertical
            # Vertical is (0, -1, 0) in world coordinates
            hip_angles = []
            for i in range(len(df)):
                if pd.notna(hip_x[i]) and pd.notna(knee_x[i]):
                    thigh_vec = np.array([
                        knee_x[i] - hip_x[i],
                        knee_y[i] - hip_y[i],
                        knee_z[i] - hip_z[i]
                    ])
                    vertical_vec = np.array([0, -1, 0])

                    thigh_norm = np.linalg.norm(thigh_vec)
                    if thigh_norm > 0:
                        cos_angle = np.dot(thigh_vec, vertical_vec) / thigh_norm
                        cos_angle = np.clip(cos_angle, -1.0, 1.0)
                        angle = np.degrees(np.arccos(cos_angle))
                        hip_angles.append(angle)
                    else:
                        hip_angles.append(np.nan)
                else:
                    hip_angles.append(np.nan)

            df[f'{side}_hip_angle'] = hip_angles

        print(f"Calculated angles for {len(df)} frames")
        return df

    def detect_heel_strikes(self, df_angles, side='right', fps=30):
        """
        Detect heel strikes for gait cycle segmentation.

        Args:
            df_angles: DataFrame with angles and landmarks
            side: 'left' or 'right'
            fps: Frame rate of the video

        Returns:
            Array of frame indices for heel strikes
        """
        heel_y = df_angles[f'y_{side}_heel'].values

        # Remove NaN values
        valid_idx = ~np.isnan(heel_y)
        if valid_idx.sum() < fps:
            return np.array([])

        valid_frames = df_angles['frame'].values[valid_idx]
        valid_heel_y = heel_y[valid_idx]

        # Smooth trajectory
        window_size = min(11, len(valid_heel_y) // 3)
        if window_size % 2 == 0:
            window_size += 1
        if window_size < 3:
            return np.array([])

        heel_y_smooth = savgol_filter(valid_heel_y, window_size, 2)

        # Find peaks (lowest points in y-coordinate = heel strikes)
        # In world coordinates, y is negative when low (near ground)
        # So we find minima by negating the signal
        min_distance = max(fps // 3, 10)
        prominence = np.std(heel_y_smooth) * 0.3

        peaks_idx, _ = find_peaks(
            -heel_y_smooth,  # Negate to find minima (heel on ground)
            distance=min_distance,
            prominence=prominence
        )

        if len(peaks_idx) == 0:
            return np.array([])

        heel_strike_frames = valid_frames[peaks_idx]
        return heel_strike_frames

    def detect_heel_strikes_fusion(self, df_angles, side='right', fps=30):
        """Improved heel strike detection using fused landmarks and velocity cues."""
        heel_col = f'y_{side}_heel'
        ankle_col = f'y_{side}_ankle'
        toe_col = f'y_{side}_foot_index'

        if any(col not in df_angles.columns for col in [heel_col, ankle_col, toe_col]):
            return self.detect_heel_strikes(df_angles, side, fps)

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

        ground_signal = 0.6 * heel_y + 0.3 * ankle_y + 0.1 * toe_y
        heel_velocity = np.gradient(heel_y, 1 / max(fps, 1))

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

        min_distance = max(fps // 3, 10)
        prominence = ground_std * 0.3 if ground_std > 0 else 0.01

        peaks_idx, _ = find_peaks(-ground_smooth, distance=min_distance, prominence=prominence)

        threshold = velocity_std * 0.5 if velocity_std > 0 else 0.01
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
        print(f"  Fusion detection: {len(peaks_idx)} raw peaks → {len(filtered_peaks)} validated heel strikes")

        return heel_strike_frames

    def segment_gait_cycles(self, df_angles, side='right', fps=30):
        """
        Segment gait cycles based on heel strikes.

        Args:
            df_angles: DataFrame with joint angles
            side: 'left' or 'right'
            fps: Frame rate

        Returns:
            List of gait cycle DataFrames
        """
        if self.use_fusion_detection:
            heel_strikes = self.detect_heel_strikes_fusion(df_angles, side, fps)
        else:
            heel_strikes = self.detect_heel_strikes(df_angles, side, fps)

        if len(heel_strikes) < 2:
            print(f"Warning: Only {len(heel_strikes)} heel strikes detected for {side} side")
            return []

        gait_cycles = []

        for i in range(len(heel_strikes) - 1):
            start_frame = heel_strikes[i]
            end_frame = heel_strikes[i + 1]

            cycle_data = df_angles[
                (df_angles['frame'] >= start_frame) &
                (df_angles['frame'] < end_frame)
            ].copy()

            if len(cycle_data) < 10:
                continue

            # Normalize to 0-100 gait cycle percentage
            cycle_data['gait_cycle'] = np.linspace(0, 100, len(cycle_data), endpoint=False)
            cycle_data['cycle_number'] = i

            gait_cycles.append(cycle_data)

        print(f"Segmented {len(gait_cycles)} gait cycles for {side} side")
        return gait_cycles

    def average_gait_cycles(self, gait_cycles, side='right', convert_to_flexion=True):
        """
        Average multiple gait cycles into a single representative cycle.

        Args:
            gait_cycles: List of gait cycle DataFrames
            side: 'left' or 'right'
            convert_to_flexion: If True, convert MediaPipe angles to flexion angles
                               (180 - angle) to match hospital data format

        Returns:
            DataFrame with averaged angles for 0-100 gait cycle
        """
        if len(gait_cycles) == 0:
            return None

        # Interpolate each cycle to 101 points (0-100)
        gait_cycle_points = np.arange(0, 101)

        interpolated_cycles = {
            'hip': [],
            'knee': [],
            'ankle': []
        }

        for cycle_df in gait_cycles:
            for joint in ['hip', 'knee', 'ankle']:
                angle_col = f'{side}_{joint}_angle'
                if angle_col in cycle_df.columns:
                    # Interpolate to 101 points
                    valid_data = cycle_df[['gait_cycle', angle_col]].dropna()
                    if len(valid_data) > 5:
                        interp_angles = np.interp(
                            gait_cycle_points,
                            valid_data['gait_cycle'].values,
                            valid_data[angle_col].values
                        )
                        interpolated_cycles[joint].append(interp_angles)

        # Average across cycles
        result_data = []

        for joint in ['hip', 'knee', 'ankle']:
            if len(interpolated_cycles[joint]) > 0:
                cycles_array = np.array(interpolated_cycles[joint])
                mean_angles = np.mean(cycles_array, axis=0)
                std_angles = np.std(cycles_array, axis=0)

                # Convert to flexion angle if requested
                # MediaPipe: 180° = extended, <180° = flexed
                # Hospital: 0° = extended, >0° = flexed
                # Conversion: flexion = 180 - mediapipe_angle
                if convert_to_flexion:
                    mean_angles = 180 - mean_angles
                    # Note: std remains the same as it's a measure of variability

                # Create long format for each gait cycle point
                for gc_point, mean_val, std_val in zip(gait_cycle_points, mean_angles, std_angles):
                    result_data.append({
                        'joint': f'{side[0]}.{joint[0:2]}.angle',  # r.kn.angle, l.hi.angle, etc.
                        'plane': 'y',  # Sagittal plane
                        'gait_cycle': int(gc_point),
                        'angle_mean': mean_val,
                        'angle_std': std_val,
                        'num_cycles': len(interpolated_cycles[joint])
                    })

        df_result = pd.DataFrame(result_data)
        print(f"Averaged {len(gait_cycles)} cycles for {side} side (flexion conversion: {convert_to_flexion})")
        return df_result

    def process_csv_file(self, csv_path, fps=30):
        """
        Complete processing pipeline for a MediaPipe CSV file.

        Args:
            csv_path: Path to MediaPipe CSV file
            fps: Frame rate (extracted from filename if possible)

        Returns:
            Dictionary with averaged gait cycle data for both sides
        """
        # Extract FPS from filename if available
        filename = Path(csv_path).name
        if 'fps' in filename:
            try:
                fps = int(filename.split('fps')[1].split('.')[0].split('_')[0])
            except:
                pass

        print(f"\n{'='*60}")
        print(f"Processing: {filename}")
        print(f"FPS: {fps}")
        print(f"{'='*60}")

        # Step 1: Load CSV
        df_wide = self.load_csv(csv_path)

        # Step 1b: Standardized preprocessing (visibility gating, resampling, smoothing)
        preprocess_result = self.preprocessor.process(df_wide, fps)
        self.last_preprocessing_result = preprocess_result
        fps = preprocess_result.fps
        df_preprocessed = preprocess_result.dataframe

        # Step 2: Calculate angles on preprocessed landmarks
        df_angles = self.calculate_joint_angles(df_preprocessed)

        # Step 3 & 4: Segment and average for both sides
        results = {
            'metadata': {
                'fps': fps,
                'csv_path': str(csv_path),
            },
            'preprocessing_log': [self._log_entry_to_dict(entry) for entry in preprocess_result.log],
        }

        for side in ['left', 'right']:
            gait_cycles = self.segment_gait_cycles(df_angles, side, fps)
            averaged_cycle = self.average_gait_cycles(gait_cycles, side)
            results[side] = {
                'averaged_cycle': averaged_cycle,
                'num_cycles': len(gait_cycles),
                'raw_cycles': gait_cycles
            }

        return results

    @staticmethod
    def _log_entry_to_dict(entry: ProcessingLogEntry) -> dict:
        """Convert preprocessing log entry to JSON-serialisable dict."""
        return {
            'step': entry.step,
            'details': {k: (float(v) if isinstance(v, (int, float)) else v) for k, v in entry.details.items()},
        }


def main():
    """Test the processor on a sample CSV file."""
    processor = MediaPipeCSVProcessor()

    # Test on subject 1
    csv_path = "/data/gait/data/1/1-2_side_pose_fps30.csv"
    results = processor.process_csv_file(csv_path)

    print("\n" + "="*60)
    print("Results Summary")
    print("="*60)

    for side in ['left', 'right']:
        print(f"\n{side.upper()} Side:")
        if results[side]['averaged_cycle'] is not None:
            df = results[side]['averaged_cycle']
            print(f"  Gait cycles detected: {results[side]['num_cycles']}")
            print(f"  Averaged data shape: {df.shape}")
            print(f"\n  Sample data:")
            print(df.head(15))
        else:
            print(f"  No gait cycles detected")

    # Save results
    output_path = "/data/gait/test_mediapipe_csv_processing.csv"
    if results['right']['averaged_cycle'] is not None:
        # Combine left and right
        df_combined = pd.concat([
            results['right']['averaged_cycle'],
            results['left']['averaged_cycle']
        ], ignore_index=True)
        df_combined.to_csv(output_path, index=False)
        print(f"\nSaved to: {output_path}")


if __name__ == "__main__":
    main()
