"""
MediaPipe CSV Processor
Loads pre-extracted MediaPipe 3D pose CSV files and calculates joint angles for gait analysis
Focuses on Knee, Hip, and Ankle angles in sagittal plane
"""

import json
import pandas as pd
import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.signal import savgol_filter, find_peaks
from typing import Optional, Tuple, Union, Dict
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from preprocessing_pipeline import PreprocessingPipeline, PreprocessingResult, ProcessingLogEntry
from angle_converter import AngleConverter
from gait_parameters import GaitParameterCalculator

class MediaPipeCSVProcessor:
    """Process MediaPipe CSV files to extract joint angles for gait analysis."""

    def __init__(
        self, 
        preprocessor: Optional[PreprocessingPipeline] = None, 
        use_fusion_detection: bool = False,
        filter_type: str = 'butterworth',
        cutoff_frequency: Union[float, Dict[str, float]] = {'ankle': 4.0, 'knee': 6.0, 'hip': 6.0, 'heel': 6.0, 'foot_index': 6.0},
        heel_strike_weights: Tuple[float, float, float] = (0.4, 0.4, 0.2),
        conversion_params_path: str = "/data/gait/angle_conversion_params.json"
    ):
        """Initialize processor."""
        if preprocessor is None:
            use_butterworth = (filter_type == 'butterworth')
            self.preprocessor = PreprocessingPipeline(
                use_butterworth=use_butterworth,
                butterworth_cutoff=cutoff_frequency
            )
        else:
            self.preprocessor = preprocessor
            
        self.use_fusion_detection = use_fusion_detection
        self.heel_strike_weights = heel_strike_weights
        self.last_preprocessing_result: Optional[PreprocessingResult] = None
        self.calibration = self._load_calibration_parameters()
        
        # Initialize AngleConverter and load params if available
        self.angle_converter = AngleConverter()
        if Path(conversion_params_path).exists():
            self.angle_converter.load_parameters(conversion_params_path)
            
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
        self.landmark_index = {name: idx for idx, name in enumerate(self.landmark_names)}

    def _load_calibration_parameters(self):
        """Load simple offset calibration parameters"""
        path = Path("calibration_parameters.json")
        if not path.exists():
            return {}
        try:
            with path.open() as f:
                return json.load(f)
        except Exception:
            return {}

    def _apply_calibration(self, joint_key: str, values):
        """
        Apply calibration: calibrated = (raw * scale) + offset
        """
        params = self.calibration.get(joint_key)
        if not params:
            return values
        
        offset = params.get("offset", 0.0)
        scale = params.get("scale", 1.0)
        
        arr = np.array(values, dtype=float)
        mask = ~np.isnan(arr)
        
        # Apply scaling and offset
        arr[mask] = (arr[mask] * scale) + offset
        return arr.tolist()

    def load_csv(self, csv_path):
        """
        Load MediaPipe CSV file and convert to wide format.

        Args:
            csv_path: Path to CSV file (format: frame,position,x,y,z,visibility)

        Returns:
            DataFrame with columns: frame, {landmark}_x, {landmark}_y, {landmark}_z, {landmark}_visibility
        """
        df = pd.read_csv(csv_path)
        
        # Check if already wide
        if 'LEFT_HIP_x' in df.columns:
            # Rename columns to match expected format: x_left_hip, y_left_hip, etc.
            # The current code expects: x_left_hip
            # The file has: LEFT_HIP_x
            # We need to map LEFT_HIP_x -> x_left_hip
            
            rename_map = {}
            for col in df.columns:
                if col == 'frame': continue
                # col: LEFT_HIP_x
                parts = col.split('_')
                if len(parts) >= 3:
                    # suffix is last part (x, y, z, visibility)
                    suffix = parts[-1]
                    # name is the rest, lowercased
                    name = '_'.join(parts[:-1]).lower()
                    new_col = f'{suffix}_{name}'
                    rename_map[col] = new_col
            
            df_wide = df.rename(columns=rename_map)
            print(f"Loaded {len(df_wide)} frames (Wide Format) from {Path(csv_path).name}")
            return df_wide
            
        # Pivot to wide format (Long format support)
        df_wide = df.pivot(index='frame', columns='position', values=['x', 'y', 'z', 'visibility'])
        df_wide.columns = [f'{col}_{pos}' for col, pos in df_wide.columns]
        df_wide = df_wide.reset_index()

        print(f"Loaded {len(df_wide)} frames (Long Format) from {Path(csv_path).name}")
        return df_wide

    def calculate_joint_angles(self, df_wide):
        """
        Calculate knee, hip, and ankle angles from wide-format landmarks.

        Args:
            df_wide: DataFrame with landmark coordinates

        Returns:
            DataFrame with joint angles added
        """
        df = df_wide.copy()
        left_hip_angles, right_hip_angles = [], []
        left_hip_add_angles, right_hip_add_angles = [], []
        left_hip_rot_angles, right_hip_rot_angles = [], []
        left_knee_angles, right_knee_angles = [], []
        left_knee_var_angles, right_knee_var_angles = [], []
        left_ankle_angles, right_ankle_angles = [], []
        left_ankle_inv_angles, right_ankle_inv_angles = [], []

        for row in df.itertuples(index=False, name='Row'):
            row_dict = row._asdict()
            landmarks = self._row_to_landmarks(row_dict)
            pelvis = self._build_pelvis_frame(landmarks)

            for side, hip_list, hip_add_list, hip_rot_list, knee_list, knee_var_list, ankle_list, ankle_inv_list in [
                ('left', left_hip_angles, left_hip_add_angles, left_hip_rot_angles, left_knee_angles, left_knee_var_angles, left_ankle_angles, left_ankle_inv_angles),
                ('right', right_hip_angles, right_hip_add_angles, right_hip_rot_angles, right_knee_angles, right_knee_var_angles, right_ankle_angles, right_ankle_inv_angles),
            ]:
                if pelvis is None:
                    hip_list.append(np.nan)
                    knee_list.append(np.nan)
                    ankle_list.append(np.nan)
                    continue

                _, pelvis_axes = pelvis
                femur_axes = self._build_femur_frame(landmarks, side, pelvis_axes)
                tibia_axes = self._build_tibia_frame(landmarks, side, pelvis_axes)
                foot_axes = self._build_foot_frame(landmarks, side)

                if femur_axes is None:
                    hip_list.append(np.nan)
                    hip_add_list.append(np.nan)
                    hip_rot_list.append(np.nan)
                else:
                    rel = self._relative_rotation(pelvis_axes, femur_axes)
                    theta_y, theta_x, theta_z = self._cardan_yxz(rel)
                    hip_list.append(theta_y)
                    hip_add_list.append(theta_x) # Adduction
                    hip_rot_list.append(theta_z) # Rotation (Internal/External)

                if femur_axes is None or tibia_axes is None:
                    knee_list.append(np.nan)
                    knee_var_list.append(np.nan)
                else:
                    rel = self._relative_rotation(femur_axes, tibia_axes)
                    theta_y, theta_x, _ = self._cardan_yxz(rel)
                    knee_list.append(theta_y)
                    knee_var_list.append(theta_x) # Varus/Valgus

                if tibia_axes is None or foot_axes is None:
                    ankle_list.append(np.nan)
                    ankle_inv_list.append(np.nan)
                else:
                    rel = self._relative_rotation(tibia_axes, foot_axes)
                    # Use YZX for Ankle per Vicon PiG
                    theta_y, theta_z, _ = self._cardan_yzx(rel)
                    ankle_list.append(theta_y)
                    ankle_inv_list.append(theta_z) # Inversion/Eversion

        df['left_hip_angle'] = self._apply_calibration('hip_flexion_extension', left_hip_angles)
        df['right_hip_angle'] = self._apply_calibration('hip_flexion_extension', right_hip_angles)
        df['left_hip_adduction'] = left_hip_add_angles
        df['right_hip_adduction'] = right_hip_add_angles
        df['left_hip_rotation'] = left_hip_rot_angles
        df['right_hip_rotation'] = right_hip_rot_angles
        
        df['left_knee_angle'] = self._apply_calibration('knee_flexion_extension', left_knee_angles)
        df['right_knee_angle'] = self._apply_calibration('knee_flexion_extension', right_knee_angles)
        df['left_knee_varus'] = left_knee_var_angles
        df['right_knee_varus'] = right_knee_var_angles
        
        df['left_ankle_angle'] = self._apply_calibration('ankle_dorsi_plantarflexion', left_ankle_angles)
        df['right_ankle_angle'] = self._apply_calibration('ankle_dorsi_plantarflexion', right_ankle_angles)
        df['left_ankle_inversion'] = left_ankle_inv_angles
        df['right_ankle_inversion'] = right_ankle_inv_angles

        print(f"Calculated angles for {len(df)} frames (Cardan YXZ)")
        return df

    def _row_to_landmarks(self, row_dict):
        coords = np.full((len(self.landmark_names), 3), np.nan, dtype=float)
        for name, idx in self.landmark_index.items():
            x = row_dict.get(f'x_{name}')
            y = row_dict.get(f'y_{name}')
            z = row_dict.get(f'z_{name}')
            if x is not None and y is not None and z is not None:
                coords[idx] = np.array([x, y, z], dtype=float)
        return coords

    def _normalize_vector(self, vec, fallback):
        if vec is None or np.any(~np.isfinite(vec)):
            return np.array(fallback, dtype=float)
        norm = np.linalg.norm(vec)
        if norm < 1e-8:
            return np.array(fallback, dtype=float)
        return vec / norm

    def _orthonormal_axes(self, axis_x, axis_y, axis_z):
        axis_x = self._normalize_vector(axis_x, [1.0, 0.0, 0.0])
        axis_y = self._normalize_vector(axis_y, [0.0, 1.0, 0.0])
        axis_z = self._normalize_vector(axis_z, [0.0, 0.0, 1.0])
        axis_x = self._normalize_vector(axis_x, [1.0, 0.0, 0.0])
        axis_y = self._normalize_vector(axis_y - np.dot(axis_y, axis_x) * axis_x, [0.0, 1.0, 0.0])
        axis_z = self._normalize_vector(axis_z - np.dot(axis_z, axis_x) * axis_x - np.dot(axis_z, axis_y) * axis_y, [0.0, 0.0, 1.0])
        return {'x': axis_x, 'y': axis_y, 'z': axis_z}

    def _axes_to_matrix(self, axes):
        return np.stack([axes['x'], axes['y'], axes['z']], axis=1)

    def _relative_rotation(self, parent_axes, child_axes):
        r_parent = self._axes_to_matrix(parent_axes)
        r_child = self._axes_to_matrix(child_axes)
        return r_parent.T @ r_child

    def _cardan_yxz(self, rotation_matrix):
        r = rotation_matrix
        theta_y = np.degrees(np.arctan2(r[0, 2], r[2, 2]))
        theta_x = np.degrees(np.arcsin(np.clip(r[2, 1], -1.0, 1.0)))
        theta_z = np.degrees(np.arctan2(-r[1, 0], r[1, 1]))
        return theta_y, theta_x, theta_z

    def _cardan_yzx(self, rotation_matrix):
        """
        Extract angles for YZX rotation sequence (Ry * Rz * Rx).
        Used for Ankle angles per Vicon Plug-in Gait standard.
        """
        r = rotation_matrix
        # Derived from R = Ry * Rz * Rx
        # R[2,0] = -sy*cz, R[0,0] = cy*cz => tan(y) = -R[2,0]/R[0,0]
        theta_y = np.degrees(np.arctan2(-r[2, 0], r[0, 0]))
        # R[1,0] = sz
        theta_z = np.degrees(np.arcsin(np.clip(r[1, 0], -1.0, 1.0)))
        # R[1,2] = -cz*sx, R[1,1] = cz*cx => tan(x) = -R[1,2]/R[1,1]
        theta_x = np.degrees(np.arctan2(-r[1, 2], r[1, 1]))
        return theta_y, theta_z, theta_x

    def _get_point(self, landmarks, name):
        idx = self.landmark_index[name]
        point = landmarks[idx]
        if np.any(~np.isfinite(point)):
            return None
        return point

    def _build_pelvis_frame(self, landmarks):
        """
        Build Pelvis coordinate frame per Vicon Plug-in Gait standard.
        Y: Left (RASI -> LASI)
        Z: Up (Perpendicular to plane)
        X: Front (Cross(Y, Z))
        """
        left_hip = self._get_point(landmarks, 'left_hip')
        right_hip = self._get_point(landmarks, 'right_hip')
        left_shoulder = self._get_point(landmarks, 'left_shoulder')
        right_shoulder = self._get_point(landmarks, 'right_shoulder')
        if left_hip is None or right_hip is None or left_shoulder is None or right_shoulder is None:
            return None
        pelvis_origin = (left_hip + right_hip) / 2
        
        # Y axis: Right Hip -> Left Hip (Left)
        axis_y = self._normalize_vector(left_hip - right_hip, [1.0, 0.0, 0.0])
        
        # Temporary Z axis: Up
        shoulder_mid = (left_shoulder + right_shoulder) / 2
        temp_up = shoulder_mid - pelvis_origin
        
        # X axis: Cross(Y, TempUp) = Left x Up = Front
        axis_x = self._normalize_vector(np.cross(axis_y, temp_up), [0.0, 0.0, 1.0]) # Hint Z? No, Front is Z? No Front is Y in global?
        # Global: Y=Up, Z=Fwd? No usually Y=Up.
        # Let's trust the cross product direction. Left x Up = Front.
        
        # Z axis: Cross(X, Y) = Front x Left = Up
        axis_z = self._normalize_vector(np.cross(axis_x, axis_y), [0.0, 1.0, 0.0])
        
        axes = self._orthonormal_axes(axis_x, axis_y, axis_z)
        return pelvis_origin, axes

    def _project_axis(self, reference, target):
        return self._normalize_vector(reference - np.dot(reference, target) * target, reference)

    def _build_femur_frame(self, landmarks, side, pelvis_axes):
        """
        Build Femur coordinate frame.
        X: Front
        Y: Left (Flexion Axis)
        Z: Up (Longitudinal)
        """
        hip_name = f'{side}_hip'
        knee_name = f'{side}_knee'
        hip = self._get_point(landmarks, hip_name)
        knee = self._get_point(landmarks, knee_name)
        if hip is None or knee is None:
            return None
            
        # Z axis: Knee -> Hip (Up)
        axis_z = self._normalize_vector(hip - knee, [0.0, 1.0, 0.0])
        
        # X axis: Project Pelvis X (Front) onto plane perpendicular to Z
        axis_x = self._project_axis(pelvis_axes['x'], axis_z)
        
        # Y axis: Cross(Z, X) = Up x Front = Left
        axis_y = self._normalize_vector(np.cross(axis_z, axis_x), [1.0, 0.0, 0.0])
        
        return self._orthonormal_axes(axis_x, axis_y, axis_z)

    def _build_tibia_frame(self, landmarks, side, pelvis_axes):
        """
        Build Tibia coordinate frame.
        X: Front
        Y: Left (Flexion Axis)
        Z: Up (Longitudinal)
        """
        knee_name = f'{side}_knee'
        ankle_name = f'{side}_ankle'
        toe_name = f'{side}_foot_index'
        knee = self._get_point(landmarks, knee_name)
        ankle = self._get_point(landmarks, ankle_name)
        toe = self._get_point(landmarks, toe_name)
        if knee is None or ankle is None or toe is None:
            return None
            
        # Z axis: Ankle -> Knee (Up)
        axis_z = self._normalize_vector(knee - ankle, [0.0, 1.0, 0.0])
        
        # Temporary Forward vector (Ankle -> Toe)
        temp_fwd = self._normalize_vector(toe - ankle, [0.0, 0.0, 1.0])
        
        # Y axis: Cross(Z, Fwd) = Up x Fwd = Left
        axis_y = self._normalize_vector(np.cross(axis_z, temp_fwd), [1.0, 0.0, 0.0])
        
        # X axis: Cross(Y, Z) = Left x Up = Front
        axis_x = self._normalize_vector(np.cross(axis_y, axis_z), [0.0, 0.0, 1.0])
        
        return self._orthonormal_axes(axis_x, axis_y, axis_z)

    def _build_foot_frame(self, landmarks, side):
        """
        Build Foot coordinate frame.
        X: Front
        Y: Left (Flexion Axis)
        Z: Up (Longitudinal)
        """
        toe_name = f'{side}_foot_index'
        ankle_name = f'{side}_ankle'
        
        toe = self._get_point(landmarks, toe_name)
        ankle = self._get_point(landmarks, ankle_name)
        
        if toe is None or ankle is None:
            return None
            
        # X axis: Ankle -> Toe (Forward)
        axis_x = self._normalize_vector(toe - ankle, [0.0, 0.0, 1.0])
        
        # Temporary Up vector (Global Y)
        global_up = np.array([0.0, 1.0, 0.0])
        
        # Y axis: Cross(Up, Fwd) = Left
        # Note: Cross(Up, Fwd) = Left
        axis_y = self._normalize_vector(np.cross(global_up, axis_x), [1.0, 0.0, 0.0])
        
        # Z axis: Cross(X, Y) = Front x Left = Up
        axis_z = self._normalize_vector(np.cross(axis_x, axis_y), [0.0, 1.0, 0.0])
        
        return self._orthonormal_axes(axis_x, axis_y, axis_z)

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

        w_heel, w_ankle, w_toe = self.heel_strike_weights
        ground_signal = w_heel * heel_y + w_ankle * ankle_y + w_toe * toe_y
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

    def check_cycle_quality(self, cycle_df: pd.DataFrame, fps: float) -> Tuple[bool, str]:
        """
        Check if a gait cycle is of good quality.
        
        Criteria:
        1. Duration: 0.6s < duration < 1.5s
        2. Knee ROM: > 20 degrees (to avoid static/noise)
        3. Continuity: No large gaps in frames
        
        Returns:
            (is_good, reason)
        """
        if len(cycle_df) < 10:
            return False, "Too few frames"
            
        duration = len(cycle_df) / fps
        if duration < 0.6 or duration > 1.5:
            return False, f"Invalid duration: {duration:.2f}s"
            
        # Check Knee ROM if available
        knee_cols = [c for c in cycle_df.columns if 'knee_angle' in c]
        if knee_cols:
            knee_rom = cycle_df[knee_cols[0]].max() - cycle_df[knee_cols[0]].min()
            if knee_rom < 20:
                return False, f"Low Knee ROM: {knee_rom:.1f}"
                
        return True, "OK"

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
        rejected_cycles = 0

        for i in range(len(heel_strikes) - 1):
            start_frame = heel_strikes[i]
            end_frame = heel_strikes[i + 1]

            cycle_data = df_angles[
                (df_angles['frame'] >= start_frame) &
                (df_angles['frame'] < end_frame)
            ].copy()

            # QC Check
            is_good, reason = self.check_cycle_quality(cycle_data, fps)
            if not is_good:
                rejected_cycles += 1
                continue

            # Normalize to 0-100 gait cycle percentage
            cycle_data['gait_cycle'] = np.linspace(0, 100, len(cycle_data), endpoint=False)
            cycle_data['cycle_number'] = i

            gait_cycles.append(cycle_data)

        print(f"Segmented {len(gait_cycles)} gait cycles for {side} side (Rejected {rejected_cycles} bad cycles)")
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
            # Process Flexion Angles
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
            
            # Process Frontal/Transverse Angles
            # Map: hip -> hip_adduction, knee -> knee_varus, ankle -> ankle_inversion
            extra_maps = {
                'hip_adduction': 'hip_adduction',
                'knee_varus': 'knee_varus',
                'ankle_inversion': 'ankle_inversion'
            }
            for key, suffix in extra_maps.items():
                col_name = f'{side}_{suffix}'
                if col_name in cycle_df.columns:
                    if key not in interpolated_cycles:
                        interpolated_cycles[key] = []
                    
                    valid_data = cycle_df[['gait_cycle', col_name]].dropna()
                    if len(valid_data) > 5:
                        interp_angles = np.interp(
                            gait_cycle_points,
                            valid_data['gait_cycle'].values,
                            valid_data[col_name].values
                        )
                        interpolated_cycles[key].append(interp_angles)

        # Average across cycles
        result_data = []

        all_keys = ['hip', 'knee', 'ankle', 'hip_adduction', 'knee_varus', 'ankle_inversion']
        for joint in all_keys:
            if joint in interpolated_cycles and len(interpolated_cycles[joint]) > 0:
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
    
        # Step 2b: Apply Angle Conversion (if configured)
        if self.angle_converter.conversion_params:
            for col in df_angles.columns:
                # Expected format: "{side}_{joint}_angle" (e.g. "left_knee_angle")
                
                parts = col.split('_')
                if len(parts) == 3 and parts[2] == 'angle':
                    side = parts[0] # left/right
                    joint = parts[1] # knee/hip/ankle
                    
                    # Map to params keys
                    # Params keys: "knee", "hip", "ankle"
                    if joint in self.angle_converter.conversion_params:
                        joint_params = self.angle_converter.conversion_params[joint].get(side)
                        if joint_params:
                            method = joint_params.get('conversion')
                            params = joint_params.get('params')
                            
                            original_values = df_angles[col].values
                            converted_values = self.angle_converter.apply_conversion(
                                original_values, method, params
                            )
                            df_angles[col] = converted_values
                            # print(f"Applied conversion to {col}: {method}")

        # Step 3 & 4: Segment and average for both sides
        results = {
            'metadata': {
                'fps': fps,
                'csv_path': str(csv_path),
            },
            'preprocessing_log': [self._log_entry_to_dict(entry) for entry in preprocess_result.log],
        }

        # Detect heel strikes for both sides first to calculate step time (requires both)
        heel_strikes = {}
        for side in ['left', 'right']:
            if self.use_fusion_detection:
                hs = self.detect_heel_strikes_fusion(df_angles, side, fps)
            else:
                hs = self.detect_heel_strikes(df_angles, side, fps)
            heel_strikes[side] = hs

        # Calculate Spatio-Temporal Parameters
        # Assuming default height 170cm for now, could be passed in init
        param_calc = GaitParameterCalculator(fps=fps, height_cm=170.0)
        temporal_params = param_calc.calculate_temporal_parameters(heel_strikes)
        spatial_params = param_calc.estimate_spatial_parameters(temporal_params)

        for side in ['left', 'right']:
            # Use already detected heel strikes for segmentation
            # Re-implement segmentation logic here to avoid re-detection or modify segment_gait_cycles
            # For simplicity, we'll call segment_gait_cycles which re-detects, 
            # but ideally we should refactor to pass heel_strikes.
            # Given the current structure, calling segment_gait_cycles is fine as detection is deterministic.
            
            gait_cycles = self.segment_gait_cycles(df_angles, side, fps)
            
            # Disable legacy flexion conversion (180-angle) because we now calculate Vicon-compliant angles (0-based)
            averaged_cycle = self.average_gait_cycles(gait_cycles, side, convert_to_flexion=False)
            
            results[side] = {
                'averaged_cycle': averaged_cycle,
                'num_cycles': len(gait_cycles),
                'raw_cycles': gait_cycles,
                'parameters': {
                    **temporal_params.get(side, {}),
                    **spatial_params.get(side, {})
                }
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
