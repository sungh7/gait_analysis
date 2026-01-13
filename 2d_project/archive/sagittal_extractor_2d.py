"""
MediaPipe Sagittal View Joint Angle Extractor
Extracts joint angles from sagittal view videos using MediaPipe Pose
Focuses on hip, knee, and ankle angles in the sagittal plane (y-axis)
"""

import cv2
import numpy as np
import pandas as pd
import mediapipe as mp
from scipy.signal import savgol_filter, find_peaks
import warnings
warnings.filterwarnings('ignore')


class MediaPipeSagittalExtractor:
    """Extract joint angles from sagittal view gait videos."""

    def __init__(self, min_detection_confidence=0.7, min_tracking_confidence=0.7):
        """
        Initialize MediaPipe Pose.

        Args:
            min_detection_confidence: Minimum confidence for pose detection
            min_tracking_confidence: Minimum confidence for pose tracking
        """
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=2,
            enable_segmentation=False,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )

        # Landmark indices
        self.landmarks = {
            'LEFT_HIP': self.mp_pose.PoseLandmark.LEFT_HIP.value,
            'RIGHT_HIP': self.mp_pose.PoseLandmark.RIGHT_HIP.value,
            'LEFT_KNEE': self.mp_pose.PoseLandmark.LEFT_KNEE.value,
            'RIGHT_KNEE': self.mp_pose.PoseLandmark.RIGHT_KNEE.value,
            'LEFT_ANKLE': self.mp_pose.PoseLandmark.LEFT_ANKLE.value,
            'RIGHT_ANKLE': self.mp_pose.PoseLandmark.RIGHT_ANKLE.value,
            'LEFT_HEEL': self.mp_pose.PoseLandmark.LEFT_HEEL.value,
            'RIGHT_HEEL': self.mp_pose.PoseLandmark.RIGHT_HEEL.value,
            'LEFT_FOOT_INDEX': self.mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value,
            'RIGHT_FOOT_INDEX': self.mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value,
            'LEFT_SHOULDER': self.mp_pose.PoseLandmark.LEFT_SHOULDER.value,
            'RIGHT_SHOULDER': self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value,
        }

    def calculate_angle_2d(self, p1, p2, p3):
        """
        Calculate angle between three points in 2D.

        Args:
            p1, p2, p3: Points as (x, y) tuples
            p2 is the vertex of the angle

        Returns:
            Angle in degrees
        """
        v1 = np.array([p1[0] - p2[0], p1[1] - p2[1]])
        v2 = np.array([p3[0] - p2[0], p3[1] - p2[1]])

        # Normalize vectors
        v1_norm = np.linalg.norm(v1)
        v2_norm = np.linalg.norm(v2)

        if v1_norm == 0 or v2_norm == 0:
            return np.nan

        # Calculate angle
        cos_angle = np.dot(v1, v2) / (v1_norm * v2_norm)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angle = np.arccos(cos_angle)

        return np.degrees(angle)

    def extract_pose_landmarks(self, video_path, max_frames=None):
        """
        Extract 2D pose landmarks from video using pose_landmarks.

        Args:
            video_path: Path to video file
            max_frames: Maximum number of frames to process

        Returns:
            DataFrame with 2D landmarks for each frame
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        video_info = {
            'fps': fps,
            'width': width,
            'height': height,
            'total_frames': total_frames,
            'duration': total_frames / fps if fps > 0 else 0
        }

        print(f"Processing video (2D Mode): {fps} FPS, {width}x{height}, {total_frames} frames")

        landmarks_data = []
        frame_count = 0

        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break
            if max_frames and frame_count >= max_frames:
                break

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.pose.process(rgb_frame)

            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                frame_data = {'frame': frame_count}

                # Extract 2D landmark positions (normalized 0-1)
                for name, idx in self.landmarks.items():
                    lm = landmarks[idx]
                    frame_data[f'{name}_x'] = lm.x
                    frame_data[f'{name}_y'] = lm.y
                    # z in pose_landmarks is depth relative to hip, still useful to detect depth changes but not for 2D angle
                    frame_data[f'{name}_z'] = lm.z 
                    frame_data[f'{name}_visibility'] = lm.visibility

                landmarks_data.append(frame_data)

            frame_count += 1

        cap.release()
    # self.pose.close() # Do not close here, reusing instance

        df = pd.DataFrame(landmarks_data)
        print(f"Extracted 2D landmarks from {len(df)} frames")

        return df, video_info

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
    def calculate_joint_angles(self, landmarks_df):
        """
        Calculate hip, knee, and ankle angles from 2D landmarks.
        Uses 2D projection (x-y plane).

        Args:
            landmarks_df: DataFrame with 2D landmark coordinates

        Returns:
            DataFrame with joint angles added
        """
        df = landmarks_df.copy()

        for side in ['LEFT', 'RIGHT']:
            # Get 2D coordinates (y is inverted in image coords? MediaPipe y is top-down (0 at top, 1 at bottom))
            # Angle calculation depends on definition.
            # Usually: 
            # Knee: Angle between thigh and shank.
            # Hip: Angle between trunk (vertical) and thigh.
            
            # Note: In image coordinates, y increases downwards. 
            # If we want standard Cartesian (y up), we should flip y or account for it.
            # Dot product angle is invariant to uniform scaling/translation but orientation matters for sign.
            # We used simple cosine rule (dot/norm) in calculate_angle_2d which returns unsigned angle [0, 180].

            hip_x = df[f'{side}_HIP_x'].values
            hip_y = df[f'{side}_HIP_y'].values

            knee_x = df[f'{side}_KNEE_x'].values
            knee_y = df[f'{side}_KNEE_y'].values

            ankle_x = df[f'{side}_ANKLE_x'].values
            ankle_y = df[f'{side}_ANKLE_y'].values
            
            # Using shoulder for Trunk-Thigh angle (Hip)
            shoulder_x = df.get(f'{side}_SHOULDER_x', np.full_like(hip_x, np.nan)).values
            shoulder_y = df.get(f'{side}_SHOULDER_y', np.full_like(hip_y, np.nan)).values
            
            # Foot index for ankle
            foot_x = df[f'{side}_FOOT_INDEX_x'].values
            foot_y = df[f'{side}_FOOT_INDEX_y'].values

            # Knee angle (hip-knee-ankle) - 2D
            # Flexion = 180 - internal angle? Or just internal angle?
            # Standard: Extension ~ 180, Flexion < 180. Or Extension ~ 0, Flexion > 0.
            # Let's compute the internal angle first.
            knee_angles = []
            for i in range(len(df)):
                if pd.notna(hip_x[i]) and pd.notna(knee_x[i]) and pd.notna(ankle_x[i]):
                    # Angle at Knee
                    angle = self.calculate_angle_2d(
                        (hip_x[i], hip_y[i]),
                        (knee_x[i], knee_y[i]),
                        (ankle_x[i], ankle_y[i])
                    )
                    # Convert to Flexion? 
                    # If straight leg is 180, then 180 - angle = flexion (0 at straight)
                    # Let's store "Flexion Angle" (0 = straight, 90 = bent)
                    # calculate_angle_2d returns internal angle (could be ~180 for straight)
                    
                    flexion_angle = 180.0 - angle
                    knee_angles.append(flexion_angle)
                else:
                    knee_angles.append(np.nan)

            df[f'{side.lower()}_knee_angle'] = knee_angles

            # Ankle angle (knee-ankle-foot) - 2D
            # Plantarflexion vs Dorsiflexion
            # Neutral is usually 90 degrees (between shank and foot)
            ankle_angles = []
            for i in range(len(df)):
                if pd.notna(knee_x[i]) and pd.notna(ankle_x[i]) and pd.notna(foot_x[i]):
                    angle = self.calculate_angle_2d(
                        (knee_x[i], knee_y[i]),
                        (ankle_x[i], ankle_y[i]),
                        (foot_x[i], foot_y[i])
                    )
                    # Convert to Clinical Ankle Angle? 
                    # Internal angle is usually ~90-120.
                    # Clinical: < 90 dorsi, > 90 plantar? Or 0 neutral?
                    # Let's keep raw internal angle for now or centered at 90.
                    # Commonly: angle - 90
                    ankle_angles.append(angle) # Keep 2D internal angle
                else:
                    ankle_angles.append(np.nan)

            df[f'{side.lower()}_ankle_angle'] = ankle_angles

            # Hip angle: Trunk-Thigh
            # If shoulder available, use Shoulder-Hip-Knee
            hip_angles = []
            for i in range(len(df)):
                if pd.notna(hip_x[i]) and pd.notna(knee_x[i]):
                    # If shoulder exists
                    if pd.notna(shoulder_x[i]):
                         angle = self.calculate_angle_2d(
                            (shoulder_x[i], shoulder_y[i]),
                            (hip_x[i], hip_y[i]),
                            (knee_x[i], knee_y[i])
                        )
                         # Straight trunk-thigh is 180. Flexion is deviation from 180.
                         flexion = 180.0 - angle
                         hip_angles.append(flexion)
                    else:
                        # Use vertical vector (assume camera upright)
                        # Vertical in image is (0, -1) [Up] or (0, 1) [Down]?
                        # MediaPipe Y is down. Hip to Knee is down.
                        # Vertical "Up" relative to hip is (0, -1).
                        # Vertical "Down" is (0, 1). 
                        # Vector Hip->Knee. Vector Hip->VerticalUp.
                        # Angle.
                        # Easier: Angle of Hip-Knee vector relative to vertical Y axis.
                        vec = np.array([knee_x[i]-hip_x[i], knee_y[i]-hip_y[i]])
                        vertical = np.array([0, 1]) # Down
                        # Dot product
                        v_norm = np.linalg.norm(vec)
                        if v_norm > 0:
                            cos_a = np.dot(vec, vertical) / v_norm
                            ang = np.degrees(np.arccos(np.clip(cos_a, -1, 1)))
                            # If leg is straight down, angle is 0. Flexion > 0.
                            hip_angles.append(ang)
                        else:
                            hip_angles.append(np.nan)
                else:
                    hip_angles.append(np.nan)

            df[f'{side.lower()}_hip_angle'] = hip_angles

        return df

    def detect_heel_strikes(self, landmarks_df, video_info, side='right'):
        """
        Detect heel strikes for gait cycle segmentation.

        Args:
            landmarks_df: DataFrame with landmarks
            video_info: Video metadata
            side: 'left' or 'right'

        Returns:
            Array of frame indices for heel strikes
        """
        fps = video_info['fps']
        if fps == 0:
            return np.array([])

        side_upper = side.upper()
        heel_y = landmarks_df[f'{side_upper}_HEEL_y'].values

        # Remove NaN values
        valid_idx = ~np.isnan(heel_y)
        if valid_idx.sum() < fps:
            return np.array([])

        valid_frames = landmarks_df['frame'].values[valid_idx]
        valid_heel_y = heel_y[valid_idx]

        # Smooth trajectory
        window_size = min(11, len(valid_heel_y) // 3)
        if window_size % 2 == 0:
            window_size += 1
        if window_size < 3:
            return np.array([])

        heel_y_smooth = savgol_filter(valid_heel_y, window_size, 2)

        # Find peaks (lowest points in image coordinates = heel strikes)
        min_distance = max(fps // 3, 10)  # At least 1/3 second between strikes
        prominence = np.std(heel_y_smooth) * 0.3

        peaks_idx, _ = find_peaks(
            heel_y_smooth,  # Lowest point = maximum in image y
            distance=min_distance,
            prominence=prominence
        )

        if len(peaks_idx) == 0:
            return np.array([])

        heel_strike_frames = valid_frames[peaks_idx]
        return heel_strike_frames

    def segment_gait_cycles(self, angles_df, video_info, side='right'):
        """
        Segment gait cycles and normalize to 0-100%.

        Args:
            angles_df: DataFrame with joint angles
            video_info: Video metadata
            side: 'left' or 'right'

        Returns:
            List of normalized gait cycle DataFrames
        """
        heel_strikes = self.detect_heel_strikes(angles_df, video_info, side)

        if len(heel_strikes) < 2:
            print(f"Warning: Only {len(heel_strikes)} heel strikes detected for {side} side")
            return []

        gait_cycles = []

        for i in range(len(heel_strikes) - 1):
            start_frame = heel_strikes[i]
            end_frame = heel_strikes[i + 1]

            cycle_data = angles_df[
                (angles_df['frame'] >= start_frame) &
                (angles_df['frame'] < end_frame)
            ].copy()

            if len(cycle_data) < 10:
                continue

            # Normalize to 0-100 gait cycle percentage
            cycle_data['gait_cycle'] = np.linspace(0, 100, len(cycle_data), endpoint=False)
            cycle_data['cycle_number'] = i

            gait_cycles.append(cycle_data)

        print(f"Segmented {len(gait_cycles)} gait cycles for {side} side")
        return gait_cycles

    def average_gait_cycles(self, gait_cycles, side='right'):
        """
        Average multiple gait cycles into a single representative cycle.

        Args:
            gait_cycles: List of gait cycle DataFrames
            side: 'left' or 'right'

        Returns:
            DataFrame with averaged angles for 0-100 gait cycle
        """
        if len(gait_cycles) == 0:
            return None

        # Interpolate each cycle to 101 points (0-100)
        gait_cycle_points = np.arange(0, 101)
        side_lower = side.lower()

        interpolated_cycles = {
            'hip': [],
            'knee': [],
            'ankle': []
        }

        for cycle_df in gait_cycles:
            for joint in ['hip', 'knee', 'ankle']:
                angle_col = f'{side_lower}_{joint}_angle'
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
        averaged_data = {
            'gait_cycle': gait_cycle_points,
            'joint': [],
            'plane': [],
            'angle_mean': [],
            'angle_std': []
        }

        for joint in ['hip', 'knee', 'ankle']:
            if len(interpolated_cycles[joint]) > 0:
                cycles_array = np.array(interpolated_cycles[joint])
                mean_angles = np.mean(cycles_array, axis=0)
                std_angles = np.std(cycles_array, axis=0)

                averaged_data['joint'].extend([f'{side_lower}.{joint[0:2]}.angle'] * 101)
                averaged_data['plane'].extend(['y'] * 101)
                averaged_data['angle_mean'].extend(mean_angles)
                averaged_data['angle_std'].extend(std_angles)

        # Replicate gait_cycle for each joint
        averaged_data['gait_cycle'] = np.tile(gait_cycle_points, 3)

        return pd.DataFrame(averaged_data)

    def process_video(self, video_path, side='right', max_frames=None):
        """
        Complete processing pipeline for a video.

        Args:
            video_path: Path to video file
            side: 'left' or 'right' for gait cycle detection
            max_frames: Limit number of frames to process
        Returns:
            DataFrame with averaged gait cycle angles
        """
        print(f"\n{'='*60}")
        print(f"Processing: {video_path}")
        print(f"{'='*60}")

        # Step 1: Extract landmarks
        landmarks_df, video_info = self.extract_pose_landmarks(video_path, max_frames=max_frames)

        # Step 2: Calculate angles
        angles_df = self.calculate_joint_angles(landmarks_df)

        # Step 3: Segment gait cycles
        gait_cycles = self.segment_gait_cycles(angles_df, video_info, side)

        # Step 4: Average cycles
        averaged_cycle = self.average_gait_cycles(gait_cycles, side)

        return averaged_cycle, gait_cycles, video_info


def main():
    """Test the extractor on a sample video."""
    extractor = MediaPipeSagittalExtractor()

    # Test on subject 1
    video_path = "/data/gait/data/1/1-2.mp4"
    averaged_cycle, gait_cycles, video_info = extractor.process_video(video_path, side='right')

    if averaged_cycle is not None:
        print("\n" + "="*60)
        print("Results Summary")
        print("="*60)
        print(f"Video: {video_info['fps']} FPS, {video_info['duration']:.1f}s")
        print(f"Gait cycles detected: {len(gait_cycles)}")
        print(f"\nAveraged cycle shape: {averaged_cycle.shape}")
        print("\nSample data:")
        print(averaged_cycle.head(10))

        # Save results
        output_path = "/data/gait/test_mediapipe_extraction.csv"
        averaged_cycle.to_csv(output_path, index=False)
        print(f"\nSaved to: {output_path}")


if __name__ == "__main__":
    main()
