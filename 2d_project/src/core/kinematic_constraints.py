#!/usr/bin/env python3
"""
Biomechanical Kinematic Constraints for Gait Analysis.

Enforces anatomically plausible joint angles and segment lengths.
This is the HIGHEST IMPACT improvement according to roadmap analysis.

Addresses QC failures by fixing anatomically impossible poses that cause
correlation drops with ground truth.
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional
from dataclasses import dataclass


@dataclass
class JointLimits:
    """Physiological joint angle limits from biomechanics literature."""
    min_angle: float  # degrees
    max_angle: float  # degrees
    max_velocity: float  # deg/sec
    max_acceleration: float  # deg/sec²


# Biomechanical constraints from literature
# Winter (2009) "Biomechanics and Motor Control of Human Movement"
JOINT_CONSTRAINTS = {
    'knee': JointLimits(
        min_angle=0,       # Full extension (cannot hyperextend in healthy gait)
        max_angle=140,     # Max flexion during swing
        max_velocity=450,  # deg/sec (Perry & Burnfield, 2010)
        max_acceleration=2500  # deg/sec²
    ),
    'hip': JointLimits(
        min_angle=-30,     # Max extension during late stance
        max_angle=120,     # Max flexion
        max_velocity=350,  # deg/sec
        max_acceleration=2000  # deg/sec²
    ),
    'ankle': JointLimits(
        min_angle=0,       # MediaPipe uses absolute geometric angles
        max_angle=180,     # Wide range to preserve ROM (not anatomical angles)
        max_velocity=400,  # deg/sec (faster than hip/knee) - NOT APPLIED, see line 111
        max_acceleration=2200  # deg/sec² - NOT APPLIED, see line 111
    )
}

# Anatomical segment length constraints (normalized by height)
# From Winter (2009) anthropometric tables
SEGMENT_LENGTH_RATIOS = {
    'thigh': (0.22, 0.28),      # 22-28% of height
    'shank': (0.22, 0.28),      # 22-28% of height
    'foot': (0.13, 0.17)        # 13-17% of height
}


class KinematicConstraintEnforcer:
    """
    Enforces biomechanical constraints on joint angles and landmarks.

    Three-stage approach:
    1. Hard constraints: Clip impossible values
    2. Soft constraints: Regularize unlikely values
    3. Kinematic chain: Ensure segment lengths are consistent
    """

    def __init__(self, fps=30, subject_height=1.70):
        """
        Args:
            fps: Video frame rate
            subject_height: Subject height in meters (for segment length constraints)
        """
        self.fps = fps
        self.dt = 1.0 / fps
        self.subject_height = subject_height

        # Compute expected segment lengths
        self.expected_thigh_length = subject_height * np.mean(SEGMENT_LENGTH_RATIOS['thigh'])
        self.expected_shank_length = subject_height * np.mean(SEGMENT_LENGTH_RATIOS['shank'])
        self.expected_foot_length = subject_height * np.mean(SEGMENT_LENGTH_RATIOS['foot'])

    def enforce_joint_angle_constraints(self, angles_df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply hard and soft constraints to joint angles.

        Args:
            angles_df: DataFrame with columns like 'right_knee_angle', 'left_hip_angle', etc.

        Returns:
            Constrained angles DataFrame
        """
        constrained_df = angles_df.copy()

        for col in angles_df.columns:
            # Identify joint type
            joint_type = self._identify_joint_type(col)
            if joint_type not in JOINT_CONSTRAINTS:
                continue

            constraints = JOINT_CONSTRAINTS[joint_type]
            signal = angles_df[col].values

            # Stage 1: Hard angle limits (apply to all joints)
            signal_clipped = self._apply_hard_limits(signal, constraints)

            # Stage 2 & 3: Velocity/Acceleration constraints
            # SKIP for ankle - already handled by improved signal processing
            # Ankle needs to preserve ROM, and these constraints over-smooth it
            if joint_type == 'ankle':
                signal_final = signal_clipped
            else:
                # Stage 2: Velocity constraints (knee & hip only)
                signal_velocity_constrained = self._constrain_velocity(
                    signal_clipped, constraints.max_velocity
                )

                # Stage 3: Acceleration constraints (knee & hip only)
                signal_final = self._constrain_acceleration(
                    signal_velocity_constrained, constraints.max_acceleration
                )

            constrained_df[col] = signal_final

        return constrained_df

    def enforce_segment_length_constraints(
        self,
        landmarks_df: pd.DataFrame,
        landmarks_col_map: Dict[str, str]
    ) -> pd.DataFrame:
        """
        Enforce anatomical segment length consistency.

        Args:
            landmarks_df: DataFrame with 3D landmark coordinates
            landmarks_col_map: Map of segment endpoints, e.g.:
                {'hip': 'LEFT_HIP', 'knee': 'LEFT_KNEE', 'ankle': 'LEFT_ANKLE'}

        Returns:
            Corrected landmarks DataFrame
        """
        corrected_df = landmarks_df.copy()

        # Check each segment
        segments = [
            ('thigh', 'hip', 'knee'),
            ('shank', 'knee', 'ankle')
        ]

        for segment_name, proximal, distal in segments:
            if proximal not in landmarks_col_map or distal not in landmarks_col_map:
                continue

            # Get landmark coordinates
            prox_lm = landmarks_col_map[proximal]
            dist_lm = landmarks_col_map[distal]

            # Compute segment vectors for each frame
            prox_coords = self._extract_coordinates(corrected_df, prox_lm)
            dist_coords = self._extract_coordinates(corrected_df, dist_lm)

            if prox_coords is None or dist_coords is None:
                continue

            # Segment vectors
            segment_vectors = dist_coords - prox_coords
            segment_lengths = np.linalg.norm(segment_vectors, axis=1)

            # Expected length range
            expected_length = getattr(self, f'expected_{segment_name}_length')
            min_length = expected_length * 0.85  # Allow 15% deviation
            max_length = expected_length * 1.15

            # Identify violations
            too_short = segment_lengths < min_length
            too_long = segment_lengths > max_length

            violations = too_short | too_long
            if violations.sum() > 0:
                print(f"⚠️  {segment_name.capitalize()}: {violations.sum()} length violations detected")

                # Correction strategy: Adjust distal landmark (assume proximal is more stable)
                for i in np.where(violations)[0]:
                    actual_length = segment_lengths[i]
                    target_length = np.clip(actual_length, min_length, max_length)

                    # Scale segment vector to correct length
                    if actual_length > 0:
                        scale_factor = target_length / actual_length
                        corrected_vector = segment_vectors[i] * scale_factor

                        # Update distal landmark
                        corrected_dist = prox_coords[i] + corrected_vector
                        self._update_coordinates(corrected_df, dist_lm, i, corrected_dist)

        return corrected_df

    def _apply_hard_limits(self, signal: np.ndarray, constraints: JointLimits) -> np.ndarray:
        """Clip angles to physiological range."""
        return np.clip(signal, constraints.min_angle, constraints.max_angle)

    def _constrain_velocity(self, signal: np.ndarray, max_velocity: float) -> np.ndarray:
        """
        Constrain angular velocity by smoothing rapid changes.

        Strategy: If velocity exceeds limit, blend with previous frame
        """
        constrained = signal.copy()

        for i in range(1, len(signal)):
            velocity = (signal[i] - signal[i-1]) / self.dt

            if abs(velocity) > max_velocity:
                # Violation: blend with previous frame
                max_change = max_velocity * self.dt
                direction = np.sign(velocity)
                constrained[i] = constrained[i-1] + direction * max_change

        return constrained

    def _constrain_acceleration(self, signal: np.ndarray, max_acceleration: float) -> np.ndarray:
        """
        Constrain angular acceleration.

        Strategy: Smooth regions with excessive acceleration
        """
        if len(signal) < 3:
            return signal

        constrained = signal.copy()

        # Compute velocity
        velocity = np.diff(constrained, prepend=constrained[0]) / self.dt

        for i in range(1, len(velocity)):
            acceleration = (velocity[i] - velocity[i-1]) / self.dt

            if abs(acceleration) > max_acceleration:
                # Violation: adjust angle to constrain acceleration
                max_vel_change = max_acceleration * self.dt
                direction = np.sign(acceleration)
                new_velocity = velocity[i-1] + direction * max_vel_change
                constrained[i] = constrained[i-1] + new_velocity * self.dt

        return constrained

    def _identify_joint_type(self, column_name: str) -> Optional[str]:
        """Extract joint type from column name."""
        col_lower = column_name.lower()
        if 'knee' in col_lower:
            return 'knee'
        elif 'hip' in col_lower:
            return 'hip'
        elif 'ankle' in col_lower:
            return 'ankle'
        return None

    def _extract_coordinates(self, df: pd.DataFrame, landmark_name: str) -> Optional[np.ndarray]:
        """Extract [x, y, z] coordinates for a landmark."""
        try:
            # Assume columns are like 'LEFT_KNEE_x', 'LEFT_KNEE_y', 'LEFT_KNEE_z'
            x_col = f'{landmark_name}_x'
            y_col = f'{landmark_name}_y'
            z_col = f'{landmark_name}_z'

            if x_col not in df.columns or y_col not in df.columns or z_col not in df.columns:
                return None

            coords = np.column_stack([
                df[x_col].values,
                df[y_col].values,
                df[z_col].values
            ])
            return coords
        except:
            return None

    def _update_coordinates(self, df: pd.DataFrame, landmark_name: str, frame_idx: int, new_coords: np.ndarray):
        """Update landmark coordinates at specific frame."""
        x_col = f'{landmark_name}_x'
        y_col = f'{landmark_name}_y'
        z_col = f'{landmark_name}_z'

        if x_col in df.columns:
            df.at[frame_idx, x_col] = new_coords[0]
            df.at[frame_idx, y_col] = new_coords[1]
            df.at[frame_idx, z_col] = new_coords[2]

    def validate_angles(self, angles_df: pd.DataFrame) -> Dict[str, Dict]:
        """
        Validate angles against constraints and report violations.

        Returns:
            Dict with violation counts for each joint
        """
        report = {}

        for col in angles_df.columns:
            joint_type = self._identify_joint_type(col)
            if joint_type not in JOINT_CONSTRAINTS:
                continue

            constraints = JOINT_CONSTRAINTS[joint_type]
            signal = angles_df[col].values

            # Check angle limits
            angle_violations = ((signal < constraints.min_angle) | (signal > constraints.max_angle)).sum()

            # Check velocity
            velocity = np.abs(np.diff(signal, prepend=signal[0]) / self.dt)
            velocity_violations = (velocity > constraints.max_velocity).sum()

            # Check acceleration
            acceleration = np.abs(np.diff(velocity, prepend=velocity[0]) / self.dt)
            accel_violations = (acceleration > constraints.max_acceleration).sum()

            report[col] = {
                'angle_violations': int(angle_violations),
                'velocity_violations': int(velocity_violations),
                'acceleration_violations': int(accel_violations),
                'total_frames': len(signal),
                'violation_rate': (angle_violations + velocity_violations + accel_violations) / (3 * len(signal))
            }

        return report

    def generate_constraint_report(self, angles_df: pd.DataFrame, output_path: str = None):
        """
        Generate detailed constraint violation report.
        """
        report = self.validate_angles(angles_df)

        print("\n" + "="*70)
        print("KINEMATIC CONSTRAINT VALIDATION REPORT")
        print("="*70)

        total_violations = 0
        total_frames = 0

        for joint, metrics in report.items():
            print(f"\n{joint}:")
            print(f"  Angle violations:        {metrics['angle_violations']:4d} / {metrics['total_frames']}")
            print(f"  Velocity violations:     {metrics['velocity_violations']:4d} / {metrics['total_frames']}")
            print(f"  Acceleration violations: {metrics['acceleration_violations']:4d} / {metrics['total_frames']}")
            print(f"  Overall violation rate:  {metrics['violation_rate']:.2%}")

            total_violations += (metrics['angle_violations'] +
                               metrics['velocity_violations'] +
                               metrics['acceleration_violations'])
            total_frames += metrics['total_frames'] * 3

            if metrics['violation_rate'] > 0.1:
                print(f"  ⚠️  HIGH violation rate - likely tracking errors")
            elif metrics['violation_rate'] > 0.05:
                print(f"  ⚠️  MODERATE violation rate")
            else:
                print(f"  ✓ Low violation rate - good tracking quality")

        print("\n" + "="*70)
        print(f"Overall Violation Rate: {total_violations}/{total_frames} ({100*total_violations/total_frames:.2f}%)")
        print("="*70)

        if output_path:
            with open(output_path, 'w') as f:
                f.write("joint,angle_violations,velocity_violations,accel_violations,total_frames,violation_rate\n")
                for joint, metrics in report.items():
                    f.write(f"{joint},{metrics['angle_violations']},{metrics['velocity_violations']},"
                           f"{metrics['acceleration_violations']},{metrics['total_frames']},"
                           f"{metrics['violation_rate']:.4f}\n")

        return report


# Example usage and testing
if __name__ == "__main__":
    from sagittal_extractor_2d import MediaPipeSagittalExtractor

    # Test with S1_08 (known bad knee tracking from QC report: r=0.443)
    VIDEO_PATH = "/data/gait/data/8/8-2.mp4"

    print("="*70)
    print("KINEMATIC CONSTRAINTS: Before vs After Comparison")
    print("="*70)

    # Extract angles
    extractor = MediaPipeSagittalExtractor()
    landmarks, _ = extractor.extract_pose_landmarks(VIDEO_PATH)
    angles_raw = extractor.calculate_joint_angles(landmarks)

    # Initialize constraint enforcer
    enforcer = KinematicConstraintEnforcer(fps=30, subject_height=1.70)

    # BEFORE: Validate raw angles
    print("\n▼ BEFORE CONSTRAINTS:")
    report_before = enforcer.generate_constraint_report(angles_raw)

    # Apply constraints
    print("\n\n▼ APPLYING CONSTRAINTS...")
    angles_constrained = enforcer.enforce_joint_angle_constraints(angles_raw)

    # AFTER: Validate constrained angles
    print("\n▼ AFTER CONSTRAINTS:")
    report_after = enforcer.generate_constraint_report(angles_constrained)

    # Compute improvement
    print("\n" + "="*70)
    print("IMPROVEMENT SUMMARY")
    print("="*70)

    for joint in report_before.keys():
        before_rate = report_before[joint]['violation_rate']
        after_rate = report_after[joint]['violation_rate']
        improvement = before_rate - after_rate

        print(f"\n{joint}:")
        print(f"  Violation rate: {before_rate:.2%} → {after_rate:.2%}")
        print(f"  Improvement: {improvement:.2%} ({100*improvement/before_rate:.1f}% reduction)")

    # Visualize
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    joints = ['right_knee_angle', 'right_hip_angle', 'right_ankle_angle']
    joint_labels = ['Knee', 'Hip', 'Ankle']

    for ax, joint, label in zip(axes, joints, joint_labels):
        # Plot raw and constrained
        ax.plot(angles_raw[joint], 'r-', alpha=0.4, label='Raw (unconstrained)', linewidth=1)
        ax.plot(angles_constrained[joint], 'b-', label='With constraints', linewidth=2)

        # Show constraint limits
        constraints = JOINT_CONSTRAINTS[label.lower()]
        ax.axhline(constraints.min_angle, color='gray', linestyle='--', alpha=0.5, label='Physiological limits')
        ax.axhline(constraints.max_angle, color='gray', linestyle='--', alpha=0.5)

        # Show violations
        signal_raw = angles_raw[joint].values
        violations = (signal_raw < constraints.min_angle) | (signal_raw > constraints.max_angle)
        if violations.sum() > 0:
            violation_frames = np.where(violations)[0]
            ax.scatter(violation_frames, signal_raw[violations],
                      c='red', s=20, marker='x', label='Violations corrected', zorder=5)

        ax.set_title(f"{label} Angle - Kinematic Constraints Applied")
        ax.set_ylabel('Angle (deg)')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel('Frame')
    plt.tight_layout()
    plt.savefig('/data/gait/2d_project/kinematic_constraints_demo.png', dpi=150)
    print("\n✓ Saved visualization to kinematic_constraints_demo.png")
