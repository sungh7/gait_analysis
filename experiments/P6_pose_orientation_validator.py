"""
P6: Pose Orientation Validator

Multi-layer validation system to verify pose direction and anatomical consistency
before label correction.

Author: Research Team
Date: 2025-10-22
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple
from scipy.signal import find_peaks


class PoseOrientationValidator:
    """
    Multi-layer pose orientation validation.

    Validates that MediaPipe pose estimation correctly identifies
    left/right and front/back orientation in lateral view videos.

    Validation Layers:
    1. Anatomical consistency (nose > shoulder > hip > heel)
    2. Foot movement direction (heel vs toe positions)
    3. Head orientation (nose vs ear positions)
    4. Temporal consistency (bidirectional walking pattern)
    """

    def __init__(self, confidence_threshold: float = 75.0):
        """
        Args:
            confidence_threshold: Minimum confidence (%) to mark as reliable
        """
        self.confidence_threshold = confidence_threshold

    def validate(self, df_angles: pd.DataFrame) -> Dict:
        """
        Run all validation layers and compute confidence score.

        Args:
            df_angles: DataFrame with x/y/z coordinates for landmarks

        Returns:
            Dict with validation results:
            {
                'checks': {layer_name: result_string},
                'confidence': float (0-100),
                'direction': str (RIGHT_DOMINANT, LEFT_DOMINANT, etc.),
                'reliable': bool
            }
        """

        results = {}

        # Layer 1: Anatomical consistency
        try:
            results['anatomical'] = self._verify_anatomical_consistency(df_angles)
        except Exception as e:
            results['anatomical'] = f'ERROR: {str(e)}'

        # Layer 2: Foot movement
        try:
            results['foot_movement'] = self._verify_foot_movement(df_angles)
        except Exception as e:
            results['foot_movement'] = f'ERROR: {str(e)}'

        # Layer 3: Head orientation
        try:
            results['head_orientation'] = self._verify_head_orientation(df_angles)
        except Exception as e:
            results['head_orientation'] = f'ERROR: {str(e)}'

        # Layer 4: Temporal consistency
        try:
            results['temporal_consistency'] = self._verify_temporal_consistency(df_angles)
        except Exception as e:
            results['temporal_consistency'] = f'ERROR: {str(e)}'

        # Compute confidence
        confidence = self._compute_confidence(results)

        # Determine direction
        direction = self._determine_direction(results)

        return {
            'checks': results,
            'confidence': confidence,
            'direction': direction,
            'reliable': confidence >= self.confidence_threshold
        }

    def _verify_anatomical_consistency(self, df: pd.DataFrame) -> str:
        """
        Layer 1: Verify anatomical structure consistency.

        In right-facing lateral view with MediaPipe coordinates:
        - X increases from anterior → posterior
        - Nose should have the smallest X value, heel the largest

        Expected X-axis order: nose < shoulder < hip < heel
        """

        # Average X positions (anterior-posterior axis)
        nose_x = df['x_nose'].dropna().mean()
        shoulder_x = df['x_left_shoulder'].dropna().mean()
        hip_x = df['x_left_hip'].dropna().mean()
        heel_x = df['x_left_heel'].dropna().mean()

        # Check if all landmarks available
        if pd.isna([nose_x, shoulder_x, hip_x, heel_x]).any():
            return 'INSUFFICIENT_DATA'

        # Allow small tolerance to handle noisy ordering
        tolerance = 1e-3

        def strictly_increasing(values):
            return all(values[i] + tolerance < values[i + 1] for i in range(len(values) - 1))

        def strictly_decreasing(values):
            return all(values[i] > values[i + 1] + tolerance for i in range(len(values) - 1))

        values = [nose_x, shoulder_x, hip_x, heel_x]

        if strictly_increasing(values):
            # MediaPipe lateral coordinates: increasing X from nose→heel indicates right-facing
            return 'FACING_RIGHT'
        elif strictly_decreasing(values):
            return 'FACING_LEFT'
        else:
            # Inconsistent order - pose corruption
            return 'POSE_CORRUPTED'

    def _verify_foot_movement(self, df: pd.DataFrame) -> str:
        """
        Layer 2: Verify walking direction from foot landmarks.

        In right-facing walk:
        - Heel should generally be ahead of toe (heel_x > toe_x)

        Ratio > 0.7 → WALKING_RIGHT
        Ratio < 0.3 → WALKING_LEFT
        """

        # Use left foot (more visible in right-facing walk)
        heel_x = df['x_left_heel'].dropna().values

        # Toe proxy: use foot index or ankle (since toe might not be reliable)
        if 'x_left_index' in df.columns:
            toe_x = df['x_left_index'].dropna().values
        elif 'x_left_foot_index' in df.columns:
            toe_x = df['x_left_foot_index'].dropna().values
        else:
            # Fallback: use ankle (should be behind heel)
            toe_x = df['x_left_ankle'].dropna().values

        if len(heel_x) == 0 or len(toe_x) == 0:
            return 'INSUFFICIENT_DATA'

        # Compare lengths and truncate to shorter
        min_len = min(len(heel_x), len(toe_x))
        heel_x = heel_x[:min_len]
        toe_x = toe_x[:min_len]

        # Compute ratio of frames where heel is ahead
        heel_ahead_ratio = (heel_x > toe_x).mean()

        if heel_ahead_ratio > 0.7:
            return 'WALKING_RIGHT'
        elif heel_ahead_ratio < 0.3:
            return 'WALKING_LEFT'
        else:
            return 'UNCERTAIN'

    def _verify_head_orientation(self, df: pd.DataFrame) -> str:
        """
        Layer 3: Verify head orientation from facial landmarks.

        In our lateral captures (right-facing):
        - Ears sit behind the nose along the MediaPipe X axis
        - nose_x < ear_x implies the face is turned to the right
        """

        nose_x = df['x_nose'].dropna().mean()
        if pd.isna(nose_x):
            return 'INSUFFICIENT_DATA'

        ear_diffs = []

        for side in ('left', 'right'):
            ear_col = f'x_{side}_ear'
            if ear_col in df.columns:
                ear_vals = df[ear_col].dropna()
                if not ear_vals.empty:
                    ear_x = ear_vals.mean()
                    if not pd.isna(ear_x):
                        ear_diffs.append(ear_x - nose_x)

        if not ear_diffs:
            return 'NO_EAR_DATA'

        avg_diff = float(np.mean(ear_diffs))

        if avg_diff > 0:
            return 'HEAD_FACING_RIGHT'
        elif avg_diff < 0:
            return 'HEAD_FACING_LEFT'
        else:
            return 'HEAD_UNCLEAR'

    def _verify_temporal_consistency(self, df: pd.DataFrame) -> str:
        """
        Layer 4: Verify bidirectional walking pattern.

        Expected for 7.5m walkway:
        - Early frames: walking one direction
        - Middle frames: turning
        - Late frames: walking opposite direction

        Check if early and late have opposite orientations.
        """

        total_frames = len(df)

        if total_frames < 90:  # Too short
            return 'INSUFFICIENT_DATA'

        # Split into thirds
        early_end = int(total_frames * 0.3)
        late_start = int(total_frames * 0.7)

        early_section = df.iloc[:early_end]
        late_section = df.iloc[late_start:]

        # Check orientation in each section
        early_orientation = self._verify_anatomical_consistency(early_section)
        late_orientation = self._verify_anatomical_consistency(late_section)

        # Expected: one RIGHT, one LEFT
        if (early_orientation == 'FACING_RIGHT' and late_orientation == 'FACING_LEFT'):
            return 'BIDIRECTIONAL_CONFIRMED'
        elif (early_orientation == 'FACING_LEFT' and late_orientation == 'FACING_RIGHT'):
            return 'BIDIRECTIONAL_REVERSED'  # Started left instead of right
        elif early_orientation == late_orientation:
            return 'UNIDIRECTIONAL_ONLY'  # Same direction throughout
        else:
            return 'INCONSISTENT'

    def _compute_confidence(self, results: Dict) -> float:
        """
        Compute confidence score (0-100) based on check results.

        Each layer contributes equally to confidence.
        "PASS" keywords: RIGHT, CONFIRMED, ALIGNED
        """

        pass_keywords = [
            'RIGHT', 'CONFIRMED', 'BIDIRECTIONAL',
            'FACING_RIGHT', 'WALKING_RIGHT', 'HEAD_FACING_RIGHT'
        ]

        total_checks = len(results)
        passed_checks = 0

        for check_result in results.values():
            if any(keyword in check_result for keyword in pass_keywords):
                passed_checks += 1

        confidence = (passed_checks / total_checks) * 100 if total_checks > 0 else 0

        return confidence

    def _determine_direction(self, results: Dict) -> str:
        """
        Determine overall direction from check results.

        Use majority voting from anatomical, foot, and head checks.
        """

        votes = []

        # Vote from anatomical
        if 'FACING_RIGHT' in results.get('anatomical', ''):
            votes.append('RIGHT')
        elif 'FACING_LEFT' in results.get('anatomical', ''):
            votes.append('LEFT')

        # Vote from foot movement
        if 'WALKING_RIGHT' in results.get('foot_movement', ''):
            votes.append('RIGHT')
        elif 'WALKING_LEFT' in results.get('foot_movement', ''):
            votes.append('LEFT')

        # Vote from head
        if 'HEAD_FACING_RIGHT' in results.get('head_orientation', ''):
            votes.append('RIGHT')
        elif 'HEAD_FACING_LEFT' in results.get('head_orientation', ''):
            votes.append('LEFT')

        if len(votes) == 0:
            return 'UNCERTAIN'

        # Count votes
        right_votes = votes.count('RIGHT')
        left_votes = votes.count('LEFT')

        if right_votes > left_votes:
            return 'RIGHT_DOMINANT'
        elif left_votes > right_votes:
            return 'LEFT_DOMINANT'
        else:
            return 'SPLIT_DECISION'


# Utility functions for testing
def test_validator_on_subject(subject_id: str, df_angles: pd.DataFrame):
    """Test validator on a single subject and print results."""

    validator = PoseOrientationValidator()
    result = validator.validate(df_angles)

    print(f"\n{'='*70}")
    print(f"Pose Orientation Validation: {subject_id}")
    print(f"{'='*70}")

    print(f"\nValidation Checks:")
    for layer, check_result in result['checks'].items():
        print(f"  {layer:25s}: {check_result}")

    print(f"\nOverall Assessment:")
    print(f"  Confidence: {result['confidence']:.0f}%")
    print(f"  Direction:  {result['direction']}")
    print(f"  Reliable:   {result['reliable']}")

    return result


if __name__ == "__main__":
    print("PoseOrientationValidator module loaded.")
    print("Import this module and use PoseOrientationValidator class.")
