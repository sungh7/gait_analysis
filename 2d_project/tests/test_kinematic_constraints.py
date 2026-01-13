"""
Unit tests for kinematic constraints module.
"""

import pytest
import numpy as np
import pandas as pd
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.kinematic_constraints import (
    KinematicConstraintEnforcer,
    JOINT_CONSTRAINTS,
    JointLimits,
)

# Helper functions for testing (since the module doesn't export standalone functions)
def apply_hard_constraints(angles, joint):
    """Helper to apply hard constraints via the enforcer."""
    import pandas as pd
    enforcer = KinematicConstraintEnforcer(fps=30)
    df = pd.DataFrame({f'{joint}_angle': angles})
    result = enforcer.enforce_joint_angle_constraints(df)
    return result[f'{joint}_angle'].values


class TestJointLimits:
    """Tests for joint limit definitions."""

    def test_knee_limits_defined(self):
        """Test knee constraints are properly defined."""
        assert 'knee' in JOINT_CONSTRAINTS
        knee = JOINT_CONSTRAINTS['knee']
        assert knee.min_angle == 0
        assert knee.max_angle == 140
        assert knee.max_velocity > 0
        assert knee.max_acceleration > 0

    def test_hip_limits_defined(self):
        """Test hip constraints are properly defined."""
        assert 'hip' in JOINT_CONSTRAINTS
        hip = JOINT_CONSTRAINTS['hip']
        assert hip.min_angle == -30
        assert hip.max_angle == 120

    def test_ankle_limits_geometric_range(self):
        """Test ankle uses geometric range (0-180), not anatomical."""
        assert 'ankle' in JOINT_CONSTRAINTS
        ankle = JOINT_CONSTRAINTS['ankle']
        # CRITICAL: Ankle should use geometric range to preserve ROM
        assert ankle.min_angle == 0
        assert ankle.max_angle == 180, "Ankle should use 0-180 geometric range"


class TestKinematicConstraintEnforcer:
    """Tests for KinematicConstraintEnforcer class."""

    @pytest.fixture
    def enforcer(self):
        """Create an enforcer instance for testing."""
        return KinematicConstraintEnforcer(fps=30, subject_height=1.70)

    def test_initialization(self, enforcer):
        """Test enforcer initialization."""
        assert enforcer.fps == 30
        assert enforcer.subject_height == 1.70
        assert enforcer.dt == 1.0 / 30

    def test_expected_segment_lengths(self, enforcer):
        """Test that expected segment lengths are calculated."""
        # Based on anthropometric ratios (0.22-0.28 for thigh/shank)
        assert 0.35 < enforcer.expected_thigh_length < 0.50
        assert 0.35 < enforcer.expected_shank_length < 0.50
        assert 0.20 < enforcer.expected_foot_length < 0.30

    def test_enforce_knee_angle_constraints(self, enforcer):
        """Test that knee angles are clipped to valid range."""
        # Create DataFrame with out-of-range knee angles
        angles_df = pd.DataFrame({
            'right_knee_angle': [-10, 50, 160, 90, 30],  # -10 and 160 are invalid
        })

        constrained = enforcer.enforce_joint_angle_constraints(angles_df)

        # Check all values are within 0-140 range
        assert constrained['right_knee_angle'].min() >= 0
        assert constrained['right_knee_angle'].max() <= 140

    def test_enforce_hip_angle_constraints(self, enforcer):
        """Test that hip angles are clipped to valid range."""
        angles_df = pd.DataFrame({
            'left_hip_angle': [-50, -20, 50, 130, 80],  # -50 and 130 are invalid
        })

        constrained = enforcer.enforce_joint_angle_constraints(angles_df)

        # Check all values are within -30 to 120 range
        assert constrained['left_hip_angle'].min() >= -30
        assert constrained['left_hip_angle'].max() <= 120

    def test_ankle_geometric_range_preserved(self, enforcer):
        """Test that ankle angles are NOT clipped to anatomical range."""
        # CRITICAL TEST: Ankle should preserve values in 0-180 range
        angles_df = pd.DataFrame({
            'right_ankle_angle': [90, 100, 110, 120, 130],  # All valid in geometric range
        })

        constrained = enforcer.enforce_joint_angle_constraints(angles_df)

        # These should be preserved (they're valid in 0-180 range)
        # Small changes allowed due to soft constraints
        np.testing.assert_array_almost_equal(
            angles_df['right_ankle_angle'].values,
            constrained['right_ankle_angle'].values,
            decimal=0  # Allow rounding
        )

    def test_multiple_joints_in_dataframe(self, enforcer):
        """Test processing DataFrame with multiple joint columns."""
        angles_df = pd.DataFrame({
            'right_knee_angle': [30, 45, 60, 75, 90],
            'left_knee_angle': [35, 50, 65, 80, 95],
            'right_hip_angle': [10, 20, 30, 40, 50],
            'left_hip_angle': [15, 25, 35, 45, 55],
            'right_ankle_angle': [85, 90, 95, 100, 105],
            'left_ankle_angle': [88, 93, 98, 103, 108],
        })

        constrained = enforcer.enforce_joint_angle_constraints(angles_df)

        # Check all columns are present
        assert set(constrained.columns) == set(angles_df.columns)
        # Check same number of rows
        assert len(constrained) == len(angles_df)

    def test_nan_handling(self, enforcer):
        """Test that NaN values are handled properly."""
        angles_df = pd.DataFrame({
            'right_knee_angle': [30, np.nan, 60, np.nan, 90],
        })

        constrained = enforcer.enforce_joint_angle_constraints(angles_df)

        # NaNs should either be preserved or filled
        assert len(constrained) == len(angles_df)


class TestConstraintEdgeCases:
    """Edge case tests for kinematic constraints."""

    @pytest.fixture
    def enforcer(self):
        return KinematicConstraintEnforcer(fps=30, subject_height=1.70)

    def test_empty_dataframe(self, enforcer):
        """Test handling of empty DataFrame."""
        angles_df = pd.DataFrame()

        constrained = enforcer.enforce_joint_angle_constraints(angles_df)

        assert len(constrained) == 0

    def test_unrecognized_column_names(self, enforcer):
        """Test that unrecognized columns are passed through unchanged."""
        angles_df = pd.DataFrame({
            'unknown_angle': [10, 20, 30, 40, 50],
            'random_data': [1, 2, 3, 4, 5],
        })

        constrained = enforcer.enforce_joint_angle_constraints(angles_df)

        # Unrecognized columns should be unchanged
        np.testing.assert_array_equal(
            angles_df['unknown_angle'].values,
            constrained['unknown_angle'].values
        )

    def test_different_subject_heights(self):
        """Test enforcer works with different subject heights."""
        for height in [1.50, 1.70, 1.90, 2.10]:
            enforcer = KinematicConstraintEnforcer(fps=30, subject_height=height)

            # Segment lengths should scale with height
            assert enforcer.expected_thigh_length > 0
            assert enforcer.expected_shank_length > 0

    def test_different_fps(self):
        """Test enforcer works with different frame rates."""
        for fps in [24, 30, 60, 120]:
            enforcer = KinematicConstraintEnforcer(fps=fps, subject_height=1.70)
            assert enforcer.fps == fps
            assert enforcer.dt == 1.0 / fps
