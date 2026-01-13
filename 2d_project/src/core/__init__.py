"""
Core algorithms for gait analysis.

This module contains the production-ready algorithms for:
- Signal processing (joint-specific filtering)
- Kinematic constraints enforcement
- Feature extraction from MediaPipe landmarks
- Landmark quality assessment
"""

from .improved_signal_processing import (
    ImprovedSignalProcessor,
    process_angles_dataframe,
)
from .kinematic_constraints import (
    KinematicConstraintEnforcer,
    JointLimits,
    JOINT_CONSTRAINTS,
)

__all__ = [
    "ImprovedSignalProcessor",
    "process_angles_dataframe",
    "KinematicConstraintEnforcer",
    "JointLimits",
    "JOINT_CONSTRAINTS",
]
