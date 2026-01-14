"""Configuration and constants for gait analysis."""

from typing import Tuple

# Pose Detection Settings
MIN_DETECTION_CONFIDENCE = 0.8
MIN_TRACKING_CONFIDENCE = 0.8

# Data Processing Settings
SAVGOL_WINDOW_LENGTH = 17
SAVGOL_POLY_ORDER = 3

# Filtering Thresholds
FRAME_DIFF_THRESHOLD = 5
SECTION_MIN_SIZE = 10
WIDTH_DIFF_QUANTILE = 0.2

# Gait Cycle Settings
STRETCH_SIZE = 100  # Normalized gait cycle length

# Plot Settings
PLOT_FIGSIZE: Tuple[int, int] = (15, 5)
XLIM: Tuple[float, float] = (-1, 1)
YLIM: Tuple[float, float] = (-1, 1)

# File Paths (can be overridden)
DEFAULT_OUTPUT_DIR = "./mp_posed"
DEFAULT_DATA_DIR = "./data"

# Body Position Sets
EXCLUDED_POSITIONS = {"INNER", "OUTER"}
SPECIAL_POSITIONS = {"NOSE", "LEFT", "RIGHT"}

# Axis mappings
FRONTAL_AXES = ("x", "y")
SAGITTAL_AXES = ("z", "y")
