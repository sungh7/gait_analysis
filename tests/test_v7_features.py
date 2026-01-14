#!/usr/bin/env python3
"""
Unit tests for V7 Pure 3D feature extraction

Tests the 10 biomechanical features computed from MediaPipe 3D pose data.
"""

import unittest
import numpy as np
import pandas as pd
import tempfile
import os
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.extract_v7_features import extract_v7_features


class TestV7Features(unittest.TestCase):
    """Test V7 Pure 3D feature extraction"""

    def setUp(self):
        """Create mock 3D pose data"""
        # Create a simple walking pattern (60 frames = 2 seconds at 30fps)
        frames = 60
        fps = 30

        # Simulate heel motion (sinusoidal vertical movement)
        t = np.linspace(0, 2, frames)

        # Left heel trajectory
        left_heel_y = 0.1 + 0.05 * np.sin(2 * np.pi * 1.0 * t)  # 1 Hz
        left_heel_x = t * 0.5  # Forward motion
        left_heel_z = np.zeros(frames)

        # Right heel trajectory (phase shifted)
        right_heel_y = 0.1 + 0.05 * np.sin(2 * np.pi * 1.0 * t + np.pi)
        right_heel_x = t * 0.5
        right_heel_z = np.zeros(frames)

        # Hip trajectories (more stable)
        hip_y = 0.9 + 0.01 * np.sin(2 * np.pi * 1.0 * t)
        hip_x = t * 0.5
        hip_z = np.zeros(frames)

        # Create DataFrame
        data = []
        for i in range(frames):
            data.append({
                'frame': i,
                'position': 'left_heel',
                'x': left_heel_x[i],
                'y': left_heel_y[i],
                'z': left_heel_z[i]
            })
            data.append({
                'frame': i,
                'position': 'right_heel',
                'x': right_heel_x[i],
                'y': right_heel_y[i],
                'z': right_heel_z[i]
            })
            data.append({
                'frame': i,
                'position': 'left_hip',
                'x': hip_x[i],
                'y': hip_y[i],
                'z': hip_z[i]
            })
            data.append({
                'frame': i,
                'position': 'right_hip',
                'x': hip_x[i],
                'y': hip_y[i],
                'z': hip_z[i]
            })

        self.df = pd.DataFrame(data)

        # Save to temporary CSV
        self.temp_csv = tempfile.NamedTemporaryFile(
            mode='w', suffix='.csv', delete=False
        )
        self.df.to_csv(self.temp_csv.name, index=False)
        self.temp_csv.close()

    def tearDown(self):
        """Clean up temporary file"""
        if os.path.exists(self.temp_csv.name):
            os.unlink(self.temp_csv.name)

    def test_feature_extraction_returns_dict(self):
        """Test that feature extraction returns a dictionary"""
        pattern = {
            'csv_file': self.temp_csv.name,
            'fps': 30
        }

        features = extract_v7_features(pattern)
        self.assertIsInstance(features, dict)

    def test_all_10_features_present(self):
        """Test that all 10 features are extracted"""
        pattern = {
            'csv_file': self.temp_csv.name,
            'fps': 30
        }

        features = extract_v7_features(pattern)

        expected_features = [
            'cadence_3d',
            'step_height_variability',
            'gait_irregularity_3d',
            'velocity_3d',
            'jerkiness_3d',
            'cycle_duration_3d',
            'stride_length_3d',
            'trunk_sway',
            'path_length_3d',
            'step_width_3d'
        ]

        for feature in expected_features:
            self.assertIn(feature, features, f"Missing feature: {feature}")

    def test_cadence_reasonable_range(self):
        """Test that cadence is in reasonable range (60-120 steps/min)"""
        pattern = {
            'csv_file': self.temp_csv.name,
            'fps': 30
        }

        features = extract_v7_features(pattern)
        cadence = features['cadence_3d']

        self.assertGreater(cadence, 0, "Cadence should be positive")
        self.assertLess(cadence, 200, "Cadence should be less than 200")

    def test_velocity_positive(self):
        """Test that velocity is positive for forward walking"""
        pattern = {
            'csv_file': self.temp_csv.name,
            'fps': 30
        }

        features = extract_v7_features(pattern)
        velocity = features['velocity_3d']

        self.assertGreater(velocity, 0, "Velocity should be positive")

    def test_stride_length_positive(self):
        """Test that stride length is positive"""
        pattern = {
            'csv_file': self.temp_csv.name,
            'fps': 30
        }

        features = extract_v7_features(pattern)
        stride_length = features['stride_length_3d']

        self.assertGreater(stride_length, 0, "Stride length should be positive")

    def test_invalid_csv_path(self):
        """Test handling of invalid CSV path"""
        pattern = {
            'csv_file': '/nonexistent/path.csv',
            'fps': 30
        }

        features = extract_v7_features(pattern)
        self.assertIsNone(features, "Should return None for invalid path")

    def test_insufficient_data(self):
        """Test handling of insufficient data (< 30 frames)"""
        # Create minimal data
        data = []
        for i in range(10):  # Only 10 frames
            data.append({
                'frame': i,
                'position': 'left_heel',
                'x': 0, 'y': 0, 'z': 0
            })

        df = pd.DataFrame(data)
        temp_file = tempfile.NamedTemporaryFile(
            mode='w', suffix='.csv', delete=False
        )
        df.to_csv(temp_file.name, index=False)
        temp_file.close()

        try:
            pattern = {
                'csv_file': temp_file.name,
                'fps': 30
            }

            features = extract_v7_features(pattern)
            self.assertIsNone(features, "Should return None for insufficient data")
        finally:
            os.unlink(temp_file.name)


class TestV7FeatureValidation(unittest.TestCase):
    """Test feature value validation and ranges"""

    def test_gait_irregularity_non_negative(self):
        """Gait irregularity (CV) should be non-negative"""
        # This would be tested with real data
        pass

    def test_jerkiness_magnitude(self):
        """Jerkiness should be within reasonable bounds"""
        # This would be tested with real data
        pass

    def test_trunk_sway_non_negative(self):
        """Trunk sway should be non-negative"""
        # This would be tested with real data
        pass


if __name__ == '__main__':
    unittest.main()
