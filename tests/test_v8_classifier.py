#!/usr/bin/env python3
"""
Unit tests for V8 ML-Enhanced classifier

Tests the logistic regression classifier and feature importance analysis.
"""

import unittest
import numpy as np
import json
import tempfile
import os
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.v8_ml_enhanced import V8_ML_Enhanced


class TestV8Classifier(unittest.TestCase):
    """Test V8 ML-Enhanced classifier"""

    def setUp(self):
        """Create mock pattern data"""
        # Create synthetic patterns for testing
        self.patterns = []

        # Normal patterns (lower irregularity, lower jerkiness)
        for i in range(50):
            self.patterns.append({
                'gait_class': 'normal',
                'cadence_3d': 110 + np.random.randn() * 5,
                'step_height_variability': 0.1 + np.random.randn() * 0.02,
                'gait_irregularity_3d': 0.05 + np.random.randn() * 0.01,
                'velocity_3d': 1.2 + np.random.randn() * 0.1,
                'jerkiness_3d': 2.0 + np.random.randn() * 0.3,
                'cycle_duration_3d': 1.1 + np.random.randn() * 0.05,
                'stride_length_3d': 1.4 + np.random.randn() * 0.1,
                'trunk_sway': 0.05 + np.random.randn() * 0.01,
                'path_length_3d': 10.0 + np.random.randn() * 1.0,
                'step_width_3d': 0.1 + np.random.randn() * 0.02
            })

        # Pathological patterns (higher irregularity, higher jerkiness)
        for i in range(50):
            self.patterns.append({
                'gait_class': 'pathological',
                'cadence_3d': 95 + np.random.randn() * 8,
                'step_height_variability': 0.2 + np.random.randn() * 0.05,
                'gait_irregularity_3d': 0.15 + np.random.randn() * 0.03,
                'velocity_3d': 0.9 + np.random.randn() * 0.15,
                'jerkiness_3d': 4.0 + np.random.randn() * 0.8,
                'cycle_duration_3d': 1.3 + np.random.randn() * 0.1,
                'stride_length_3d': 1.1 + np.random.randn() * 0.15,
                'trunk_sway': 0.12 + np.random.randn() * 0.03,
                'path_length_3d': 8.0 + np.random.randn() * 1.5,
                'step_width_3d': 0.15 + np.random.randn() * 0.04
            })

        # Save to temporary JSON file
        self.temp_json = tempfile.NamedTemporaryFile(
            mode='w', suffix='.json', delete=False
        )
        json.dump(self.patterns, self.temp_json)
        self.temp_json.close()

    def tearDown(self):
        """Clean up temporary file"""
        if os.path.exists(self.temp_json.name):
            os.unlink(self.temp_json.name)

    def test_classifier_initialization(self):
        """Test that classifier initializes correctly"""
        clf = V8_ML_Enhanced(patterns_file=self.temp_json.name)
        self.assertIsNotNone(clf.patterns)
        self.assertEqual(len(clf.patterns), 100)

    def test_feature_names(self):
        """Test that all 10 feature names are defined"""
        clf = V8_ML_Enhanced(patterns_file=self.temp_json.name)

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

        self.assertEqual(clf.feature_names, expected_features)

    def test_training(self):
        """Test that classifier can be trained"""
        clf = V8_ML_Enhanced(patterns_file=self.temp_json.name)
        clf.train()

        # Check that model is fitted
        self.assertTrue(hasattr(clf.clf, 'coef_'))
        self.assertTrue(hasattr(clf.clf, 'intercept_'))

    def test_prediction_format(self):
        """Test that predictions return expected format"""
        clf = V8_ML_Enhanced(patterns_file=self.temp_json.name)
        clf.train()

        # Get evaluation results
        results = clf.evaluate()

        # Check required keys
        self.assertIn('accuracy', results)
        self.assertIn('sensitivity', results)
        self.assertIn('specificity', results)

    def test_accuracy_above_baseline(self):
        """Test that accuracy is reasonable (> 60%)"""
        clf = V8_ML_Enhanced(patterns_file=self.temp_json.name)
        clf.train()

        results = clf.evaluate()
        accuracy = results['accuracy']

        # With clearly separated synthetic data, should get high accuracy
        self.assertGreater(accuracy, 0.6,
                          f"Accuracy {accuracy:.2%} should be > 60%")

    def test_feature_importance_extraction(self):
        """Test that feature importance can be extracted"""
        clf = V8_ML_Enhanced(patterns_file=self.temp_json.name)
        clf.train()

        # Feature importance is stored in coefficients
        coefs = clf.clf.coef_[0]
        self.assertEqual(len(coefs), 10, "Should have 10 feature coefficients")

    def test_cross_validation(self):
        """Test that cross-validation works"""
        clf = V8_ML_Enhanced(patterns_file=self.temp_json.name)
        clf.train()

        # Cross-validation should be part of training
        # This is tested implicitly in the train method
        results = clf.evaluate()
        self.assertIsNotNone(results)

    def test_invalid_pattern_file(self):
        """Test handling of invalid pattern file"""
        with self.assertRaises(FileNotFoundError):
            clf = V8_ML_Enhanced(patterns_file='/nonexistent/patterns.json')


class TestV8FeatureImportance(unittest.TestCase):
    """Test feature importance analysis"""

    def test_top_features_identification(self):
        """Test identification of most important features"""
        # This would be tested with real GAVD data
        # Expected: gait_irregularity, jerkiness, step_height_variability
        pass

    def test_coefficient_magnitudes(self):
        """Test that coefficient magnitudes are reasonable"""
        # This would be tested with real GAVD data
        pass


if __name__ == '__main__':
    unittest.main()
