#!/usr/bin/env python3
"""
Pathological Gait Detector - STAGE 1-C: Baseline Implementation
=================================================================

Z-score based anomaly detection using scalar gait features.

Input:
    - Patient gait parameters (from V5 pipeline)
    - Normal gait reference (from build_normal_reference.py)

Output:
    - Classification: Normal (0) or Pathological (1)
    - Confidence score
    - Detailed feature deviations
    - Clinical interpretation

Author: Gait Analysis System
Version: 1.0
Date: 2025-10-27
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum


class Severity(Enum):
    """Deviation severity levels based on Z-scores"""
    NORMAL = "Normal"           # |Z| < 1.0
    MILD = "Mild"              # 1.0 ≤ |Z| < 2.0
    MODERATE = "Moderate"       # 2.0 ≤ |Z| < 3.0
    SEVERE = "Severe"          # |Z| ≥ 3.0


@dataclass
class FeatureDeviation:
    """Single feature deviation analysis"""
    feature_name: str
    value: float
    reference_mean: float
    reference_std: float
    z_score: float
    severity: Severity
    clinical_interpretation: str


@dataclass
class DetectionResult:
    """Complete pathological gait detection result"""
    is_pathological: bool
    confidence: float
    overall_severity: Severity
    max_z_score: float
    mean_z_score: float
    deviations: List[FeatureDeviation]
    summary: str


class PathologicalGaitDetector:
    """
    Baseline pathological gait detector using Z-score analysis.

    Detection Strategy:
        1. Calculate Z-scores for all features
        2. Apply severity thresholds
        3. Decision rules:
           - Any feature |Z| ≥ 3.0 → Pathological (High confidence)
           - Mean |Z| ≥ 2.0 → Pathological (Medium confidence)
           - Max |Z| ≥ 2.0 + asymmetry → Pathological (Medium confidence)
           - Otherwise → Normal
    """

    def __init__(self, reference_path: str = "normal_gait_reference.json"):
        """
        Initialize detector with normal gait reference.

        Args:
            reference_path: Path to normal gait reference JSON file
        """
        self.reference = self._load_reference(reference_path)

        # Detection thresholds (optimized for sensitivity/specificity balance)
        self.SEVERE_THRESHOLD = 3.0      # Any feature > 3 SD → Pathological
        self.MODERATE_THRESHOLD = 2.0    # Mean > 2 SD → Pathological
        self.MILD_THRESHOLD = 1.0        # For severity classification

        # Asymmetry thresholds (L/R ratio)
        self.ASYMMETRY_NORMAL = (0.95, 1.05)
        self.ASYMMETRY_MODERATE = (0.85, 1.15)

    def _load_reference(self, path: str) -> dict:
        """Load normal gait reference from JSON file"""
        with open(path, 'r') as f:
            return json.load(f)

    def _calculate_z_score(self, value: float, mean: float, std: float) -> float:
        """Calculate Z-score with minimum std to prevent extreme values"""
        if std == 0:
            return 0.0
        # Use minimum std to prevent extremely high Z-scores from tiny variations
        # especially for ratio features which have very small natural std
        min_std = 0.05  # 5% minimum variability
        effective_std = max(std, min_std)
        return (value - mean) / effective_std

    def _classify_severity(self, z_score: float) -> Severity:
        """Classify deviation severity based on Z-score"""
        abs_z = abs(z_score)
        if abs_z < self.MILD_THRESHOLD:
            return Severity.NORMAL
        elif abs_z < self.MODERATE_THRESHOLD:
            return Severity.MILD
        elif abs_z < self.SEVERE_THRESHOLD:
            return Severity.MODERATE
        else:
            return Severity.SEVERE

    def _get_clinical_interpretation(self, feature: str, value: float,
                                     z_score: float, severity: Severity) -> str:
        """Generate clinical interpretation for feature deviation"""
        direction = "increased" if z_score > 0 else "decreased"

        interpretations = {
            'step_length': f"Step length {direction} ({value:.1f} cm, {abs(z_score):.1f} SD from normal)",
            'cadence': f"Cadence {direction} ({value:.1f} steps/min, {abs(z_score):.1f} SD from normal)",
            'stance': f"Stance phase {direction} ({value:.1f}%, {abs(z_score):.1f} SD from normal)",
            'velocity': f"Walking velocity {direction} ({value:.1f} cm/s, {abs(z_score):.1f} SD from normal)",
            'asymmetry': f"Significant asymmetry detected (ratio: {value:.3f}, {abs(z_score):.1f} SD from normal)"
        }

        # Get base interpretation
        for key, interpretation in interpretations.items():
            if key in feature.lower():
                return interpretation

        return f"{feature} {direction} ({abs(z_score):.1f} SD from normal)"

    def analyze_feature(self, feature_path: List[str], value: float) -> Optional[FeatureDeviation]:
        """
        Analyze a single feature and calculate deviation.

        Args:
            feature_path: Path to feature in reference dict (e.g., ['step_length_cm', 'left'])
            value: Patient's feature value

        Returns:
            FeatureDeviation object or None if feature not found
        """
        # Navigate to reference statistics
        ref = self.reference
        for key in feature_path:
            if key not in ref:
                return None
            ref = ref[key]

        # Extract statistics
        mean = ref.get('mean')
        std = ref.get('std')

        if mean is None or std is None:
            return None

        # Calculate Z-score and severity
        z_score = self._calculate_z_score(value, mean, std)
        severity = self._classify_severity(z_score)

        # Generate clinical interpretation
        feature_name = '_'.join(feature_path)
        interpretation = self._get_clinical_interpretation(
            feature_name, value, z_score, severity
        )

        return FeatureDeviation(
            feature_name=feature_name,
            value=value,
            reference_mean=mean,
            reference_std=std,
            z_score=z_score,
            severity=severity,
            clinical_interpretation=interpretation
        )

    def detect(self, patient_data: Dict) -> DetectionResult:
        """
        Detect pathological gait from patient data.

        Args:
            patient_data: Dictionary with gait parameters
                Expected keys:
                - step_length_left, step_length_right (cm)
                - cadence_left, cadence_right (steps/min)
                - stance_left, stance_right (%)
                - velocity_left, velocity_right (cm/s)

        Returns:
            DetectionResult with classification and detailed analysis
        """
        deviations = []

        # Define features to analyze
        features_to_check = [
            (['step_length_cm', 'left'], 'step_length_left'),
            (['step_length_cm', 'right'], 'step_length_right'),
            (['cadence_steps_min', 'left'], 'cadence_left'),
            (['cadence_steps_min', 'right'], 'cadence_right'),
            (['stance_percent', 'left'], 'stance_left'),
            (['stance_percent', 'right'], 'stance_right'),
            (['velocity_cm_s', 'left'], 'velocity_left'),
            (['velocity_cm_s', 'right'], 'velocity_right'),
        ]

        # Analyze each feature
        for ref_path, data_key in features_to_check:
            if data_key in patient_data:
                deviation = self.analyze_feature(ref_path, patient_data[data_key])
                if deviation:
                    deviations.append(deviation)

        # Calculate asymmetry indices
        asymmetry_deviations = self._analyze_asymmetry(patient_data)
        deviations.extend(asymmetry_deviations)

        # Calculate aggregate scores
        z_scores = [abs(d.z_score) for d in deviations]
        max_z = max(z_scores) if z_scores else 0.0
        mean_z = np.mean(z_scores) if z_scores else 0.0

        # Decision logic
        is_pathological, confidence, severity = self._make_decision(
            max_z, mean_z, deviations
        )

        # Generate summary
        summary = self._generate_summary(is_pathological, confidence,
                                         max_z, mean_z, deviations)

        return DetectionResult(
            is_pathological=is_pathological,
            confidence=confidence,
            overall_severity=severity,
            max_z_score=max_z,
            mean_z_score=mean_z,
            deviations=deviations,
            summary=summary
        )

    def _analyze_asymmetry(self, patient_data: Dict) -> List[FeatureDeviation]:
        """Analyze left/right asymmetry"""
        asymmetry_deviations = []

        # Define asymmetry calculations
        asymmetries = [
            ('step_length_left', 'step_length_right', 'step_length_lr_ratio'),
            ('cadence_left', 'cadence_right', 'cadence_lr_ratio'),
            ('stance_left', 'stance_right', 'stance_lr_ratio'),
        ]

        for left_key, right_key, ref_key in asymmetries:
            if left_key in patient_data and right_key in patient_data:
                left_val = patient_data[left_key]
                right_val = patient_data[right_key]

                if right_val > 0:
                    ratio = left_val / right_val

                    # Analyze against reference
                    deviation = self.analyze_feature(
                        ['asymmetry_index', ref_key],
                        ratio
                    )

                    if deviation:
                        asymmetry_deviations.append(deviation)

        return asymmetry_deviations

    def _make_decision(self, max_z: float, mean_z: float,
                      deviations: List[FeatureDeviation]) -> Tuple[bool, float, Severity]:
        """
        Make classification decision based on Z-scores.

        Returns:
            (is_pathological, confidence, overall_severity)
        """
        # Rule 1: Any severe deviation → Pathological (high confidence)
        severe_deviations = [d for d in deviations if d.severity == Severity.SEVERE]
        if severe_deviations:
            confidence = min(0.95, 0.7 + (max_z - 3.0) * 0.05)
            return True, confidence, Severity.SEVERE

        # Rule 2: Mean Z-score high → Pathological (medium confidence)
        if mean_z >= self.MODERATE_THRESHOLD:
            confidence = min(0.85, 0.5 + (mean_z - 2.0) * 0.1)
            return True, confidence, Severity.MODERATE

        # Rule 3: Multiple moderate deviations → Pathological
        moderate_deviations = [d for d in deviations if d.severity == Severity.MODERATE]
        if len(moderate_deviations) >= 3:
            confidence = 0.6 + len(moderate_deviations) * 0.05
            return True, confidence, Severity.MODERATE

        # Rule 4: High max Z + asymmetry → Pathological
        asymmetry_deviations = [d for d in deviations if 'asymmetry' in d.feature_name]
        moderate_asymmetry = [d for d in asymmetry_deviations
                             if d.severity in [Severity.MODERATE, Severity.SEVERE]]

        if max_z >= 2.0 and moderate_asymmetry:
            confidence = 0.65
            return True, confidence, Severity.MODERATE

        # Otherwise → Normal
        if mean_z < 1.0:
            confidence = 0.9
        else:
            confidence = 0.7 - (mean_z - 1.0) * 0.2

        return False, confidence, Severity.NORMAL

    def _generate_summary(self, is_pathological: bool, confidence: float,
                         max_z: float, mean_z: float,
                         deviations: List[FeatureDeviation]) -> str:
        """Generate human-readable summary"""
        classification = "Pathological" if is_pathological else "Normal"

        summary = f"Classification: {classification} (Confidence: {confidence:.1%})\n"
        summary += f"Max Z-score: {max_z:.2f}, Mean Z-score: {mean_z:.2f}\n\n"

        # Group deviations by severity
        severe = [d for d in deviations if d.severity == Severity.SEVERE]
        moderate = [d for d in deviations if d.severity == Severity.MODERATE]
        mild = [d for d in deviations if d.severity == Severity.MILD]

        if severe:
            summary += "SEVERE DEVIATIONS:\n"
            for d in severe:
                summary += f"  - {d.clinical_interpretation}\n"
            summary += "\n"

        if moderate:
            summary += "MODERATE DEVIATIONS:\n"
            for d in moderate:
                summary += f"  - {d.clinical_interpretation}\n"
            summary += "\n"

        if mild and is_pathological:
            summary += "MILD DEVIATIONS:\n"
            for d in mild[:3]:  # Show top 3 only
                summary += f"  - {d.clinical_interpretation}\n"

        return summary.strip()

    def batch_detect(self, patients_data: List[Dict]) -> List[DetectionResult]:
        """
        Detect pathological gait for multiple patients.

        Args:
            patients_data: List of patient data dictionaries

        Returns:
            List of DetectionResult objects
        """
        return [self.detect(patient) for patient in patients_data]

    def evaluate_performance(self, test_data: List[Tuple[Dict, int]]) -> Dict:
        """
        Evaluate detector performance on labeled test data.

        Args:
            test_data: List of (patient_data, true_label) tuples
                       true_label: 0 = Normal, 1 = Pathological

        Returns:
            Dictionary with performance metrics
        """
        predictions = []
        true_labels = []
        confidences = []

        for patient_data, true_label in test_data:
            result = self.detect(patient_data)
            predictions.append(1 if result.is_pathological else 0)
            true_labels.append(true_label)
            confidences.append(result.confidence)

        # Calculate metrics
        predictions = np.array(predictions)
        true_labels = np.array(true_labels)

        # Confusion matrix
        tp = np.sum((predictions == 1) & (true_labels == 1))
        tn = np.sum((predictions == 0) & (true_labels == 0))
        fp = np.sum((predictions == 1) & (true_labels == 0))
        fn = np.sum((predictions == 0) & (true_labels == 1))

        # Metrics
        accuracy = (tp + tn) / len(true_labels) if len(true_labels) > 0 else 0
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0  # Recall for pathological
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0  # Recall for normal
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        f1_score = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0

        return {
            'accuracy': accuracy,
            'sensitivity': sensitivity,  # True Positive Rate
            'specificity': specificity,  # True Negative Rate
            'precision': precision,
            'f1_score': f1_score,
            'confusion_matrix': {
                'tp': int(tp),
                'tn': int(tn),
                'fp': int(fp),
                'fn': int(fn)
            },
            'n_samples': len(true_labels),
            'mean_confidence': float(np.mean(confidences))
        }


def main():
    """Example usage"""

    # Initialize detector
    detector = PathologicalGaitDetector("normal_gait_reference.json")

    # Example 1: Normal gait
    print("=" * 80)
    print("EXAMPLE 1: Normal Gait")
    print("=" * 80)

    normal_patient = {
        'step_length_left': 65.0,
        'step_length_right': 64.5,
        'cadence_left': 112.0,
        'cadence_right': 113.0,
        'stance_left': 61.5,
        'stance_right': 61.8,
        'velocity_left': 123.0,
        'velocity_right': 122.5,
    }

    result = detector.detect(normal_patient)
    print(result.summary)
    print()

    # Example 2: Pathological gait (reduced step length, slow cadence)
    print("=" * 80)
    print("EXAMPLE 2: Pathological Gait (Parkinson's-like)")
    print("=" * 80)

    pathological_patient = {
        'step_length_left': 45.0,  # Very short
        'step_length_right': 43.0,  # Very short
        'cadence_left': 95.0,  # Slow
        'cadence_right': 94.0,  # Slow
        'stance_left': 63.0,
        'stance_right': 62.5,
        'velocity_left': 85.0,  # Slow
        'velocity_right': 83.0,  # Slow
    }

    result = detector.detect(pathological_patient)
    print(result.summary)
    print()

    # Example 3: Asymmetric gait (stroke-like)
    print("=" * 80)
    print("EXAMPLE 3: Asymmetric Gait (Stroke-like)")
    print("=" * 80)

    asymmetric_patient = {
        'step_length_left': 55.0,  # Affected side
        'step_length_right': 68.0,  # Healthy side
        'cadence_left': 105.0,
        'cadence_right': 115.0,
        'stance_left': 64.0,  # Longer stance on affected side
        'stance_right': 60.0,
        'velocity_left': 100.0,
        'velocity_right': 120.0,
    }

    result = detector.detect(asymmetric_patient)
    print(result.summary)
    print()


if __name__ == "__main__":
    main()
