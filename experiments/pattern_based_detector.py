#!/usr/bin/env python3
"""
Pathological Gait Detector - STAGE 2: Pattern-Based Detection
==============================================================

Combines scalar features (STAGE 1) with time-series pattern analysis
for improved pathological gait detection.

Features:
1. Scalar features (step length, cadence, stance, velocity, asymmetry)
2. Time-series patterns (heel height, joint angles)
3. DTW-based template matching
4. Multi-class pathology classification

Author: Gait Analysis System
Version: 2.0
Date: 2025-10-27
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
from scipy.signal import savgol_filter
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean

# Import STAGE 1 detector
from pathological_gait_detector import (
    PathologicalGaitDetector,
    Severity,
    FeatureDeviation,
    DetectionResult
)


class PathologyType(Enum):
    """Specific pathology types"""
    NORMAL = "Normal"
    PARKINSONS = "Parkinson's Disease"
    STROKE = "Stroke (Hemiplegia)"
    CEREBRAL_PALSY = "Cerebral Palsy"
    MYOPATHIC = "Myopathic Gait"
    ANTALGIC = "Antalgic Gait"
    GENERAL_ABNORMAL = "General Abnormal"
    UNKNOWN = "Unknown Pathology"


@dataclass
class PatternFeatures:
    """Time-series pattern features"""
    # Heel height pattern
    heel_height_pattern: np.ndarray  # 101 points (0-100% gait cycle)
    heel_height_amplitude: float     # Peak-to-peak amplitude
    heel_height_peak_time: float     # % of cycle at peak

    # Variability features
    step_time_cv: float              # Coefficient of variation
    stride_regularity: float         # Pattern consistency

    # DTW distances to reference templates
    dtw_normal: float
    dtw_parkinsons: float
    dtw_stroke: float

    # Asymmetry in patterns
    lr_pattern_correlation: float    # Left-right pattern similarity


@dataclass
class EnhancedDetectionResult:
    """Enhanced detection result with pathology classification"""
    # STAGE 1 results
    is_pathological: bool
    confidence: float
    overall_severity: Severity
    scalar_deviations: List[FeatureDeviation]

    # STAGE 2 additions
    pathology_type: PathologyType
    pathology_confidence: float
    pattern_features: Optional[PatternFeatures]

    # Combined reasoning
    summary: str
    clinical_interpretation: str


class PatternBasedDetector:
    """
    Enhanced pathological gait detector combining scalar and pattern features.

    Workflow:
    1. STAGE 1: Scalar feature analysis (baseline detector)
    2. STAGE 2: Pattern feature extraction and analysis
    3. Combine both for final classification
    """

    def __init__(self, reference_path: str = "normal_gait_reference.json"):
        """Initialize enhanced detector"""
        # STAGE 1: Scalar detector
        self.scalar_detector = PathologicalGaitDetector(reference_path)

        # STAGE 2: Pattern templates (to be loaded/created)
        self.pattern_templates = {}
        self._initialize_pattern_templates()

        # Classification thresholds
        self.PATTERN_SIMILARITY_THRESHOLD = 0.7
        self.MULTICLASS_CONFIDENCE_THRESHOLD = 0.6

    def _initialize_pattern_templates(self):
        """
        Initialize reference pattern templates for each pathology.

        In production, these would be loaded from a database or created
        from training data. For now, we create synthetic templates based
        on known biomechanical characteristics.
        """
        # Template: 101-point normalized heel height pattern (0-100% gait cycle)

        # Normal gait: smooth sinusoidal pattern
        cycle = np.linspace(0, 2*np.pi, 101)
        self.pattern_templates['normal'] = {
            'heel_height': np.sin(cycle),  # Smooth swing
            'amplitude': 1.0,
            'peak_time': 50.0,  # Mid-swing
            'regularity': 0.95
        }

        # Parkinson's: reduced amplitude, shuffling
        self.pattern_templates['parkinsons'] = {
            'heel_height': 0.4 * np.sin(cycle),  # Low amplitude
            'amplitude': 0.4,
            'peak_time': 45.0,  # Earlier peak
            'regularity': 0.85  # More variable
        }

        # Stroke: asymmetric pattern
        stroke_pattern = np.sin(cycle)
        stroke_pattern[50:] *= 0.6  # Reduced swing on affected side
        self.pattern_templates['stroke'] = {
            'heel_height': stroke_pattern,
            'amplitude': 0.8,
            'peak_time': 55.0,  # Delayed
            'regularity': 0.75  # Very variable
        }

        # Cerebral Palsy: irregular, spastic
        cp_pattern = np.sin(cycle) + 0.3 * np.sin(3 * cycle)  # High frequency tremor
        self.pattern_templates['cerebral_palsy'] = {
            'heel_height': cp_pattern,
            'amplitude': 0.7,
            'peak_time': 52.0,
            'regularity': 0.70  # High variability
        }

    def extract_pattern_features(self, time_series_data: Dict) -> Optional[PatternFeatures]:
        """
        Extract pattern features from time-series gait data.

        Args:
            time_series_data: Dictionary with:
                - 'heel_height': array of heel heights over multiple cycles
                - 'timestamps': array of timestamps
                - 'side': 'left' or 'right'

        Returns:
            PatternFeatures object or None if insufficient data
        """
        if 'heel_height' not in time_series_data:
            return None

        heel_height = time_series_data['heel_height']

        if len(heel_height) < 50:  # Minimum data requirement
            return None

        # Smooth signal
        window_size = min(11, max(3, len(heel_height) // 10))
        if window_size % 2 == 0:
            window_size += 1

        heel_smooth = savgol_filter(heel_height, window_size, 2) if len(heel_height) >= window_size else heel_height

        # Normalize to 101 points (standard gait cycle)
        pattern_normalized = np.interp(
            np.linspace(0, len(heel_smooth) - 1, 101),
            np.arange(len(heel_smooth)),
            heel_smooth
        )

        # Z-score normalize
        pattern_mean = np.mean(pattern_normalized)
        pattern_std = np.std(pattern_normalized)
        if pattern_std > 0:
            pattern_normalized = (pattern_normalized - pattern_mean) / pattern_std

        # Calculate features
        amplitude = np.max(pattern_normalized) - np.min(pattern_normalized)
        peak_idx = np.argmax(pattern_normalized)
        peak_time = (peak_idx / 100.0) * 100.0  # Percentage

        # Variability (simulate from pattern characteristics)
        step_time_cv = amplitude * 5.0  # Rough estimate: higher amplitude -> more regular
        stride_regularity = 1.0 - (np.std(pattern_normalized) / (amplitude + 1e-6))

        # Calculate DTW distances to templates
        dtw_normal, _ = fastdtw(
            pattern_normalized.reshape(-1, 1),
            self.pattern_templates['normal']['heel_height'].reshape(-1, 1),
            dist=euclidean
        )

        dtw_parkinsons, _ = fastdtw(
            pattern_normalized.reshape(-1, 1),
            self.pattern_templates['parkinsons']['heel_height'].reshape(-1, 1),
            dist=euclidean
        )

        dtw_stroke, _ = fastdtw(
            pattern_normalized.reshape(-1, 1),
            self.pattern_templates['stroke']['heel_height'].reshape(-1, 1),
            dist=euclidean
        )

        # Left-right correlation (simulated)
        lr_correlation = 0.9 if amplitude > 0.5 else 0.7

        return PatternFeatures(
            heel_height_pattern=pattern_normalized,
            heel_height_amplitude=amplitude,
            heel_height_peak_time=peak_time,
            step_time_cv=step_time_cv,
            stride_regularity=stride_regularity,
            dtw_normal=dtw_normal,
            dtw_parkinsons=dtw_parkinsons,
            dtw_stroke=dtw_stroke,
            lr_pattern_correlation=lr_correlation
        )

    def classify_pathology_from_patterns(self, pattern_features: PatternFeatures,
                                        scalar_result: DetectionResult) -> Tuple[PathologyType, float]:
        """
        Classify specific pathology type using pattern features.

        Args:
            pattern_features: Extracted pattern features
            scalar_result: Result from STAGE 1 scalar detector

        Returns:
            (pathology_type, confidence)
        """
        # If scalar detector says normal, trust it
        if not scalar_result.is_pathological:
            return PathologyType.NORMAL, scalar_result.confidence

        # Calculate similarity scores to each pathology template
        dtw_distances = {
            'normal': pattern_features.dtw_normal,
            'parkinsons': pattern_features.dtw_parkinsons,
            'stroke': pattern_features.dtw_stroke,
        }

        # Find closest match
        min_pathology = min(dtw_distances, key=dtw_distances.get)
        min_distance = dtw_distances[min_pathology]

        # Calculate confidence (inverse of distance, normalized)
        total_distance = sum(dtw_distances.values())
        if total_distance > 0:
            confidence = 1.0 - (min_distance / total_distance)
        else:
            confidence = 0.5

        # Additional rules based on scalar features
        # Check for characteristic patterns

        # Parkinson's: short steps + slow cadence + reduced amplitude
        short_steps = any('step_length' in d.feature_name and d.z_score < -3.0
                         for d in scalar_result.deviations)
        slow_cadence = any('cadence' in d.feature_name and d.z_score < -2.0
                          for d in scalar_result.deviations)

        if short_steps and slow_cadence:
            return PathologyType.PARKINSONS, max(confidence, 0.80)

        # Stroke: strong asymmetry in scalar features
        asymmetry_deviations = [d for d in scalar_result.deviations
                               if 'asymmetry' in d.feature_name or 'ratio' in d.feature_name]
        severe_asymmetry = any(abs(d.z_score) > 3.0 for d in asymmetry_deviations)

        if severe_asymmetry:
            return PathologyType.STROKE, max(confidence, 0.8)

        # If pattern match is strong, use it
        if confidence > self.MULTICLASS_CONFIDENCE_THRESHOLD:
            pathology_map = {
                'normal': PathologyType.NORMAL,
                'parkinsons': PathologyType.PARKINSONS,
                'stroke': PathologyType.STROKE,
            }
            return pathology_map.get(min_pathology, PathologyType.UNKNOWN), confidence

        # Otherwise, general abnormal
        return PathologyType.GENERAL_ABNORMAL, scalar_result.confidence

    def detect_enhanced(self, patient_data: Dict,
                       time_series_data: Optional[Dict] = None) -> EnhancedDetectionResult:
        """
        Enhanced detection combining scalar and pattern features.

        Args:
            patient_data: Scalar gait parameters (same as STAGE 1)
            time_series_data: Optional time-series data for pattern analysis

        Returns:
            EnhancedDetectionResult with pathology classification
        """
        # STAGE 1: Scalar feature analysis
        scalar_result = self.scalar_detector.detect(patient_data)

        # STAGE 2: Pattern analysis (if available)
        pattern_features = None
        pathology_type = PathologyType.UNKNOWN
        pathology_confidence = scalar_result.confidence

        if time_series_data is not None:
            pattern_features = self.extract_pattern_features(time_series_data)

            if pattern_features is not None:
                pathology_type, pathology_confidence = self.classify_pathology_from_patterns(
                    pattern_features, scalar_result
                )
        else:
            # No pattern data: classify based on scalar features only
            if not scalar_result.is_pathological:
                pathology_type = PathologyType.NORMAL
            else:
                # Use scalar features to infer pathology type
                pathology_type = self._infer_pathology_from_scalars(scalar_result)

        # Generate clinical interpretation
        clinical_interpretation = self._generate_clinical_interpretation(
            scalar_result, pathology_type, pattern_features
        )

        # Combined summary
        summary = f"{scalar_result.summary}\n\n"
        summary += f"PATHOLOGY CLASSIFICATION:\n"
        summary += f"  Type: {pathology_type.value}\n"
        summary += f"  Confidence: {pathology_confidence:.1%}\n"

        if pattern_features is not None:
            summary += f"\nPATTERN ANALYSIS:\n"
            summary += f"  Heel amplitude: {pattern_features.heel_height_amplitude:.2f}\n"
            summary += f"  Peak timing: {pattern_features.heel_height_peak_time:.1f}% of cycle\n"
            summary += f"  Stride regularity: {pattern_features.stride_regularity:.1%}\n"

        return EnhancedDetectionResult(
            is_pathological=scalar_result.is_pathological,
            confidence=scalar_result.confidence,
            overall_severity=scalar_result.overall_severity,
            scalar_deviations=scalar_result.deviations,
            pathology_type=pathology_type,
            pathology_confidence=pathology_confidence,
            pattern_features=pattern_features,
            summary=summary,
            clinical_interpretation=clinical_interpretation
        )

    def _infer_pathology_from_scalars(self, scalar_result: DetectionResult) -> PathologyType:
        """Infer pathology type from scalar features only (fallback)"""
        if not scalar_result.is_pathological:
            return PathologyType.NORMAL

        # Check for asymmetry (stroke)
        asymmetry_severe = any(
            'asymmetry' in d.feature_name and abs(d.z_score) > 3.0
            for d in scalar_result.deviations
        )
        if asymmetry_severe:
            return PathologyType.STROKE

        # Check for short steps + slow cadence (Parkinson's)
        short_steps = any(
            'step_length' in d.feature_name and d.z_score < -3.0
            for d in scalar_result.deviations
        )
        slow_cadence = any(
            'cadence' in d.feature_name and d.z_score < -2.0
            for d in scalar_result.deviations
        )
        if short_steps and slow_cadence:
            return PathologyType.PARKINSONS

        return PathologyType.GENERAL_ABNORMAL

    def _generate_clinical_interpretation(self, scalar_result: DetectionResult,
                                         pathology_type: PathologyType,
                                         pattern_features: Optional[PatternFeatures]) -> str:
        """Generate detailed clinical interpretation"""
        interpretation = f"CLINICAL ASSESSMENT\n"
        interpretation += f"=" * 60 + "\n\n"

        interpretation += f"Classification: {pathology_type.value}\n"
        interpretation += f"Severity: {scalar_result.overall_severity.value}\n"
        interpretation += f"Confidence: {scalar_result.confidence:.1%}\n\n"

        # Pathology-specific interpretation
        if pathology_type == PathologyType.NORMAL:
            interpretation += "Assessment: All gait parameters within normal limits.\n"
            interpretation += "Recommendation: No intervention required.\n"

        elif pathology_type == PathologyType.PARKINSONS:
            interpretation += "Assessment: Gait pattern consistent with Parkinsonian features.\n"
            interpretation += "Characteristic findings:\n"
            interpretation += "  - Reduced step length (shuffling gait)\n"
            interpretation += "  - Reduced walking velocity (bradykinesia)\n"
            if pattern_features:
                interpretation += f"  - Reduced swing amplitude ({pattern_features.heel_height_amplitude:.2f})\n"
            interpretation += "Recommendation: Neurology consultation, fall risk assessment.\n"

        elif pathology_type == PathologyType.STROKE:
            interpretation += "Assessment: Gait pattern consistent with hemiplegic gait.\n"
            interpretation += "Characteristic findings:\n"
            interpretation += "  - Significant left-right asymmetry\n"
            interpretation += "  - Affected side: reduced step length, prolonged stance\n"
            interpretation += "Recommendation: Physical therapy, assistive device evaluation.\n"

        elif pathology_type == PathologyType.CEREBRAL_PALSY:
            interpretation += "Assessment: Gait pattern consistent with spastic gait.\n"
            interpretation += "Characteristic findings:\n"
            interpretation += "  - Reduced step length\n"
            interpretation += "  - Increased stance phase (spasticity)\n"
            if pattern_features:
                interpretation += f"  - Irregular pattern (regularity: {pattern_features.stride_regularity:.1%})\n"
            interpretation += "Recommendation: Comprehensive gait analysis, orthotic assessment.\n"

        else:
            interpretation += "Assessment: Abnormal gait pattern detected.\n"
            interpretation += "Recommendation: Comprehensive clinical evaluation.\n"

        return interpretation


def main():
    """Example usage of enhanced pattern-based detector"""

    print("=" * 80)
    print("PATHOLOGICAL GAIT DETECTOR - STAGE 2: Pattern-Based Detection")
    print("=" * 80)
    print()

    # Initialize detector
    detector = PatternBasedDetector("normal_gait_reference.json")

    # Example 1: Normal gait
    print("-" * 80)
    print("EXAMPLE 1: Normal Gait")
    print("-" * 80)

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

    # Simulate normal pattern
    normal_pattern = {
        'heel_height': np.sin(np.linspace(0, 4*np.pi, 200)) + np.random.normal(0, 0.05, 200),
        'timestamps': np.linspace(0, 4.0, 200),
        'side': 'left'
    }

    result = detector.detect_enhanced(normal_patient, normal_pattern)
    print(result.summary)
    print()
    print(result.clinical_interpretation)
    print()

    # Example 2: Parkinson's gait
    print("-" * 80)
    print("EXAMPLE 2: Parkinson's Disease")
    print("-" * 80)

    parkinsons_patient = {
        'step_length_left': 45.0,
        'step_length_right': 43.0,
        'cadence_left': 95.0,
        'cadence_right': 94.0,
        'stance_left': 63.0,
        'stance_right': 62.5,
        'velocity_left': 85.0,
        'velocity_right': 83.0,
    }

    # Simulate shuffling pattern (reduced amplitude)
    parkinsons_pattern = {
        'heel_height': 0.3 * np.sin(np.linspace(0, 4*np.pi, 200)) + np.random.normal(0, 0.1, 200),
        'timestamps': np.linspace(0, 5.0, 200),  # Slower
        'side': 'left'
    }

    result = detector.detect_enhanced(parkinsons_patient, parkinsons_pattern)
    print(result.summary)
    print()
    print(result.clinical_interpretation)
    print()

    # Example 3: Stroke (hemiplegic gait)
    print("-" * 80)
    print("EXAMPLE 3: Stroke (Hemiplegia)")
    print("-" * 80)

    stroke_patient = {
        'step_length_left': 55.0,
        'step_length_right': 68.0,
        'cadence_left': 105.0,
        'cadence_right': 115.0,
        'stance_left': 64.0,
        'stance_right': 60.0,
        'velocity_left': 100.0,
        'velocity_right': 120.0,
    }

    # Simulate asymmetric pattern
    stroke_pattern_left = np.sin(np.linspace(0, 4*np.pi, 200))
    stroke_pattern_left[::2] *= 0.6  # Affected side reduced
    stroke_pattern = {
        'heel_height': stroke_pattern_left + np.random.normal(0, 0.08, 200),
        'timestamps': np.linspace(0, 4.5, 200),
        'side': 'left'
    }

    result = detector.detect_enhanced(stroke_patient, stroke_pattern)
    print(result.summary)
    print()
    print(result.clinical_interpretation)
    print()


if __name__ == "__main__":
    main()
