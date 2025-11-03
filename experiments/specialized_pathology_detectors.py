#!/usr/bin/env python3
"""
Specialized Pathology Detectors
================================

Develops targeted detectors for specific gait pathologies using
hybrid approach: DTW pattern matching + scalar feature analysis.

Focus pathologies:
- Prosthetic gait (80% baseline accuracy)
- Hemiplegic gait (Stroke, 43% baseline)
- Cerebral Palsy

Author: Gait Analysis System
Version: 1.0
Date: 2025-10-30
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw
from sklearn.metrics import classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')


@dataclass
class PathologySignature:
    """Signature features for a specific pathology"""
    pathology: str

    # DTW template
    mean_pattern_left: np.ndarray
    mean_pattern_right: np.ndarray
    std_pattern_left: np.ndarray
    std_pattern_right: np.ndarray

    # Scalar features
    mean_amplitude_left: float
    mean_amplitude_right: float
    std_amplitude_left: float
    std_amplitude_right: float

    # Asymmetry features
    mean_lr_amplitude_diff: float  # |L - R|
    std_lr_amplitude_diff: float

    # Temporal features
    mean_peak_time_left: float
    mean_peak_time_right: float
    mean_peak_time_diff: float

    # Sample statistics
    n_samples: int


class SpecializedDetector:
    """Base class for pathology-specific detectors"""

    def __init__(self, pathology: str, patterns_file: str):
        """
        Initialize detector for specific pathology.

        Args:
            pathology: Target pathology (e.g., 'prosthetic', 'stroke')
            patterns_file: Real GAVD patterns file
        """
        self.pathology = pathology

        with open(patterns_file, 'r') as f:
            all_patterns = json.load(f)

        # Separate target pathology from others
        self.target_patterns = [
            p for p in all_patterns
            if p['gait_class'] == pathology
            and p['normalized_pattern_left']
            and p['normalized_pattern_right']
        ]

        self.normal_patterns = [
            p for p in all_patterns
            if p['gait_class'] == 'normal'
            and p['normalized_pattern_left']
            and p['normalized_pattern_right']
        ]

        print(f"\n{pathology.upper()} Detector Initialized:")
        print(f"  Target samples: {len(self.target_patterns)}")
        print(f"  Normal samples: {len(self.normal_patterns)}")

        # Build signatures
        self.target_signature = self._build_signature(self.target_patterns, pathology)
        self.normal_signature = self._build_signature(self.normal_patterns, 'normal')

    def _build_signature(self, patterns: List[dict], label: str) -> PathologySignature:
        """Build comprehensive signature from patterns"""

        # DTW templates
        left_patterns = np.array([p['normalized_pattern_left'] for p in patterns])
        right_patterns = np.array([p['normalized_pattern_right'] for p in patterns])

        # Scalar features
        amplitudes_left = [p['amplitude_left'] for p in patterns]
        amplitudes_right = [p['amplitude_right'] for p in patterns]
        lr_diffs = [abs(p['amplitude_left'] - p['amplitude_right']) for p in patterns]

        # Temporal features
        peak_times_left = [p['peak_time_left'] for p in patterns]
        peak_times_right = [p['peak_time_right'] for p in patterns]
        peak_time_diffs = [abs(p['peak_time_left'] - p['peak_time_right']) for p in patterns]

        signature = PathologySignature(
            pathology=label,
            mean_pattern_left=np.mean(left_patterns, axis=0),
            mean_pattern_right=np.mean(right_patterns, axis=0),
            std_pattern_left=np.std(left_patterns, axis=0),
            std_pattern_right=np.std(right_patterns, axis=0),
            mean_amplitude_left=np.mean(amplitudes_left),
            mean_amplitude_right=np.mean(amplitudes_right),
            std_amplitude_left=np.std(amplitudes_left),
            std_amplitude_right=np.std(amplitudes_right),
            mean_lr_amplitude_diff=np.mean(lr_diffs),
            std_lr_amplitude_diff=np.std(lr_diffs),
            mean_peak_time_left=np.mean(peak_times_left),
            mean_peak_time_right=np.mean(peak_times_right),
            mean_peak_time_diff=np.mean(peak_time_diffs),
            n_samples=len(patterns)
        )

        return signature

    def compute_dtw_score(self, pattern: dict, signature: PathologySignature) -> float:
        """Compute DTW similarity score (lower = more similar)"""
        pattern_left = np.array(pattern['normalized_pattern_left'])
        pattern_right = np.array(pattern['normalized_pattern_right'])

        dist_left, _ = fastdtw(
            pattern_left.reshape(-1, 1),
            signature.mean_pattern_left.reshape(-1, 1),
            dist=euclidean
        )

        dist_right, _ = fastdtw(
            pattern_right.reshape(-1, 1),
            signature.mean_pattern_right.reshape(-1, 1),
            dist=euclidean
        )

        return (dist_left + dist_right) / 2

    def compute_scalar_score(self, pattern: dict, signature: PathologySignature) -> float:
        """Compute scalar feature similarity score (Z-score based)"""

        # Amplitude Z-score
        amp_left_z = abs(pattern['amplitude_left'] - signature.mean_amplitude_left) / (signature.std_amplitude_left + 1e-6)
        amp_right_z = abs(pattern['amplitude_right'] - signature.mean_amplitude_right) / (signature.std_amplitude_right + 1e-6)

        # Asymmetry Z-score
        lr_diff = abs(pattern['amplitude_left'] - pattern['amplitude_right'])
        asym_z = abs(lr_diff - signature.mean_lr_amplitude_diff) / (signature.std_lr_amplitude_diff + 1e-6)

        # Combined Z-score (lower = more similar)
        combined_z = (amp_left_z + amp_right_z + asym_z) / 3

        return combined_z

    def compute_hybrid_score(self, pattern: dict, signature: PathologySignature,
                           dtw_weight: float = 0.5) -> float:
        """
        Compute hybrid score combining DTW and scalar features.

        Args:
            pattern: Gait pattern to score
            signature: Reference signature
            dtw_weight: Weight for DTW score (0-1)

        Returns:
            Hybrid score (lower = more similar)
        """
        dtw_score = self.compute_dtw_score(pattern, signature)
        scalar_score = self.compute_scalar_score(pattern, signature)

        # Normalize DTW score (typical range: 50-100)
        dtw_normalized = dtw_score / 100.0

        # Combine (both are "lower is better")
        hybrid = dtw_weight * dtw_normalized + (1 - dtw_weight) * scalar_score

        return hybrid

    def detect(self, pattern: dict, method: str = 'hybrid', dtw_weight: float = 0.5) -> Tuple[str, float]:
        """
        Detect if pattern matches target pathology.

        Args:
            pattern: Gait pattern to classify
            method: 'dtw', 'scalar', or 'hybrid'
            dtw_weight: Weight for DTW in hybrid mode

        Returns:
            (predicted_class, confidence_score)
        """
        if not pattern['normalized_pattern_left'] or not pattern['normalized_pattern_right']:
            return 'unknown', 0.0

        if method == 'dtw':
            target_score = self.compute_dtw_score(pattern, self.target_signature)
            normal_score = self.compute_dtw_score(pattern, self.normal_signature)
        elif method == 'scalar':
            target_score = self.compute_scalar_score(pattern, self.target_signature)
            normal_score = self.compute_scalar_score(pattern, self.normal_signature)
        else:  # hybrid
            target_score = self.compute_hybrid_score(pattern, self.target_signature, dtw_weight)
            normal_score = self.compute_hybrid_score(pattern, self.normal_signature, dtw_weight)

        # Lower score = more similar
        if target_score < normal_score:
            predicted = self.pathology
            confidence = (normal_score - target_score) / (normal_score + target_score + 1e-6)
        else:
            predicted = 'normal'
            confidence = (target_score - normal_score) / (normal_score + target_score + 1e-6)

        return predicted, confidence

    def evaluate(self, test_patterns: List[dict], method: str = 'hybrid',
                dtw_weight: float = 0.5) -> Dict:
        """
        Evaluate detector on test patterns.

        Args:
            test_patterns: Patterns to test
            method: Detection method
            dtw_weight: Weight for DTW in hybrid mode

        Returns:
            Evaluation metrics
        """
        true_labels = []
        pred_labels = []
        confidences = []

        for pattern in test_patterns:
            if not pattern['normalized_pattern_left'] or not pattern['normalized_pattern_right']:
                continue

            # True label (binary: target pathology vs normal)
            true_label = self.pathology if pattern['gait_class'] == self.pathology else 'normal'
            true_labels.append(true_label)

            # Prediction
            pred_label, confidence = self.detect(pattern, method, dtw_weight)
            pred_labels.append(pred_label)
            confidences.append(confidence)

        # Compute metrics
        true_labels = np.array(true_labels)
        pred_labels = np.array(pred_labels)

        accuracy = np.mean(true_labels == pred_labels)

        # Pathology-specific metrics
        target_mask = true_labels == self.pathology
        normal_mask = true_labels == 'normal'

        sensitivity = np.mean(pred_labels[target_mask] == self.pathology) if target_mask.any() else 0
        specificity = np.mean(pred_labels[normal_mask] == 'normal') if normal_mask.any() else 0

        # Confusion matrix
        tp = np.sum((true_labels == self.pathology) & (pred_labels == self.pathology))
        tn = np.sum((true_labels == 'normal') & (pred_labels == 'normal'))
        fp = np.sum((true_labels == 'normal') & (pred_labels == self.pathology))
        fn = np.sum((true_labels == self.pathology) & (pred_labels == 'normal'))

        results = {
            'pathology': self.pathology,
            'method': method,
            'dtw_weight': dtw_weight if method == 'hybrid' else None,
            'accuracy': accuracy,
            'sensitivity': sensitivity,
            'specificity': specificity,
            'tp': int(tp),
            'tn': int(tn),
            'fp': int(fp),
            'fn': int(fn),
            'n_samples': len(true_labels),
            'mean_confidence': np.mean(confidences)
        }

        return results

    def analyze_signature(self):
        """Analyze and print signature differences"""
        print(f"\n{'='*70}")
        print(f"{self.pathology.upper()} SIGNATURE ANALYSIS")
        print(f"{'='*70}")

        print(f"\nAmplitude (mean ± std):")
        print(f"  {self.pathology}:")
        print(f"    Left: {self.target_signature.mean_amplitude_left:.2f} ± {self.target_signature.std_amplitude_left:.2f}")
        print(f"    Right: {self.target_signature.mean_amplitude_right:.2f} ± {self.target_signature.std_amplitude_right:.2f}")
        print(f"  Normal:")
        print(f"    Left: {self.normal_signature.mean_amplitude_left:.2f} ± {self.normal_signature.std_amplitude_left:.2f}")
        print(f"    Right: {self.normal_signature.mean_amplitude_right:.2f} ± {self.normal_signature.std_amplitude_right:.2f}")

        amp_diff_left = abs(self.target_signature.mean_amplitude_left - self.normal_signature.mean_amplitude_left)
        amp_diff_right = abs(self.target_signature.mean_amplitude_right - self.normal_signature.mean_amplitude_right)
        print(f"  Difference: L={amp_diff_left:.2f}, R={amp_diff_right:.2f}")

        print(f"\nAsymmetry (|L-R|):")
        print(f"  {self.pathology}: {self.target_signature.mean_lr_amplitude_diff:.2f} ± {self.target_signature.std_lr_amplitude_diff:.2f}")
        print(f"  Normal: {self.normal_signature.mean_lr_amplitude_diff:.2f} ± {self.normal_signature.std_lr_amplitude_diff:.2f}")
        print(f"  Difference: {abs(self.target_signature.mean_lr_amplitude_diff - self.normal_signature.mean_lr_amplitude_diff):.2f}")

        print(f"\nPeak Timing:")
        print(f"  {self.pathology}: L={self.target_signature.mean_peak_time_left:.1f}, R={self.target_signature.mean_peak_time_right:.1f}")
        print(f"  Normal: L={self.normal_signature.mean_peak_time_left:.1f}, R={self.normal_signature.mean_peak_time_right:.1f}")

        # DTW template distance
        dtw_dist, _ = fastdtw(
            self.target_signature.mean_pattern_left.reshape(-1, 1),
            self.normal_signature.mean_pattern_left.reshape(-1, 1),
            dist=euclidean
        )
        print(f"\nDTW Template Distance (left): {dtw_dist:.2f}")


def optimize_detector(detector: SpecializedDetector, all_patterns: List[dict]) -> Dict:
    """
    Optimize detector by testing different methods and weights.

    Args:
        detector: Specialized detector
        all_patterns: All available patterns for testing

    Returns:
        Best configuration and results
    """
    print(f"\n{'='*70}")
    print(f"OPTIMIZING {detector.pathology.upper()} DETECTOR")
    print(f"{'='*70}")

    # Prepare test data
    test_patterns = [
        p for p in all_patterns
        if p['normalized_pattern_left'] and p['normalized_pattern_right']
    ]

    best_accuracy = 0
    best_config = None
    best_results = None

    # Test different methods
    methods = ['dtw', 'scalar', 'hybrid']
    dtw_weights = [0.3, 0.5, 0.7]

    for method in methods:
        if method == 'hybrid':
            for dtw_weight in dtw_weights:
                results = detector.evaluate(test_patterns, method, dtw_weight)

                print(f"\n{method.upper()} (DTW weight={dtw_weight}):")
                print(f"  Accuracy: {results['accuracy']*100:.1f}%")
                print(f"  Sensitivity: {results['sensitivity']*100:.1f}%")
                print(f"  Specificity: {results['specificity']*100:.1f}%")

                if results['accuracy'] > best_accuracy:
                    best_accuracy = results['accuracy']
                    best_config = {'method': method, 'dtw_weight': dtw_weight}
                    best_results = results
        else:
            results = detector.evaluate(test_patterns, method)

            print(f"\n{method.upper()}:")
            print(f"  Accuracy: {results['accuracy']*100:.1f}%")
            print(f"  Sensitivity: {results['sensitivity']*100:.1f}%")
            print(f"  Specificity: {results['specificity']*100:.1f}%")

            if results['accuracy'] > best_accuracy:
                best_accuracy = results['accuracy']
                best_config = {'method': method, 'dtw_weight': None}
                best_results = results

    print(f"\n{'='*70}")
    print(f"BEST CONFIGURATION:")
    print(f"  Method: {best_config['method']}")
    if best_config['dtw_weight']:
        print(f"  DTW Weight: {best_config['dtw_weight']}")
    print(f"  Accuracy: {best_accuracy*100:.1f}%")
    print(f"  Sensitivity: {best_results['sensitivity']*100:.1f}%")
    print(f"  Specificity: {best_results['specificity']*100:.1f}%")
    print(f"{'='*70}")

    return {
        'pathology': detector.pathology,
        'best_config': best_config,
        'best_results': best_results
    }


def main():
    """Main evaluation pipeline for specialized detectors"""

    print("="*80)
    print("SPECIALIZED PATHOLOGY DETECTORS")
    print("="*80)

    patterns_file = "gavd_real_patterns.json"

    # Load all patterns
    with open(patterns_file, 'r') as f:
        all_patterns = json.load(f)

    # Target pathologies with sufficient samples
    pathologies = ['prosthetic', 'stroke', 'cerebral palsy']

    all_results = {}

    for pathology in pathologies:
        print(f"\n\n{'#'*80}")
        print(f"# {pathology.upper()} DETECTOR")
        print(f"{'#'*80}")

        # Create detector
        detector = SpecializedDetector(pathology, patterns_file)

        # Analyze signature
        detector.analyze_signature()

        # Optimize
        optimization_results = optimize_detector(detector, all_patterns)
        all_results[pathology] = optimization_results

    # Save results
    output_file = "specialized_detectors_results.json"

    # Convert numpy types
    def convert_numpy(obj):
        if isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [convert_numpy(v) for v in obj]
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj

    with open(output_file, 'w') as f:
        json.dump(convert_numpy(all_results), f, indent=2)

    # Summary
    print(f"\n\n{'='*80}")
    print("FINAL SUMMARY - SPECIALIZED DETECTORS")
    print(f"{'='*80}")

    for pathology, results in all_results.items():
        best = results['best_results']
        print(f"\n{pathology.upper()}:")
        print(f"  Method: {results['best_config']['method']}")
        print(f"  Accuracy: {best['accuracy']*100:.1f}%")
        print(f"  Sensitivity: {best['sensitivity']*100:.1f}%")
        print(f"  Specificity: {best['specificity']*100:.1f}%")

    print(f"\nResults saved to: {output_file}")
    print()


if __name__ == "__main__":
    main()
