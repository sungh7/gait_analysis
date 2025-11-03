#!/usr/bin/env python3
"""
Pure Pathological Gait Detection
=================================

Evaluates gait detection after excluding:
- Prosthetic gait (mechanical, not disease)
- Exercise gait (normal variant, not pathology)

Focus on pure pathological gaits:
- Stroke (hemiplegic)
- Cerebral Palsy
- Abnormal (general pathology)
- Antalgic (pain-related)
- Inebriated (toxin-related)

Author: Gait Analysis System
Version: 1.0
Date: 2025-10-30
"""

import json
import numpy as np
from typing import Dict, List
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')


# Classes to EXCLUDE
EXCLUDE_CLASSES = ['prosthetic', 'exercise']

# Pure pathological classes
PATHOLOGICAL_CLASSES = ['abnormal', 'stroke', 'cerebral palsy', 'antalgic', 'inebriated']


@dataclass
class EvaluationResult:
    """Evaluation results"""
    method: str
    accuracy: float
    sensitivity: float
    specificity: float
    tp: int
    tn: int
    fp: int
    fn: int
    n_normal: int
    n_pathological: int


class PurePathologicalDetector:
    """Detector for pure pathological gaits (excluding prosthetic/exercise)"""

    def __init__(self, patterns_file: str):
        """Load and filter patterns"""
        with open(patterns_file, 'r') as f:
            all_patterns = json.load(f)

        # Filter out prosthetic and exercise
        self.patterns = [
            p for p in all_patterns
            if p['gait_class'] not in EXCLUDE_CLASSES
            and p['normalized_pattern_left']
            and p['normalized_pattern_right']
        ]

        # Separate normal and pathological
        self.normal_patterns = [p for p in self.patterns if p['gait_class'] == 'normal']
        self.pathological_patterns = [p for p in self.patterns if p['gait_class'] in PATHOLOGICAL_CLASSES]

        print(f"Pure Pathological Detector Initialized:")
        print(f"  Normal samples: {len(self.normal_patterns)}")
        print(f"  Pathological samples: {len(self.pathological_patterns)}")
        print(f"  Excluded: prosthetic ({sum(1 for p in all_patterns if p['gait_class'] == 'prosthetic')}), " +
              f"exercise ({sum(1 for p in all_patterns if p['gait_class'] == 'exercise')})")

        # Build signatures
        self._build_signatures()

    def _build_signatures(self):
        """Build normal and pathological signatures"""

        # Normal signature
        normal_left = np.array([p['normalized_pattern_left'] for p in self.normal_patterns])
        normal_right = np.array([p['normalized_pattern_right'] for p in self.normal_patterns])
        normal_amp_left = [p['amplitude_left'] for p in self.normal_patterns]
        normal_amp_right = [p['amplitude_right'] for p in self.normal_patterns]
        normal_asym = [abs(p['amplitude_left'] - p['amplitude_right']) for p in self.normal_patterns]

        self.normal_sig = {
            'pattern_left': np.mean(normal_left, axis=0),
            'pattern_right': np.mean(normal_right, axis=0),
            'amplitude_left': np.mean(normal_amp_left),
            'amplitude_right': np.mean(normal_amp_right),
            'std_amplitude_left': np.std(normal_amp_left),
            'std_amplitude_right': np.std(normal_amp_right),
            'asymmetry': np.mean(normal_asym),
            'std_asymmetry': np.std(normal_asym)
        }

        # Pathological signature
        path_left = np.array([p['normalized_pattern_left'] for p in self.pathological_patterns])
        path_right = np.array([p['normalized_pattern_right'] for p in self.pathological_patterns])
        path_amp_left = [p['amplitude_left'] for p in self.pathological_patterns]
        path_amp_right = [p['amplitude_right'] for p in self.pathological_patterns]
        path_asym = [abs(p['amplitude_left'] - p['amplitude_right']) for p in self.pathological_patterns]

        self.pathological_sig = {
            'pattern_left': np.mean(path_left, axis=0),
            'pattern_right': np.mean(path_right, axis=0),
            'amplitude_left': np.mean(path_amp_left),
            'amplitude_right': np.mean(path_amp_right),
            'std_amplitude_left': np.std(path_amp_left),
            'std_amplitude_right': np.std(path_amp_right),
            'asymmetry': np.mean(path_asym),
            'std_asymmetry': np.std(path_asym)
        }

        print(f"\nSignature Analysis:")
        print(f"  Normal amplitude: L={self.normal_sig['amplitude_left']:.2f}±{self.normal_sig['std_amplitude_left']:.2f}, " +
              f"R={self.normal_sig['amplitude_right']:.2f}±{self.normal_sig['std_amplitude_right']:.2f}")
        print(f"  Pathological amplitude: L={self.pathological_sig['amplitude_left']:.2f}±{self.pathological_sig['std_amplitude_left']:.2f}, " +
              f"R={self.pathological_sig['amplitude_right']:.2f}±{self.pathological_sig['std_amplitude_right']:.2f}")
        print(f"  Normal asymmetry: {self.normal_sig['asymmetry']:.2f}±{self.normal_sig['std_asymmetry']:.2f}")
        print(f"  Pathological asymmetry: {self.pathological_sig['asymmetry']:.2f}±{self.pathological_sig['std_asymmetry']:.2f}")

        # DTW distance between templates
        dtw_dist, _ = fastdtw(
            self.normal_sig['pattern_left'].reshape(-1, 1),
            self.pathological_sig['pattern_left'].reshape(-1, 1),
            dist=euclidean
        )
        print(f"  DTW template distance: {dtw_dist:.2f}")

    def detect_dtw(self, pattern: dict) -> str:
        """Detect using DTW only"""
        pattern_left = np.array(pattern['normalized_pattern_left'])
        pattern_right = np.array(pattern['normalized_pattern_right'])

        # Distance to normal
        dist_normal_left, _ = fastdtw(pattern_left.reshape(-1, 1),
                                       self.normal_sig['pattern_left'].reshape(-1, 1),
                                       dist=euclidean)
        dist_normal_right, _ = fastdtw(pattern_right.reshape(-1, 1),
                                        self.normal_sig['pattern_right'].reshape(-1, 1),
                                        dist=euclidean)
        dist_normal = (dist_normal_left + dist_normal_right) / 2

        # Distance to pathological
        dist_path_left, _ = fastdtw(pattern_left.reshape(-1, 1),
                                     self.pathological_sig['pattern_left'].reshape(-1, 1),
                                     dist=euclidean)
        dist_path_right, _ = fastdtw(pattern_right.reshape(-1, 1),
                                      self.pathological_sig['pattern_right'].reshape(-1, 1),
                                      dist=euclidean)
        dist_path = (dist_path_left + dist_path_right) / 2

        return 'normal' if dist_normal < dist_path else 'pathological'

    def detect_scalar(self, pattern: dict) -> str:
        """Detect using scalar features only"""
        # Z-scores relative to normal
        amp_left_z = abs(pattern['amplitude_left'] - self.normal_sig['amplitude_left']) / (self.normal_sig['std_amplitude_left'] + 1e-6)
        amp_right_z = abs(pattern['amplitude_right'] - self.normal_sig['amplitude_right']) / (self.normal_sig['std_amplitude_right'] + 1e-6)
        asym = abs(pattern['amplitude_left'] - pattern['amplitude_right'])
        asym_z = abs(asym - self.normal_sig['asymmetry']) / (self.normal_sig['std_asymmetry'] + 1e-6)

        z_score_normal = (amp_left_z + amp_right_z + asym_z) / 3

        # Z-scores relative to pathological
        amp_left_z = abs(pattern['amplitude_left'] - self.pathological_sig['amplitude_left']) / (self.pathological_sig['std_amplitude_left'] + 1e-6)
        amp_right_z = abs(pattern['amplitude_right'] - self.pathological_sig['amplitude_right']) / (self.pathological_sig['std_amplitude_right'] + 1e-6)
        asym_z = abs(asym - self.pathological_sig['asymmetry']) / (self.pathological_sig['std_asymmetry'] + 1e-6)

        z_score_path = (amp_left_z + amp_right_z + asym_z) / 3

        return 'normal' if z_score_normal < z_score_path else 'pathological'

    def detect_hybrid(self, pattern: dict, dtw_weight: float = 0.5) -> str:
        """Detect using hybrid approach"""
        # DTW scores
        pattern_left = np.array(pattern['normalized_pattern_left'])
        pattern_right = np.array(pattern['normalized_pattern_right'])

        dist_normal_left, _ = fastdtw(pattern_left.reshape(-1, 1),
                                       self.normal_sig['pattern_left'].reshape(-1, 1),
                                       dist=euclidean)
        dist_normal_right, _ = fastdtw(pattern_right.reshape(-1, 1),
                                        self.normal_sig['pattern_right'].reshape(-1, 1),
                                        dist=euclidean)
        dtw_normal = (dist_normal_left + dist_normal_right) / 2 / 100.0  # Normalize

        dist_path_left, _ = fastdtw(pattern_left.reshape(-1, 1),
                                     self.pathological_sig['pattern_left'].reshape(-1, 1),
                                     dist=euclidean)
        dist_path_right, _ = fastdtw(pattern_right.reshape(-1, 1),
                                      self.pathological_sig['pattern_right'].reshape(-1, 1),
                                      dist=euclidean)
        dtw_path = (dist_path_left + dist_path_right) / 2 / 100.0  # Normalize

        # Scalar scores
        amp_left_z = abs(pattern['amplitude_left'] - self.normal_sig['amplitude_left']) / (self.normal_sig['std_amplitude_left'] + 1e-6)
        amp_right_z = abs(pattern['amplitude_right'] - self.normal_sig['amplitude_right']) / (self.normal_sig['std_amplitude_right'] + 1e-6)
        asym = abs(pattern['amplitude_left'] - pattern['amplitude_right'])
        asym_z = abs(asym - self.normal_sig['asymmetry']) / (self.normal_sig['std_asymmetry'] + 1e-6)
        scalar_normal = (amp_left_z + amp_right_z + asym_z) / 3

        amp_left_z = abs(pattern['amplitude_left'] - self.pathological_sig['amplitude_left']) / (self.pathological_sig['std_amplitude_left'] + 1e-6)
        amp_right_z = abs(pattern['amplitude_right'] - self.pathological_sig['amplitude_right']) / (self.pathological_sig['std_amplitude_right'] + 1e-6)
        asym_z = abs(asym - self.pathological_sig['asymmetry']) / (self.pathological_sig['std_asymmetry'] + 1e-6)
        scalar_path = (amp_left_z + amp_right_z + asym_z) / 3

        # Combined scores
        score_normal = dtw_weight * dtw_normal + (1 - dtw_weight) * scalar_normal
        score_path = dtw_weight * dtw_path + (1 - dtw_weight) * scalar_path

        return 'normal' if score_normal < score_path else 'pathological'

    def evaluate(self, method: str = 'scalar', dtw_weight: float = 0.5) -> EvaluationResult:
        """Evaluate detector"""
        true_labels = []
        pred_labels = []

        for pattern in self.patterns:
            # True label
            true_label = 'normal' if pattern['gait_class'] == 'normal' else 'pathological'
            true_labels.append(true_label)

            # Prediction
            if method == 'dtw':
                pred = self.detect_dtw(pattern)
            elif method == 'scalar':
                pred = self.detect_scalar(pattern)
            else:  # hybrid
                pred = self.detect_hybrid(pattern, dtw_weight)

            pred_labels.append(pred)

        # Compute metrics
        true_labels = np.array(true_labels)
        pred_labels = np.array(pred_labels)

        accuracy = np.mean(true_labels == pred_labels)

        # Sensitivity & Specificity
        path_mask = true_labels == 'pathological'
        normal_mask = true_labels == 'normal'

        sensitivity = np.mean(pred_labels[path_mask] == 'pathological') if path_mask.any() else 0
        specificity = np.mean(pred_labels[normal_mask] == 'normal') if normal_mask.any() else 0

        # Confusion matrix
        tp = np.sum((true_labels == 'pathological') & (pred_labels == 'pathological'))
        tn = np.sum((true_labels == 'normal') & (pred_labels == 'normal'))
        fp = np.sum((true_labels == 'normal') & (pred_labels == 'pathological'))
        fn = np.sum((true_labels == 'pathological') & (pred_labels == 'normal'))

        result = EvaluationResult(
            method=method,
            accuracy=accuracy,
            sensitivity=sensitivity,
            specificity=specificity,
            tp=int(tp),
            tn=int(tn),
            fp=int(fp),
            fn=int(fn),
            n_normal=int(normal_mask.sum()),
            n_pathological=int(path_mask.sum())
        )

        return result


def main():
    """Main evaluation pipeline"""

    print("="*80)
    print("PURE PATHOLOGICAL GAIT DETECTION")
    print("(Excluding Prosthetic and Exercise)")
    print("="*80)
    print()

    patterns_file = "gavd_real_patterns.json"

    # Create detector
    detector = PurePathologicalDetector(patterns_file)

    # Test all methods
    print("\n" + "="*80)
    print("EVALUATION RESULTS")
    print("="*80)

    methods = [
        ('dtw', None),
        ('scalar', None),
        ('hybrid', 0.3),
        ('hybrid', 0.5),
        ('hybrid', 0.7)
    ]

    results = []

    for method, weight in methods:
        result = detector.evaluate(method, weight if weight else 0.5)
        results.append(result)

        method_name = f"{method.upper()}" if method != 'hybrid' else f"HYBRID (DTW weight={weight})"

        print(f"\n{method_name}:")
        print(f"  Accuracy: {result.accuracy*100:.1f}%")
        print(f"  Sensitivity: {result.sensitivity*100:.1f}%")
        print(f"  Specificity: {result.specificity*100:.1f}%")
        print(f"  TP: {result.tp}, TN: {result.tn}, FP: {result.fp}, FN: {result.fn}")

    # Find best
    best_result = max(results, key=lambda r: r.accuracy)

    print("\n" + "="*80)
    print("BEST CONFIGURATION")
    print("="*80)
    print(f"  Method: {best_result.method.upper()}")
    print(f"  Accuracy: {best_result.accuracy*100:.1f}%")
    print(f"  Sensitivity: {best_result.sensitivity*100:.1f}%")
    print(f"  Specificity: {best_result.specificity*100:.1f}%")

    # Save results
    output = {
        'excluded_classes': EXCLUDE_CLASSES,
        'pathological_classes': PATHOLOGICAL_CLASSES,
        'n_normal': best_result.n_normal,
        'n_pathological': best_result.n_pathological,
        'best_method': best_result.method,
        'best_accuracy': best_result.accuracy,
        'best_sensitivity': best_result.sensitivity,
        'best_specificity': best_result.specificity,
        'all_results': [
            {
                'method': r.method,
                'accuracy': r.accuracy,
                'sensitivity': r.sensitivity,
                'specificity': r.specificity,
                'tp': r.tp,
                'tn': r.tn,
                'fp': r.fp,
                'fn': r.fn
            }
            for r in results
        ]
    }

    with open('pure_pathological_results.json', 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to: pure_pathological_results.json")
    print()


if __name__ == "__main__":
    main()
