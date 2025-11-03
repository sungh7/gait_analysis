#!/usr/bin/env python3
"""
Baseline V3 Robust: Current Best Method
========================================

This is the BEST performing method after discovering the fake 76.6%.

Performance (on gavd_real_patterns_fixed.json):
  - Accuracy: 62.0%
  - Sensitivity: 69.8%
  - Specificity: 55.4%
  - Threshold: 0.75

Features (6 total):
  1. Cadence (step frequency)
  2. Variability (peak height consistency)
  3. Irregularity (stride interval consistency)
  4. Velocity (vertical heel speed)
  5. Jerkiness (acceleration magnitude)
  6. Cycle duration (time per stride)

Method: Robust Z-score using Median/MAD instead of Mean/Std

This is the starting point for all future improvements.

Author: Gait Analysis System
Date: 2025-10-30
Version: 3.0 Robust (BASELINE)
"""

import json
import numpy as np
from scipy.signal import find_peaks
from scipy.stats import median_abs_deviation
from typing import Dict, List, Tuple
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')


@dataclass
class GaitFeatures:
    """6 enhanced gait features"""
    # Core features (v2)
    cadence: float
    variability: float
    irregularity: float
    # Enhanced features (v3)
    velocity: float
    jerkiness: float
    cycle_duration: float


class BaselineV3Robust:
    """
    Best baseline method: 6 features + Robust MAD-Z

    Performance: 62.0% accuracy (69.8% sensitivity, 55.4% specificity)
    """

    def __init__(self, patterns_file: str = "gavd_real_patterns_fixed.json"):
        """Initialize with clean patterns (no NaN)"""

        print("="*80)
        print("Baseline V3 Robust - Best Current Method")
        print("="*80)
        print()

        with open(patterns_file, 'r') as f:
            all_patterns = json.load(f)

        # Filter valid patterns
        self.patterns = [p for p in all_patterns
                        if p['heel_height_left'] and p['heel_height_right']
                        and len(p['heel_height_left']) > 10
                        and p['gait_class'] not in ['prosthetic', 'exercise']]

        print(f"Loaded {len(self.patterns)} clean patterns")

        # Verify no NaN
        n_nan = sum(1 for p in self.patterns
                   if np.any(np.isnan(p['heel_height_left']))
                   or np.any(np.isnan(p['heel_height_right'])))

        if n_nan > 0:
            raise ValueError(f"Found {n_nan} patterns with NaN! Use gavd_real_patterns_fixed.json")

        # Extract features for all patterns
        print("Extracting 6 enhanced features...")
        for p in self.patterns:
            p['features'] = self._extract_features(p)

        # Build robust baseline
        self._build_robust_baseline()

    def _extract_features(self, pattern: dict) -> GaitFeatures:
        """Extract all 6 features"""

        heel_left = np.array(pattern['heel_height_left'])
        heel_right = np.array(pattern['heel_height_right'])
        n_frames = len(heel_left)
        fps = pattern.get('fps', 30)
        duration = n_frames / fps
        dt = 1.0 / fps

        # 1. CADENCE
        peaks_left, _ = find_peaks(heel_left, height=np.mean(heel_left), distance=5)
        peaks_right, _ = find_peaks(heel_right, height=np.mean(heel_right), distance=5)
        n_steps = len(peaks_left) + len(peaks_right)
        cadence = (n_steps / duration) * 60 if duration > 0 else 0

        # 2. VARIABILITY
        if len(peaks_left) > 1:
            var_left = np.std(heel_left[peaks_left]) / (np.mean(heel_left[peaks_left]) + 1e-6)
        else:
            var_left = 0

        if len(peaks_right) > 1:
            var_right = np.std(heel_right[peaks_right]) / (np.mean(heel_right[peaks_right]) + 1e-6)
        else:
            var_right = 0

        variability = (var_left + var_right) / 2

        # 3. IRREGULARITY
        if len(peaks_left) > 2:
            intervals = np.diff(peaks_left)
            irreg_left = np.std(intervals) / (np.mean(intervals) + 1e-6)
        else:
            irreg_left = 0

        if len(peaks_right) > 2:
            intervals = np.diff(peaks_right)
            irreg_right = np.std(intervals) / (np.mean(intervals) + 1e-6)
        else:
            irreg_right = 0

        irregularity = (irreg_left + irreg_right) / 2

        # 4. VELOCITY
        vel_left = np.diff(heel_left) / dt
        vel_right = np.diff(heel_right) / dt
        velocity = (np.mean(np.abs(vel_left)) + np.mean(np.abs(vel_right))) / 2

        # 5. JERKINESS
        accel_left = np.diff(vel_left) / dt
        accel_right = np.diff(vel_right) / dt
        jerkiness = (np.mean(np.abs(accel_left)) + np.mean(np.abs(accel_right))) / 2

        # 6. CYCLE DURATION
        if len(peaks_left) > 1:
            intervals = np.diff(peaks_left) / fps
            cycle_duration = np.mean(intervals)
        else:
            cycle_duration = 0

        return GaitFeatures(
            cadence=cadence,
            variability=variability,
            irregularity=irregularity,
            velocity=velocity,
            jerkiness=jerkiness,
            cycle_duration=cycle_duration
        )

    def _build_robust_baseline(self):
        """Build baseline using robust statistics (Median/MAD)"""

        normal_patterns = [p for p in self.patterns if p['gait_class'] == 'normal']

        print(f"\nBuilding robust baseline from {len(normal_patterns)} normal patterns...")

        # Extract feature arrays
        feature_names = ['cadence', 'variability', 'irregularity',
                        'velocity', 'jerkiness', 'cycle_duration']

        self.baseline = {}

        for feat_name in feature_names:
            vals = np.array([getattr(p['features'], feat_name) for p in normal_patterns])

            # Robust statistics
            median_val = np.median(vals)
            mad_val = median_abs_deviation(vals, scale='normal')

            self.baseline[f'{feat_name}_median'] = median_val
            self.baseline[f'{feat_name}_mad'] = mad_val

        self.baseline['n_samples'] = len(normal_patterns)

        print(f"\nRobust Baseline Statistics (Median ± MAD):")
        print(f"  Cadence: {self.baseline['cadence_median']:.1f} ± {self.baseline['cadence_mad']:.1f} steps/min")
        print(f"  Variability: {self.baseline['variability_median']:.3f} ± {self.baseline['variability_mad']:.3f}")
        print(f"  Irregularity: {self.baseline['irregularity_median']:.3f} ± {self.baseline['irregularity_mad']:.3f}")
        print(f"  Velocity: {self.baseline['velocity_median']:.3f} ± {self.baseline['velocity_mad']:.3f}")
        print(f"  Jerkiness: {self.baseline['jerkiness_median']:.3f} ± {self.baseline['jerkiness_mad']:.3f}")
        print(f"  Cycle Duration: {self.baseline['cycle_duration_median']:.3f} ± {self.baseline['cycle_duration_mad']:.3f}")
        print()

    def compute_robust_z_score(self, features: GaitFeatures) -> float:
        """Compute composite robust Z-score (MAD-Z)"""

        feature_names = ['cadence', 'variability', 'irregularity',
                        'velocity', 'jerkiness', 'cycle_duration']

        z_scores = []

        for feat_name in feature_names:
            feat_val = getattr(features, feat_name)
            median_val = self.baseline[f'{feat_name}_median']
            mad_val = self.baseline[f'{feat_name}_mad']

            # Robust Z-score
            z = abs(feat_val - median_val) / (mad_val + 1e-6)
            z_scores.append(z)

        # Equal-weighted average
        composite_z = np.mean(z_scores)

        return composite_z

    def detect(self, pattern: dict, threshold: float = 0.75) -> Tuple[str, float]:
        """
        Detect pathological gait using robust Z-score.

        Args:
            pattern: Gait pattern with features
            threshold: Z-score threshold (default: 0.75, optimized)

        Returns:
            (predicted_class, z_score)
        """
        features = pattern['features']
        z_score = self.compute_robust_z_score(features)

        predicted = 'pathological' if z_score > threshold else 'normal'

        return predicted, z_score

    def evaluate(self, threshold: float = 0.75) -> Dict:
        """Evaluate detector on all patterns"""

        print(f"\n{'='*80}")
        print(f"BASELINE V3 ROBUST EVALUATION (threshold={threshold})")
        print(f"{'='*80}")

        tp, tn, fp, fn = 0, 0, 0, 0

        normal_z_scores = []
        path_z_scores = []

        for p in self.patterns:
            true_label = 'normal' if p['gait_class'] == 'normal' else 'pathological'
            pred_label, z_score = self.detect(p, threshold)

            # Collect Z-scores
            if true_label == 'normal':
                normal_z_scores.append(z_score)
            else:
                path_z_scores.append(z_score)

            # Confusion matrix
            if true_label == 'pathological' and pred_label == 'pathological':
                tp += 1
            elif true_label == 'normal' and pred_label == 'normal':
                tn += 1
            elif true_label == 'normal' and pred_label == 'pathological':
                fp += 1
            else:
                fn += 1

        # Metrics
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

        results = {
            'threshold': threshold,
            'accuracy': float(accuracy),
            'sensitivity': float(sensitivity),
            'specificity': float(specificity),
            'tp': int(tp),
            'tn': int(tn),
            'fp': int(fp),
            'fn': int(fn),
            'n_samples': len(self.patterns),
            'normal_z_mean': float(np.mean(normal_z_scores)),
            'normal_z_std': float(np.std(normal_z_scores)),
            'path_z_mean': float(np.mean(path_z_scores)),
            'path_z_std': float(np.std(path_z_scores))
        }

        print(f"\nResults on {len(self.patterns)} patterns:")
        print(f"  Accuracy: {accuracy*100:.1f}%")
        print(f"  Sensitivity: {sensitivity*100:.1f}%")
        print(f"  Specificity: {specificity*100:.1f}%")
        print(f"\nConfusion Matrix:")
        print(f"  TP: {tp}, TN: {tn}, FP: {fp}, FN: {fn}")
        print(f"\nZ-score Analysis:")
        print(f"  Normal: {np.mean(normal_z_scores):.2f} ± {np.std(normal_z_scores):.2f}")
        print(f"  Pathological: {np.mean(path_z_scores):.2f} ± {np.std(path_z_scores):.2f}")
        print()

        return results


def main():
    """Main evaluation pipeline"""

    # Initialize detector
    detector = BaselineV3Robust("gavd_real_patterns_fixed.json")

    # Evaluate at optimized threshold
    results = detector.evaluate(threshold=0.75)

    # Save results
    output = {
        'version': 'baseline_v3_robust',
        'features': ['cadence', 'variability', 'irregularity',
                    'velocity', 'jerkiness', 'cycle_duration'],
        'method': 'Robust Z-score (Median/MAD)',
        'threshold': 0.75,
        'performance': results,
        'baseline': detector.baseline
    }

    with open('baseline_v3_robust_results.json', 'w') as f:
        json.dump(output, f, indent=2)

    print("="*80)
    print("BASELINE ESTABLISHED")
    print("="*80)
    print()
    print(f"Method: Baseline V3 Robust (6 features + MAD-Z)")
    print(f"Accuracy: {results['accuracy']*100:.1f}%")
    print(f"Sensitivity: {results['sensitivity']*100:.1f}%")
    print(f"Specificity: {results['specificity']*100:.1f}%")
    print()
    print("This is the starting point for all improvements.")
    print(f"Results saved to: baseline_v3_robust_results.json")
    print()


if __name__ == "__main__":
    main()
