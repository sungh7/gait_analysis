#!/usr/bin/env python3
"""
Improved V5: Quality-of-Signal (QoS) Weighted Features
=======================================================

Builds on Baseline V3 Robust (66.3%) by adding QoS weighting.

QoS measures:
  1. Peak detection quality (how many peaks found)
  2. Signal smoothness (low jitter)
  3. Temporal consistency (regular stride intervals)

Features with higher QoS get higher weight in Z-score computation.

Expected improvement: +2-4%
Target: 68-70% accuracy

Author: Gait Analysis System
Date: 2025-10-30
Version: 5.0 - QoS Weighting
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
    """6 features with QoS scores"""
    # Core features
    cadence: float
    variability: float
    irregularity: float
    velocity: float
    jerkiness: float
    cycle_duration: float
    # QoS scores (0-1, higher = better quality)
    qos_cadence: float = 1.0
    qos_variability: float = 1.0
    qos_irregularity: float = 1.0
    qos_velocity: float = 1.0
    qos_jerkiness: float = 1.0
    qos_cycle_duration: float = 1.0


class ImprovedV5QoS:
    """
    V5: QoS-weighted robust Z-score

    Target: 68-70% accuracy (baseline 66.3% + 2-4%)
    """

    def __init__(self, patterns_file: str = "gavd_real_patterns_fixed.json"):
        """Initialize with clean patterns"""

        print("="*80)
        print("Improved V5: QoS-Weighted Features")
        print("="*80)
        print()

        with open(patterns_file, 'r') as f:
            all_patterns = json.load(f)

        self.patterns = [p for p in all_patterns
                        if p['heel_height_left'] and p['heel_height_right']
                        and len(p['heel_height_left']) > 10
                        and p['gait_class'] not in ['prosthetic', 'exercise']]

        print(f"Loaded {len(self.patterns)} clean patterns")

        # Extract features with QoS
        print("Extracting features with QoS scores...")
        for p in self.patterns:
            p['features'] = self._extract_features_with_qos(p)

        # Build robust baseline
        self._build_robust_baseline()

    def _compute_qos(self, heel_left, heel_right, peaks_left, peaks_right) -> Dict[str, float]:
        """
        Compute Quality-of-Signal scores for each feature.

        QoS = combination of:
          1. Peak detection quality (enough peaks found?)
          2. Signal smoothness (low noise/jitter?)
          3. Temporal consistency (regular intervals?)
        """

        n_frames = len(heel_left)
        expected_peaks = max(3, n_frames // 15)  # Expect ~1 peak per 15 frames

        # 1. Peak detection quality
        peak_quality_left = min(1.0, len(peaks_left) / expected_peaks)
        peak_quality_right = min(1.0, len(peaks_right) / expected_peaks)
        peak_quality = (peak_quality_left + peak_quality_right) / 2

        # 2. Signal smoothness (inverse of jitter)
        diff_left = np.diff(heel_left)
        diff_right = np.diff(heel_right)
        jitter_left = np.std(np.diff(diff_left)) if len(diff_left) > 1 else 0
        jitter_right = np.std(np.diff(diff_right)) if len(diff_right) > 1 else 0
        avg_jitter = (jitter_left + jitter_right) / 2
        smoothness = np.exp(-avg_jitter)  # High jitter → low smoothness

        # 3. Temporal consistency (regularity of stride intervals)
        if len(peaks_left) > 2:
            intervals_left = np.diff(peaks_left)
            cv_left = np.std(intervals_left) / (np.mean(intervals_left) + 1e-6)
            consistency_left = np.exp(-cv_left)
        else:
            consistency_left = 0.5

        if len(peaks_right) > 2:
            intervals_right = np.diff(peaks_right)
            cv_right = np.std(intervals_right) / (np.mean(intervals_right) + 1e-6)
            consistency_right = np.exp(-cv_right)
        else:
            consistency_right = 0.5

        consistency = (consistency_left + consistency_right) / 2

        # Composite QoS
        overall_qos = (0.4 * peak_quality + 0.3 * smoothness + 0.3 * consistency)

        return {
            'qos_cadence': overall_qos,
            'qos_variability': peak_quality,  # Needs peaks
            'qos_irregularity': consistency,  # Needs regular intervals
            'qos_velocity': smoothness,  # Sensitive to noise
            'qos_jerkiness': smoothness,  # Sensitive to noise
            'qos_cycle_duration': peak_quality  # Needs peaks
        }

    def _extract_features_with_qos(self, pattern: dict) -> GaitFeatures:
        """Extract features and compute QoS for each"""

        heel_left = np.array(pattern['heel_height_left'])
        heel_right = np.array(pattern['heel_height_right'])
        n_frames = len(heel_left)
        fps = pattern.get('fps', 30)
        duration = n_frames / fps
        dt = 1.0 / fps

        # Find peaks
        peaks_left, _ = find_peaks(heel_left, height=np.mean(heel_left), distance=5)
        peaks_right, _ = find_peaks(heel_right, height=np.mean(heel_right), distance=5)

        # Compute QoS
        qos = self._compute_qos(heel_left, heel_right, peaks_left, peaks_right)

        # 1. CADENCE
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
            cycle_duration=cycle_duration,
            qos_cadence=qos['qos_cadence'],
            qos_variability=qos['qos_variability'],
            qos_irregularity=qos['qos_irregularity'],
            qos_velocity=qos['qos_velocity'],
            qos_jerkiness=qos['qos_jerkiness'],
            qos_cycle_duration=qos['qos_cycle_duration']
        )

    def _build_robust_baseline(self):
        """Build baseline using robust statistics"""

        normal_patterns = [p for p in self.patterns if p['gait_class'] == 'normal']

        print(f"\nBuilding robust baseline from {len(normal_patterns)} normal patterns...")

        feature_names = ['cadence', 'variability', 'irregularity',
                        'velocity', 'jerkiness', 'cycle_duration']

        self.baseline = {}

        for feat_name in feature_names:
            vals = np.array([getattr(p['features'], feat_name) for p in normal_patterns])

            median_val = np.median(vals)
            mad_val = median_abs_deviation(vals, scale='normal')

            self.baseline[f'{feat_name}_median'] = median_val
            self.baseline[f'{feat_name}_mad'] = mad_val

        # Average QoS across normal patterns
        for feat_name in feature_names:
            qos_vals = np.array([getattr(p['features'], f'qos_{feat_name}') for p in normal_patterns])
            self.baseline[f'{feat_name}_avg_qos'] = np.mean(qos_vals)

        self.baseline['n_samples'] = len(normal_patterns)

        print(f"\nRobust Baseline Statistics (Median ± MAD):")
        print(f"  Cadence: {self.baseline['cadence_median']:.1f} ± {self.baseline['cadence_mad']:.1f} (QoS: {self.baseline['cadence_avg_qos']:.3f})")
        print(f"  Variability: {self.baseline['variability_median']:.3f} ± {self.baseline['variability_mad']:.3f} (QoS: {self.baseline['variability_avg_qos']:.3f})")
        print(f"  Irregularity: {self.baseline['irregularity_median']:.3f} ± {self.baseline['irregularity_mad']:.3f} (QoS: {self.baseline['irregularity_avg_qos']:.3f})")
        print(f"  Velocity: {self.baseline['velocity_median']:.3f} ± {self.baseline['velocity_mad']:.3f} (QoS: {self.baseline['velocity_avg_qos']:.3f})")
        print(f"  Jerkiness: {self.baseline['jerkiness_median']:.3f} ± {self.baseline['jerkiness_mad']:.3f} (QoS: {self.baseline['jerkiness_avg_qos']:.3f})")
        print(f"  Cycle Duration: {self.baseline['cycle_duration_median']:.3f} ± {self.baseline['cycle_duration_mad']:.3f} (QoS: {self.baseline['cycle_duration_avg_qos']:.3f})")
        print()

    def compute_qos_weighted_z_score(self, features: GaitFeatures) -> float:
        """Compute QoS-weighted robust Z-score"""

        feature_names = ['cadence', 'variability', 'irregularity',
                        'velocity', 'jerkiness', 'cycle_duration']

        weighted_z_sum = 0
        weight_sum = 0

        for feat_name in feature_names:
            feat_val = getattr(features, feat_name)
            qos_val = getattr(features, f'qos_{feat_name}')

            median_val = self.baseline[f'{feat_name}_median']
            mad_val = self.baseline[f'{feat_name}_mad']

            # Robust Z-score
            z = abs(feat_val - median_val) / (mad_val + 1e-6)

            # QoS weight (higher QoS → more trust → higher weight)
            weight = qos_val

            weighted_z_sum += weight * z
            weight_sum += weight

        # Weighted average
        composite_z = weighted_z_sum / (weight_sum + 1e-6)

        return composite_z

    def detect(self, pattern: dict, threshold: float = 0.75) -> Tuple[str, float]:
        """Detect pathological gait"""
        features = pattern['features']
        z_score = self.compute_qos_weighted_z_score(features)
        predicted = 'pathological' if z_score > threshold else 'normal'
        return predicted, z_score

    def evaluate(self, threshold: float = 0.75) -> Dict:
        """Evaluate detector"""

        print(f"\n{'='*80}")
        print(f"IMPROVED V5 EVALUATION (QoS-weighted, threshold={threshold})")
        print(f"{'='*80}")

        tp, tn, fp, fn = 0, 0, 0, 0
        normal_z_scores = []
        path_z_scores = []

        for p in self.patterns:
            true_label = 'normal' if p['gait_class'] == 'normal' else 'pathological'
            pred_label, z_score = self.detect(p, threshold)

            if true_label == 'normal':
                normal_z_scores.append(z_score)
            else:
                path_z_scores.append(z_score)

            if true_label == 'pathological' and pred_label == 'pathological':
                tp += 1
            elif true_label == 'normal' and pred_label == 'normal':
                tn += 1
            elif true_label == 'normal' and pred_label == 'pathological':
                fp += 1
            else:
                fn += 1

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
            'fn': int(fn)
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

        return results

    def optimize_threshold(self) -> Dict:
        """Find optimal threshold"""

        print(f"\n{'='*80}")
        print("THRESHOLD OPTIMIZATION")
        print(f"{'='*80}")

        thresholds = [0.5, 0.6, 0.7, 0.75, 0.8, 0.9, 1.0, 1.25, 1.5]
        best_acc = 0
        best_result = None
        best_threshold = 0.75

        for threshold in thresholds:
            result = self.evaluate(threshold)

            if result['accuracy'] > best_acc:
                best_acc = result['accuracy']
                best_result = result
                best_threshold = threshold

        print(f"\n{'='*80}")
        print("BEST CONFIGURATION")
        print(f"{'='*80}")
        print(f"  Threshold: {best_threshold}")
        print(f"  Accuracy: {best_acc*100:.1f}%")
        print(f"  Sensitivity: {best_result['sensitivity']*100:.1f}%")
        print(f"  Specificity: {best_result['specificity']*100:.1f}%")

        return {
            'best_threshold': best_threshold,
            'best_result': best_result
        }


def main():
    """Main evaluation"""

    detector = ImprovedV5QoS("gavd_real_patterns_fixed.json")
    optimization = detector.optimize_threshold()

    # Comparison
    print(f"\n{'='*80}")
    print("COMPARISON WITH BASELINE")
    print(f"{'='*80}")
    print(f"\n  Baseline V3 (equal weight): 66.3% accuracy")
    print(f"  Improved V5 (QoS-weighted): {optimization['best_result']['accuracy']*100:.1f}% accuracy")
    print(f"\n  Improvement: {(optimization['best_result']['accuracy'] - 0.663)*100:+.1f}%")
    print()

    # Save
    output = {
        'version': 'improved_v5_qos_weighting',
        'method': 'QoS-weighted robust Z-score',
        'best_threshold': optimization['best_threshold'],
        'performance': optimization['best_result'],
        'baseline_comparison': {
            'v3_accuracy': 0.663,
            'v5_accuracy': optimization['best_result']['accuracy'],
            'improvement': optimization['best_result']['accuracy'] - 0.663
        }
    }

    with open('improved_v5_results.json', 'w') as f:
        json.dump(output, f, indent=2)

    print(f"Results saved to: improved_v5_results.json")
    print()


if __name__ == "__main__":
    main()
