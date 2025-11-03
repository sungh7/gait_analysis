#!/usr/bin/env python3
"""
Improved V6: Real 3D Features from GAVD
========================================

Adds TRUE 3D features:
  7. 3D Stride Length (Cohen's d = 1.120) ✓ LARGE effect
  8. Trunk Sway (Cohen's d = 0.138) - small but may help

Total: 8 features with QoS weighting

Expected improvement: +3-5%
Target: 72-74% accuracy

Author: Gait Analysis System
Date: 2025-10-30
Version: 6.0 - True 3D Features
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
    """8 features including 3D stride length and trunk sway"""
    # Core features (v2)
    cadence: float
    variability: float
    irregularity: float
    # Enhanced features (v3)
    velocity: float
    jerkiness: float
    cycle_duration: float
    # 3D features (v6) - NEW!
    stride_length_3d: float
    trunk_sway: float
    # QoS scores
    qos_cadence: float = 1.0
    qos_variability: float = 1.0
    qos_irregularity: float = 1.0
    qos_velocity: float = 1.0
    qos_jerkiness: float = 1.0
    qos_cycle_duration: float = 1.0
    qos_stride_length_3d: float = 1.0
    qos_trunk_sway: float = 1.0


class ImprovedV6_3D:
    """
    V6: 8 features with true 3D stride length

    Target: 72-74% accuracy (baseline 69.0% + 3-5%)
    """

    def __init__(self, patterns_file: str = "gavd_3d_patterns.json"):
        """Initialize with GAVD 3D patterns"""

        print("="*80)
        print("Improved V6: True 3D Features from GAVD")
        print("="*80)
        print()

        with open(patterns_file, 'r') as f:
            self.patterns = json.load(f)

        print(f"Loaded {len(self.patterns)} patterns from GAVD")

        normal = [p for p in self.patterns if p['gait_class'] == 'normal']
        pathological = [p for p in self.patterns if p['gait_class'] == 'pathological']

        print(f"  Normal: {len(normal)}")
        print(f"  Pathological: {len(pathological)}")
        print()

        # Extract all features
        print("Extracting 8 features (6 baseline + 2 3D features)...")
        for p in self.patterns:
            p['features'] = self._extract_all_features(p)

        # Build robust baseline
        self._build_robust_baseline()

    def _compute_qos(self, heel_left, heel_right, peaks_left, peaks_right) -> Dict[str, float]:
        """Compute QoS scores"""

        n_frames = len(heel_left)
        expected_peaks = max(3, n_frames // 15)

        # Peak quality
        peak_quality_left = min(1.0, len(peaks_left) / expected_peaks)
        peak_quality_right = min(1.0, len(peaks_right) / expected_peaks)
        peak_quality = (peak_quality_left + peak_quality_right) / 2

        # Smoothness
        diff_left = np.diff(heel_left)
        diff_right = np.diff(heel_right)
        jitter_left = np.std(np.diff(diff_left)) if len(diff_left) > 1 else 0
        jitter_right = np.std(np.diff(diff_right)) if len(diff_right) > 1 else 0
        avg_jitter = (jitter_left + jitter_right) / 2
        smoothness = np.exp(-avg_jitter)

        # Consistency
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

        overall_qos = (0.4 * peak_quality + 0.3 * smoothness + 0.3 * consistency)

        return {
            'qos_cadence': overall_qos,
            'qos_variability': peak_quality,
            'qos_irregularity': consistency,
            'qos_velocity': smoothness,
            'qos_jerkiness': smoothness,
            'qos_cycle_duration': peak_quality,
            'qos_stride_length_3d': overall_qos,  # Same as overall
            'qos_trunk_sway': smoothness  # Sensitive to noise
        }

    def _extract_all_features(self, pattern: dict) -> GaitFeatures:
        """Extract all 8 features"""

        heel_left = np.array(pattern['heel_height_left'])
        heel_right = np.array(pattern['heel_height_right'])
        n_frames = len(heel_left)
        fps = pattern.get('fps', 30)
        duration = n_frames / fps
        dt = 1.0 / fps

        # Find peaks
        peaks_left, _ = find_peaks(heel_left, height=np.mean(heel_left), distance=5)
        peaks_right, _ = find_peaks(heel_right, height=np.mean(heel_right), distance=5)

        # QoS
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

        # 7. 3D STRIDE LENGTH (NEW!)
        stride_length_3d = pattern['stride_length_3d']

        # 8. TRUNK SWAY (NEW!)
        trunk_sway = pattern['trunk_sway']

        return GaitFeatures(
            cadence=cadence,
            variability=variability,
            irregularity=irregularity,
            velocity=velocity,
            jerkiness=jerkiness,
            cycle_duration=cycle_duration,
            stride_length_3d=stride_length_3d,
            trunk_sway=trunk_sway,
            qos_cadence=qos['qos_cadence'],
            qos_variability=qos['qos_variability'],
            qos_irregularity=qos['qos_irregularity'],
            qos_velocity=qos['qos_velocity'],
            qos_jerkiness=qos['qos_jerkiness'],
            qos_cycle_duration=qos['qos_cycle_duration'],
            qos_stride_length_3d=qos['qos_stride_length_3d'],
            qos_trunk_sway=qos['qos_trunk_sway']
        )

    def _build_robust_baseline(self):
        """Build baseline using robust statistics"""

        normal_patterns = [p for p in self.patterns if p['gait_class'] == 'normal']

        print(f"Building robust baseline from {len(normal_patterns)} normal patterns...")

        feature_names = ['cadence', 'variability', 'irregularity',
                        'velocity', 'jerkiness', 'cycle_duration',
                        'stride_length_3d', 'trunk_sway']

        self.baseline = {}

        for feat_name in feature_names:
            vals = np.array([getattr(p['features'], feat_name) for p in normal_patterns])

            median_val = np.median(vals)
            mad_val = median_abs_deviation(vals, scale='normal')

            self.baseline[f'{feat_name}_median'] = median_val
            self.baseline[f'{feat_name}_mad'] = mad_val

            # Average QoS
            qos_vals = np.array([getattr(p['features'], f'qos_{feat_name}') for p in normal_patterns])
            self.baseline[f'{feat_name}_avg_qos'] = np.mean(qos_vals)

        self.baseline['n_samples'] = len(normal_patterns)

        print(f"\nRobust Baseline (Median ± MAD):")
        for feat_name in feature_names:
            median = self.baseline[f'{feat_name}_median']
            mad = self.baseline[f'{feat_name}_mad']
            qos = self.baseline[f'{feat_name}_avg_qos']
            marker = " ← NEW!" if feat_name in ['stride_length_3d', 'trunk_sway'] else ""
            print(f"  {feat_name}: {median:.6f} ± {mad:.6f} (QoS: {qos:.3f}){marker}")
        print()

    def compute_qos_weighted_z_score(self, features: GaitFeatures) -> float:
        """Compute QoS-weighted robust Z-score for all 8 features"""

        feature_names = ['cadence', 'variability', 'irregularity',
                        'velocity', 'jerkiness', 'cycle_duration',
                        'stride_length_3d', 'trunk_sway']

        weighted_z_sum = 0
        weight_sum = 0

        for feat_name in feature_names:
            feat_val = getattr(features, feat_name)
            qos_val = getattr(features, f'qos_{feat_name}')

            median_val = self.baseline[f'{feat_name}_median']
            mad_val = self.baseline[f'{feat_name}_mad']

            z = abs(feat_val - median_val) / (mad_val + 1e-6)
            weight = qos_val

            weighted_z_sum += weight * z
            weight_sum += weight

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
        print(f"IMPROVED V6 EVALUATION (8 features, QoS-weighted, threshold={threshold})")
        print(f"{'='*80}")

        tp, tn, fp, fn = 0, 0, 0, 0
        normal_z_scores = []
        path_z_scores = []

        for p in self.patterns:
            true_label = p['gait_class']
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

    detector = ImprovedV6_3D("gavd_3d_patterns.json")
    optimization = detector.optimize_threshold()

    # Comparison
    print(f"\n{'='*80}")
    print("COMPARISON WITH PREVIOUS VERSIONS")
    print(f"{'='*80}")
    print(f"\n  V5 (6 features, QoS): 69.0% accuracy (on different dataset)")
    print(f"  V6 (8 features, 3D, QoS): {optimization['best_result']['accuracy']*100:.1f}% accuracy (on GAVD)")
    print()
    print(f"  New 3D features:")
    print(f"    - Stride Length 3D (Cohen's d = 1.120) ← LARGE effect!")
    print(f"    - Trunk Sway (Cohen's d = 0.138)")
    print()

    # Save
    output = {
        'version': 'improved_v6_3d_features',
        'dataset': 'GAVD (real 3D pose)',
        'features': 8,
        'new_features': ['stride_length_3d', 'trunk_sway'],
        'best_threshold': optimization['best_threshold'],
        'performance': optimization['best_result']
    }

    with open('improved_v6_results.json', 'w') as f:
        json.dump(output, f, indent=2)

    print(f"Results saved to: improved_v6_results.json")
    print()


if __name__ == "__main__":
    main()
