#!/usr/bin/env python3
"""
STAGE 1 v3: Enhanced Detector with ALL Available Features
==========================================================

Adds even more features that humans can see:
- Cadence, Variability, Irregularity (from v2)
- Vertical velocity (NEW)
- Acceleration/Jerkiness (NEW)
- Gait cycle duration (NEW)

Goal: Push accuracy beyond 80%!

Author: Gait Analysis System
Version: 3.0 - ENHANCED
Date: 2025-10-30
"""

import json
import numpy as np
from scipy.signal import find_peaks
from typing import Dict
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')


@dataclass
class EnhancedGaitFeatures:
    """All available gait features"""
    # Core features (from v2)
    cadence: float
    variability_avg: float
    irregularity_avg: float

    # NEW features
    vertical_velocity_avg: float
    acceleration_std_avg: float
    cycle_duration_avg: float


class Stage1V3Detector:
    """STAGE 1 v3 with ENHANCED features"""

    def __init__(self, patterns_file: str):
        """Initialize"""
        with open(patterns_file, 'r') as f:
            all_patterns = json.load(f)

        self.patterns = [p for p in all_patterns
                        if p['heel_height_left'] and p['heel_height_right']
                        and len(p['heel_height_left']) > 10
                        and p['gait_class'] not in ['prosthetic', 'exercise']]

        print(f"Stage 1 v3 Initialized: {len(self.patterns)} patterns")

        # Extract ALL features
        print("\nExtracting ENHANCED features...")
        for p in self.patterns:
            p['features'] = self._extract_features(p)

        self._build_baseline()

    def _extract_features(self, pattern: dict) -> EnhancedGaitFeatures:
        """Extract all available features"""

        heel_left = np.array(pattern['heel_height_left'])
        heel_right = np.array(pattern['heel_height_right'])
        fps = pattern.get('fps', 30)
        duration = len(heel_left) / fps

        # === CORE FEATURES (from v2) ===

        # Cadence
        peaks_left, _ = find_peaks(heel_left, height=np.mean(heel_left), distance=5)
        peaks_right, _ = find_peaks(heel_right, height=np.mean(heel_right), distance=5)
        n_steps = len(peaks_left) + len(peaks_right)
        cadence = (n_steps / duration) * 60 if duration > 0 else 0

        # Variability
        def compute_variability(heel, peaks):
            if len(peaks) > 1:
                peak_heights = heel[peaks]
                return np.std(peak_heights) / (np.mean(peak_heights) + 1e-6)
            return 0

        var_left = compute_variability(heel_left, peaks_left)
        var_right = compute_variability(heel_right, peaks_right)
        var_avg = (var_left + var_right) / 2

        # Irregularity
        def compute_irregularity(peaks):
            if len(peaks) > 2:
                intervals = np.diff(peaks)
                return np.std(intervals) / (np.mean(intervals) + 1e-6)
            return 0

        irreg_left = compute_irregularity(peaks_left)
        irreg_right = compute_irregularity(peaks_right)
        irreg_avg = (irreg_left + irreg_right) / 2

        # === NEW FEATURES ===

        # 1. VERTICAL VELOCITY
        def compute_vertical_velocity(heel):
            if len(heel) > 1:
                velocity = np.diff(heel) * fps  # units/second
                return np.mean(np.abs(velocity))
            return 0

        vel_left = compute_vertical_velocity(heel_left)
        vel_right = compute_vertical_velocity(heel_right)
        vel_avg = (vel_left + vel_right) / 2

        # 2. ACCELERATION / JERKINESS
        def compute_jerkiness(heel):
            if len(heel) > 2:
                acceleration = np.diff(np.diff(heel)) * fps * fps  # units/second²
                return np.std(acceleration)  # Higher = more jerky
            return 0

        jerk_left = compute_jerkiness(heel_left)
        jerk_right = compute_jerkiness(heel_right)
        jerk_avg = (jerk_left + jerk_right) / 2

        # 3. GAIT CYCLE DURATION
        def compute_cycle_duration(peaks):
            if len(peaks) > 1:
                intervals = np.diff(peaks)
                return np.mean(intervals) / fps  # seconds
            return 0

        cycle_left = compute_cycle_duration(peaks_left)
        cycle_right = compute_cycle_duration(peaks_right)
        cycle_avg = (cycle_left + cycle_right) / 2

        return EnhancedGaitFeatures(
            cadence=cadence,
            variability_avg=var_avg,
            irregularity_avg=irreg_avg,
            vertical_velocity_avg=vel_avg,
            acceleration_std_avg=jerk_avg,
            cycle_duration_avg=cycle_avg
        )

    def _build_baseline(self):
        """Build baseline from normal patterns"""

        normal_patterns = [p for p in self.patterns if p['gait_class'] == 'normal']

        print(f"\nBuilding baseline from {len(normal_patterns)} normal patterns...")

        # Extract all feature values
        features_dict = {
            'cadence': [p['features'].cadence for p in normal_patterns],
            'variability': [p['features'].variability_avg for p in normal_patterns],
            'irregularity': [p['features'].irregularity_avg for p in normal_patterns],
            'velocity': [p['features'].vertical_velocity_avg for p in normal_patterns],
            'jerkiness': [p['features'].acceleration_std_avg for p in normal_patterns],
            'cycle_duration': [p['features'].cycle_duration_avg for p in normal_patterns]
        }

        # Remove outliers and compute stats
        self.baseline = {}

        for name, vals in features_dict.items():
            mean_val = np.mean(vals)
            std_val = np.std(vals)
            # Remove > 3 std
            clean_vals = [v for v in vals if abs(v - mean_val) < 3*std_val]

            self.baseline[f'{name}_mean'] = np.mean(clean_vals)
            self.baseline[f'{name}_std'] = np.std(clean_vals)

        self.baseline['n_samples'] = len(normal_patterns)

        print(f"\nBaseline Statistics:")
        print(f"  Cadence: {self.baseline['cadence_mean']:.1f} ± {self.baseline['cadence_std']:.1f}")
        print(f"  Variability: {self.baseline['variability_mean']:.3f} ± {self.baseline['variability_std']:.3f}")
        print(f"  Irregularity: {self.baseline['irregularity_mean']:.3f} ± {self.baseline['irregularity_std']:.3f}")
        print(f"  Velocity: {self.baseline['velocity_mean']:.3f} ± {self.baseline['velocity_std']:.3f}")
        print(f"  Jerkiness: {self.baseline['jerkiness_mean']:.4f} ± {self.baseline['jerkiness_std']:.4f}")
        print(f"  Cycle Duration: {self.baseline['cycle_duration_mean']:.2f} ± {self.baseline['cycle_duration_std']:.2f}s")

    def compute_z_score(self, features: EnhancedGaitFeatures) -> float:
        """Compute composite Z-score with ALL features"""

        z_scores = []

        # Core features
        z_scores.append(abs(features.cadence - self.baseline['cadence_mean']) / (self.baseline['cadence_std'] + 1e-6))
        z_scores.append(abs(features.variability_avg - self.baseline['variability_mean']) / (self.baseline['variability_std'] + 1e-6))
        z_scores.append(abs(features.irregularity_avg - self.baseline['irregularity_mean']) / (self.baseline['irregularity_std'] + 1e-6))

        # NEW features
        z_scores.append(abs(features.vertical_velocity_avg - self.baseline['velocity_mean']) / (self.baseline['velocity_std'] + 1e-6))
        z_scores.append(abs(features.acceleration_std_avg - self.baseline['jerkiness_mean']) / (self.baseline['jerkiness_std'] + 1e-6))
        z_scores.append(abs(features.cycle_duration_avg - self.baseline['cycle_duration_mean']) / (self.baseline['cycle_duration_std'] + 1e-6))

        # Average of all Z-scores
        composite_z = np.mean(z_scores)

        return composite_z

    def evaluate(self, threshold: float = 1.5) -> Dict:
        """Evaluate detector"""

        true_labels = []
        pred_labels = []

        for p in self.patterns:
            true_label = 'normal' if p['gait_class'] == 'normal' else 'pathological'
            z_score = self.compute_z_score(p['features'])
            pred_label = 'pathological' if z_score > threshold else 'normal'

            true_labels.append(true_label)
            pred_labels.append(pred_label)

        true_labels = np.array(true_labels)
        pred_labels = np.array(pred_labels)

        accuracy = np.mean(true_labels == pred_labels)

        path_mask = true_labels == 'pathological'
        normal_mask = true_labels == 'normal'

        sensitivity = np.mean(pred_labels[path_mask] == 'pathological')
        specificity = np.mean(pred_labels[normal_mask] == 'normal')

        tp = np.sum((true_labels == 'pathological') & (pred_labels == 'pathological'))
        tn = np.sum((true_labels == 'normal') & (pred_labels == 'normal'))
        fp = np.sum((true_labels == 'normal') & (pred_labels == 'pathological'))
        fn = np.sum((true_labels == 'pathological') & (pred_labels == 'normal'))

        return {
            'threshold': threshold,
            'accuracy': float(accuracy),
            'sensitivity': float(sensitivity),
            'specificity': float(specificity),
            'tp': int(tp),
            'tn': int(tn),
            'fp': int(fp),
            'fn': int(fn)
        }

    def optimize(self):
        """Find best threshold"""

        print(f"\n{'='*80}")
        print("THRESHOLD OPTIMIZATION")
        print(f"{'='*80}")

        thresholds = [1.0, 1.25, 1.5, 1.75, 2.0, 2.5]
        results = []

        for threshold in thresholds:
            result = self.evaluate(threshold)
            results.append(result)

            print(f"\nThreshold={threshold}:")
            print(f"  Accuracy: {result['accuracy']*100:.1f}%")
            print(f"  Sensitivity: {result['sensitivity']*100:.1f}%")
            print(f"  Specificity: {result['specificity']*100:.1f}%")

        # Best by F1 score
        best = max(results, key=lambda r: 2*r['sensitivity']*r['specificity']/(r['sensitivity']+r['specificity']+1e-6))

        print(f"\n{'='*80}")
        print("BEST CONFIGURATION")
        print(f"{'='*80}")
        print(f"  Threshold: {best['threshold']}")
        print(f"  Accuracy: {best['accuracy']*100:.1f}%")
        print(f"  Sensitivity: {best['sensitivity']*100:.1f}%")
        print(f"  Specificity: {best['specificity']*100:.1f}%")
        print(f"  TP: {best['tp']}, TN: {best['tn']}, FP: {best['fp']}, FN: {best['fn']}")

        return best


def main():
    """Main pipeline"""

    print("="*80)
    print("STAGE 1 v3: Enhanced Detector with ALL Features")
    print("="*80)
    print()

    detector = Stage1V3Detector("gavd_real_patterns.json")
    best_result = detector.optimize()

    # Save
    with open('stage1_v3_results.json', 'w') as f:
        json.dump(best_result, f, indent=2)

    print(f"\n{'='*80}")
    print("FINAL COMPARISON")
    print(f"{'='*80}")
    print(f"\nWrong features (amplitude, asymmetry): 57.0%")
    print(f"STAGE 1 v2 (3 correct features): 76.6%")
    print(f"STAGE 1 v3 (6 enhanced features): {best_result['accuracy']*100:.1f}%")

    improvement = (best_result['accuracy'] - 0.766) * 100
    print(f"Improvement from v2: {improvement:+.1f}%")

    print(f"\nResults saved to: stage1_v3_results.json")
    print()


if __name__ == "__main__":
    main()
