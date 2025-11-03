#!/usr/bin/env python3
"""
Improved V6 Final: 8 Features with Equal Weight
================================================

Using all 8 features with equal-weight MAD-Z (no QoS):
1-6. Previous features (cadence, variability, irregularity, velocity, jerkiness, cycle_duration)
7. 3D Stride Length (Cohen's d = 1.120) ✓
8. Trunk Sway (Cohen's d = 0.138)

Dataset: GAVD 3D pose (182 patterns: 101 normal, 81 pathological)

Author: Gait Analysis System
Date: 2025-10-30
Version: 6.0 Final
"""

import json
import numpy as np
from scipy.signal import find_peaks
from scipy.stats import median_abs_deviation
from typing import Dict
import warnings
warnings.filterwarnings('ignore')


class ImprovedV6Final:
    """V6: 8 features with equal-weight MAD-Z"""

    def __init__(self, patterns_file: str = "gavd_3d_patterns.json"):
        print("="*80)
        print("Improved V6 Final: 8 Features (Equal Weight)")
        print("="*80)
        print()

        with open(patterns_file, 'r') as f:
            self.patterns = json.load(f)

        normal = [p for p in self.patterns if p['gait_class'] == 'normal']
        pathological = [p for p in self.patterns if p['gait_class'] == 'pathological']

        print(f"GAVD Dataset: {len(self.patterns)} patterns")
        print(f"  Normal: {len(normal)}")
        print(f"  Pathological: {len(pathological)}")
        print()

        # Extract all 8 features
        print("Extracting 8 features...")
        for p in self.patterns:
            p['features'] = self._extract_features(p)

        self._build_baseline()

    def _extract_features(self, p: dict) -> Dict:
        """Extract all 8 features"""
        heel_left = np.array(p['heel_height_left'])
        heel_right = np.array(p['heel_height_right'])
        fps = p.get('fps', 30)
        dt = 1.0 / fps
        duration = len(heel_left) / fps

        peaks_left, _ = find_peaks(heel_left, height=np.mean(heel_left), distance=5)
        peaks_right, _ = find_peaks(heel_right, height=np.mean(heel_right), distance=5)

        # 1. Cadence
        n_steps = len(peaks_left) + len(peaks_right)
        cadence = (n_steps / duration) * 60 if duration > 0 else 0

        # 2. Variability
        var_left = np.std(heel_left[peaks_left]) / (np.mean(heel_left[peaks_left]) + 1e-6) if len(peaks_left) > 1 else 0
        var_right = np.std(heel_right[peaks_right]) / (np.mean(heel_right[peaks_right]) + 1e-6) if len(peaks_right) > 1 else 0
        variability = (var_left + var_right) / 2

        # 3. Irregularity
        irreg_left = np.std(np.diff(peaks_left)) / (np.mean(np.diff(peaks_left)) + 1e-6) if len(peaks_left) > 2 else 0
        irreg_right = np.std(np.diff(peaks_right)) / (np.mean(np.diff(peaks_right)) + 1e-6) if len(peaks_right) > 2 else 0
        irregularity = (irreg_left + irreg_right) / 2

        # 4. Velocity
        vel_left = np.diff(heel_left) / dt
        vel_right = np.diff(heel_right) / dt
        velocity = (np.mean(np.abs(vel_left)) + np.mean(np.abs(vel_right))) / 2

        # 5. Jerkiness
        accel_left = np.diff(vel_left) / dt
        accel_right = np.diff(vel_right) / dt
        jerkiness = (np.mean(np.abs(accel_left)) + np.mean(np.abs(accel_right))) / 2

        # 6. Cycle Duration
        cycle_duration = np.mean(np.diff(peaks_left) / fps) if len(peaks_left) > 1 else 0

        # 7. 3D Stride Length (NEW!)
        stride_length_3d = p['stride_length_3d']

        # 8. Trunk Sway (NEW!)
        trunk_sway = p['trunk_sway']

        return {
            'cadence': cadence,
            'variability': variability,
            'irregularity': irregularity,
            'velocity': velocity,
            'jerkiness': jerkiness,
            'cycle_duration': cycle_duration,
            'stride_length_3d': stride_length_3d,
            'trunk_sway': trunk_sway
        }

    def _build_baseline(self):
        """Build robust baseline"""
        normal = [p for p in self.patterns if p['gait_class'] == 'normal']

        print(f"Building baseline from {len(normal)} normal patterns...")

        features = ['cadence', 'variability', 'irregularity', 'velocity',
                   'jerkiness', 'cycle_duration', 'stride_length_3d', 'trunk_sway']

        self.baseline = {}
        for feat in features:
            vals = np.array([p['features'][feat] for p in normal])
            self.baseline[f'{feat}_median'] = np.median(vals)
            self.baseline[f'{feat}_mad'] = median_abs_deviation(vals, scale='normal')

        print("\nBaseline (Median ± MAD):")
        for feat in features:
            med = self.baseline[f'{feat}_median']
            mad = self.baseline[f'{feat}_mad']
            new = " ← NEW!" if feat in ['stride_length_3d', 'trunk_sway'] else ""
            print(f"  {feat}: {med:.6f} ± {mad:.6f}{new}")
        print()

    def compute_z(self, p: dict) -> float:
        """Compute equal-weight MAD-Z"""
        features = ['cadence', 'variability', 'irregularity', 'velocity',
                   'jerkiness', 'cycle_duration', 'stride_length_3d', 'trunk_sway']

        z_scores = []
        for feat in features:
            val = p['features'][feat]
            med = self.baseline[f'{feat}_median']
            mad = self.baseline[f'{feat}_mad']
            z = abs(val - med) / (mad + 1e-10)
            z_scores.append(z)

        return np.mean(z_scores)

    def evaluate(self, threshold: float = 1.5) -> Dict:
        """Evaluate detector"""
        tp, tn, fp, fn = 0, 0, 0, 0
        normal_z, path_z = [], []

        for p in self.patterns:
            z = self.compute_z(p)
            pred = 'pathological' if z > threshold else 'normal'
            true = p['gait_class']

            if true == 'normal':
                normal_z.append(z)
            else:
                path_z.append(z)

            if true == 'pathological' and pred == 'pathological':
                tp += 1
            elif true == 'normal' and pred == 'normal':
                tn += 1
            elif true == 'normal' and pred == 'pathological':
                fp += 1
            else:
                fn += 1

        acc = (tp + tn) / (tp + tn + fp + fn)
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0

        print(f"\nThreshold={threshold}:")
        print(f"  Accuracy: {acc*100:.1f}%")
        print(f"  Sensitivity: {sens*100:.1f}%")
        print(f"  Specificity: {spec*100:.1f}%")
        print(f"  Z-scores: Normal={np.mean(normal_z):.2f}±{np.std(normal_z):.2f}, Path={np.mean(path_z):.2f}±{np.std(path_z):.2f}")

        return {'threshold': threshold, 'accuracy': acc, 'sensitivity': sens, 'specificity': spec,
                'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn}

    def optimize(self):
        """Find best threshold"""
        print("\n" + "="*80)
        print("THRESHOLD OPTIMIZATION")
        print("="*80)

        thresholds = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5]
        best = None
        best_acc = 0

        for t in thresholds:
            result = self.evaluate(t)
            if result['accuracy'] > best_acc:
                best_acc = result['accuracy']
                best = result

        print("\n" + "="*80)
        print("BEST RESULT")
        print("="*80)
        print(f"  Threshold: {best['threshold']}")
        print(f"  Accuracy: {best['accuracy']*100:.1f}%")
        print(f"  Sensitivity: {best['sensitivity']*100:.1f}%")
        print(f"  Specificity: {best['specificity']*100:.1f}%")
        print(f"  TP={best['tp']}, TN={best['tn']}, FP={best['fp']}, FN={best['fn']}")

        return best


def main():
    detector = ImprovedV6Final("gavd_3d_patterns.json")
    best = detector.optimize()

    print("\n" + "="*80)
    print("FINAL V6 SUMMARY")
    print("="*80)
    print(f"\nDataset: GAVD 3D pose (182 patterns)")
    print(f"Features: 8 (6 baseline + 3D stride length + trunk sway)")
    print(f"Method: Equal-weight MAD-Z")
    print(f"\nPerformance:")
    print(f"  Accuracy: {best['accuracy']*100:.1f}%")
    print(f"  Sensitivity: {best['sensitivity']*100:.1f}%")
    print(f"  Specificity: {best['specificity']*100:.1f}%")
    print()

    # Save
    with open('improved_v6_final_results.json', 'w') as f:
        json.dump(best, f, indent=2)

    print("Results saved to: improved_v6_final_results.json\n")


if __name__ == "__main__":
    main()
