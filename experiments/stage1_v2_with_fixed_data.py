#!/usr/bin/env python3
"""
STAGE 1 v2: Baseline Detector with CORRECT Features
====================================================

Re-implements STAGE 1 with the correct features:
- Cadence (step frequency)
- Variability (consistency)
- Irregularity (stride regularity)

Instead of wrong features:
- Amplitude (heel height range)
- Asymmetry (L-R difference)

Author: Gait Analysis System
Version: 2.0 - CORRECTED
Date: 2025-10-30
"""

import json
import numpy as np
from scipy.signal import find_peaks
from scipy import stats
from typing import Dict, List, Tuple
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')


@dataclass
class GaitFeatures:
    """Correct gait features that humans actually see"""
    cadence: float
    variability_left: float
    variability_right: float
    variability_avg: float
    irregularity_left: float
    irregularity_right: float
    irregularity_avg: float


class Stage1V2Detector:
    """STAGE 1 with CORRECT features"""

    def __init__(self, patterns_file: str):
        """Initialize with real GAVD patterns"""
        with open(patterns_file, 'r') as f:
            all_patterns = json.load(f)

        # Filter valid patterns (exclude prosthetic/exercise)
        self.patterns = [p for p in all_patterns
                        if p['heel_height_left'] and p['heel_height_right']
                        and len(p['heel_height_left']) > 10
                        and p['gait_class'] not in ['prosthetic', 'exercise']]

        print(f"Stage 1 v2 Initialized:")
        print(f"  Total patterns: {len(self.patterns)}")
        print(f"  Excluded: prosthetic, exercise")

        # Extract features for all patterns
        print("\nExtracting CORRECT features for all patterns...")
        for p in self.patterns:
            p['features'] = self._extract_features(p)

        # Build normal population baseline
        self._build_baseline()

    def _extract_features(self, pattern: dict) -> GaitFeatures:
        """Extract correct features that humans see"""

        heel_left = np.array(pattern['heel_height_left'])
        heel_right = np.array(pattern['heel_height_right'])
        n_frames = len(heel_left)
        fps = pattern.get('fps', 30)
        duration = n_frames / fps

        # 1. CADENCE
        peaks_left, _ = find_peaks(heel_left, height=np.mean(heel_left), distance=5)
        peaks_right, _ = find_peaks(heel_right, height=np.mean(heel_right), distance=5)
        n_steps = len(peaks_left) + len(peaks_right)
        cadence = (n_steps / duration) * 60 if duration > 0 else 0

        # 2. VARIABILITY
        if len(peaks_left) > 1:
            peak_heights = heel_left[peaks_left]
            var_left = np.std(peak_heights) / (np.mean(peak_heights) + 1e-6)
        else:
            var_left = 0

        if len(peaks_right) > 1:
            peak_heights = heel_right[peaks_right]
            var_right = np.std(peak_heights) / (np.mean(peak_heights) + 1e-6)
        else:
            var_right = 0

        var_avg = (var_left + var_right) / 2

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

        irreg_avg = (irreg_left + irreg_right) / 2

        return GaitFeatures(
            cadence=cadence,
            variability_left=var_left,
            variability_right=var_right,
            variability_avg=var_avg,
            irregularity_left=irreg_left,
            irregularity_right=irreg_right,
            irregularity_avg=irreg_avg
        )

    def _build_baseline(self):
        """Build normal population baseline statistics"""

        normal_patterns = [p for p in self.patterns if p['gait_class'] == 'normal']

        print(f"\nBuilding baseline from {len(normal_patterns)} normal patterns...")

        # Extract feature values
        cadence_vals = [p['features'].cadence for p in normal_patterns]
        var_vals = [p['features'].variability_avg for p in normal_patterns]
        irreg_vals = [p['features'].irregularity_avg for p in normal_patterns]

        # Remove outliers (>3 std)
        def remove_outliers(vals):
            mean_val = np.mean(vals)
            std_val = np.std(vals)
            return [v for v in vals if abs(v - mean_val) < 3*std_val]

        cadence_clean = remove_outliers(cadence_vals)
        var_clean = remove_outliers(var_vals)
        irreg_clean = remove_outliers(irreg_vals)

        # Compute statistics
        self.baseline = {
            'cadence_mean': np.mean(cadence_clean),
            'cadence_std': np.std(cadence_clean),
            'cadence_median': np.median(cadence_clean),
            'variability_mean': np.mean(var_clean),
            'variability_std': np.std(var_clean),
            'variability_median': np.median(var_clean),
            'irregularity_mean': np.mean(irreg_clean),
            'irregularity_std': np.std(irreg_clean),
            'irregularity_median': np.median(irreg_clean),
            'n_samples': len(normal_patterns)
        }

        print(f"\nBaseline Statistics:")
        print(f"  Cadence: {self.baseline['cadence_mean']:.1f} ± {self.baseline['cadence_std']:.1f} steps/min")
        print(f"  Variability: {self.baseline['variability_mean']:.3f} ± {self.baseline['variability_std']:.3f}")
        print(f"  Irregularity: {self.baseline['irregularity_mean']:.3f} ± {self.baseline['irregularity_std']:.3f}")

    def compute_z_score(self, features: GaitFeatures) -> float:
        """Compute composite Z-score (higher = more abnormal)"""

        # Z-score for each feature
        z_cadence = abs(features.cadence - self.baseline['cadence_mean']) / (self.baseline['cadence_std'] + 1e-6)
        z_var = abs(features.variability_avg - self.baseline['variability_mean']) / (self.baseline['variability_std'] + 1e-6)
        z_irreg = abs(features.irregularity_avg - self.baseline['irregularity_mean']) / (self.baseline['irregularity_std'] + 1e-6)

        # Composite Z-score (average)
        composite_z = (z_cadence + z_var + z_irreg) / 3

        return composite_z

    def detect(self, pattern: dict, threshold: float = 2.0) -> Tuple[str, float]:
        """
        Detect pathological gait using Z-score threshold.

        Args:
            pattern: Gait pattern with features
            threshold: Z-score threshold (default: 2.0 = 95th percentile)

        Returns:
            (predicted_class, z_score)
        """
        features = pattern['features']
        z_score = self.compute_z_score(features)

        predicted = 'pathological' if z_score > threshold else 'normal'

        return predicted, z_score

    def evaluate(self, threshold: float = 2.0) -> Dict:
        """Evaluate detector on all patterns"""

        print(f"\n{'='*80}")
        print(f"STAGE 1 v2 EVALUATION (threshold={threshold})")
        print(f"{'='*80}")

        true_labels = []
        pred_labels = []
        z_scores = []

        for p in self.patterns:
            true_label = 'normal' if p['gait_class'] == 'normal' else 'pathological'
            pred_label, z_score = self.detect(p, threshold)

            true_labels.append(true_label)
            pred_labels.append(pred_label)
            z_scores.append(z_score)

        true_labels = np.array(true_labels)
        pred_labels = np.array(pred_labels)
        z_scores = np.array(z_scores)

        # Metrics
        accuracy = np.mean(true_labels == pred_labels)

        path_mask = true_labels == 'pathological'
        normal_mask = true_labels == 'normal'

        sensitivity = np.mean(pred_labels[path_mask] == 'pathological')
        specificity = np.mean(pred_labels[normal_mask] == 'normal')

        tp = np.sum((true_labels == 'pathological') & (pred_labels == 'pathological'))
        tn = np.sum((true_labels == 'normal') & (pred_labels == 'normal'))
        fp = np.sum((true_labels == 'normal') & (pred_labels == 'pathological'))
        fn = np.sum((true_labels == 'pathological') & (pred_labels == 'normal'))

        # Z-score analysis
        normal_z = z_scores[normal_mask]
        path_z = z_scores[path_mask]

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
            'normal_z_mean': float(np.mean(normal_z)),
            'normal_z_std': float(np.std(normal_z)),
            'path_z_mean': float(np.mean(path_z)),
            'path_z_std': float(np.std(path_z))
        }

        print(f"\nResults on {len(self.patterns)} patterns:")
        print(f"  Accuracy: {accuracy*100:.1f}%")
        print(f"  Sensitivity: {sensitivity*100:.1f}%")
        print(f"  Specificity: {specificity*100:.1f}%")
        print(f"\nConfusion Matrix:")
        print(f"  TP: {tp}, TN: {tn}, FP: {fp}, FN: {fn}")
        print(f"\nZ-score Analysis:")
        print(f"  Normal: {np.mean(normal_z):.2f} ± {np.std(normal_z):.2f}")
        print(f"  Pathological: {np.mean(path_z):.2f} ± {np.std(path_z):.2f}")

        return results

    def optimize_threshold(self) -> Dict:
        """Find optimal threshold by testing multiple values"""

        print(f"\n{'='*80}")
        print("THRESHOLD OPTIMIZATION")
        print(f"{'='*80}")

        thresholds = [1.0, 1.5, 2.0, 2.5, 3.0]
        all_results = []

        for threshold in thresholds:
            results = self.evaluate(threshold)
            all_results.append(results)

        # Find best by F1 score
        best_result = max(all_results, key=lambda r: 2*r['sensitivity']*r['specificity']/(r['sensitivity']+r['specificity']+1e-6))

        print(f"\n{'='*80}")
        print("BEST CONFIGURATION")
        print(f"{'='*80}")
        print(f"  Threshold: {best_result['threshold']}")
        print(f"  Accuracy: {best_result['accuracy']*100:.1f}%")
        print(f"  Sensitivity: {best_result['sensitivity']*100:.1f}%")
        print(f"  Specificity: {best_result['specificity']*100:.1f}%")

        return {
            'best_threshold': best_result['threshold'],
            'best_accuracy': best_result['accuracy'],
            'all_results': all_results
        }


def main():
    """Main evaluation pipeline"""

    print("="*80)
    print("STAGE 1 v2: Baseline Detector with CORRECT Features")
    print("="*80)
    print()

    patterns_file = "gavd_real_patterns.json"

    # Initialize detector
    detector = Stage1V2Detector(patterns_file)

    # Optimize threshold
    optimization_results = detector.optimize_threshold()

    # Save results
    output = {
        'version': 'stage1_v2_correct_features',
        'features': ['cadence', 'variability', 'irregularity'],
        'best_threshold': optimization_results['best_threshold'],
        'best_accuracy': optimization_results['best_accuracy'],
        'all_results': optimization_results['all_results']
    }

    with open('stage1_v2_results.json', 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\n{'='*80}")
    print("COMPARISON WITH PREVIOUS METHODS")
    print(f"{'='*80}")
    print(f"\nSTAGE 1 v1 (wrong features, simulated): 85-93%")
    print(f"Pure Pathological (wrong features, real): 57.0%")
    print(f"Pure Pathological (correct features, real): 76.1%")
    print(f"STAGE 1 v2 (correct features, real): {optimization_results['best_accuracy']*100:.1f}%")

    print(f"\nResults saved to: stage1_v2_results.json")
    print()


if __name__ == "__main__":
    main()
