#!/usr/bin/env python3
"""
Improved Detector v1: Feature-Weighted Z-score
================================================

Improvement: Weight features by their discriminative power (Cohen's d)

Expected improvement: +3-5% (76.6% → 79-81%)

Key change:
- Before: Z = (z1 + z2 + z3) / 3 (equal weights)
- After:  Z = 0.50*z1 + 0.20*z2 + 0.30*z3 (weighted by Cohen's d)
"""

import json
import numpy as np
from scipy.signal import find_peaks
from typing import Dict, Tuple
from dataclasses import dataclass


@dataclass
class GaitFeatures:
    cadence: float
    variability_avg: float
    irregularity_avg: float


class ImprovedDetectorV1:
    """Weighted Z-score detector"""

    def __init__(self, patterns_file: str = 'gavd_real_patterns_fixed.json'):
        with open(patterns_file, 'r') as f:
            all_patterns = json.load(f)

        # Filter
        self.patterns = [p for p in all_patterns
                        if p['heel_height_left'] and p['heel_height_right']
                        and len(p['heel_height_left']) > 10
                        and p['gait_class'] not in ['prosthetic', 'exercise']]

        print(f"✅ Loaded {len(self.patterns)} patterns")

        # Extract features
        for p in self.patterns:
            p['features'] = self._extract_features(p)

        # Build baseline
        self._build_baseline()

        # Feature weights from Cohen's d
        # Cadence: d=0.85, Variability: d=0.35, Irregularity: d=0.51
        # Normalized: 0.85/(0.85+0.35+0.51) = 0.50, 0.35/1.71=0.20, 0.51/1.71=0.30
        self.weights = {
            'cadence': 0.50,
            'variability': 0.20,
            'irregularity': 0.30
        }

        print(f"\n✅ Feature weights (Cohen's d based):")
        print(f"   Cadence: {self.weights['cadence']:.2f}")
        print(f"   Variability: {self.weights['variability']:.2f}")
        print(f"   Irregularity: {self.weights['irregularity']:.2f}")

    def _extract_features(self, pattern: dict) -> GaitFeatures:
        """Extract 3 core features"""
        heel_left = np.array(pattern['heel_height_left'])
        heel_right = np.array(pattern['heel_height_right'])
        fps = pattern.get('fps', 30)
        duration = len(heel_left) / fps

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

        return GaitFeatures(cadence=cadence, variability_avg=var_avg, irregularity_avg=irreg_avg)

    def _build_baseline(self):
        """Build baseline from normal patterns"""
        normal_patterns = [p for p in self.patterns if p['gait_class'] == 'normal']
        print(f"\n✅ Building baseline from {len(normal_patterns)} normal patterns")

        # Extract feature values
        features_dict = {
            'cadence': [p['features'].cadence for p in normal_patterns],
            'variability': [p['features'].variability_avg for p in normal_patterns],
            'irregularity': [p['features'].irregularity_avg for p in normal_patterns]
        }

        # Remove outliers and compute stats
        self.baseline = {}
        for name, vals in features_dict.items():
            mean_val = np.mean(vals)
            std_val = np.std(vals)
            clean_vals = [v for v in vals if abs(v - mean_val) < 3*std_val]

            self.baseline[f'{name}_mean'] = np.mean(clean_vals)
            self.baseline[f'{name}_std'] = np.std(clean_vals)

        print(f"\n   Cadence: {self.baseline['cadence_mean']:.1f} ± {self.baseline['cadence_std']:.1f}")
        print(f"   Variability: {self.baseline['variability_mean']:.3f} ± {self.baseline['variability_std']:.3f}")
        print(f"   Irregularity: {self.baseline['irregularity_mean']:.3f} ± {self.baseline['irregularity_std']:.3f}")

    def compute_z_score(self, features: GaitFeatures) -> Dict:
        """Compute WEIGHTED Z-score"""

        # Individual Z-scores
        z_cadence = abs(features.cadence - self.baseline['cadence_mean']) / (self.baseline['cadence_std'] + 1e-6)
        z_var = abs(features.variability_avg - self.baseline['variability_mean']) / (self.baseline['variability_std'] + 1e-6)
        z_irreg = abs(features.irregularity_avg - self.baseline['irregularity_mean']) / (self.baseline['irregularity_std'] + 1e-6)

        # OLD: Equal weighting
        z_equal = (z_cadence + z_var + z_irreg) / 3

        # NEW: Weighted by Cohen's d
        z_weighted = (self.weights['cadence'] * z_cadence +
                     self.weights['variability'] * z_var +
                     self.weights['irregularity'] * z_irreg)

        return {
            'z_cadence': z_cadence,
            'z_variability': z_var,
            'z_irregularity': z_irreg,
            'z_equal': z_equal,
            'z_weighted': z_weighted
        }

    def evaluate(self, threshold: float = 1.5, use_weighted: bool = True) -> Dict:
        """Evaluate detector"""

        true_labels = []
        pred_labels = []
        z_scores = []

        for p in self.patterns:
            true_label = 'normal' if p['gait_class'] == 'normal' else 'pathological'
            z_dict = self.compute_z_score(p['features'])

            # Choose Z-score type
            z_score = z_dict['z_weighted'] if use_weighted else z_dict['z_equal']
            pred_label = 'pathological' if z_score > threshold else 'normal'

            true_labels.append(true_label)
            pred_labels.append(pred_label)
            z_scores.append(z_score)

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
            'method': 'weighted' if use_weighted else 'equal',
            'threshold': threshold,
            'accuracy': float(accuracy),
            'sensitivity': float(sensitivity),
            'specificity': float(specificity),
            'tp': int(tp),
            'tn': int(tn),
            'fp': int(fp),
            'fn': int(fn),
            'mean_z_normal': float(np.mean([z for z, label in zip(z_scores, true_labels) if label == 'normal'])),
            'mean_z_path': float(np.mean([z for z, label in zip(z_scores, true_labels) if label == 'pathological']))
        }

    def optimize(self):
        """Compare equal vs weighted Z-score"""

        print(f"\n{'='*80}")
        print("COMPARISON: Equal vs Weighted Z-score")
        print(f"{'='*80}")

        thresholds = [1.0, 1.25, 1.5, 1.75, 2.0]

        print(f"\n{'Threshold':<12} {'Method':<12} {'Accuracy':<12} {'Sensitivity':<12} {'Specificity':<12}")
        print("-"*80)

        best_equal = None
        best_weighted = None

        for threshold in thresholds:
            # Equal weighting
            result_equal = self.evaluate(threshold, use_weighted=False)
            print(f"{threshold:<12.2f} {'Equal':<12} {result_equal['accuracy']*100:<12.1f} "
                  f"{result_equal['sensitivity']*100:<12.1f} {result_equal['specificity']*100:<12.1f}")

            if best_equal is None or result_equal['accuracy'] > best_equal['accuracy']:
                best_equal = result_equal

            # Weighted
            result_weighted = self.evaluate(threshold, use_weighted=True)
            print(f"{threshold:<12.2f} {'Weighted':<12} {result_weighted['accuracy']*100:<12.1f} "
                  f"{result_weighted['sensitivity']*100:<12.1f} {result_weighted['specificity']*100:<12.1f}")

            if best_weighted is None or result_weighted['accuracy'] > best_weighted['accuracy']:
                best_weighted = result_weighted

        print(f"\n{'='*80}")
        print("BEST RESULTS")
        print(f"{'='*80}")

        print(f"\n📊 Equal Weighting (v2 baseline):")
        print(f"   Threshold: {best_equal['threshold']}")
        print(f"   Accuracy: {best_equal['accuracy']*100:.1f}%")
        print(f"   Sensitivity: {best_equal['sensitivity']*100:.1f}%")
        print(f"   Specificity: {best_equal['specificity']*100:.1f}%")

        print(f"\n📊 Weighted by Cohen's d (v1 improved):")
        print(f"   Threshold: {best_weighted['threshold']}")
        print(f"   Accuracy: {best_weighted['accuracy']*100:.1f}%")
        print(f"   Sensitivity: {best_weighted['sensitivity']*100:.1f}%")
        print(f"   Specificity: {best_weighted['specificity']*100:.1f}%")

        improvement = (best_weighted['accuracy'] - best_equal['accuracy']) * 100
        print(f"\n{'='*80}")
        print(f"IMPROVEMENT: {improvement:+.1f}%")
        print(f"{'='*80}")

        if improvement > 0:
            print(f"\n✅ Weighted Z-score IMPROVES performance by {improvement:.1f}%!")
        elif improvement < 0:
            print(f"\n❌ Weighted Z-score DECREASES performance by {abs(improvement):.1f}%")
        else:
            print(f"\n⚠️  No improvement. Equal weighting is sufficient.")

        # Save results
        results = {
            'equal_best': best_equal,
            'weighted_best': best_weighted,
            'improvement': improvement
        }

        with open('improved_detector_v1_results.json', 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\n✅ Results saved to: improved_detector_v1_results.json")

        return results


def main():
    """Main evaluation"""

    print("="*80)
    print("IMPROVED DETECTOR V1: Feature-Weighted Z-score")
    print("="*80)
    print()
    print("Hypothesis: Weighting features by Cohen's d improves performance")
    print("  • Cadence (d=0.85) → weight=0.50")
    print("  • Variability (d=0.35) → weight=0.20")
    print("  • Irregularity (d=0.51) → weight=0.30")
    print()

    detector = ImprovedDetectorV1('gavd_real_patterns_fixed.json')
    results = detector.optimize()

    print("\n" + "="*80)
    print("CONCLUSION")
    print("="*80)

    if results['improvement'] > 2.0:
        print("\n✅ SUCCESS! Feature weighting significantly improves performance.")
        print(f"   Gain: {results['improvement']:.1f}%")
        print(f"\n   Recommendation: Deploy v1 (weighted) instead of v2 (equal)")
    elif results['improvement'] > 0:
        print("\n⚠️  MARGINAL improvement. Feature weighting helps slightly.")
        print(f"   Gain: {results['improvement']:.1f}%")
        print(f"\n   Recommendation: Consider deployment if interpretability maintained")
    else:
        print("\n❌ NO improvement. Stick with v2 (equal weighting).")
        print(f"   Change: {results['improvement']:.1f}%")
        print(f"\n   Reason: Equal weighting is simpler and performs as well")

    print()


if __name__ == "__main__":
    main()
