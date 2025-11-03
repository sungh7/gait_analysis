#!/usr/bin/env python3
"""
Improved Detector v2: Adding Stride Length Feature
===================================================

Improvement: Add stride length (hip-ankle distance) as 4th feature

Expected Cohen's d: 0.9-1.1 (LARGE effect)
Expected improvement: +3-5% (79-81% → 82-85%)

New feature:
- Stride Length: Distance from hip to ankle at heel strike
- Normal: 0.8-1.2m (longer strides)
- Pathological: 0.4-0.8m (shorter strides, shuffling)
"""

import json
import numpy as np
from scipy.signal import find_peaks
from scipy.stats import pearsonr
from typing import Dict, Tuple
from dataclasses import dataclass


@dataclass
class EnhancedGaitFeatures:
    cadence: float
    variability_avg: float
    irregularity_avg: float
    stride_length_avg: float  # NEW!


class ImprovedDetectorV2:
    """Detector with stride length feature"""

    def __init__(self, patterns_file: str = 'gavd_real_patterns_fixed.json'):
        with open(patterns_file, 'r') as f:
            all_patterns = json.load(f)

        # Filter
        self.patterns = [p for p in all_patterns
                        if p['heel_height_left'] and p['heel_height_right']
                        and len(p['heel_height_left']) > 10
                        and p['gait_class'] not in ['prosthetic', 'exercise']]

        print(f"✅ Loaded {len(self.patterns)} patterns")

        # Extract features (including stride length)
        print("\n📏 Computing stride length for all patterns...")
        for p in self.patterns:
            p['features'] = self._extract_features(p)

        # Check if stride length is computable
        valid_stride = [p for p in self.patterns if p['features'].stride_length_avg > 0]
        print(f"   ✅ {len(valid_stride)}/{len(self.patterns)} patterns have valid stride length")

        # Build baseline
        self._build_baseline()

        # Assess feature quality
        self._assess_feature_quality()

    def _extract_features(self, pattern: dict) -> EnhancedGaitFeatures:
        """Extract 4 features (3 core + stride length)"""

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

        # === NEW FEATURE: Stride Length ===
        # Note: In MediaPipe, normalized coordinates [0,1]
        # Stride length = average distance from peak to valley
        # (approximation: heel height range as proxy for stride length)

        def compute_stride_length_proxy(heel, peaks):
            """
            Stride length proxy: amplitude of heel movement

            In side view:
            - Larger amplitude = longer stride (heel goes higher in swing phase)
            - Smaller amplitude = shorter stride (shuffling gait)
            """
            if len(peaks) > 1:
                # Find valleys (minimum heel height)
                valleys, _ = find_peaks(-heel, distance=5)

                if len(valleys) > 0:
                    # Stride length ≈ vertical heel displacement
                    amplitudes = []
                    for i in range(min(len(peaks), len(valleys))):
                        amplitude = heel[peaks[i]] - heel[valleys[i]]
                        amplitudes.append(amplitude)

                    return np.mean(amplitudes)

            # Fallback: use range
            return np.max(heel) - np.min(heel)

        stride_left = compute_stride_length_proxy(heel_left, peaks_left)
        stride_right = compute_stride_length_proxy(heel_right, peaks_right)
        stride_avg = (stride_left + stride_right) / 2

        return EnhancedGaitFeatures(
            cadence=cadence,
            variability_avg=var_avg,
            irregularity_avg=irreg_avg,
            stride_length_avg=stride_avg
        )

    def _build_baseline(self):
        """Build baseline from normal patterns"""

        normal_patterns = [p for p in self.patterns if p['gait_class'] == 'normal']
        print(f"\n✅ Building baseline from {len(normal_patterns)} normal patterns")

        # Extract feature values
        features_dict = {
            'cadence': [p['features'].cadence for p in normal_patterns],
            'variability': [p['features'].variability_avg for p in normal_patterns],
            'irregularity': [p['features'].irregularity_avg for p in normal_patterns],
            'stride_length': [p['features'].stride_length_avg for p in normal_patterns]
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
        print(f"   Stride Length: {self.baseline['stride_length_mean']:.3f} ± {self.baseline['stride_length_std']:.3f}")

    def _assess_feature_quality(self):
        """Compute Cohen's d for new feature"""

        print(f"\n{'='*80}")
        print("FEATURE QUALITY ASSESSMENT")
        print(f"{'='*80}")

        normal = [p for p in self.patterns if p['gait_class'] == 'normal']
        pathological = [p for p in self.patterns if p['gait_class'] != 'normal']

        features_to_check = [
            ('cadence', lambda p: p['features'].cadence),
            ('variability', lambda p: p['features'].variability_avg),
            ('irregularity', lambda p: p['features'].irregularity_avg),
            ('stride_length', lambda p: p['features'].stride_length_avg)
        ]

        print(f"\n{'Feature':<20} {'Normal Mean':<15} {'Path Mean':<15} {'Cohen d':<10} {'Quality':<10}")
        print("-"*80)

        for name, extractor in features_to_check:
            normal_vals = [extractor(p) for p in normal]
            path_vals = [extractor(p) for p in pathological]

            # Remove outliers
            def remove_outliers(vals):
                mean_val = np.mean(vals)
                std_val = np.std(vals)
                return [v for v in vals if abs(v - mean_val) < 3*std_val]

            normal_clean = remove_outliers(normal_vals)
            path_clean = remove_outliers(path_vals)

            # Cohen's d
            mean1, mean2 = np.mean(normal_clean), np.mean(path_clean)
            std1, std2 = np.std(normal_clean), np.std(path_clean)
            pooled_std = np.sqrt((std1**2 + std2**2) / 2)
            d = abs(mean1 - mean2) / (pooled_std + 1e-6)

            quality = "✅ LARGE" if d > 0.8 else ("⚠️ MEDIUM" if d > 0.5 else "❌ SMALL")

            print(f"{name:<20} {mean1:<15.2f} {mean2:<15.2f} {d:<10.2f} {quality:<10}")

            # Store stride length Cohen's d
            if name == 'stride_length':
                self.stride_cohens_d = d

        print(f"\n💡 KEY FINDING:")
        if self.stride_cohens_d > 0.8:
            print(f"   ✅ Stride length has LARGE effect size (d={self.stride_cohens_d:.2f})")
            print(f"      → Expected to improve performance!")
        elif self.stride_cohens_d > 0.5:
            print(f"   ⚠️ Stride length has MEDIUM effect size (d={self.stride_cohens_d:.2f})")
            print(f"      → May provide some improvement")
        else:
            print(f"   ❌ Stride length has SMALL effect size (d={self.stride_cohens_d:.2f})")
            print(f"      → Unlikely to improve performance")

    def compute_z_score(self, features: EnhancedGaitFeatures, use_stride: bool = True) -> float:
        """Compute Z-score (with or without stride length)"""

        # Core 3 features
        z_cadence = abs(features.cadence - self.baseline['cadence_mean']) / (self.baseline['cadence_std'] + 1e-6)
        z_var = abs(features.variability_avg - self.baseline['variability_mean']) / (self.baseline['variability_std'] + 1e-6)
        z_irreg = abs(features.irregularity_avg - self.baseline['irregularity_mean']) / (self.baseline['irregularity_std'] + 1e-6)

        if use_stride:
            # Add stride length
            z_stride = abs(features.stride_length_avg - self.baseline['stride_length_mean']) / (self.baseline['stride_length_std'] + 1e-6)
            Z = (z_cadence + z_var + z_irreg + z_stride) / 4
        else:
            # Without stride
            Z = (z_cadence + z_var + z_irreg) / 3

        return Z

    def evaluate(self, threshold: float = 1.5, use_stride: bool = True) -> Dict:
        """Evaluate detector"""

        true_labels = []
        pred_labels = []

        for p in self.patterns:
            true_label = 'normal' if p['gait_class'] == 'normal' else 'pathological'
            z_score = self.compute_z_score(p['features'], use_stride=use_stride)
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
            'method': '4-features' if use_stride else '3-features',
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
        """Compare 3 features vs 4 features"""

        print(f"\n{'='*80}")
        print("COMPARISON: 3 Features vs 4 Features (+ Stride Length)")
        print(f"{'='*80}")

        thresholds = [1.0, 1.25, 1.5, 1.75, 2.0]

        print(f"\n{'Threshold':<12} {'Method':<15} {'Accuracy':<12} {'Sensitivity':<12} {'Specificity':<12}")
        print("-"*80)

        best_3feat = None
        best_4feat = None

        for threshold in thresholds:
            # 3 features (baseline)
            result_3 = self.evaluate(threshold, use_stride=False)
            print(f"{threshold:<12.2f} {'3-feat':<15} {result_3['accuracy']*100:<12.1f} "
                  f"{result_3['sensitivity']*100:<12.1f} {result_3['specificity']*100:<12.1f}")

            if best_3feat is None or result_3['accuracy'] > best_3feat['accuracy']:
                best_3feat = result_3

            # 4 features (with stride)
            result_4 = self.evaluate(threshold, use_stride=True)
            print(f"{threshold:<12.2f} {'4-feat (stride)':<15} {result_4['accuracy']*100:<12.1f} "
                  f"{result_4['sensitivity']*100:<12.1f} {result_4['specificity']*100:<12.1f}")

            if best_4feat is None or result_4['accuracy'] > best_4feat['accuracy']:
                best_4feat = result_4

        print(f"\n{'='*80}")
        print("BEST RESULTS")
        print(f"{'='*80}")

        print(f"\n📊 3 Features (baseline):")
        print(f"   Threshold: {best_3feat['threshold']}")
        print(f"   Accuracy: {best_3feat['accuracy']*100:.1f}%")
        print(f"   Sensitivity: {best_3feat['sensitivity']*100:.1f}%")
        print(f"   Specificity: {best_3feat['specificity']*100:.1f}%")

        print(f"\n📊 4 Features (+ stride length):")
        print(f"   Threshold: {best_4feat['threshold']}")
        print(f"   Accuracy: {best_4feat['accuracy']*100:.1f}%")
        print(f"   Sensitivity: {best_4feat['sensitivity']*100:.1f}%")
        print(f"   Specificity: {best_4feat['specificity']*100:.1f}%")

        improvement = (best_4feat['accuracy'] - best_3feat['accuracy']) * 100
        print(f"\n{'='*80}")
        print(f"IMPROVEMENT: {improvement:+.1f}%")
        print(f"Stride Length Cohen's d: {self.stride_cohens_d:.2f}")
        print(f"{'='*80}")

        if improvement > 2.0:
            print(f"\n✅ SUCCESS! Stride length SIGNIFICANTLY improves performance by {improvement:.1f}%!")
        elif improvement > 0:
            print(f"\n⚠️  Stride length provides MARGINAL improvement ({improvement:.1f}%)")
        else:
            print(f"\n❌ Stride length does NOT improve performance ({improvement:.1f}%)")

        # Save results
        results = {
            '3feat_best': best_3feat,
            '4feat_best': best_4feat,
            'improvement': improvement,
            'stride_cohens_d': self.stride_cohens_d
        }

        with open('improved_detector_v2_results.json', 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\n✅ Results saved to: improved_detector_v2_results.json")

        return results


def main():
    """Main evaluation"""

    print("="*80)
    print("IMPROVED DETECTOR V2: Adding Stride Length Feature")
    print("="*80)
    print()
    print("Hypothesis: Stride length has large effect size (d>0.8)")
    print("  → Should improve detection performance")
    print()
    print("Stride length = vertical heel displacement")
    print("  • Normal: Large amplitude (longer strides)")
    print("  • Pathological: Small amplitude (shorter strides, shuffling)")
    print()

    detector = ImprovedDetectorV2('gavd_real_patterns_fixed.json')
    results = detector.optimize()

    print("\n" + "="*80)
    print("CONCLUSION")
    print("="*80)

    if results['stride_cohens_d'] > 0.8 and results['improvement'] > 2.0:
        print("\n✅ HYPOTHESIS CONFIRMED!")
        print(f"   Stride length has LARGE effect size (d={results['stride_cohens_d']:.2f})")
        print(f"   Performance improves by {results['improvement']:.1f}%")
        print(f"\n   Recommendation: Add stride length to production system")
    elif results['stride_cohens_d'] > 0.8 and results['improvement'] > 0:
        print("\n⚠️  PARTIAL SUCCESS")
        print(f"   Stride length has LARGE effect size (d={results['stride_cohens_d']:.2f})")
        print(f"   But improvement is small ({results['improvement']:.1f}%)")
        print(f"\n   Possible reason: Correlation with existing features")
    elif results['stride_cohens_d'] < 0.8:
        print("\n❌ HYPOTHESIS REJECTED")
        print(f"   Stride length has SMALL/MEDIUM effect size (d={results['stride_cohens_d']:.2f})")
        print(f"   Performance change: {results['improvement']:.1f}%")
        print(f"\n   Reason: Proxy metric (heel height amplitude) != true stride length")
        print(f"   Need full 3D hip-ankle distance for accurate measurement")
    else:
        print("\n❌ NO IMPROVEMENT despite good Cohen's d")
        print(f"   Stride length d={results['stride_cohens_d']:.2f}, but {results['improvement']:.1f}% change")
        print(f"\n   Possible reason: Feature correlation dilutes signal (same as v3)")

    print()


if __name__ == "__main__":
    main()
