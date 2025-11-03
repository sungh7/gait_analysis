#!/usr/bin/env python3
"""
Pathological Gait Detection with CORRECT Features
=================================================

Uses features that humans actually see:
- Cadence (step frequency)
- Variability (consistency)
- Irregularity (stride regularity)

NOT the wrong features we were using:
- Amplitude (heel height range)
- Asymmetry (L-R difference)

Author: Gait Analysis System
Version: 2.0 - CORRECTED
Date: 2025-10-30
"""

import json
import numpy as np
from scipy.signal import find_peaks
from typing import Dict, List
import warnings
warnings.filterwarnings('ignore')


def extract_correct_features(pattern: dict) -> Dict:
    """Extract features that humans can actually see"""

    heel_left = np.array(pattern['heel_height_left'])
    heel_right = np.array(pattern['heel_height_right'])
    n_frames = len(heel_left)
    fps = pattern.get('fps', 30)
    duration = n_frames / fps

    features = {}

    # 1. CADENCE (steps per minute) - humans see speed!
    peaks_left, _ = find_peaks(heel_left, height=np.mean(heel_left), distance=5)
    peaks_right, _ = find_peaks(heel_right, height=np.mean(heel_right), distance=5)
    n_steps = len(peaks_left) + len(peaks_right)
    features['cadence'] = (n_steps / duration) * 60 if duration > 0 else 0

    # 2. VARIABILITY (consistency) - humans see shakiness!
    if len(peaks_left) > 1:
        peak_heights = heel_left[peaks_left]
        features['variability_left'] = np.std(peak_heights) / (np.mean(peak_heights) + 1e-6)
    else:
        features['variability_left'] = 0

    if len(peaks_right) > 1:
        peak_heights = heel_right[peaks_right]
        features['variability_right'] = np.std(peak_heights) / (np.mean(peak_heights) + 1e-6)
    else:
        features['variability_right'] = 0

    features['variability_avg'] = (features['variability_left'] + features['variability_right']) / 2

    # 3. IRREGULARITY (rhythm consistency) - humans see stumbling!
    if len(peaks_left) > 2:
        intervals = np.diff(peaks_left)
        features['irregularity_left'] = np.std(intervals) / (np.mean(intervals) + 1e-6)
    else:
        features['irregularity_left'] = 0

    if len(peaks_right) > 2:
        intervals = np.diff(peaks_right)
        features['irregularity_right'] = np.std(intervals) / (np.mean(intervals) + 1e-6)
    else:
        features['irregularity_right'] = 0

    features['irregularity_avg'] = (features['irregularity_left'] + features['irregularity_right']) / 2

    return features


def detect_with_correct_features(pattern: dict, normal_stats: Dict, path_stats: Dict) -> str:
    """Detect using correct features (Z-score based)"""

    features = extract_correct_features(pattern)

    # Z-scores relative to normal
    z_cadence_normal = abs(features['cadence'] - normal_stats['cadence_mean']) / (normal_stats['cadence_std'] + 1e-6)
    z_var_normal = abs(features['variability_avg'] - normal_stats['var_mean']) / (normal_stats['var_std'] + 1e-6)
    z_irreg_normal = abs(features['irregularity_avg'] - normal_stats['irreg_mean']) / (normal_stats['irreg_std'] + 1e-6)

    score_normal = (z_cadence_normal + z_var_normal + z_irreg_normal) / 3

    # Z-scores relative to pathological
    z_cadence_path = abs(features['cadence'] - path_stats['cadence_mean']) / (path_stats['cadence_std'] + 1e-6)
    z_var_path = abs(features['variability_avg'] - path_stats['var_mean']) / (path_stats['var_std'] + 1e-6)
    z_irreg_path = abs(features['irregularity_avg'] - path_stats['irreg_mean']) / (path_stats['irreg_std'] + 1e-6)

    score_path = (z_cadence_path + z_var_path + z_irreg_path) / 3

    return 'normal' if score_normal < score_path else 'pathological'


def main():
    """Main evaluation with CORRECT features"""

    print("="*80)
    print("PATHOLOGICAL GAIT DETECTION - WITH CORRECT FEATURES")
    print("="*80)
    print()

    # Load patterns
    with open('gavd_real_patterns.json', 'r') as f:
        all_patterns = json.load(f)

    # Filter valid patterns (exclude prosthetic/exercise)
    patterns = [p for p in all_patterns
                if p['heel_height_left'] and p['heel_height_right']
                and len(p['heel_height_left']) > 10
                and p['gait_class'] not in ['prosthetic', 'exercise']]

    print(f"Loaded {len(patterns)} patterns (excluded prosthetic/exercise)")

    # Extract features for all
    print("\nExtracting correct features...")
    for p in patterns:
        p['correct_features'] = extract_correct_features(p)

    # Separate normal vs pathological
    normal_patterns = [p for p in patterns if p['gait_class'] == 'normal']
    path_patterns = [p for p in patterns if p['gait_class'] in ['abnormal', 'stroke', 'cerebral palsy', 'antalgic']]

    # Remove outliers and compute statistics
    def get_clean_values(patterns_list, feature_key):
        vals = [p['correct_features'][feature_key] for p in patterns_list]
        mean_val = np.mean(vals)
        std_val = np.std(vals)
        # Remove values > 3 std
        clean_vals = [v for v in vals if abs(v - mean_val) < 3*std_val]
        return clean_vals

    normal_cadence = get_clean_values(normal_patterns, 'cadence')
    normal_var = get_clean_values(normal_patterns, 'variability_avg')
    normal_irreg = get_clean_values(normal_patterns, 'irregularity_avg')

    path_cadence = get_clean_values(path_patterns, 'cadence')
    path_var = get_clean_values(path_patterns, 'variability_avg')
    path_irreg = get_clean_values(path_patterns, 'irregularity_avg')

    normal_stats = {
        'cadence_mean': np.mean(normal_cadence),
        'cadence_std': np.std(normal_cadence),
        'var_mean': np.mean(normal_var),
        'var_std': np.std(normal_var),
        'irreg_mean': np.mean(normal_irreg),
        'irreg_std': np.std(normal_irreg)
    }

    path_stats = {
        'cadence_mean': np.mean(path_cadence),
        'cadence_std': np.std(path_cadence),
        'var_mean': np.mean(path_var),
        'var_std': np.std(path_var),
        'irreg_mean': np.mean(path_irreg),
        'irreg_std': np.std(path_irreg)
    }

    print(f"\n{'='*80}")
    print("FEATURE STATISTICS")
    print(f"{'='*80}")
    print(f"\nNORMAL (n={len(normal_patterns)}):")
    print(f"  Cadence: {normal_stats['cadence_mean']:.1f} ± {normal_stats['cadence_std']:.1f} steps/min")
    print(f"  Variability: {normal_stats['var_mean']:.3f} ± {normal_stats['var_std']:.3f}")
    print(f"  Irregularity: {normal_stats['irreg_mean']:.3f} ± {normal_stats['irreg_std']:.3f}")

    print(f"\nPATHOLOGICAL (n={len(path_patterns)}):")
    print(f"  Cadence: {path_stats['cadence_mean']:.1f} ± {path_stats['cadence_std']:.1f} steps/min")
    print(f"  Variability: {path_stats['var_mean']:.3f} ± {path_stats['var_std']:.3f}")
    print(f"  Irregularity: {path_stats['irreg_mean']:.3f} ± {path_stats['irreg_std']:.3f}")

    # Evaluate
    print(f"\n{'='*80}")
    print("EVALUATION RESULTS")
    print(f"{'='*80}")

    true_labels = []
    pred_labels = []

    for p in patterns:
        true_label = 'normal' if p['gait_class'] == 'normal' else 'pathological'
        pred_label = detect_with_correct_features(p, normal_stats, path_stats)

        true_labels.append(true_label)
        pred_labels.append(pred_label)

    true_labels = np.array(true_labels)
    pred_labels = np.array(pred_labels)

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

    print(f"\nResults on {len(patterns)} samples:")
    print(f"  Accuracy: {accuracy*100:.1f}%")
    print(f"  Sensitivity: {sensitivity*100:.1f}%")
    print(f"  Specificity: {specificity*100:.1f}%")
    print(f"\nConfusion Matrix:")
    print(f"  TP (pathological detected): {tp}")
    print(f"  TN (normal detected): {tn}")
    print(f"  FP (normal → pathological): {fp}")
    print(f"  FN (pathological → normal): {fn}")

    # Save results
    results = {
        'method': 'correct_features',
        'features': ['cadence', 'variability', 'irregularity'],
        'accuracy': float(accuracy),
        'sensitivity': float(sensitivity),
        'specificity': float(specificity),
        'tp': int(tp),
        'tn': int(tn),
        'fp': int(fp),
        'fn': int(fn),
        'n_samples': len(patterns)
    }

    with open('correct_features_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*80}")
    print("COMPARISON")
    print(f"{'='*80}")
    print(f"\nWRONG features (amplitude, asymmetry): 57.0%")
    print(f"CORRECT features (cadence, variability, irregularity): {accuracy*100:.1f}%")
    print(f"IMPROVEMENT: +{(accuracy - 0.57)*100:.1f}%")

    if accuracy > 0.75:
        print(f"\n✅ SUCCESS! We CAN detect pathological gait!")
        print(f"   Problem was: we were measuring the WRONG features.")
    else:
        print(f"\n⚠️  Still not great. Need more investigation.")

    print(f"\nResults saved to: correct_features_results.json")
    print()


if __name__ == "__main__":
    main()
