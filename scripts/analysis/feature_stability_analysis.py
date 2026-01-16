#!/usr/bin/env python3
"""
Feature Stability Analysis
==========================

Analyzes coefficient stability across CV folds to identify which features
are consistently important vs. those with high variance.

Outputs:
- Coefficient matrix (fold x feature)
- Rank stability scores
- Feature importance with uncertainty

Author: Gait Analysis System
Date: 2026-01-15
"""

import json
import numpy as np
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression


def load_data(patterns_file):
    """Load patterns and prepare features."""
    with open(patterns_file, 'r') as f:
        patterns = json.load(f)

    feature_names = [
        'cadence_3d', 'step_height_variability', 'gait_irregularity_3d',
        'velocity_3d', 'jerkiness_3d', 'cycle_duration_3d',
        'stride_length_3d', 'trunk_sway', 'path_length_3d', 'step_width_3d'
    ]

    X, y = [], []
    for p in patterns:
        features = [p.get(f, 0.0) for f in feature_names]
        if all(np.isfinite(features)):
            X.append(features)
            y.append(1 if p.get('gait_class') == 'pathological' else 0)

    return np.array(X), np.array(y), feature_names


def analyze_feature_stability(patterns_file, n_folds=5, output_file=None):
    """Analyze feature coefficient stability across CV folds."""
    print("="*70)
    print("Feature Stability Analysis")
    print("="*70)
    print()

    X, y, feature_names = load_data(patterns_file)
    n_features = len(feature_names)

    print(f"Total samples: {len(X)}")
    print(f"Number of features: {n_features}")
    print(f"Cross-validation folds: {n_folds}")
    print()

    # Collect coefficients from each fold
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    coef_matrix = []
    fold_accs = []

    for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X, y)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        clf = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
        clf.fit(X_train_scaled, y_train)

        coef_matrix.append(clf.coef_[0])
        fold_accs.append(clf.score(X_test_scaled, y_test))

    coef_matrix = np.array(coef_matrix)

    # Compute rank matrix (1 = most important, n = least important)
    rank_matrix = np.zeros_like(coef_matrix, dtype=int)
    for fold_idx in range(n_folds):
        abs_coefs = np.abs(coef_matrix[fold_idx])
        rank_matrix[fold_idx] = np.argsort(np.argsort(-abs_coefs)) + 1

    # Compute stability metrics
    mean_abs_coef = np.mean(np.abs(coef_matrix), axis=0)
    std_coef = np.std(coef_matrix, axis=0)
    mean_coef = np.mean(coef_matrix, axis=0)

    # Rank stability: lower std in ranks = more stable
    rank_std = np.std(rank_matrix, axis=0)
    rank_mean = np.mean(rank_matrix, axis=0)

    # Stability score: coefficient of variation (lower = more stable)
    cv_score = std_coef / (np.abs(mean_coef) + 1e-10)

    # Stability categories
    def get_stability(cv_val, rank_range):
        if cv_val < 0.3 and rank_range <= 2:
            return 'High'
        elif cv_val < 0.5 and rank_range <= 4:
            return 'Moderate'
        else:
            return 'Low'

    # Sort by mean absolute coefficient
    sorted_indices = np.argsort(-mean_abs_coef)

    print("Fold Accuracies:", [f"{acc*100:.1f}%" for acc in fold_accs])
    print(f"Mean CV Accuracy: {np.mean(fold_accs)*100:.1f}% (+/- {np.std(fold_accs)*100:.1f}%)")
    print()

    print("Feature Stability Analysis Results:")
    print("="*80)
    print(f"{'Rank':<5} {'Feature':<25} {'Mean |B|':>10} {'SD':>8} "
          f"{'Rank Range':>12} {'Stability':>10}")
    print("-"*80)

    results = {'features': [], 'coef_matrix': coef_matrix.tolist()}

    for rank, idx in enumerate(sorted_indices, 1):
        rank_min = rank_matrix[:, idx].min()
        rank_max = rank_matrix[:, idx].max()
        rank_range = rank_max - rank_min

        stability = get_stability(cv_score[idx], rank_range)

        print(f"{rank:<5} {feature_names[idx]:<25} {mean_abs_coef[idx]:>10.3f} "
              f"{std_coef[idx]:>8.3f} {str(rank_min)+'-'+str(rank_max):>10} {stability:>10}")

        results['features'].append({
            'name': feature_names[idx],
            'rank': rank,
            'mean_abs_coef': round(mean_abs_coef[idx], 4),
            'mean_coef': round(mean_coef[idx], 4),
            'std_coef': round(std_coef[idx], 4),
            'rank_min': int(rank_min),
            'rank_max': int(rank_max),
            'stability': stability
        })

    print()
    print("Coefficient Matrix (Folds x Features):")
    print("-"*80)

    # Print header
    short_names = ['Cadence', 'StepHtVar', 'GaitIrreg', 'Velocity', 'Jerkiness',
                   'CycleDur', 'StrideLn', 'TrunkSway', 'PathLen', 'StepWidth']
    print(f"{'Fold':<8}", end='')
    for name in short_names:
        print(f"{name:>9}", end='')
    print()

    for fold_idx in range(n_folds):
        print(f"Fold {fold_idx+1:<3}", end='')
        for coef in coef_matrix[fold_idx]:
            print(f"{coef:>9.2f}", end='')
        print()

    # Summary statistics
    print()
    print("Summary:")
    print("-"*80)
    high_stability = sum(1 for f in results['features'] if f['stability'] == 'High')
    mod_stability = sum(1 for f in results['features'] if f['stability'] == 'Moderate')
    low_stability = sum(1 for f in results['features'] if f['stability'] == 'Low')

    print(f"High stability features: {high_stability}")
    print(f"Moderate stability features: {mod_stability}")
    print(f"Low stability features: {low_stability}")

    top3 = [results['features'][i]['name'] for i in range(3)]
    print(f"Top 3 most important: {', '.join(top3)}")

    results['summary'] = {
        'high_stability_count': high_stability,
        'moderate_stability_count': mod_stability,
        'low_stability_count': low_stability,
        'top3_features': top3,
        'mean_cv_accuracy': round(np.mean(fold_accs) * 100, 1),
        'std_cv_accuracy': round(np.std(fold_accs) * 100, 1)
    }

    # Save results
    if output_file:
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {output_file}")

    return results


if __name__ == "__main__":
    patterns_file = "/data/gait/output/results/root_dumps/gavd_patterns_with_v7_features.json"
    output_file = "/data/gait/docs/papers/figures/feature_stability_results.json"

    results = analyze_feature_stability(patterns_file, output_file=output_file)
