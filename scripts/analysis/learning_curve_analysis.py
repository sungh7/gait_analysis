#!/usr/bin/env python3
"""
Learning Curve Analysis
=======================

Shows how classification performance changes with training set size.
Helps identify if more data would improve performance (high variance)
or if model has reached its capacity (high bias).

Author: Gait Analysis System
Date: 2026-01-15
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import learning_curve, StratifiedKFold


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


def generate_learning_curves(patterns_file, output_dir):
    """Generate learning curves showing accuracy vs sample size."""
    print("="*70)
    print("Learning Curve Analysis")
    print("="*70)
    print()

    X, y, feature_names = load_data(patterns_file)

    print(f"Total samples: {len(X)}")
    print(f"Class distribution: {sum(y)} pathological, {len(y)-sum(y)} normal")
    print()

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42))
    ])

    # Define training sizes (10% to 100%)
    train_sizes = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])

    print("Computing learning curves...")
    train_sizes_abs, train_scores, test_scores = learning_curve(
        pipeline, X, y,
        train_sizes=train_sizes,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        scoring='accuracy',
        n_jobs=-1,
        random_state=42
    )

    # Calculate statistics
    train_mean = train_scores.mean(axis=1)
    train_std = train_scores.std(axis=1)
    test_mean = test_scores.mean(axis=1)
    test_std = test_scores.std(axis=1)

    print("\nLearning Curve Results:")
    print("="*60)
    print(f"{'N Training':>12} {'Train Acc':>12} {'CV Acc':>12} {'Gap':>10}")
    print("-"*60)

    for n, train_acc, test_acc in zip(train_sizes_abs, train_mean, test_mean):
        gap = train_acc - test_acc
        print(f"{int(n):>12} {train_acc*100:>11.1f}% {test_acc*100:>11.1f}% "
              f"{gap*100:>9.1f}%")

    print()

    # Analysis
    print("Bias-Variance Analysis:")
    print("-"*60)

    # Gap between train and test at full data
    final_gap = train_mean[-1] - test_mean[-1]
    print(f"Final train-test gap: {final_gap*100:.1f}%")

    if final_gap < 0.05:
        print("  -> Low variance: Model generalizes well")
    elif final_gap < 0.10:
        print("  -> Moderate variance: Some overfitting present")
    else:
        print("  -> High variance: Significant overfitting, more data may help")

    # Improvement from 50% to 100% data
    idx_50 = np.argmin(np.abs(train_sizes - 0.5))
    improvement = test_mean[-1] - test_mean[idx_50]
    print(f"\nImprovement from 50% to 100% data: {improvement*100:+.1f}%")

    if improvement > 0.03:
        print("  -> Performance still improving with more data")
        print("  -> Collecting more samples would likely help")
    else:
        print("  -> Performance plateau reached")
        print("  -> More data unlikely to significantly improve results")

    # Create figure
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 6))

    # Plot training score
    plt.plot(train_sizes_abs, train_mean, 'o-', color='#2E86AB', linewidth=2,
             markersize=8, label='Training Score')
    plt.fill_between(train_sizes_abs,
                     train_mean - train_std,
                     train_mean + train_std,
                     alpha=0.2, color='#2E86AB')

    # Plot CV score
    plt.plot(train_sizes_abs, test_mean, 'o-', color='#E94F37', linewidth=2,
             markersize=8, label='Cross-validation Score')
    plt.fill_between(train_sizes_abs,
                     test_mean - test_std,
                     test_mean + test_std,
                     alpha=0.2, color='#E94F37')

    plt.xlabel('Training Set Size', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.title('Learning Curve: Accuracy vs. Training Set Size', fontsize=14)
    plt.legend(loc='lower right', fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.ylim(0.7, 1.0)

    # Add annotation for final performance
    plt.annotate(f'{test_mean[-1]*100:.1f}%',
                 xy=(train_sizes_abs[-1], test_mean[-1]),
                 xytext=(train_sizes_abs[-1] - 30, test_mean[-1] - 0.05),
                 fontsize=11, fontweight='bold',
                 arrowprops=dict(arrowstyle='->', color='#E94F37'))

    plt.tight_layout()
    plt.savefig(output_path / 'figure_learning_curve.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_path / 'figure_learning_curve.pdf', bbox_inches='tight')
    plt.close()

    print(f"\nFigure saved to: {output_path / 'figure_learning_curve.png'}")

    # Save numerical results
    results = {
        'train_sizes': train_sizes_abs.tolist(),
        'train_scores_mean': train_mean.tolist(),
        'train_scores_std': train_std.tolist(),
        'test_scores_mean': test_mean.tolist(),
        'test_scores_std': test_std.tolist(),
        'analysis': {
            'final_gap': round(final_gap * 100, 1),
            'improvement_50_to_100': round(improvement * 100, 1),
            'final_cv_accuracy': round(test_mean[-1] * 100, 1)
        }
    }

    results_file = output_path / 'learning_curve_results.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to: {results_file}")

    return results


if __name__ == "__main__":
    patterns_file = "/data/gait/output/results/root_dumps/gavd_patterns_with_v7_features.json"
    output_dir = "/data/gait/docs/papers/figures"

    results = generate_learning_curves(patterns_file, output_dir)
