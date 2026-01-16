#!/usr/bin/env python3
"""
Ablation Study: Performance with Feature Subsets
=================================================

Evaluates model performance with varying numbers of features (top-3, 5, 7, 10)
to understand feature contribution and model parsimony.

Author: Gait Analysis System
Date: 2026-01-15
"""

import json
import numpy as np
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import accuracy_score, recall_score, roc_auc_score


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


def run_ablation_study(patterns_file, k_values=None, output_file=None):
    """Run ablation study with varying feature counts."""
    if k_values is None:
        k_values = [3, 5, 7, 10]

    print("="*70)
    print("Ablation Study: Performance vs Feature Count")
    print("="*70)
    print()

    X, y, feature_names = load_data(patterns_file)

    print(f"Total samples: {len(X)}")
    print(f"Class distribution: {sum(y)} pathological, {len(y)-sum(y)} normal")
    print()

    # Get feature importance from full model
    full_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42))
    ])
    full_pipeline.fit(X, y)
    importance = np.abs(full_pipeline.named_steps['clf'].coef_[0])
    sorted_indices = np.argsort(-importance)

    print("Feature Importance Ranking (by |coefficient|):")
    print("-"*50)
    for rank, idx in enumerate(sorted_indices, 1):
        print(f"  {rank:2}. {feature_names[idx]:<25} ({importance[idx]:.4f})")
    print()

    # Run ablation for each k
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    results = {
        'feature_ranking': [feature_names[i] for i in sorted_indices],
        'ablation_results': []
    }

    print("Ablation Study Results:")
    print("="*70)
    print(f"{'Features':>10} {'Accuracy':>10} {'Sensitivity':>12} "
          f"{'Specificity':>12} {'AUC':>8}")
    print("-"*70)

    for k in k_values:
        top_k_indices = sorted_indices[:k]
        X_subset = X[:, top_k_indices]
        top_k_names = [feature_names[i] for i in top_k_indices]

        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('clf', LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42))
        ])

        # Cross-validated predictions
        y_pred = cross_val_predict(pipeline, X_subset, y, cv=cv)

        # For AUC, we need probability predictions
        y_prob = np.zeros(len(y))
        for train_idx, test_idx in cv.split(X_subset, y):
            pipeline.fit(X_subset[train_idx], y[train_idx])
            y_prob[test_idx] = pipeline.predict_proba(X_subset[test_idx])[:, 1]

        acc = accuracy_score(y, y_pred)
        sens = recall_score(y, y_pred, pos_label=1)
        spec = recall_score(y, y_pred, pos_label=0)
        auc = roc_auc_score(y, y_prob)

        print(f"Top {k:>2}     {acc*100:>9.1f}%  {sens*100:>11.1f}%  "
              f"{spec*100:>11.1f}%  {auc:>7.3f}")

        results['ablation_results'].append({
            'k': k,
            'features': top_k_names,
            'accuracy': round(acc * 100, 1),
            'sensitivity': round(sens * 100, 1),
            'specificity': round(spec * 100, 1),
            'auc': round(auc, 3)
        })

    print()

    # Performance drop analysis
    print("Performance Drop Analysis:")
    print("-"*70)
    full_acc = results['ablation_results'][-1]['accuracy']
    full_sens = results['ablation_results'][-1]['sensitivity']

    for r in results['ablation_results'][:-1]:
        acc_drop = full_acc - r['accuracy']
        sens_drop = full_sens - r['sensitivity']
        print(f"Top {r['k']:>2} vs Full: Accuracy -{acc_drop:.1f}%, Sensitivity -{sens_drop:.1f}%")

    print()

    # Key findings
    print("Key Findings:")
    print("-"*70)

    # Find minimal k with <5% accuracy drop
    for r in results['ablation_results']:
        if full_acc - r['accuracy'] < 5:
            print(f"Minimal feature set with <5% accuracy drop: Top {r['k']} features")
            print(f"  Features: {', '.join(r['features'])}")
            break

    # Top 3 features
    top3 = results['ablation_results'][0]
    print(f"\nTop 3 features achieve {top3['accuracy']}% accuracy "
          f"({top3['sensitivity']}% sensitivity)")
    print(f"  - {top3['features'][0]}")
    print(f"  - {top3['features'][1]}")
    print(f"  - {top3['features'][2]}")

    # Save results
    if output_file:
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {output_file}")

    return results


if __name__ == "__main__":
    patterns_file = "/data/gait/output/results/root_dumps/gavd_patterns_with_v7_features.json"
    output_file = "/data/gait/docs/papers/figures/ablation_results.json"

    results = run_ablation_study(patterns_file, output_file=output_file)
