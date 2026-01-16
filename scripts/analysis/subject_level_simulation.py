#!/usr/bin/env python3
"""
Subject-Level Validation Simulation
====================================

Since GAVD lacks subject identifiers, we simulate subject-level validation by:
1. Clustering patterns that may belong to the same subject
2. Performing Leave-One-Cluster-Out cross-validation
3. Estimating expected performance degradation

This provides a more realistic estimate of generalization performance.

Author: Gait Analysis System
Date: 2026-01-16
"""

import json
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.model_selection import cross_val_predict, LeaveOneGroupOut, StratifiedKFold
from sklearn.metrics import accuracy_score, recall_score, roc_auc_score
import warnings
warnings.filterwarnings('ignore')


def load_data(patterns_file):
    """Load patterns and prepare features."""
    with open(patterns_file, 'r') as f:
        patterns = json.load(f)

    feature_names = [
        'cadence_3d', 'step_height_variability', 'gait_irregularity_3d',
        'velocity_3d', 'jerkiness_3d', 'cycle_duration_3d',
        'stride_length_3d', 'trunk_sway', 'path_length_3d', 'step_width_3d'
    ]

    X, y, pathologies = [], [], []
    for p in patterns:
        features = [p.get(f, 0.0) for f in feature_names]
        if all(np.isfinite(features)):
            X.append(features)
            y.append(1 if p.get('gait_class') == 'pathological' else 0)
            pathologies.append(p.get('gait_pathology', 'normal'))

    return np.array(X), np.array(y), pathologies, feature_names


def create_subject_clusters(X, y, pathologies, n_clusters_per_class=None):
    """
    Cluster patterns within each class to simulate subjects.

    Assumption: Patterns from the same subject will be similar in feature space.
    We cluster within each pathology type to find potential subjects.
    """
    if n_clusters_per_class is None:
        # Estimate: assume ~2-3 patterns per subject on average
        n_clusters_per_class = {}
        for path in set(pathologies):
            count = sum(1 for p in pathologies if p == path)
            n_clusters_per_class[path] = max(2, count // 3)

    subject_ids = np.zeros(len(X), dtype=int)
    current_id = 0

    for path in set(pathologies):
        mask = np.array([p == path for p in pathologies])
        X_path = X[mask]
        indices = np.where(mask)[0]

        n_clusters = min(n_clusters_per_class.get(path, 5), len(X_path))

        if n_clusters < 2:
            # Only one sample, assign unique ID
            for idx in indices:
                subject_ids[idx] = current_id
                current_id += 1
        else:
            # Cluster using Agglomerative (captures similarity structure)
            clustering = AgglomerativeClustering(n_clusters=n_clusters)
            labels = clustering.fit_predict(X_path)

            for i, idx in enumerate(indices):
                subject_ids[idx] = current_id + labels[i]

            current_id += n_clusters

    return subject_ids


def evaluate_with_logo(X, y, subject_ids, pipeline, name="Model"):
    """Evaluate using Leave-One-Group-Out (simulated subject-level)."""
    logo = LeaveOneGroupOut()

    y_pred = np.zeros(len(y))
    y_prob = np.zeros(len(y))

    n_groups = len(np.unique(subject_ids))
    n_folds = 0

    for train_idx, test_idx in logo.split(X, y, subject_ids):
        # Skip if test set has only one class
        if len(np.unique(y[train_idx])) < 2:
            continue

        pipeline.fit(X[train_idx], y[train_idx])
        y_pred[test_idx] = pipeline.predict(X[test_idx])

        if hasattr(pipeline, 'predict_proba'):
            y_prob[test_idx] = pipeline.predict_proba(X[test_idx])[:, 1]
        n_folds += 1

    # Calculate metrics only for predicted samples
    valid_mask = y_prob != 0
    if sum(valid_mask) < len(y) * 0.8:
        # Too many skipped, use all
        valid_mask = np.ones(len(y), dtype=bool)

    acc = accuracy_score(y, y_pred)
    sens = recall_score(y, y_pred, pos_label=1)
    spec = recall_score(y, y_pred, pos_label=0)

    try:
        auc = roc_auc_score(y[valid_mask], y_prob[valid_mask])
    except:
        auc = 0.0

    return {
        'name': name,
        'n_groups': n_groups,
        'accuracy': acc,
        'sensitivity': sens,
        'specificity': spec,
        'auc': auc
    }


def main():
    print("="*70)
    print("Subject-Level Validation Simulation")
    print("="*70)
    print()

    # Load data
    patterns_file = "/data/gait/output/results/root_dumps/gavd_patterns_with_v7_features.json"
    X, y, pathologies, feature_names = load_data(patterns_file)

    print(f"Total patterns: {len(X)}")
    print(f"Normal: {sum(y==0)}, Pathological: {sum(y==1)}")
    print()

    # Standardize features for clustering
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Create subject clusters with different granularities
    print("Creating simulated subject clusters...")
    print("-"*70)

    cluster_configs = [
        ("Conservative (many subjects)", 2),   # ~2 patterns per subject
        ("Moderate", 3),                        # ~3 patterns per subject
        ("Liberal (few subjects)", 5),          # ~5 patterns per subject
    ]

    results = []

    # First, get baseline (pattern-level CV)
    print("\nBaseline (Pattern-level 5-fold CV):")
    print("-"*70)

    models = {
        'Logistic Regression': Pipeline([
            ('scaler', StandardScaler()),
            ('clf', LogisticRegression(C=1.0, class_weight='balanced', max_iter=1000, random_state=42))
        ]),
        'Random Forest': Pipeline([
            ('scaler', StandardScaler()),
            ('clf', RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42))
        ])
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for model_name, pipeline in models.items():
        y_pred = cross_val_predict(pipeline, X, y, cv=cv)

        # Get probabilities
        y_prob = np.zeros(len(y))
        for train_idx, test_idx in cv.split(X, y):
            pipeline.fit(X[train_idx], y[train_idx])
            y_prob[test_idx] = pipeline.predict_proba(X[test_idx])[:, 1]

        acc = accuracy_score(y, y_pred)
        sens = recall_score(y, y_pred, pos_label=1)
        spec = recall_score(y, y_pred, pos_label=0)
        auc = roc_auc_score(y, y_prob)

        print(f"{model_name:<25} Acc: {acc*100:.1f}% | Sens: {sens*100:.1f}% | "
              f"Spec: {spec*100:.1f}% | AUC: {auc:.3f}")

        results.append({
            'validation': 'Pattern-level',
            'model': model_name,
            'n_groups': 296,
            'accuracy': round(acc * 100, 1),
            'sensitivity': round(sens * 100, 1),
            'specificity': round(spec * 100, 1),
            'auc': round(auc, 3)
        })

    print()

    # Subject-level simulations
    for config_name, patterns_per_subject in cluster_configs:
        print(f"\n{config_name} (~{patterns_per_subject} patterns/subject):")
        print("-"*70)

        # Create clusters
        n_clusters_per_class = {}
        for path in set(pathologies):
            count = sum(1 for p in pathologies if p == path)
            n_clusters_per_class[path] = max(2, count // patterns_per_subject)

        subject_ids = create_subject_clusters(X_scaled, y, pathologies, n_clusters_per_class)
        n_subjects = len(np.unique(subject_ids))

        print(f"Simulated subjects: {n_subjects}")

        for model_name, pipeline in models.items():
            result = evaluate_with_logo(X, y, subject_ids, pipeline, model_name)

            print(f"{model_name:<25} Acc: {result['accuracy']*100:.1f}% | "
                  f"Sens: {result['sensitivity']*100:.1f}% | "
                  f"Spec: {result['specificity']*100:.1f}% | AUC: {result['auc']:.3f}")

            results.append({
                'validation': f'Subject-level ({config_name})',
                'model': model_name,
                'n_groups': n_subjects,
                'accuracy': round(result['accuracy'] * 100, 1),
                'sensitivity': round(result['sensitivity'] * 100, 1),
                'specificity': round(result['specificity'] * 100, 1),
                'auc': round(result['auc'], 3)
            })

    # Performance degradation analysis
    print("\n" + "="*70)
    print("PERFORMANCE DEGRADATION ANALYSIS")
    print("="*70)

    for model_name in ['Logistic Regression', 'Random Forest']:
        pattern_result = [r for r in results if r['validation'] == 'Pattern-level'
                         and r['model'] == model_name][0]

        print(f"\n{model_name}:")
        print("-"*50)

        for r in results:
            if r['model'] == model_name and r['validation'] != 'Pattern-level':
                acc_drop = pattern_result['accuracy'] - r['accuracy']
                sens_drop = pattern_result['sensitivity'] - r['sensitivity']
                spec_drop = pattern_result['specificity'] - r['specificity']

                print(f"  {r['validation'][:30]:<35}")
                print(f"    Accuracy:    {r['accuracy']:.1f}% (Δ = {-acc_drop:+.1f}%)")
                print(f"    Sensitivity: {r['sensitivity']:.1f}% (Δ = {-sens_drop:+.1f}%)")
                print(f"    Specificity: {r['specificity']:.1f}% (Δ = {-spec_drop:+.1f}%)")

    # Estimated real-world performance
    print("\n" + "="*70)
    print("ESTIMATED REAL-WORLD PERFORMANCE")
    print("="*70)

    # Take conservative estimate (average of moderate and liberal)
    lr_moderate = [r for r in results if 'Moderate' in r['validation'] and r['model'] == 'Logistic Regression'][0]
    rf_moderate = [r for r in results if 'Moderate' in r['validation'] and r['model'] == 'Random Forest'][0]

    print("\nBased on moderate clustering simulation:")
    print(f"\nLogistic Regression (Conservative Estimate):")
    print(f"  Accuracy:    {lr_moderate['accuracy']:.1f}%")
    print(f"  Sensitivity: {lr_moderate['sensitivity']:.1f}%")
    print(f"  Specificity: {lr_moderate['specificity']:.1f}%")

    print(f"\nRandom Forest (Conservative Estimate):")
    print(f"  Accuracy:    {rf_moderate['accuracy']:.1f}%")
    print(f"  Sensitivity: {rf_moderate['sensitivity']:.1f}%")
    print(f"  Specificity: {rf_moderate['specificity']:.1f}%")

    print("\n⚠️  Note: These are simulated estimates. True subject-level")
    print("   performance requires proper subject identifiers in the dataset.")

    # Save results
    output = {
        'results': results,
        'methodology': 'Simulated subject-level validation using feature-space clustering',
        'warning': 'These are estimates based on clustering. True validation requires subject IDs.'
    }

    output_file = "/data/gait/docs/papers/figures/subject_level_simulation_results.json"
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()
