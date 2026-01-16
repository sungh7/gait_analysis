#!/usr/bin/env python3
"""
Multi-class Pathology Classification Analysis
=============================================

Classifies gait by specific pathology type (not just binary normal/pathological).
Groups pathologies by neurological basis for statistical power.

Classes:
- Normal (n=142)
- Neurological (CP + Parkinson's + Stroke = 41)
- Myopathic (n=20)
- Other Abnormal (Antalgic + Generic = 89)

Author: Gait Analysis System
Date: 2026-01-15
"""

import json
import numpy as np
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, precision_recall_fscore_support
)
from sklearn.preprocessing import label_binarize


def load_and_prepare_multiclass_data(patterns_file):
    """Load patterns and create multi-class labels."""
    with open(patterns_file, 'r') as f:
        patterns = json.load(f)

    # Define class mapping for statistical power
    class_mapping = {
        'normal': 'Normal',
        'cerebral palsy': 'Neurological',
        'parkinsons': 'Neurological',
        'stroke': 'Neurological',
        'myopathic': 'Myopathic',
        'antalgic': 'Other_Abnormal',
        'abnormal': 'Other_Abnormal'
    }

    # Exclude lifestyle/temporary conditions (too few samples)
    excluded = {'inebriated', 'pregnant'}

    feature_names = [
        'cadence_3d', 'step_height_variability', 'gait_irregularity_3d',
        'velocity_3d', 'jerkiness_3d', 'cycle_duration_3d',
        'stride_length_3d', 'trunk_sway', 'path_length_3d', 'step_width_3d'
    ]

    X, y, pathology_details = [], [], []
    for p in patterns:
        pathology = p.get('gait_pathology', 'normal').lower().strip()

        if pathology in excluded:
            continue
        if pathology not in class_mapping:
            continue

        features = [p.get(f, 0.0) for f in feature_names]
        if not all(np.isfinite(features)):
            continue

        X.append(features)
        y.append(class_mapping[pathology])
        pathology_details.append(pathology)

    return np.array(X), np.array(y), feature_names, pathology_details


def run_multiclass_analysis(patterns_file, output_file=None):
    """Run complete multi-class classification analysis."""
    print("="*70)
    print("Multi-class Pathology Classification Analysis")
    print("="*70)
    print()

    X, y, feature_names, pathology_details = load_and_prepare_multiclass_data(patterns_file)

    # Show class distribution
    print("Class Distribution:")
    unique, counts = np.unique(y, return_counts=True)
    for cls, cnt in zip(unique, counts):
        print(f"  {cls}: {cnt}")
    print()

    # Show pathology breakdown
    print("Original Pathology Breakdown:")
    unique_path, counts_path = np.unique(pathology_details, return_counts=True)
    for path, cnt in sorted(zip(unique_path, counts_path), key=lambda x: -x[1]):
        print(f"  {path}: {cnt}")
    print()

    # Encode labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    classes = le.classes_
    n_classes = len(classes)

    print(f"Total samples: {len(X)}")
    print(f"Number of classes: {n_classes}")
    print(f"Classes: {list(classes)}")
    print()

    # Create pipeline with OneVsRest strategy
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', OneVsRestClassifier(
            LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
        ))
    ])

    # 5-fold stratified CV
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    y_pred = cross_val_predict(pipeline, X, y_encoded, cv=cv)

    # Classification report
    print("Multi-class Classification Report:")
    print("-"*70)
    print(classification_report(y_encoded, y_pred, target_names=classes, digits=3))

    # Confusion matrix
    cm = confusion_matrix(y_encoded, y_pred)
    print("Confusion Matrix:")
    print(f"{'':15}", end='')
    for cls in classes:
        print(f"{cls[:10]:>12}", end='')
    print()
    for i, cls in enumerate(classes):
        print(f"{cls:15}", end='')
        for j in range(n_classes):
            print(f"{cm[i,j]:>12}", end='')
        print()
    print()

    # Per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        y_encoded, y_pred, average=None
    )

    # Fit for AUC calculation
    pipeline.fit(X, y_encoded)
    y_prob = pipeline.predict_proba(X)
    y_bin = label_binarize(y_encoded, classes=range(n_classes))

    print("Per-class Performance Summary:")
    print("-"*70)
    print(f"{'Class':<15} {'N':>6} {'Sens':>8} {'Spec':>8} {'PPV':>8} {'NPV':>8} {'AUC':>8}")
    print("-"*70)

    results = {'classes': list(classes), 'metrics': []}

    for i, cls in enumerate(classes):
        n = support[i]
        sens = recall[i]  # Sensitivity = Recall
        ppv = precision[i]  # PPV = Precision

        # Calculate specificity and NPV
        tp = cm[i, i]
        fn = np.sum(cm[i, :]) - tp
        fp = np.sum(cm[:, i]) - tp
        tn = np.sum(cm) - tp - fn - fp

        spec = tn / (tn + fp) if (tn + fp) > 0 else 0
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0

        # AUC (One-vs-Rest)
        auc = roc_auc_score(y_bin[:, i], y_prob[:, i])

        print(f"{cls:<15} {n:>6} {sens*100:>7.1f}% {spec*100:>7.1f}% "
              f"{ppv*100:>7.1f}% {npv*100:>7.1f}% {auc:>8.3f}")

        results['metrics'].append({
            'class': cls,
            'n': int(n),
            'sensitivity': round(sens * 100, 1),
            'specificity': round(spec * 100, 1),
            'ppv': round(ppv * 100, 1),
            'npv': round(npv * 100, 1),
            'auc': round(auc, 3)
        })

    # Macro averages
    macro_sens = np.mean(recall)
    macro_prec = np.mean(precision)
    macro_f1 = np.mean(f1)
    macro_auc = roc_auc_score(y_bin, y_prob, average='macro')

    print("-"*70)
    print(f"{'Macro Avg':<15} {len(X):>6} {macro_sens*100:>7.1f}% {'':>8} "
          f"{macro_prec*100:>7.1f}% {'':>8} {macro_auc:>8.3f}")

    results['macro_avg'] = {
        'sensitivity': round(macro_sens * 100, 1),
        'precision': round(macro_prec * 100, 1),
        'f1': round(macro_f1, 3),
        'auc': round(macro_auc, 3)
    }
    results['confusion_matrix'] = cm.tolist()

    # Save results
    if output_file:
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {output_file}")

    return results


if __name__ == "__main__":
    patterns_file = "/data/gait/output/results/root_dumps/gavd_patterns_with_v7_features.json"
    output_file = "/data/gait/docs/papers/figures/multiclass_results.json"

    results = run_multiclass_analysis(patterns_file, output_file)
