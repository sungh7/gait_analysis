#!/usr/bin/env python3
"""
Improve Classification Specificity
===================================

Goal: Increase specificity from 81% to 90%+ while maintaining high sensitivity.

Strategies:
1. Ensemble methods (Voting Classifier)
2. Threshold optimization
3. Feature engineering (interaction terms, ratios)
4. Calibrated classifiers

Author: Gait Analysis System
Date: 2026-01-16
"""

import json
import numpy as np
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier, VotingClassifier,
    GradientBoostingClassifier, StackingClassifier
)
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import (
    accuracy_score, recall_score, precision_score,
    f1_score, roc_auc_score, confusion_matrix, roc_curve
)
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

    X, y = [], []
    for p in patterns:
        features = [p.get(f, 0.0) for f in feature_names]
        if all(np.isfinite(features)):
            X.append(features)
            y.append(1 if p.get('gait_class') == 'pathological' else 0)

    return np.array(X), np.array(y), feature_names


def add_interaction_features(X, feature_names):
    """Add interaction and ratio features."""
    X_new = X.copy()
    new_names = feature_names.copy()

    # Key ratios
    # Cadence / Cycle Duration (should be ~constant if consistent)
    cadence_idx = feature_names.index('cadence_3d')
    cycle_idx = feature_names.index('cycle_duration_3d')
    ratio1 = X[:, cadence_idx] / (X[:, cycle_idx] + 1e-6)
    X_new = np.column_stack([X_new, ratio1])
    new_names.append('cadence_cycle_ratio')

    # Gait irregularity * Jerkiness (compounding instability)
    irreg_idx = feature_names.index('gait_irregularity_3d')
    jerk_idx = feature_names.index('jerkiness_3d')
    interaction1 = X[:, irreg_idx] * X[:, jerk_idx]
    X_new = np.column_stack([X_new, interaction1])
    new_names.append('irreg_jerk_interaction')

    # Velocity / Stride length (step frequency proxy)
    vel_idx = feature_names.index('velocity_3d')
    stride_idx = feature_names.index('stride_length_3d')
    ratio2 = X[:, vel_idx] / (X[:, stride_idx] + 1e-6)
    X_new = np.column_stack([X_new, ratio2])
    new_names.append('vel_stride_ratio')

    # Trunk sway / Step width (lateral stability)
    trunk_idx = feature_names.index('trunk_sway')
    width_idx = feature_names.index('step_width_3d')
    ratio3 = X[:, trunk_idx] / (X[:, width_idx] + 1e-6)
    X_new = np.column_stack([X_new, ratio3])
    new_names.append('trunk_width_ratio')

    return X_new, new_names


def evaluate_model(name, pipeline, X, y, cv):
    """Evaluate model with cross-validation."""
    y_pred = cross_val_predict(pipeline, X, y, cv=cv)

    # Get probabilities for threshold analysis
    y_prob = np.zeros(len(y))
    for train_idx, test_idx in cv.split(X, y):
        pipeline.fit(X[train_idx], y[train_idx])
        if hasattr(pipeline, 'predict_proba'):
            y_prob[test_idx] = pipeline.predict_proba(X[test_idx])[:, 1]
        else:
            y_prob[test_idx] = pipeline.decision_function(X[test_idx])

    acc = accuracy_score(y, y_pred)
    sens = recall_score(y, y_pred, pos_label=1)
    spec = recall_score(y, y_pred, pos_label=0)
    prec = precision_score(y, y_pred)
    f1 = f1_score(y, y_pred)
    auc = roc_auc_score(y, y_prob)

    return {
        'name': name,
        'accuracy': acc,
        'sensitivity': sens,
        'specificity': spec,
        'precision': prec,
        'f1': f1,
        'auc': auc,
        'y_prob': y_prob
    }


def find_optimal_threshold(y_true, y_prob, target_sensitivity=0.90):
    """Find threshold that maximizes specificity while maintaining target sensitivity."""
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)

    # Find thresholds where sensitivity >= target
    valid_idx = tpr >= target_sensitivity
    if not any(valid_idx):
        return 0.5, tpr[0], 1 - fpr[0]

    # Among valid thresholds, find the one with highest specificity (lowest FPR)
    valid_fpr = fpr[valid_idx]
    valid_tpr = tpr[valid_idx]
    valid_thresh = thresholds[valid_idx]

    best_idx = np.argmin(valid_fpr)
    best_threshold = valid_thresh[best_idx]
    best_sens = valid_tpr[best_idx]
    best_spec = 1 - valid_fpr[best_idx]

    return best_threshold, best_sens, best_spec


def main():
    print("="*70)
    print("Specificity Improvement Analysis")
    print("="*70)
    print()

    # Load data
    patterns_file = "/data/gait/output/results/root_dumps/gavd_patterns_with_v7_features.json"
    X, y, feature_names = load_data(patterns_file)

    print(f"Samples: {len(X)} (Normal: {sum(y==0)}, Pathological: {sum(y==1)})")
    print()

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Define models to test
    models = {
        'Baseline (LR)': Pipeline([
            ('scaler', StandardScaler()),
            ('clf', LogisticRegression(C=1.0, class_weight='balanced', max_iter=1000, random_state=42))
        ]),

        'Random Forest': Pipeline([
            ('scaler', StandardScaler()),
            ('clf', RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42))
        ]),

        'Gradient Boosting': Pipeline([
            ('scaler', StandardScaler()),
            ('clf', GradientBoostingClassifier(n_estimators=100, random_state=42))
        ]),

        'SVM (RBF)': Pipeline([
            ('scaler', StandardScaler()),
            ('clf', SVC(kernel='rbf', class_weight='balanced', probability=True, random_state=42))
        ]),

        'Voting (LR+RF+SVM)': Pipeline([
            ('scaler', StandardScaler()),
            ('clf', VotingClassifier(
                estimators=[
                    ('lr', LogisticRegression(C=1.0, class_weight='balanced', max_iter=1000, random_state=42)),
                    ('rf', RandomForestClassifier(n_estimators=50, class_weight='balanced', random_state=42)),
                    ('svm', SVC(kernel='rbf', class_weight='balanced', probability=True, random_state=42))
                ],
                voting='soft'
            ))
        ]),

        'Stacking (LR+RF→LR)': Pipeline([
            ('scaler', StandardScaler()),
            ('clf', StackingClassifier(
                estimators=[
                    ('lr', LogisticRegression(C=1.0, class_weight='balanced', max_iter=1000, random_state=42)),
                    ('rf', RandomForestClassifier(n_estimators=50, class_weight='balanced', random_state=42))
                ],
                final_estimator=LogisticRegression(max_iter=1000),
                cv=3
            ))
        ])
    }

    # Evaluate all models
    print("Model Comparison (Default Threshold = 0.5):")
    print("-"*85)
    print(f"{'Model':<25} {'Accuracy':>10} {'Sens':>8} {'Spec':>8} {'Prec':>8} {'F1':>8} {'AUC':>8}")
    print("-"*85)

    results = []
    for name, pipeline in models.items():
        result = evaluate_model(name, pipeline, X, y, cv)
        results.append(result)
        print(f"{name:<25} {result['accuracy']*100:>9.1f}% {result['sensitivity']*100:>7.1f}% "
              f"{result['specificity']*100:>7.1f}% {result['precision']*100:>7.1f}% "
              f"{result['f1']:>7.3f} {result['auc']:>7.3f}")

    print()

    # Try with interaction features
    print("With Interaction Features:")
    print("-"*85)

    X_enhanced, enhanced_names = add_interaction_features(X, feature_names)
    print(f"Features: {len(feature_names)} → {len(enhanced_names)}")

    enhanced_models = {
        'LR + Interactions': Pipeline([
            ('scaler', StandardScaler()),
            ('clf', LogisticRegression(C=1.0, class_weight='balanced', max_iter=1000, random_state=42))
        ]),
        'Voting + Interactions': Pipeline([
            ('scaler', StandardScaler()),
            ('clf', VotingClassifier(
                estimators=[
                    ('lr', LogisticRegression(C=1.0, class_weight='balanced', max_iter=1000, random_state=42)),
                    ('rf', RandomForestClassifier(n_estimators=50, class_weight='balanced', random_state=42)),
                    ('svm', SVC(kernel='rbf', class_weight='balanced', probability=True, random_state=42))
                ],
                voting='soft'
            ))
        ])
    }

    for name, pipeline in enhanced_models.items():
        result = evaluate_model(name, pipeline, X_enhanced, y, cv)
        results.append(result)
        print(f"{name:<25} {result['accuracy']*100:>9.1f}% {result['sensitivity']*100:>7.1f}% "
              f"{result['specificity']*100:>7.1f}% {result['precision']*100:>7.1f}% "
              f"{result['f1']:>7.3f} {result['auc']:>7.3f}")

    print()

    # Threshold optimization
    print("Threshold Optimization (Target: 90% Sensitivity):")
    print("-"*70)

    best_result = max(results, key=lambda x: x['auc'])
    print(f"Best model by AUC: {best_result['name']} (AUC = {best_result['auc']:.3f})")

    for target_sens in [0.95, 0.90, 0.85]:
        thresh, actual_sens, actual_spec = find_optimal_threshold(
            y, best_result['y_prob'], target_sensitivity=target_sens
        )
        print(f"  Target Sens ≥{target_sens*100:.0f}%: Threshold={thresh:.3f} → "
              f"Sens={actual_sens*100:.1f}%, Spec={actual_spec*100:.1f}%")

    print()

    # Final recommendation
    print("="*70)
    print("RECOMMENDATIONS")
    print("="*70)

    # Find best model with spec > 85%
    high_spec_models = [r for r in results if r['specificity'] > 0.85]
    if high_spec_models:
        best_high_spec = max(high_spec_models, key=lambda x: x['sensitivity'])
        print(f"\n1. Best model with Spec > 85%:")
        print(f"   {best_high_spec['name']}")
        print(f"   Accuracy: {best_high_spec['accuracy']*100:.1f}%")
        print(f"   Sensitivity: {best_high_spec['sensitivity']*100:.1f}%")
        print(f"   Specificity: {best_high_spec['specificity']*100:.1f}%")

    # Best overall by balanced metric
    best_balanced = max(results, key=lambda x: (x['sensitivity'] + x['specificity']) / 2)
    print(f"\n2. Best balanced model:")
    print(f"   {best_balanced['name']}")
    print(f"   Accuracy: {best_balanced['accuracy']*100:.1f}%")
    print(f"   Sensitivity: {best_balanced['sensitivity']*100:.1f}%")
    print(f"   Specificity: {best_balanced['specificity']*100:.1f}%")

    # Threshold-optimized recommendation
    print(f"\n3. Threshold optimization with {best_result['name']}:")
    thresh, sens, spec = find_optimal_threshold(y, best_result['y_prob'], target_sensitivity=0.90)
    print(f"   Threshold: {thresh:.3f}")
    print(f"   Sensitivity: {sens*100:.1f}%")
    print(f"   Specificity: {spec*100:.1f}%")

    # Save results
    output = {
        'models': [{k: v for k, v in r.items() if k != 'y_prob'} for r in results],
        'best_auc_model': best_result['name'],
        'threshold_optimization': {
            'model': best_result['name'],
            'thresholds': {}
        }
    }

    for target_sens in [0.95, 0.90, 0.85]:
        thresh, sens, spec = find_optimal_threshold(y, best_result['y_prob'], target_sensitivity=target_sens)
        output['threshold_optimization']['thresholds'][f'sens_{int(target_sens*100)}'] = {
            'threshold': round(thresh, 3),
            'sensitivity': round(sens * 100, 1),
            'specificity': round(spec * 100, 1)
        }

    output_file = "/data/gait/docs/papers/figures/specificity_improvement_results.json"
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()
