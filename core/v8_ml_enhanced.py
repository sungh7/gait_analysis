#!/usr/bin/env python3
"""
V8 ML-Enhanced Gait Detector (Corrected)
=========================================

Machine Learning enhanced version of V7 Pure 3D algorithm.
CORRECTED: Fixed data leakage in cross-validation by using sklearn Pipeline.

Key Fixes:
1. Pipeline ensures scaler fits only on training data per fold
2. Bootstrap confidence intervals for robust metrics
3. Proper cross-validation without information leakage
4. Multiple comparison awareness

Author: Gait Analysis System
Date: 2025-11-04 (Updated: 2026-01-15)
Version: 8.1 - Corrected Methodology
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import (
    StratifiedKFold, cross_val_score, cross_val_predict,
    cross_validate
)
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    accuracy_score, recall_score, precision_score, f1_score,
    make_scorer
)
import warnings
warnings.filterwarnings('ignore')


class V8_ML_Enhanced:
    """
    ML-enhanced gait detector using Logistic Regression.

    METHODOLOGY FIXES (v8.1):
    - Uses sklearn Pipeline to prevent data leakage
    - Scaler is fit ONLY on training fold, transformed on test fold
    - Bootstrap confidence intervals for all metrics
    - Explicit acknowledgment of pattern-level (not subject-level) CV
    """

    def __init__(self, patterns_file: str = "gavd_patterns_with_v7_features.json"):
        print("="*80)
        print("V8 ML-Enhanced Gait Detector (v8.1 - Corrected Methodology)")
        print("="*80)
        print()
        print("⚠️  IMPORTANT METHODOLOGICAL NOTES:")
        print("    - Cross-validation is PATTERN-LEVEL (not subject-level)")
        print("    - GAVD lacks subject identifiers; same subject may appear in train/test")
        print("    - Results represent proof-of-concept, not clinical deployment readiness")
        print()

        with open(patterns_file, 'r') as f:
            self.patterns = json.load(f)

        print(f"Loaded {len(self.patterns)} patterns")

        # Filter valid patterns with all 10 features
        valid_patterns = []
        for p in self.patterns:
            if self._has_all_features(p):
                valid_patterns.append(p)

        self.patterns = valid_patterns

        normal = [p for p in self.patterns if p['gait_class'] == 'normal']
        pathological = [p for p in self.patterns if p['gait_class'] == 'pathological']

        print(f"Valid patterns: {len(self.patterns)}")
        print(f"  Normal: {len(normal)}")
        print(f"  Pathological: {len(pathological)}")
        print()

        # Feature names (10 features from V7)
        self.feature_names = [
            'cadence_3d',
            'step_height_variability',
            'gait_irregularity_3d',
            'velocity_3d',
            'jerkiness_3d',
            'cycle_duration_3d',
            'stride_length_3d',
            'trunk_sway',
            'path_length_3d',
            'step_width_3d'
        ]

        # Create Pipeline (FIXES DATA LEAKAGE)
        # StandardScaler will be fit ONLY on training data within each CV fold
        self.pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', LogisticRegression(
                random_state=42,
                max_iter=1000,
                class_weight='balanced',
                C=1.0,
                solver='lbfgs'
            ))
        ])

        # Keep separate scaler/clf for final model
        self.scaler = StandardScaler()
        self.clf = LogisticRegression(
            random_state=42,
            max_iter=1000,
            class_weight='balanced',
            C=1.0,
            solver='lbfgs'
        )

    def _has_all_features(self, pattern: dict) -> bool:
        """Check if pattern has all required features"""
        required = [
            'cadence_3d', 'step_height_variability', 'gait_irregularity_3d',
            'velocity_3d', 'jerkiness_3d', 'cycle_duration_3d',
            'stride_length_3d', 'trunk_sway', 'path_length_3d', 'step_width_3d'
        ]
        return all(f in pattern for f in required)

    def extract_features(self, pattern: dict) -> np.ndarray:
        """Extract feature vector from pattern"""
        features = []
        for fname in self.feature_names:
            val = pattern.get(fname, 0.0)
            if not np.isfinite(val):
                val = 0.0
            features.append(val)
        return np.array(features)

    def prepare_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare feature matrix X and labels y"""
        X = []
        y = []

        for p in self.patterns:
            features = self.extract_features(p)
            X.append(features)
            label = 1 if p['gait_class'] == 'pathological' else 0
            y.append(label)

        return np.array(X), np.array(y)

    def _bootstrap_ci(self, y_true: np.ndarray, y_pred: np.ndarray,
                      metric_func, n_bootstrap: int = 1000,
                      ci: float = 0.95) -> Tuple[float, float, float]:
        """
        Calculate bootstrap confidence interval for a metric.

        Returns:
            (point_estimate, ci_lower, ci_upper)
        """
        np.random.seed(42)
        n = len(y_true)
        scores = []

        for _ in range(n_bootstrap):
            indices = np.random.choice(n, size=n, replace=True)
            y_true_boot = y_true[indices]
            y_pred_boot = y_pred[indices]

            # Skip if only one class in bootstrap sample
            if len(np.unique(y_true_boot)) < 2:
                continue

            try:
                score = metric_func(y_true_boot, y_pred_boot)
                scores.append(score)
            except:
                continue

        if len(scores) < 100:
            return metric_func(y_true, y_pred), np.nan, np.nan

        point_est = metric_func(y_true, y_pred)
        alpha = (1 - ci) / 2
        ci_lower = np.percentile(scores, alpha * 100)
        ci_upper = np.percentile(scores, (1 - alpha) * 100)

        return point_est, ci_lower, ci_upper

    def train(self) -> Tuple[np.ndarray, np.ndarray, dict]:
        """
        Train Logistic Regression model with CORRECTED cross-validation.

        KEY FIX: Uses Pipeline to ensure scaler fits only on training folds.
        """
        print("Training ML model (CORRECTED methodology)...")
        print()

        X, y = self.prepare_data()

        print(f"Feature matrix shape: {X.shape}")
        print(f"Class distribution: Normal={np.sum(y==0)}, Pathological={np.sum(y==1)}")
        print()

        # Cross-validation with Pipeline (NO DATA LEAKAGE)
        print("5-Fold Stratified Cross-Validation (Pipeline - No Data Leakage):")
        print("-" * 60)

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        # Multiple metrics
        scoring = {
            'accuracy': 'accuracy',
            'sensitivity': 'recall',
            'specificity': make_scorer(
                lambda y_t, y_p: recall_score(y_t, y_p, pos_label=0)
            ),
            'precision': 'precision',
            'f1': 'f1',
            'roc_auc': 'roc_auc'
        }

        # Cross-validate with Pipeline (CORRECT WAY)
        cv_results = cross_validate(
            self.pipeline, X, y, cv=cv, scoring=scoring,
            return_train_score=False
        )

        # Store CV results
        self.cv_results = {}

        for metric in ['accuracy', 'sensitivity', 'specificity', 'precision', 'f1', 'roc_auc']:
            key = f'test_{metric}'
            scores = cv_results[key]
            mean_score = scores.mean()
            std_score = scores.std()

            # Compute 95% CI from fold scores
            ci_lower = mean_score - 1.96 * std_score / np.sqrt(5)
            ci_upper = mean_score + 1.96 * std_score / np.sqrt(5)

            self.cv_results[metric] = {
                'mean': mean_score,
                'std': std_score,
                'ci_lower': ci_lower,
                'ci_upper': ci_upper,
                'fold_scores': scores.tolist()
            }

            print(f"  {metric:12s}: {mean_score:.3f} ± {std_score:.3f}  "
                  f"[95% CI: {ci_lower:.3f}, {ci_upper:.3f}]")

        print()

        # Get cross-validated predictions for confusion matrix
        y_pred_cv = cross_val_predict(self.pipeline, X, y, cv=cv)

        # Train final model on all data (for deployment)
        print("Training final model on all data...")
        X_scaled = self.scaler.fit_transform(X)
        self.clf.fit(X_scaled, y)

        # Analyze feature importance
        self._analyze_feature_importance()

        return X_scaled, y, {'y_pred_cv': y_pred_cv}

    def _analyze_feature_importance(self):
        """Analyze and display feature importance with stability assessment"""
        print("Feature Importance (Logistic Regression Coefficients):")
        print("-" * 60)

        coefficients = self.clf.coef_[0]

        importance = [(name, coef) for name, coef in zip(self.feature_names, coefficients)]
        importance.sort(key=lambda x: abs(x[1]), reverse=True)

        for rank, (name, coef) in enumerate(importance, 1):
            direction = "↑ Path." if coef > 0 else "↓ Normal"
            print(f"{rank:2d}. {name:30s}: {coef:+.4f}  {direction}")

        print()
        self.feature_importance = dict(importance)

    def evaluate(self, X_scaled: np.ndarray, y: np.ndarray,
                 y_pred_cv: Optional[np.ndarray] = None) -> dict:
        """
        Evaluate model performance with bootstrap confidence intervals.

        Uses CV predictions if available (more honest assessment).
        """
        print("="*80)
        print("Performance Evaluation")
        print("="*80)
        print()

        # Use CV predictions if available, otherwise use training predictions
        if y_pred_cv is not None:
            print("📊 Using Cross-Validated Predictions (unbiased estimate)")
            y_pred = y_pred_cv
            y_proba = None  # CV doesn't give probabilities easily
        else:
            print("⚠️  Using Training Predictions (may be optimistic)")
            y_pred = self.clf.predict(X_scaled)
            y_proba = self.clf.predict_proba(X_scaled)[:, 1]

        print()

        # Classification report
        print(classification_report(
            y, y_pred,
            target_names=['Normal', 'Pathological'],
            digits=3
        ))

        # Confusion matrix
        cm = confusion_matrix(y, y_pred)
        tn, fp, fn, tp = cm.ravel()

        print("Confusion Matrix:")
        print(f"  True Negatives (Normal correct):      {tn}")
        print(f"  False Positives (Normal → Path):      {fp}")
        print(f"  False Negatives (Path missed):        {fn}")
        print(f"  True Positives (Path detected):       {tp}")
        print()

        # Calculate metrics with bootstrap CIs
        print("Metrics with 95% Bootstrap Confidence Intervals:")
        print("-" * 60)

        metrics = {}

        # Accuracy
        acc, acc_lo, acc_hi = self._bootstrap_ci(y, y_pred, accuracy_score)
        metrics['accuracy'] = {'value': acc, 'ci_lower': acc_lo, 'ci_upper': acc_hi}
        print(f"  Accuracy:    {acc:.3f} [{acc_lo:.3f}, {acc_hi:.3f}]")

        # Sensitivity (Recall for positive class)
        sens, sens_lo, sens_hi = self._bootstrap_ci(y, y_pred, recall_score)
        metrics['sensitivity'] = {'value': sens, 'ci_lower': sens_lo, 'ci_upper': sens_hi}
        print(f"  Sensitivity: {sens:.3f} [{sens_lo:.3f}, {sens_hi:.3f}]")

        # Specificity (Recall for negative class)
        spec_func = lambda yt, yp: recall_score(yt, yp, pos_label=0)
        spec, spec_lo, spec_hi = self._bootstrap_ci(y, y_pred, spec_func)
        metrics['specificity'] = {'value': spec, 'ci_lower': spec_lo, 'ci_upper': spec_hi}
        print(f"  Specificity: {spec:.3f} [{spec_lo:.3f}, {spec_hi:.3f}]")

        # Precision
        prec, prec_lo, prec_hi = self._bootstrap_ci(y, y_pred, precision_score)
        metrics['precision'] = {'value': prec, 'ci_lower': prec_lo, 'ci_upper': prec_hi}
        print(f"  Precision:   {prec:.3f} [{prec_lo:.3f}, {prec_hi:.3f}]")

        # F1
        f1, f1_lo, f1_hi = self._bootstrap_ci(y, y_pred, f1_score)
        metrics['f1'] = {'value': f1, 'ci_lower': f1_lo, 'ci_upper': f1_hi}
        print(f"  F1 Score:    {f1:.3f} [{f1_lo:.3f}, {f1_hi:.3f}]")

        print()

        return {
            'metrics': metrics,
            'cv_results': self.cv_results,
            'confusion_matrix': cm,
            'y_pred': y_pred
        }

    def evaluate_by_pathology(self, y_pred: np.ndarray) -> dict:
        """Evaluate performance by specific pathology type"""
        print("="*80)
        print("Performance by Pathology Type")
        print("="*80)
        print()

        # Group patterns by pathology
        pathology_groups = {}
        for i, p in enumerate(self.patterns):
            path_type = p.get('gait_pathology', 'normal')
            if path_type not in pathology_groups:
                pathology_groups[path_type] = []
            label = 1 if p['gait_class'] == 'pathological' else 0
            pathology_groups[path_type].append((i, label, y_pred[i]))

        results = {}
        for path_type, indices in sorted(pathology_groups.items()):
            total = len(indices)

            if path_type == 'normal':
                correct = sum(1 for _, y_true, y_p in indices if y_p == 0)
                results[path_type] = {
                    'total': total,
                    'detected': total - correct,
                    'missed': 0,
                    'rate': correct / total if total > 0 else 0,
                    'metric': 'specificity'
                }
            else:
                detected = sum(1 for _, y_true, y_p in indices if y_p == 1)
                missed = total - detected
                results[path_type] = {
                    'total': total,
                    'detected': detected,
                    'missed': missed,
                    'rate': detected / total if total > 0 else 0,
                    'metric': 'sensitivity'
                }

        for path_type, res in sorted(results.items()):
            rate_pct = res['rate'] * 100
            metric_name = res['metric']
            if path_type == 'normal':
                print(f"{path_type:20s}: {res['total']:3d} cases, "
                      f"{rate_pct:5.1f}% {metric_name}")
            else:
                print(f"{path_type:20s}: {res['total']:3d} cases, "
                      f"{res['detected']:3d}/{res['total']:3d} detected "
                      f"({rate_pct:5.1f}% {metric_name})")

        print()
        return results

    def predict(self, features: np.ndarray) -> Tuple[int, float]:
        """Predict pathology for new sample"""
        if features.ndim == 1:
            features = features.reshape(1, -1)

        features_scaled = self.scaler.transform(features)
        prediction = self.clf.predict(features_scaled)[0]
        probability = self.clf.predict_proba(features_scaled)[0, 1]

        return int(prediction), float(probability)

    def save_model(self, output_file: str = "v8_ml_model.json"):
        """Save trained model parameters"""
        model_data = {
            'version': '8.1',
            'methodology': 'Corrected - Pipeline CV without data leakage',
            'algorithm': 'Logistic Regression',
            'feature_names': self.feature_names,
            'scaler_mean': self.scaler.mean_.tolist(),
            'scaler_scale': self.scaler.scale_.tolist(),
            'coefficients': self.clf.coef_[0].tolist(),
            'intercept': float(self.clf.intercept_[0]),
            'feature_importance': self.feature_importance,
            'cv_results': self.cv_results,
            'notes': [
                'Cross-validation is PATTERN-LEVEL, not subject-level',
                'GAVD dataset lacks subject identifiers',
                'Results are proof-of-concept, not clinical-grade'
            ]
        }

        with open(output_file, 'w') as f:
            json.dump(model_data, f, indent=2)

        print(f"Model saved to {output_file}")


def main():
    """Train and evaluate V8 ML-Enhanced model with corrected methodology"""

    print("\n" + "="*80)
    print("V8 ML-Enhanced Gait Detector - CORRECTED METHODOLOGY")
    print("="*80 + "\n")

    # Initialize
    v8 = V8_ML_Enhanced("gavd_patterns_with_v7_features.json")

    # Train with CORRECTED cross-validation (Pipeline)
    X_scaled, y, cv_data = v8.train()

    # Evaluate using CV predictions (unbiased)
    results = v8.evaluate(X_scaled, y, y_pred_cv=cv_data['y_pred_cv'])

    # Evaluate by pathology
    pathology_results = v8.evaluate_by_pathology(cv_data['y_pred_cv'])

    # Save model
    v8.save_model("v8_ml_model.json")

    # Summary comparison
    print("="*80)
    print("Summary: Corrected vs Original Results")
    print("="*80)
    print()
    print("⚠️  KEY METHODOLOGICAL IMPROVEMENTS:")
    print("    1. Fixed data leakage (scaler now fit per fold)")
    print("    2. Added 95% confidence intervals")
    print("    3. Using CV predictions for evaluation (unbiased)")
    print("    4. Explicit warnings about pattern-level CV")
    print()

    # Display CV results
    print("Cross-Validated Performance (Corrected):")
    print("-" * 60)
    for metric, data in v8.cv_results.items():
        print(f"  {metric:12s}: {data['mean']:.3f} ± {data['std']:.3f}  "
              f"[{data['ci_lower']:.3f}, {data['ci_upper']:.3f}]")

    print()
    print("✅ V8.1 Corrected model training complete!")
    print()


if __name__ == "__main__":
    main()
