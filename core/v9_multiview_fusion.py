#!/usr/bin/env python3
"""
V9 Multi-View Fusion Gait Detector
====================================

Fuses multiple camera views (front + side) for improved accuracy.

Strategy:
1. Extract features from each view independently
2. Combine features using different fusion methods:
   - Early fusion: Concatenate features
   - Late fusion: Average predictions
   - Hybrid fusion: Weighted combination

Expected Performance:
- Overall Accuracy: 92-95% (vs 89.5% single-view)
- Overall Sensitivity: 97%+ (vs 96.1% single-view)

Author: Gait Analysis System
Date: 2025-11-04
Version: 9.0 - Multi-View Fusion
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')


class V9_MultiView_Fusion:
    """Multi-view fusion gait detector"""

    def __init__(self, patterns_file: str = "gavd_patterns_with_v7_features.json"):
        print("="*80)
        print("V9 Multi-View Fusion Gait Detector")
        print("="*80)
        print()

        with open(patterns_file, 'r') as f:
            self.patterns = json.load(f)

        print(f"Loaded {len(self.patterns)} total patterns")

        # Group patterns by sequence ID (same subject, different views)
        self.grouped_patterns = self._group_by_sequence()

        print(f"Grouped into {len(self.grouped_patterns)} unique sequences")
        print()

        # Feature names (10 features from V7)
        self.feature_names = [
            'cadence_3d', 'step_height_variability', 'gait_irregularity_3d',
            'velocity_3d', 'jerkiness_3d', 'cycle_duration_3d',
            'stride_length_3d', 'trunk_sway', 'path_length_3d', 'step_width_3d'
        ]

        # Initialize models for each fusion strategy
        self.scaler_early = StandardScaler()
        self.clf_early = LogisticRegression(
            random_state=42, max_iter=1000, class_weight='balanced', C=1.0
        )

        self.scaler_single = StandardScaler()
        self.clf_single = LogisticRegression(
            random_state=42, max_iter=1000, class_weight='balanced', C=1.0
        )

    def _group_by_sequence(self) -> Dict:
        """Group patterns by video ID (same subject, different camera views)"""
        grouped = defaultdict(list)

        for p in self.patterns:
            # Extract video ID from CSV path
            # Example: .../cljan9b4p00043n6ligceanyp_B5hrxKe2nP8.csv
            # Video ID: B5hrxKe2nP8
            csv_path = p.get('csv_file', '')
            if '_' in csv_path:
                video_id = csv_path.split('_')[-1].replace('.csv', '')
                grouped[video_id].append(p)

        return dict(grouped)

    def extract_features(self, pattern: dict) -> np.ndarray:
        """Extract 10D feature vector from pattern"""
        features = []
        for fname in self.feature_names:
            val = pattern.get(fname, 0.0)
            if not np.isfinite(val):
                val = 0.0
            features.append(val)
        return np.array(features)

    def prepare_multiview_data(self) -> Tuple[np.ndarray, np.ndarray, List]:
        """
        Prepare multi-view data for training

        Returns:
            X_multi: Feature matrix with concatenated views
            y: Labels
            seq_ids: Sequence IDs for tracking
        """
        X_multi = []
        y = []
        seq_ids = []

        # Only keep sequences with exactly 2 views (for consistent feature dimension)
        for seq_id, patterns_list in self.grouped_patterns.items():
            if len(patterns_list) != 2:
                continue  # Skip single-view or 3+ view sequences

            # Get label (should be same for all views of same sequence)
            label = 1 if patterns_list[0]['gait_class'] == 'pathological' else 0

            # Sort by camera view for consistency
            patterns_list = sorted(patterns_list, key=lambda p: p.get('camera_view', ''))

            # Extract features from both views
            features_view1 = self.extract_features(patterns_list[0])
            features_view2 = self.extract_features(patterns_list[1])

            # Early fusion: Concatenate features from both views
            # 10 + 10 = 20 features
            fused_features = np.concatenate([features_view1, features_view2])

            X_multi.append(fused_features)
            y.append(label)
            seq_ids.append(seq_id)

        return np.array(X_multi), np.array(y), seq_ids

    def prepare_single_view_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare single-view data for baseline comparison"""
        X = []
        y = []

        for p in self.patterns:
            features = self.extract_features(p)
            label = 1 if p['gait_class'] == 'pathological' else 0

            X.append(features)
            y.append(label)

        return np.array(X), np.array(y)

    def train_early_fusion(self):
        """Train early fusion model (concatenated features)"""
        print("Training Early Fusion Model...")
        print("-" * 60)

        X_multi, y, seq_ids = self.prepare_multiview_data()

        print(f"Multi-view sequences: {len(X_multi)}")
        print(f"Feature dimension: {X_multi.shape[1]}")
        print(f"Class distribution: Normal={np.sum(y==0)}, Pathological={np.sum(y==1)}")
        print()

        # Standardize
        X_scaled = self.scaler_early.fit_transform(X_multi)

        # Cross-validation
        print("5-Fold Cross-Validation:")
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(self.clf_early, X_scaled, y, cv=cv, scoring='accuracy')
        print(f"  Accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std():.3f})")

        cv_sens = cross_val_score(self.clf_early, X_scaled, y, cv=cv, scoring='recall')
        print(f"  Sensitivity: {cv_sens.mean():.3f} (+/- {cv_sens.std():.3f})")
        print()

        # Train final model
        self.clf_early.fit(X_scaled, y)

        return X_scaled, y, seq_ids

    def train_single_view_baseline(self):
        """Train single-view baseline for comparison"""
        print("Training Single-View Baseline...")
        print("-" * 60)

        X, y = self.prepare_single_view_data()

        print(f"Single-view patterns: {len(X)}")
        print(f"Feature dimension: {X.shape[1]}")
        print(f"Class distribution: Normal={np.sum(y==0)}, Pathological={np.sum(y==1)}")
        print()

        # Standardize
        X_scaled = self.scaler_single.fit_transform(X)

        # Cross-validation
        print("5-Fold Cross-Validation:")
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(self.clf_single, X_scaled, y, cv=cv, scoring='accuracy')
        print(f"  Accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std():.3f})")

        cv_sens = cross_val_score(self.clf_single, X_scaled, y, cv=cv, scoring='recall')
        print(f"  Sensitivity: {cv_sens.mean():.3f} (+/- {cv_sens.std():.3f})")
        print()

        # Train final model
        self.clf_single.fit(X_scaled, y)

        return X_scaled, y

    def evaluate(self, X_scaled: np.ndarray, y: np.ndarray, model_name: str, clf):
        """Evaluate model performance"""
        print("="*80)
        print(f"{model_name} - Performance")
        print("="*80)
        print()

        # Predictions
        y_pred = clf.predict(X_scaled)
        y_proba = clf.predict_proba(X_scaled)[:, 1]

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
        print(f"  True Negatives:  {tn}")
        print(f"  False Positives: {fp}")
        print(f"  False Negatives: {fn}")
        print(f"  True Positives:  {tp}")
        print()

        # Metrics
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0

        print(f"Accuracy:    {accuracy:.3f} ({accuracy*100:.1f}%)")
        print(f"Sensitivity: {sensitivity:.3f} ({sensitivity*100:.1f}%)")
        print(f"Specificity: {specificity:.3f} ({specificity*100:.1f}%)")
        print(f"Precision:   {precision:.3f} ({precision*100:.1f}%)")

        # ROC AUC
        auc = roc_auc_score(y, y_proba)
        print(f"ROC AUC:     {auc:.3f}")
        print()

        return {
            'accuracy': accuracy,
            'sensitivity': sensitivity,
            'specificity': specificity,
            'precision': precision,
            'auc': auc
        }

    def compare_models(self, results_single: dict, results_multi: dict):
        """Compare single-view vs multi-view performance"""
        print("="*80)
        print("Single-View vs Multi-View Fusion Comparison")
        print("="*80)
        print()

        print(f"{'Metric':<20s} {'Single-View':<15s} {'Multi-View':<15s} {'Improvement':<15s}")
        print("-" * 70)

        for metric in ['accuracy', 'sensitivity', 'specificity', 'precision', 'auc']:
            single_val = results_single[metric]
            multi_val = results_multi[metric]
            improvement = (multi_val - single_val) * 100

            print(f"{metric.capitalize():<20s} {single_val:<15.3f} {multi_val:<15.3f} "
                  f"{improvement:+.1f}%")

        print()

    def save_model(self, output_file: str = "v9_multiview_model.json"):
        """Save trained multi-view model"""
        model_data = {
            'version': '9.0',
            'algorithm': 'Multi-View Fusion (Early Fusion)',
            'feature_names': self.feature_names,
            'scaler_mean': self.scaler_early.mean_.tolist(),
            'scaler_scale': self.scaler_early.scale_.tolist(),
            'coefficients': self.clf_early.coef_[0].tolist(),
            'intercept': float(self.clf_early.intercept_[0])
        }

        with open(output_file, 'w') as f:
            json.dump(model_data, f, indent=2)

        print(f"Model saved to {output_file}")


def main():
    """Train and evaluate V9 Multi-View Fusion"""

    # Initialize
    v9 = V9_MultiView_Fusion("gavd_patterns_with_v7_features.json")

    # Train single-view baseline
    X_single, y_single = v9.train_single_view_baseline()
    results_single = v9.evaluate(X_single, y_single, "Single-View Baseline (V8)", v9.clf_single)

    # Train multi-view fusion
    X_multi, y_multi, seq_ids = v9.train_early_fusion()
    results_multi = v9.evaluate(X_multi, y_multi, "Multi-View Fusion (V9)", v9.clf_early)

    # Compare
    v9.compare_models(results_single, results_multi)

    # Save model
    v9.save_model("v9_multiview_model.json")

    print("✅ V9 Multi-View Fusion training complete!")
    print()


if __name__ == "__main__":
    main()
