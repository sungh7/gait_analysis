#!/usr/bin/env python3
"""
STAGE 2 Evaluation with Real GAVD Data
======================================

Builds population-based templates from real GAVD patterns and
re-evaluates the STAGE 2 Pattern-Based Detector.

Author: Gait Analysis System
Version: 2.0
Date: 2025-10-30
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from dataclasses import dataclass
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw
import warnings
warnings.filterwarnings('ignore')


@dataclass
class PatternTemplate:
    """Population-based pattern template"""
    gait_class: str
    mean_pattern_left: np.ndarray
    mean_pattern_right: np.ndarray
    std_pattern_left: np.ndarray
    std_pattern_right: np.ndarray
    n_samples: int
    mean_amplitude_left: float
    mean_amplitude_right: float


class RealDataTemplateBuilder:
    """Build templates from real GAVD patterns"""

    def __init__(self, patterns_file: str):
        """Load real GAVD patterns"""
        with open(patterns_file, 'r') as f:
            self.patterns = json.load(f)

        print(f"Loaded {len(self.patterns)} real patterns from GAVD dataset")

    def build_templates(self) -> Dict[str, PatternTemplate]:
        """
        Build population-based templates for each gait class.

        Returns:
            Dictionary mapping gait class to template
        """
        print("\n" + "="*70)
        print("BUILDING POPULATION-BASED TEMPLATES FROM REAL DATA")
        print("="*70)

        # Group patterns by class
        patterns_by_class = {}
        for pattern in self.patterns:
            gait_class = pattern['gait_class']
            if gait_class not in patterns_by_class:
                patterns_by_class[gait_class] = []
            patterns_by_class[gait_class].append(pattern)

        templates = {}

        for gait_class, class_patterns in patterns_by_class.items():
            # Filter valid patterns
            valid_patterns = [
                p for p in class_patterns
                if p['normalized_pattern_left'] and p['normalized_pattern_right']
            ]

            if len(valid_patterns) < 2:
                print(f"\nSkipping {gait_class}: insufficient samples ({len(valid_patterns)})")
                continue

            # Extract pattern arrays
            left_patterns = np.array([p['normalized_pattern_left'] for p in valid_patterns])
            right_patterns = np.array([p['normalized_pattern_right'] for p in valid_patterns])

            # Compute mean and std
            mean_left = np.mean(left_patterns, axis=0)
            mean_right = np.mean(right_patterns, axis=0)
            std_left = np.std(left_patterns, axis=0)
            std_right = np.std(right_patterns, axis=0)

            # Compute mean amplitudes
            amplitudes_left = [p['amplitude_left'] for p in valid_patterns]
            amplitudes_right = [p['amplitude_right'] for p in valid_patterns]

            template = PatternTemplate(
                gait_class=gait_class,
                mean_pattern_left=mean_left,
                mean_pattern_right=mean_right,
                std_pattern_left=std_left,
                std_pattern_right=std_right,
                n_samples=len(valid_patterns),
                mean_amplitude_left=np.mean(amplitudes_left),
                mean_amplitude_right=np.mean(amplitudes_right)
            )

            templates[gait_class] = template

            print(f"\n{gait_class}:")
            print(f"  Samples: {len(valid_patterns)}")
            print(f"  Mean amplitude (L/R): {template.mean_amplitude_left:.2f} / {template.mean_amplitude_right:.2f}")
            print(f"  Pattern variability (L/R): {np.mean(std_left):.2f} / {np.mean(std_right):.2f}")

        return templates

    def create_binary_templates(self, templates: Dict[str, PatternTemplate]) -> Dict[str, PatternTemplate]:
        """
        Create binary (normal vs abnormal) templates.

        Args:
            templates: Multi-class templates

        Returns:
            Binary templates (normal, abnormal)
        """
        print("\n" + "="*70)
        print("CREATING BINARY TEMPLATES")
        print("="*70)

        # Define normal vs abnormal classes
        normal_classes = ['normal']
        abnormal_classes = [cls for cls in templates.keys() if cls != 'normal']

        binary_templates = {}

        # Normal template (already exists)
        if 'normal' in templates:
            binary_templates['normal'] = templates['normal']
            print(f"\nNormal template: {templates['normal'].n_samples} samples")

        # Aggregate abnormal template
        abnormal_patterns_left = []
        abnormal_patterns_right = []
        abnormal_amplitudes_left = []
        abnormal_amplitudes_right = []

        for pattern in self.patterns:
            if pattern['gait_class'] in abnormal_classes:
                if pattern['normalized_pattern_left'] and pattern['normalized_pattern_right']:
                    abnormal_patterns_left.append(pattern['normalized_pattern_left'])
                    abnormal_patterns_right.append(pattern['normalized_pattern_right'])
                    abnormal_amplitudes_left.append(pattern['amplitude_left'])
                    abnormal_amplitudes_right.append(pattern['amplitude_right'])

        if len(abnormal_patterns_left) > 0:
            abnormal_patterns_left = np.array(abnormal_patterns_left)
            abnormal_patterns_right = np.array(abnormal_patterns_right)

            binary_templates['abnormal'] = PatternTemplate(
                gait_class='abnormal',
                mean_pattern_left=np.mean(abnormal_patterns_left, axis=0),
                mean_pattern_right=np.mean(abnormal_patterns_right, axis=0),
                std_pattern_left=np.std(abnormal_patterns_left, axis=0),
                std_pattern_right=np.std(abnormal_patterns_right, axis=0),
                n_samples=len(abnormal_patterns_left),
                mean_amplitude_left=np.mean(abnormal_amplitudes_left),
                mean_amplitude_right=np.mean(abnormal_amplitudes_right)
            )

            print(f"\nAbnormal template: {len(abnormal_patterns_left)} samples")
            print(f"  Includes classes: {abnormal_classes}")

        return binary_templates


class RealDataEvaluator:
    """Evaluate STAGE 2 detector with real data"""

    def __init__(self, templates: Dict[str, PatternTemplate], patterns_file: str):
        """
        Initialize evaluator.

        Args:
            templates: Pattern templates for matching
            patterns_file: Real GAVD patterns for testing
        """
        self.templates = templates

        with open(patterns_file, 'r') as f:
            self.patterns = json.load(f)

    def compute_dtw_distance(self, pattern1: np.ndarray, pattern2: np.ndarray) -> float:
        """Compute DTW distance between two patterns"""
        distance, _ = fastdtw(
            pattern1.reshape(-1, 1),
            pattern2.reshape(-1, 1),
            dist=euclidean
        )
        return distance

    def classify_pattern(self, pattern: dict, classification_type: str = 'binary') -> str:
        """
        Classify a gait pattern using DTW template matching.

        Args:
            pattern: Pattern to classify
            classification_type: 'binary' or 'multiclass'

        Returns:
            Predicted class
        """
        if not pattern['normalized_pattern_left'] or not pattern['normalized_pattern_right']:
            return 'unknown'

        pattern_left = np.array(pattern['normalized_pattern_left'])
        pattern_right = np.array(pattern['normalized_pattern_right'])

        # Compute DTW distances to all templates
        distances = {}

        for gait_class, template in self.templates.items():
            # Average DTW distance for left and right
            dist_left = self.compute_dtw_distance(pattern_left, template.mean_pattern_left)
            dist_right = self.compute_dtw_distance(pattern_right, template.mean_pattern_right)
            distances[gait_class] = (dist_left + dist_right) / 2

        # Find closest template
        predicted_class = min(distances, key=distances.get)

        return predicted_class

    def evaluate_binary(self) -> Dict:
        """
        Evaluate binary classification (normal vs abnormal).

        Returns:
            Evaluation metrics
        """
        print("\n" + "="*70)
        print("BINARY CLASSIFICATION EVALUATION")
        print("="*70)

        # Map classes to binary
        normal_classes = ['normal']

        true_labels = []
        pred_labels = []

        for pattern in self.patterns:
            # Skip invalid patterns
            if not pattern['normalized_pattern_left'] or not pattern['normalized_pattern_right']:
                continue

            # True label
            true_class = 'normal' if pattern['gait_class'] in normal_classes else 'abnormal'
            true_labels.append(true_class)

            # Predicted label
            pred_class = self.classify_pattern(pattern, 'binary')
            pred_labels.append(pred_class)

        # Compute metrics
        true_labels = np.array(true_labels)
        pred_labels = np.array(pred_labels)

        # Overall accuracy
        accuracy = np.mean(true_labels == pred_labels)

        # Class-specific metrics
        normal_mask = true_labels == 'normal'
        abnormal_mask = true_labels == 'abnormal'

        sensitivity = np.mean(pred_labels[abnormal_mask] == 'abnormal') if abnormal_mask.any() else 0
        specificity = np.mean(pred_labels[normal_mask] == 'normal') if normal_mask.any() else 0

        # Confusion matrix
        tp = np.sum((true_labels == 'abnormal') & (pred_labels == 'abnormal'))
        tn = np.sum((true_labels == 'normal') & (pred_labels == 'normal'))
        fp = np.sum((true_labels == 'normal') & (pred_labels == 'abnormal'))
        fn = np.sum((true_labels == 'abnormal') & (pred_labels == 'normal'))

        results = {
            'accuracy': accuracy,
            'sensitivity': sensitivity,
            'specificity': specificity,
            'tp': int(tp),
            'tn': int(tn),
            'fp': int(fp),
            'fn': int(fn),
            'n_samples': len(true_labels)
        }

        # Print results
        print(f"\nResults on {len(true_labels)} samples:")
        print(f"  Accuracy: {accuracy*100:.1f}%")
        print(f"  Sensitivity: {sensitivity*100:.1f}%")
        print(f"  Specificity: {specificity*100:.1f}%")
        print(f"\nConfusion Matrix:")
        print(f"  True Positive (abnormal detected): {tp}")
        print(f"  True Negative (normal detected): {tn}")
        print(f"  False Positive (normal → abnormal): {fp}")
        print(f"  False Negative (abnormal → normal): {fn}")

        return results

    def evaluate_multiclass(self) -> Dict:
        """
        Evaluate multi-class classification.

        Returns:
            Evaluation metrics
        """
        print("\n" + "="*70)
        print("MULTI-CLASS CLASSIFICATION EVALUATION")
        print("="*70)

        # Only evaluate classes with templates
        valid_classes = set(self.templates.keys())

        true_labels = []
        pred_labels = []

        for pattern in self.patterns:
            # Skip invalid patterns
            if not pattern['normalized_pattern_left'] or not pattern['normalized_pattern_right']:
                continue

            # Skip classes without templates
            if pattern['gait_class'] not in valid_classes:
                continue

            true_labels.append(pattern['gait_class'])
            pred_labels.append(self.classify_pattern(pattern, 'multiclass'))

        # Compute metrics
        true_labels = np.array(true_labels)
        pred_labels = np.array(pred_labels)

        # Overall accuracy
        accuracy = np.mean(true_labels == pred_labels)

        # Per-class accuracy
        class_accuracies = {}
        for gait_class in valid_classes:
            mask = true_labels == gait_class
            if mask.any():
                class_acc = np.mean(pred_labels[mask] == gait_class)
                class_accuracies[gait_class] = class_acc

        results = {
            'accuracy': accuracy,
            'class_accuracies': class_accuracies,
            'n_samples': len(true_labels),
            'n_classes': len(valid_classes)
        }

        # Print results
        print(f"\nResults on {len(true_labels)} samples, {len(valid_classes)} classes:")
        print(f"  Overall Accuracy: {accuracy*100:.1f}%")
        print(f"\nPer-class accuracy:")
        for gait_class, acc in sorted(class_accuracies.items(), key=lambda x: x[1], reverse=True):
            n_samples = np.sum(true_labels == gait_class)
            print(f"  {gait_class}: {acc*100:.1f}% ({n_samples} samples)")

        return results


def main():
    """Main evaluation pipeline"""

    print("="*80)
    print("STAGE 2 EVALUATION WITH REAL GAVD DATA")
    print("="*80)
    print()

    patterns_file = "gavd_real_patterns.json"

    # Build templates
    builder = RealDataTemplateBuilder(patterns_file)

    # Multi-class templates
    multiclass_templates = builder.build_templates()

    # Binary templates
    binary_templates = builder.create_binary_templates(multiclass_templates)

    # Evaluate binary classification
    binary_evaluator = RealDataEvaluator(binary_templates, patterns_file)
    binary_results = binary_evaluator.evaluate_binary()

    # Evaluate multi-class classification
    multiclass_evaluator = RealDataEvaluator(multiclass_templates, patterns_file)
    multiclass_results = multiclass_evaluator.evaluate_multiclass()

    # Save results
    results = {
        'binary': binary_results,
        'multiclass': multiclass_results,
        'templates': {
            'binary_classes': list(binary_templates.keys()),
            'multiclass_classes': list(multiclass_templates.keys()),
            'n_binary_templates': len(binary_templates),
            'n_multiclass_templates': len(multiclass_templates)
        }
    }

    output_file = "stage2_real_data_results.json"
    with open(output_file, 'w') as f:
        # Convert numpy types to native Python types
        def convert_numpy(obj):
            if isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [convert_numpy(v) for v in obj]
            elif isinstance(obj, (np.integer, np.floating)):
                return obj.item()
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            else:
                return obj

        json.dump(convert_numpy(results), f, indent=2)

    print("\n" + "="*80)
    print("EVALUATION COMPLETE!")
    print("="*80)
    print(f"\nResults saved to: {output_file}")
    print(f"\nSummary:")
    print(f"  Binary Classification: {binary_results['accuracy']*100:.1f}%")
    print(f"  Multi-class Classification: {multiclass_results['accuracy']*100:.1f}%")
    print()


if __name__ == "__main__":
    main()
