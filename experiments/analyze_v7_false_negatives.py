#!/usr/bin/env python3
"""
Analyze V7 False Negatives
===========================

Analyze the 12 pathological cases that V7 missed (false negatives).
Understand what makes them hard to detect.

Author: Gait Analysis System
Date: 2025-10-31
"""

import json
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from scipy.stats import median_abs_deviation
from pathlib import Path
from collections import Counter
import warnings
warnings.filterwarnings('ignore')


class V7FalseNegativeAnalyzer:
    """Analyze false negatives from V7"""

    def __init__(self):
        print("="*80)
        print("V7 False Negative Analysis")
        print("="*80)
        print()

        # Load patterns
        with open('gavd_all_views_patterns.json', 'r') as f:
            self.patterns = json.load(f)

        # Load V7 evaluation
        self.gavd_root = Path("/data/datasets/GAVD")
        self.pose_dir = self.gavd_root / "mediapipe_pose"

        print(f"Loaded {len(self.patterns)} patterns")
        print()

        # Extract features
        self._extract_all_features()
        self._build_baseline()

    def _load_3d_pose(self, csv_path: str) -> pd.DataFrame:
        """Load 3D pose CSV"""
        try:
            return pd.read_csv(csv_path)
        except:
            return None

    def _extract_pure_3d_features(self, pattern: dict) -> dict:
        """Extract all 10 3D features"""

        csv_path = pattern.get('csv_file')
        if not csv_path:
            return None

        df = self._load_3d_pose(csv_path)
        if df is None or len(df) < 30:
            return None

        fps = pattern.get('fps', 30)
        dt = 1.0 / fps

        # Get trajectories
        left_heel_df = df[df['position'] == 'left_heel'].sort_values('frame')
        right_heel_df = df[df['position'] == 'right_heel'].sort_values('frame')

        if len(left_heel_df) < 10 or len(right_heel_df) < 10:
            return None

        left_heel_3d = left_heel_df[['x', 'y', 'z']].values
        right_heel_3d = right_heel_df[['x', 'y', 'z']].values

        left_hip_df = df[df['position'] == 'left_hip'].sort_values('frame')
        right_hip_df = df[df['position'] == 'right_hip'].sort_values('frame')

        # Extract features
        left_y = left_heel_3d[:, 1]
        right_y = right_heel_3d[:, 1]

        peaks_left, _ = find_peaks(left_y, height=np.mean(left_y), distance=5)
        peaks_right, _ = find_peaks(right_y, height=np.mean(right_y), distance=5)

        # 1. Cadence
        n_steps = len(peaks_left) + len(peaks_right)
        duration = len(left_y) / fps
        cadence_3d = (n_steps / duration) * 60 if duration > 0 else 0

        # 2. Step height variability
        var_left = np.std(left_y[peaks_left]) / (np.mean(left_y[peaks_left]) + 1e-6) if len(peaks_left) > 1 else 0
        var_right = np.std(right_y[peaks_right]) / (np.mean(right_y[peaks_right]) + 1e-6) if len(peaks_right) > 1 else 0
        step_height_variability = (var_left + var_right) / 2

        # 3. Gait irregularity
        irreg_left = np.std(np.diff(peaks_left)) / (np.mean(np.diff(peaks_left)) + 1e-6) if len(peaks_left) > 2 else 0
        irreg_right = np.std(np.diff(peaks_right)) / (np.mean(np.diff(peaks_right)) + 1e-6) if len(peaks_right) > 2 else 0
        gait_irregularity_3d = (irreg_left + irreg_right) / 2

        # 4. 3D Velocity
        vel_left_3d = np.diff(left_heel_3d, axis=0) / dt
        vel_right_3d = np.diff(right_heel_3d, axis=0) / dt
        vel_left_mag = np.sqrt(np.sum(vel_left_3d**2, axis=1))
        vel_right_mag = np.sqrt(np.sum(vel_right_3d**2, axis=1))
        velocity_3d = (np.mean(vel_left_mag) + np.mean(vel_right_mag)) / 2

        # 5. 3D Jerkiness
        accel_left_3d = np.diff(vel_left_3d, axis=0) / dt
        accel_right_3d = np.diff(vel_right_3d, axis=0) / dt
        accel_left_mag = np.sqrt(np.sum(accel_left_3d**2, axis=1))
        accel_right_mag = np.sqrt(np.sum(accel_right_3d**2, axis=1))
        jerkiness_3d = (np.mean(accel_left_mag) + np.mean(accel_right_mag)) / 2

        # 6. Cycle duration
        cycle_duration_3d = np.mean(np.diff(peaks_left) / fps) if len(peaks_left) > 1 else 0

        # 7. Stride length (from pattern)
        stride_length_3d = pattern['stride_length_3d']

        # 8. Trunk sway
        trunk_sway = pattern['trunk_sway']

        # 9. Path length
        path_left = np.sum(np.sqrt(np.sum(np.diff(left_heel_3d, axis=0)**2, axis=1)))
        path_right = np.sum(np.sqrt(np.sum(np.diff(right_heel_3d, axis=0)**2, axis=1)))
        path_length_3d = (path_left + path_right) / 2 / duration

        # 10. Step width
        if len(left_hip_df) > 0 and len(right_hip_df) > 0:
            left_hip_x = left_hip_df['x'].values
            right_hip_x = right_hip_df['x'].values
            min_len = min(len(left_hip_x), len(right_hip_x))
            step_width_3d = np.mean(np.abs(left_hip_x[:min_len] - right_hip_x[:min_len]))
        else:
            step_width_3d = 0

        return {
            'cadence_3d': cadence_3d,
            'step_height_variability': step_height_variability,
            'gait_irregularity_3d': gait_irregularity_3d,
            'velocity_3d': velocity_3d,
            'jerkiness_3d': jerkiness_3d,
            'cycle_duration_3d': cycle_duration_3d,
            'stride_length_3d': stride_length_3d,
            'trunk_sway': trunk_sway,
            'path_length_3d': path_length_3d,
            'step_width_3d': step_width_3d
        }

    def _extract_all_features(self):
        """Extract features for all patterns"""
        print("Extracting features...")
        valid_patterns = []
        for p in self.patterns:
            features = self._extract_pure_3d_features(p)
            if features is not None:
                p['features_3d'] = features
                valid_patterns.append(p)
        self.patterns = valid_patterns
        print(f"Valid patterns: {len(self.patterns)}")
        print()

    def _build_baseline(self):
        """Build baseline from normal patterns"""
        normal = [p for p in self.patterns if p['gait_class'] == 'normal']

        features = ['cadence_3d', 'step_height_variability', 'gait_irregularity_3d',
                   'velocity_3d', 'jerkiness_3d', 'cycle_duration_3d',
                   'stride_length_3d', 'trunk_sway', 'path_length_3d', 'step_width_3d']

        self.baseline = {}
        for feat in features:
            vals = np.array([p['features_3d'][feat] for p in normal])
            self.baseline[f'{feat}_median'] = np.median(vals)
            self.baseline[f'{feat}_mad'] = median_abs_deviation(vals, scale='normal')

    def compute_z(self, p: dict) -> float:
        """Compute composite Z-score"""
        features = ['cadence_3d', 'step_height_variability', 'gait_irregularity_3d',
                   'velocity_3d', 'jerkiness_3d', 'cycle_duration_3d',
                   'stride_length_3d', 'trunk_sway', 'path_length_3d', 'step_width_3d']

        z_scores = []
        for feat in features:
            val = p['features_3d'][feat]
            med = self.baseline[f'{feat}_median']
            mad = self.baseline[f'{feat}_mad']
            z = abs(val - med) / (mad + 1e-10)
            z_scores.append(z)

        return np.mean(z_scores)

    def analyze(self):
        """Analyze false negatives"""

        print("="*80)
        print("FALSE NEGATIVE ANALYSIS")
        print("="*80)
        print()

        threshold = 0.75
        false_negatives = []
        true_positives = []

        for p in self.patterns:
            if p['gait_class'] != 'pathological':
                continue

            z = self.compute_z(p)
            pred = 'pathological' if z > threshold else 'normal'

            if pred == 'normal':  # False negative
                p['z_score'] = z
                false_negatives.append(p)
            else:  # True positive
                p['z_score'] = z
                true_positives.append(p)

        print(f"Total pathological cases: {len(false_negatives) + len(true_positives)}")
        print(f"True positives: {len(true_positives)} ({len(true_positives)/(len(false_negatives)+len(true_positives))*100:.1f}%)")
        print(f"False negatives: {len(false_negatives)} ({len(false_negatives)/(len(false_negatives)+len(true_positives))*100:.1f}%)")
        print()

        # Analyze false negatives
        print("="*80)
        print("FALSE NEGATIVE DETAILS")
        print("="*80)
        print()

        # By pathology
        fn_pathologies = [p['gait_pathology'] for p in false_negatives]
        pathology_counts = Counter(fn_pathologies)

        print("False Negatives by Pathology:")
        for pathology, count in sorted(pathology_counts.items(), key=lambda x: -x[1]):
            # Total of this pathology
            total = sum(1 for p in self.patterns if p['gait_class'] == 'pathological' and p['gait_pathology'] == pathology)
            pct = count / total * 100 if total > 0 else 0
            print(f"  {pathology:20s}: {count:2d} / {total:3d} missed ({pct:5.1f}%)")
        print()

        # By camera view
        fn_views = [p['camera_view'] for p in false_negatives]
        view_counts = Counter(fn_views)

        print("False Negatives by Camera View:")
        for view, count in sorted(view_counts.items(), key=lambda x: -x[1]):
            total = sum(1 for p in self.patterns if p['gait_class'] == 'pathological' and p['camera_view'] == view)
            pct = count / total * 100 if total > 0 else 0
            print(f"  {view:15s}: {count:2d} / {total:3d} missed ({pct:5.1f}%)")
        print()

        # Z-score analysis
        fn_z_scores = [p['z_score'] for p in false_negatives]
        tp_z_scores = [p['z_score'] for p in true_positives]

        print("Z-score Analysis:")
        print(f"  False Negatives: {np.mean(fn_z_scores):.3f} ± {np.std(fn_z_scores):.3f} (range: {np.min(fn_z_scores):.3f} - {np.max(fn_z_scores):.3f})")
        print(f"  True Positives:  {np.mean(tp_z_scores):.3f} ± {np.std(tp_z_scores):.3f} (range: {np.min(tp_z_scores):.3f} - {np.max(tp_z_scores):.3f})")
        print(f"  Threshold: {threshold}")
        print()

        # Individual false negatives
        print("="*80)
        print("INDIVIDUAL FALSE NEGATIVES (sorted by Z-score)")
        print("="*80)
        print()

        false_negatives_sorted = sorted(false_negatives, key=lambda x: x['z_score'])

        for i, fn in enumerate(false_negatives_sorted):
            print(f"{i+1}. Seq: {fn['seq_id']}, View: {fn['camera_view']}, Pathology: {fn['gait_pathology']}, Z={fn['z_score']:.3f}")

            # Feature breakdown
            features = fn['features_3d']
            print(f"   Features:")
            for feat_name in ['cadence_3d', 'stride_length_3d', 'gait_irregularity_3d', 'trunk_sway']:
                val = features[feat_name]
                med = self.baseline[f'{feat_name}_median']
                mad = self.baseline[f'{feat_name}_mad']
                z = abs(val - med) / (mad + 1e-10)
                print(f"     {feat_name:25s}: {val:8.4f} (baseline: {med:8.4f}, Z={z:.2f})")
            print()

        # Summary statistics
        print("="*80)
        print("SUMMARY")
        print("="*80)
        print()

        print("Most commonly missed pathologies:")
        for pathology, count in sorted(pathology_counts.items(), key=lambda x: -x[1])[:3]:
            total = sum(1 for p in self.patterns if p['gait_class'] == 'pathological' and p['gait_pathology'] == pathology)
            print(f"  {pathology}: {count}/{total} ({count/total*100:.1f}% miss rate)")
        print()

        print("Most problematic camera view:")
        for view, count in sorted(view_counts.items(), key=lambda x: -x[1])[:1]:
            total = sum(1 for p in self.patterns if p['gait_class'] == 'pathological' and p['camera_view'] == view)
            print(f"  {view}: {count}/{total} ({count/total*100:.1f}% miss rate)")
        print()

        print("Recommendations:")
        print("  1. Lower threshold for high-miss pathologies")
        print("  2. View-specific thresholds (especially for problematic views)")
        print("  3. Pathology-specific detectors for commonly missed types")
        print()

        # Save results
        results = {
            'total_pathological': len(false_negatives) + len(true_positives),
            'true_positives': len(true_positives),
            'false_negatives': len(false_negatives),
            'miss_rate': len(false_negatives) / (len(false_negatives) + len(true_positives)),
            'threshold': threshold,
            'false_negative_z_mean': float(np.mean(fn_z_scores)),
            'false_negative_z_std': float(np.std(fn_z_scores)),
            'true_positive_z_mean': float(np.mean(tp_z_scores)),
            'true_positive_z_std': float(np.std(tp_z_scores)),
            'pathology_breakdown': dict(pathology_counts),
            'view_breakdown': dict(view_counts),
            'false_negative_cases': [
                {
                    'seq_id': fn['seq_id'],
                    'pathology': fn['gait_pathology'],
                    'view': fn['camera_view'],
                    'z_score': float(fn['z_score']),
                    'features': {k: float(v) for k, v in fn['features_3d'].items()}
                }
                for fn in false_negatives_sorted
            ]
        }

        with open('v7_false_negative_analysis.json', 'w') as f:
            json.dump(results, f, indent=2)

        print("Saved analysis to: v7_false_negative_analysis.json")
        print()


def main():
    analyzer = V7FalseNegativeAnalyzer()
    analyzer.analyze()


if __name__ == "__main__":
    main()
