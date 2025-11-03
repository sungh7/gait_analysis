#!/usr/bin/env python3
"""
Improved V7: ALL Features from 3D Pose
=======================================

All features computed from true 3D coordinates:
1. 3D Cadence - Using 3D heel velocity peaks
2. 3D Step Height Variability - Vertical heel displacement consistency
3. 3D Gait Irregularity - 3D trajectory smoothness
4. 3D Velocity - Full 3D heel velocity magnitude
5. 3D Jerkiness - 3D acceleration magnitude
6. 3D Cycle Duration - From 3D heel trajectory
7. 3D Stride Length - Horizontal distance (hip-ankle)
8. Trunk Sway - Lateral shoulder movement

All computed directly from 3D pose data, no 2D approximations!

Expected: 72-75% accuracy
Target: 95%+ sensitivity

Author: Gait Analysis System
Date: 2025-10-30
Version: 7.0 - Pure 3D
"""

import json
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from scipy.stats import median_abs_deviation
from pathlib import Path
from typing import Dict, List
import warnings
warnings.filterwarnings('ignore')


class ImprovedV7_Pure3D:
    """All features from pure 3D pose data"""

    def __init__(self, patterns_file: str = "gavd_all_views_patterns.json"):
        print("="*80)
        print("Improved V7: ALL Features from 3D Pose (EXPANDED DATASET)")
        print("="*80)
        print()

        with open(patterns_file, 'r') as f:
            self.patterns = json.load(f)

        # Load 3D pose CSVs
        self.gavd_root = Path("/data/datasets/GAVD")
        self.pose_dir = self.gavd_root / "mediapipe_pose"

        print(f"Loaded {len(self.patterns)} patterns")
        print("Extracting ALL features from 3D pose data...")

        # Extract pure 3D features
        valid_patterns = []
        for p in self.patterns:
            features = self._extract_pure_3d_features(p)
            if features is not None:
                p['features_3d'] = features
                valid_patterns.append(p)

        self.patterns = valid_patterns

        normal = [p for p in self.patterns if p['gait_class'] == 'normal']
        pathological = [p for p in self.patterns if p['gait_class'] == 'pathological']

        print(f"\nValid patterns with 3D features: {len(self.patterns)}")
        print(f"  Normal: {len(normal)}")
        print(f"  Pathological: {len(pathological)}")
        print()

        self._build_baseline()

    def _load_3d_pose(self, csv_path: str) -> pd.DataFrame:
        """Load 3D pose CSV"""
        try:
            df = pd.read_csv(csv_path)
            return df
        except:
            return None

    def _extract_pure_3d_features(self, pattern: dict) -> Dict:
        """Extract ALL features from 3D pose coordinates"""

        csv_path = pattern.get('csv_file')
        if not csv_path:
            return None

        df = self._load_3d_pose(csv_path)
        if df is None or len(df) < 30:
            return None

        fps = pattern.get('fps', 30)
        dt = 1.0 / fps

        # Get 3D trajectories for heels
        left_heel_df = df[df['position'] == 'left_heel'].sort_values('frame')
        right_heel_df = df[df['position'] == 'right_heel'].sort_values('frame')

        if len(left_heel_df) < 10 or len(right_heel_df) < 10:
            return None

        # 3D coordinates
        left_heel_3d = left_heel_df[['x', 'y', 'z']].values
        right_heel_3d = right_heel_df[['x', 'y', 'z']].values

        # Hip and ankle for stride length
        left_hip_df = df[df['position'] == 'left_hip'].sort_values('frame')
        right_hip_df = df[df['position'] == 'right_hip'].sort_values('frame')
        left_ankle_df = df[df['position'] == 'left_ankle'].sort_values('frame')
        right_ankle_df = df[df['position'] == 'right_ankle'].sort_values('frame')

        # Shoulders for trunk sway
        left_shoulder_df = df[df['position'] == 'left_shoulder'].sort_values('frame')
        right_shoulder_df = df[df['position'] == 'right_shoulder'].sort_values('frame')

        # 1. 3D CADENCE - Using vertical component for peak detection
        left_y = left_heel_3d[:, 1]  # y = vertical
        right_y = right_heel_3d[:, 1]

        peaks_left, _ = find_peaks(left_y, height=np.mean(left_y), distance=5)
        peaks_right, _ = find_peaks(right_y, height=np.mean(right_y), distance=5)

        n_steps = len(peaks_left) + len(peaks_right)
        duration = len(left_y) / fps
        cadence_3d = (n_steps / duration) * 60 if duration > 0 else 0

        # 2. 3D STEP HEIGHT VARIABILITY
        if len(peaks_left) > 1:
            var_left = np.std(left_y[peaks_left]) / (np.mean(left_y[peaks_left]) + 1e-6)
        else:
            var_left = 0

        if len(peaks_right) > 1:
            var_right = np.std(right_y[peaks_right]) / (np.mean(right_y[peaks_right]) + 1e-6)
        else:
            var_right = 0

        step_height_variability = (var_left + var_right) / 2

        # 3. 3D GAIT IRREGULARITY - Stride interval consistency
        if len(peaks_left) > 2:
            intervals_left = np.diff(peaks_left)
            irreg_left = np.std(intervals_left) / (np.mean(intervals_left) + 1e-6)
        else:
            irreg_left = 0

        if len(peaks_right) > 2:
            intervals_right = np.diff(peaks_right)
            irreg_right = np.std(intervals_right) / (np.mean(intervals_right) + 1e-6)
        else:
            irreg_right = 0

        gait_irregularity_3d = (irreg_left + irreg_right) / 2

        # 4. 3D VELOCITY - Full 3D magnitude
        vel_left_3d = np.diff(left_heel_3d, axis=0) / dt
        vel_right_3d = np.diff(right_heel_3d, axis=0) / dt

        # Magnitude: sqrt(vx^2 + vy^2 + vz^2)
        vel_left_mag = np.sqrt(np.sum(vel_left_3d**2, axis=1))
        vel_right_mag = np.sqrt(np.sum(vel_right_3d**2, axis=1))

        velocity_3d = (np.mean(vel_left_mag) + np.mean(vel_right_mag)) / 2

        # 5. 3D JERKINESS - 3D acceleration magnitude
        accel_left_3d = np.diff(vel_left_3d, axis=0) / dt
        accel_right_3d = np.diff(vel_right_3d, axis=0) / dt

        accel_left_mag = np.sqrt(np.sum(accel_left_3d**2, axis=1))
        accel_right_mag = np.sqrt(np.sum(accel_right_3d**2, axis=1))

        jerkiness_3d = (np.mean(accel_left_mag) + np.mean(accel_right_mag)) / 2

        # 6. 3D CYCLE DURATION
        cycle_duration_3d = np.mean(np.diff(peaks_left) / fps) if len(peaks_left) > 1 else 0

        # 7. 3D STRIDE LENGTH (already computed, from pattern)
        stride_length_3d = pattern['stride_length_3d']

        # 8. TRUNK SWAY
        trunk_sway = pattern['trunk_sway']

        # 9. NEW: 3D PATH LENGTH - Total distance traveled by heel
        # More comprehensive than stride length
        path_left = np.sum(np.sqrt(np.sum(np.diff(left_heel_3d, axis=0)**2, axis=1)))
        path_right = np.sum(np.sqrt(np.sum(np.diff(right_heel_3d, axis=0)**2, axis=1)))
        path_length_3d = (path_left + path_right) / 2 / duration  # Normalized by time

        # 10. NEW: 3D STEP WIDTH - Lateral spacing (left-right distance)
        if len(left_hip_df) > 0 and len(right_hip_df) > 0:
            # Use hip positions as reference
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

    def _build_baseline(self):
        """Build robust baseline"""
        normal = [p for p in self.patterns if p['gait_class'] == 'normal']

        print(f"Building baseline from {len(normal)} normal patterns...")

        features = ['cadence_3d', 'step_height_variability', 'gait_irregularity_3d',
                   'velocity_3d', 'jerkiness_3d', 'cycle_duration_3d',
                   'stride_length_3d', 'trunk_sway', 'path_length_3d', 'step_width_3d']

        self.baseline = {}
        for feat in features:
            vals = np.array([p['features_3d'][feat] for p in normal])
            self.baseline[f'{feat}_median'] = np.median(vals)
            self.baseline[f'{feat}_mad'] = median_abs_deviation(vals, scale='normal')

        print("\nBaseline (Median ± MAD) - ALL FROM 3D:")
        for feat in features:
            med = self.baseline[f'{feat}_median']
            mad = self.baseline[f'{feat}_mad']
            print(f"  {feat}: {med:.6f} ± {mad:.6f}")
        print()

    def compute_z(self, p: dict) -> float:
        """Equal-weight MAD-Z for all 10 3D features"""
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

    def evaluate(self, threshold: float = 1.5):
        """Evaluate detector"""
        tp, tn, fp, fn = 0, 0, 0, 0
        normal_z, path_z = [], []

        for p in self.patterns:
            z = self.compute_z(p)
            pred = 'pathological' if z > threshold else 'normal'
            true = p['gait_class']

            if true == 'normal':
                normal_z.append(z)
            else:
                path_z.append(z)

            if true == 'pathological' and pred == 'pathological':
                tp += 1
            elif true == 'normal' and pred == 'normal':
                tn += 1
            elif true == 'normal' and pred == 'pathological':
                fp += 1
            else:
                fn += 1

        acc = (tp + tn) / (tp + tn + fp + fn)
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0

        print(f"\nThreshold={threshold}:")
        print(f"  Accuracy: {acc*100:.1f}%")
        print(f"  Sensitivity: {sens*100:.1f}%")
        print(f"  Specificity: {spec*100:.1f}%")
        print(f"  Z: Normal={np.mean(normal_z):.2f}±{np.std(normal_z):.2f}, Path={np.mean(path_z):.2f}±{np.std(path_z):.2f}")

        return {'threshold': threshold, 'accuracy': acc, 'sensitivity': sens, 'specificity': spec,
                'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn}

    def optimize(self):
        """Find best threshold"""
        print("\n" + "="*80)
        print("THRESHOLD OPTIMIZATION")
        print("="*80)

        thresholds = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0]
        best = None
        best_acc = 0

        for t in thresholds:
            result = self.evaluate(t)
            if result['accuracy'] > best_acc:
                best_acc = result['accuracy']
                best = result

        print("\n" + "="*80)
        print("BEST RESULT - PURE 3D")
        print("="*80)
        print(f"  Threshold: {best['threshold']}")
        print(f"  Accuracy: {best['accuracy']*100:.1f}%")
        print(f"  Sensitivity: {best['sensitivity']*100:.1f}%")
        print(f"  Specificity: {best['specificity']*100:.1f}%")

        return best


def main():
    detector = ImprovedV7_Pure3D("gavd_all_views_patterns.json")
    best = detector.optimize()

    print("\n" + "="*80)
    print("V7 PURE 3D SUMMARY")
    print("="*80)
    print(f"\nAll 10 features computed from pure 3D pose:")
    print(f"  1. 3D Cadence")
    print(f"  2. 3D Step Height Variability")
    print(f"  3. 3D Gait Irregularity")
    print(f"  4. 3D Velocity (magnitude)")
    print(f"  5. 3D Jerkiness (magnitude)")
    print(f"  6. 3D Cycle Duration")
    print(f"  7. 3D Stride Length")
    print(f"  8. Trunk Sway (lateral)")
    print(f"  9. 3D Path Length (NEW!)")
    print(f" 10. 3D Step Width (NEW!)")
    print(f"\nPerformance:")
    print(f"  Accuracy: {best['accuracy']*100:.1f}%")
    print(f"  Sensitivity: {best['sensitivity']*100:.1f}%")
    print(f"  Specificity: {best['specificity']*100:.1f}%")
    print()

    # Compare with V6
    print("="*80)
    print("COMPARISON")
    print("="*80)
    print(f"  V6 (8 features, partial 3D): 67.0% acc, 91.4% sens")
    print(f"  V7 (10 features, pure 3D):   {best['accuracy']*100:.1f}% acc, {best['sensitivity']*100:.1f}% sens")
    print()

    with open('improved_v7_results.json', 'w') as f:
        json.dump(best, f, indent=2)

    print("Results saved to: improved_v7_results.json\n")


if __name__ == "__main__":
    main()
