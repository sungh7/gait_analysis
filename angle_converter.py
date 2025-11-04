"""
MediaPipe-to-hospital angle conversion module.

Provides multiple kinematic definitions (joint/segment-based) and a
cross-validation workflow to identify the best-performing conversion for
knee, hip, or ankle angles while guarding against overfitting.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import nnls, minimize
from scipy.stats import pearsonr
from scipy.interpolate import UnivariateSpline
from sklearn.linear_model import Ridge


@dataclass
class MethodMetrics:
    rmse: float
    mae: float
    correlation: float
    icc: float


class AngleConverter:
    """Convert MediaPipe 3D landmark trajectories into hospital angle formats."""

    def __init__(self, coordinate_report: Optional[str] = None) -> None:
        self.conversion_params: Dict[str, Dict[str, Dict[str, float]]] = {}
        self.axis_flips = {'x': 1.0, 'y': 1.0, 'z': 1.0}
        if coordinate_report is not None:
            self._load_coordinate_report(coordinate_report)

        self.method_registry = {
            'knee': ['joint_angle', 'joint_angle_flexion', 'joint_angle_inverted', 'projected_2d'],
            'hip': ['segment_to_vertical', 'joint_angle', 'trunk_relative', 'pelvic_tilt'],
            'ankle': ['joint_angle', 'joint_angle_flexion', 'segment_to_horizontal', 'foot_ground_angle'],
        }

        # Conversion methods: linear, polynomial, piecewise, spline
        self.conversion_methods = ['linear', 'polynomial_2nd', 'piecewise', 'ridge']

    # ------------------------------------------------------------------
    # Coordinate conventions
    # ------------------------------------------------------------------
    def _load_coordinate_report(self, report_path: str) -> None:
        report = json.loads(Path(report_path).read_text(encoding='utf-8'))
        instructions = report.get('recommendation', {}).get('transform_instructions', {})
        for axis, config in instructions.items():
            scale = config.get('scale', 1.0)
            if axis in self.axis_flips:
                self.axis_flips[axis] = scale

    def _apply_axis_flips(self, coords: np.ndarray) -> np.ndarray:
        flipped = coords.copy()
        flipped[:, 0] *= self.axis_flips['x']
        flipped[:, 1] *= self.axis_flips['y']
        flipped[:, 2] *= self.axis_flips['z']
        return flipped

    # ------------------------------------------------------------------
    # Angle computation helpers
    # ------------------------------------------------------------------
    def _extract_point(self, landmarks: pd.DataFrame, side: str, joint: str) -> np.ndarray:
        cols = [f'x_{side}_{joint}', f'y_{side}_{joint}', f'z_{side}_{joint}']
        if any(col not in landmarks.columns for col in cols):
            raise KeyError(f"Missing coordinates for {side} {joint}")
        points = landmarks[cols].to_numpy(dtype=float)
        return self._apply_axis_flips(points)

    @staticmethod
    def _joint_angle(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> np.ndarray:
        v1 = p1 - p2
        v2 = p3 - p2
        v1_norm = np.linalg.norm(v1, axis=1)
        v2_norm = np.linalg.norm(v2, axis=1)
        valid = (v1_norm > 0) & (v2_norm > 0)
        cos_angle = np.ones(len(p1))
        cos_angle[valid] = np.sum(v1[valid] * v2[valid], axis=1) / (v1_norm[valid] * v2_norm[valid])
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angles = np.degrees(np.arccos(cos_angle))
        angles[~valid] = np.nan
        return angles

    @staticmethod
    def _segment_angle_to_axis(segment: np.ndarray, axis_vector: np.ndarray) -> np.ndarray:
        seg_norm = np.linalg.norm(segment, axis=1)
        valid = seg_norm > 0
        dot = np.zeros(len(segment))
        dot[valid] = np.sum(segment[valid] * axis_vector, axis=1) / seg_norm[valid]
        dot = np.clip(dot, -1.0, 1.0)
        angles = np.degrees(np.arccos(dot))
        angles[~valid] = np.nan
        return angles

    @staticmethod
    def _vector_angle(v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
        """Calculate angle between two vectors."""
        v1_norm = np.linalg.norm(v1, axis=1)
        v2_norm = np.linalg.norm(v2, axis=1)
        valid = (v1_norm > 0) & (v2_norm > 0)
        cos_angle = np.ones(len(v1))
        cos_angle[valid] = np.sum(v1[valid] * v2[valid], axis=1) / (v1_norm[valid] * v2_norm[valid])
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angles = np.degrees(np.arccos(cos_angle))
        angles[~valid] = np.nan
        return angles

    def compute_series(self, landmarks: pd.DataFrame, method: str, joint_type: str, side: str) -> np.ndarray:
        joint_type = joint_type.lower()
        side = side.lower()
        if joint_type not in self.method_registry:
            raise ValueError(f"Unsupported joint type: {joint_type}")
        if method not in self.method_registry[joint_type]:
            raise ValueError(f"Unsupported method '{method}' for joint '{joint_type}'")

        if joint_type == 'knee':
            hip = self._extract_point(landmarks, side, 'hip')
            knee = self._extract_point(landmarks, side, 'knee')
            ankle = self._extract_point(landmarks, side, 'ankle')
            angle = self._joint_angle(hip, knee, ankle)
            if method == 'joint_angle_flexion':
                return 180.0 - angle
            if method == 'joint_angle_inverted':
                return 180.0 - angle
            if method == 'projected_2d':
                hip_2d = hip[:, 1:3]
                knee_2d = knee[:, 1:3]
                ankle_2d = ankle[:, 1:3]
                return self._projected_angle(hip_2d, knee_2d, ankle_2d)
            return angle

        if joint_type == 'hip':
            hip = self._extract_point(landmarks, side, 'hip')
            knee = self._extract_point(landmarks, side, 'knee')

            if method == 'segment_to_vertical':
                segment = knee - hip
                vertical = np.array([0.0, -1.0, 0.0]) * np.array([
                    self.axis_flips['x'], self.axis_flips['y'], self.axis_flips['z']
                ])
                return self._segment_angle_to_axis(segment, vertical)

            if method == 'joint_angle':
                shoulder = self._extract_point(landmarks, side, 'shoulder')
                return self._joint_angle(shoulder, hip, knee)

            if method == 'trunk_relative':
                # Trunk vector: midpoint of shoulders to midpoint of hips
                left_shoulder = self._extract_point(landmarks, 'left', 'shoulder')
                right_shoulder = self._extract_point(landmarks, 'right', 'shoulder')
                left_hip = self._extract_point(landmarks, 'left', 'hip')
                right_hip = self._extract_point(landmarks, 'right', 'hip')

                shoulder_mid = (left_shoulder + right_shoulder) / 2
                hip_mid = (left_hip + right_hip) / 2
                trunk = hip_mid - shoulder_mid  # Pointing down

                # Thigh vector: hip to knee
                thigh = knee - hip

                # Angle between trunk and thigh
                return self._vector_angle(trunk, thigh)

            if method == 'pelvic_tilt':
                # Pelvic tilt: angle of line connecting left-right hips from horizontal
                left_hip = self._extract_point(landmarks, 'left', 'hip')
                right_hip = self._extract_point(landmarks, 'right', 'hip')

                # Hip line vector
                hip_line = right_hip - left_hip

                # Project to sagittal plane (y-z)
                hip_line_sagittal = hip_line[:, [1, 2]]  # y, z

                # Horizontal in sagittal plane
                horizontal = np.array([0.0, 1.0])  # z-axis (forward)

                # Calculate tilt angle
                norms = np.linalg.norm(hip_line_sagittal, axis=1)
                valid = norms > 0
                angles = np.zeros(len(hip_line_sagittal))
                angles[valid] = np.degrees(np.arcsin(
                    np.clip(hip_line_sagittal[valid, 0] / norms[valid], -1.0, 1.0)
                ))
                angles[~valid] = np.nan

                # Combined with thigh angle
                thigh = knee - hip
                vertical = np.array([0.0, -1.0, 0.0]) * np.array([
                    self.axis_flips['x'], self.axis_flips['y'], self.axis_flips['z']
                ])
                thigh_angle = self._segment_angle_to_axis(thigh, vertical)

                return thigh_angle + angles  # Pelvic tilt correction

        if joint_type == 'ankle':
            knee = self._extract_point(landmarks, side, 'knee')
            ankle = self._extract_point(landmarks, side, 'ankle')
            foot = self._extract_point(landmarks, side, 'foot_index')
            heel = self._extract_point(landmarks, side, 'heel')
            if method == 'joint_angle_flexion':
                base = self._joint_angle(knee, ankle, foot)
                return 180.0 - base

            if method == 'segment_to_horizontal':
                segment = foot - ankle
                forward = np.array([0.0, 0.0, 1.0]) * np.array([
                    self.axis_flips['x'], self.axis_flips['y'], self.axis_flips['z']
                ])
                return self._segment_angle_to_axis(segment, forward) - 90.0

            if method == 'foot_ground_angle':
                # Foot vector: heel to toe
                foot_vector = foot - heel

                # Ground plane in sagittal view: horizontal (z-axis)
                # Project foot vector to sagittal plane (y-z)
                foot_sagittal = foot_vector[:, [1, 2]]  # y, z components

                # Calculate angle from horizontal
                norms = np.linalg.norm(foot_sagittal, axis=1)
                valid = norms > 0
                angles = np.zeros(len(foot_sagittal))

                # Angle from horizontal (positive = dorsiflexion, negative = plantarflexion)
                angles[valid] = np.degrees(np.arctan2(
                    foot_sagittal[valid, 0],  # y (vertical)
                    foot_sagittal[valid, 1]   # z (forward)
                ))
                angles[~valid] = np.nan

                # Adjust to ankle dorsiflexion convention (0° = neutral, +ve = dorsiflexion)
                return 90.0 - angles

            return self._joint_angle(knee, ankle, foot)

        raise ValueError(f"Unhandled joint/method combination: {joint_type} / {method}")

    @staticmethod
    def _projected_angle(hip: np.ndarray, knee: np.ndarray, ankle: np.ndarray) -> np.ndarray:
        v1 = hip - knee
        v2 = ankle - knee
        v1_norm = np.linalg.norm(v1, axis=1)
        v2_norm = np.linalg.norm(v2, axis=1)
        valid = (v1_norm > 0) & (v2_norm > 0)
        cos_angle = np.ones(len(hip))
        cos_angle[valid] = np.sum(v1[valid] * v2[valid], axis=1) / (v1_norm[valid] * v2_norm[valid])
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angles = np.degrees(np.arccos(cos_angle))
        angles[~valid] = np.nan
        return 180.0 - angles

    # ------------------------------------------------------------------
    # Cross-validation workflow
    # ------------------------------------------------------------------
    def fit_and_select(
        self,
        train_data: Sequence[Dict[str, object]],
        val_data: Sequence[Dict[str, object]],
        joint_type: str,
        side: str = 'right',
    ) -> List[Dict[str, object]]:
        joint_type = joint_type.lower()
        results: List[Dict[str, object]] = []

        for method in self.method_registry[joint_type]:
            train_preds = [self.compute_series(sample['mp_landmarks'], method, joint_type, side) for sample in train_data]
            train_targets = [np.asarray(sample['hospital_angles'], dtype=float) for sample in train_data]

            # Try different conversion methods
            for conversion_method in self.conversion_methods:
                if conversion_method == 'linear':
                    offset, scale = self._fit_linear_alignment(train_preds, train_targets)
                    val_summary = self._evaluate_dataset_linear(
                        val_data, method, joint_type, side, offset, scale
                    )
                    results.append({
                        'method': method,
                        'conversion': 'linear',
                        'offset': offset,
                        'scale': scale,
                        'validation': val_summary,
                    })

                elif conversion_method == 'polynomial_2nd':
                    coeffs = self._fit_polynomial_alignment(train_preds, train_targets, degree=2)
                    val_summary = self._evaluate_dataset_polynomial(
                        val_data, method, joint_type, side, coeffs
                    )
                    results.append({
                        'method': method,
                        'conversion': 'polynomial_2nd',
                        'coeffs': coeffs.tolist(),
                        'validation': val_summary,
                    })

                elif conversion_method == 'piecewise':
                    segments = self._fit_piecewise_alignment(train_preds, train_targets, n_segments=2)
                    val_summary = self._evaluate_dataset_piecewise(
                        val_data, method, joint_type, side, segments
                    )
                    results.append({
                        'method': method,
                        'conversion': 'piecewise',
                        'segments': segments,
                        'validation': val_summary,
                    })

                elif conversion_method == 'ridge':
                    # Ridge with higher alpha for more regularization
                    coeffs = self._fit_polynomial_alignment(train_preds, train_targets, degree=2)
                    ridge = Ridge(alpha=1.0)
                    pred_concat = np.concatenate(train_preds)
                    targ_concat = np.concatenate(train_targets)
                    mask = ~np.isnan(pred_concat) & ~np.isnan(targ_concat)
                    if mask.sum() > 3:
                        X = np.column_stack([pred_concat[mask]**i for i in range(3)])
                        ridge.fit(X, targ_concat[mask])
                        coeffs = ridge.coef_

                    val_summary = self._evaluate_dataset_polynomial(
                        val_data, method, joint_type, side, coeffs
                    )
                    results.append({
                        'method': method,
                        'conversion': 'ridge',
                        'coeffs': coeffs.tolist(),
                        'validation': val_summary,
                    })

        results.sort(key=lambda item: item['validation']['overall']['icc'], reverse=True)
        return results

    def _fit_linear_alignment(
        self,
        predictions: Sequence[np.ndarray],
        targets: Sequence[np.ndarray],
    ) -> Tuple[float, float]:
        pred_concat = np.concatenate(predictions)
        targ_concat = np.concatenate(targets)
        mask = ~np.isnan(pred_concat) & ~np.isnan(targ_concat)
        pred = pred_concat[mask]
        targ = targ_concat[mask]
        if len(pred) < 2:
            return 0.0, 1.0
        A = np.column_stack([np.ones_like(pred), pred])
        # Non-negative least squares for stability (allows scale >= 0)
        coeffs, _ = nnls(A, targ)
        offset, scale = float(coeffs[0]), float(coeffs[1])
        if scale < 1e-3:
            # Fallback to unconstrained least squares to avoid degenerate scaling
            coeffs_ls, _, _, _ = np.linalg.lstsq(A, targ, rcond=None)
            offset, scale = float(coeffs_ls[0]), float(coeffs_ls[1])
            if abs(scale) < 1e-3:
                scale = 1.0
        return offset, scale

    def _fit_polynomial_alignment(
        self,
        predictions: Sequence[np.ndarray],
        targets: Sequence[np.ndarray],
        degree: int = 2
    ) -> np.ndarray:
        """Fit polynomial transformation: y = c0 + c1*x + c2*x^2 + ..."""
        pred_concat = np.concatenate(predictions)
        targ_concat = np.concatenate(targets)
        mask = ~np.isnan(pred_concat) & ~np.isnan(targ_concat)
        pred = pred_concat[mask]
        targ = targ_concat[mask]
        if len(pred) < degree + 1:
            return np.array([0.0, 1.0] + [0.0] * (degree - 1))

        # Polynomial regression with Ridge regularization
        ridge = Ridge(alpha=0.1)
        X = np.column_stack([pred**i for i in range(degree + 1)])
        ridge.fit(X, targ)
        return ridge.coef_

    def _fit_piecewise_alignment(
        self,
        predictions: Sequence[np.ndarray],
        targets: Sequence[np.ndarray],
        n_segments: int = 2
    ) -> List[Tuple[float, float, float]]:
        """Fit piecewise linear: different (offset, scale) for each segment."""
        pred_concat = np.concatenate(predictions)
        targ_concat = np.concatenate(targets)
        mask = ~np.isnan(pred_concat) & ~np.isnan(targ_concat)
        pred = pred_concat[mask]
        targ = targ_concat[mask]

        if len(pred) < n_segments * 3:
            return [(0.0, 1.0, 100.0)] * n_segments

        # Split by quantiles
        quantiles = np.linspace(0, 100, n_segments + 1)
        boundaries = np.percentile(pred, quantiles)

        segments = []
        for i in range(n_segments):
            lower = boundaries[i]
            upper = boundaries[i + 1] if i < n_segments - 1 else np.inf

            mask_seg = (pred >= lower) & (pred < upper)
            if mask_seg.sum() < 2:
                segments.append((0.0, 1.0, upper))
                continue

            pred_seg = pred[mask_seg]
            targ_seg = targ[mask_seg]

            A = np.column_stack([np.ones_like(pred_seg), pred_seg])
            coeffs, _ = nnls(A, targ_seg)
            offset, scale = float(coeffs[0]), float(coeffs[1])
            segments.append((offset, scale, upper))

        return segments

    def _apply_polynomial_transform(self, pred: np.ndarray, coeffs: np.ndarray) -> np.ndarray:
        """Apply polynomial transformation."""
        result = np.zeros_like(pred, dtype=float)
        for i, c in enumerate(coeffs):
            result += c * (pred ** i)
        return result

    def _apply_piecewise_transform(
        self,
        pred: np.ndarray,
        segments: List[Tuple[float, float, float]]
    ) -> np.ndarray:
        """Apply piecewise linear transformation."""
        result = np.zeros_like(pred, dtype=float)
        for i, (offset, scale, upper_bound) in enumerate(segments):
            if i == len(segments) - 1:
                mask = pred >= segments[i - 1][2] if i > 0 else np.ones_like(pred, dtype=bool)
            else:
                lower = segments[i - 1][2] if i > 0 else -np.inf
                mask = (pred >= lower) & (pred < upper_bound)
            result[mask] = offset + scale * pred[mask]
        return result

    def apply_conversion(
        self,
        raw_series: np.ndarray,
        conversion: str,
        params: Dict[str, object],
    ) -> np.ndarray:
        """Apply a stored conversion to a raw angle series."""
        if raw_series is None:
            return None

        series = np.asarray(raw_series, dtype=float)
        converted = np.full_like(series, np.nan, dtype=float)
        valid = ~np.isnan(series)
        if not valid.any():
            return converted

        if conversion == 'linear':
            offset = float(params.get('offset', 0.0))
            scale = float(params.get('scale', 1.0))
            converted[valid] = offset + scale * series[valid]
            return converted

        if conversion in {'polynomial_2nd', 'ridge'}:
            coeffs = params.get('coeffs')
            if coeffs is None:
                converted[valid] = series[valid]
                return converted
            coeffs = np.asarray(coeffs, dtype=float)
            order = len(coeffs)
            X = np.column_stack([series[valid] ** i for i in range(order)])
            converted[valid] = X @ coeffs
            return converted

        if conversion == 'piecewise':
            segments = params.get('segments') or []
            if not segments:
                converted[valid] = series[valid]
                return converted
            normalized_segments = [
                (
                    float(seg.get('offset') if isinstance(seg, dict) else seg[0]),
                    float(seg.get('scale') if isinstance(seg, dict) else seg[1]),
                    float(seg.get('upper') if isinstance(seg, dict) else seg[2]),
                )
                for seg in segments
            ]
            converted[valid] = self._apply_piecewise_transform(series[valid], normalized_segments)
            return converted

        # Fallback: no conversion applied
        converted[valid] = series[valid]
        return converted

    def _evaluate_dataset_linear(
        self,
        dataset: Sequence[Dict[str, object]],
        method: str,
        joint_type: str,
        side: str,
        offset: float,
        scale: float,
    ) -> Dict[str, object]:
        per_subject: List[Dict[str, float]] = []
        all_pred = []
        all_true = []
        for sample in dataset:
            pred = self.compute_series(sample['mp_landmarks'], method, joint_type, side)
            pred = pred * scale + offset
            true = np.asarray(sample['hospital_angles'], dtype=float)
            mask = ~np.isnan(pred) & ~np.isnan(true)
            if mask.sum() == 0:
                continue
            pred_masked = pred[mask]
            true_masked = true[mask]
            metrics = self._calculate_metrics(pred_masked, true_masked)
            per_subject.append(metrics.__dict__)
            all_pred.append(pred_masked)
            all_true.append(true_masked)
        if not all_pred:
            return {
                'overall': {'rmse': float('nan'), 'mae': float('nan'), 'correlation': float('nan'), 'icc': float('nan')},
                'per_subject': per_subject,
            }
        overall_metrics = self._calculate_metrics(np.concatenate(all_pred), np.concatenate(all_true))
        return {
            'overall': overall_metrics.__dict__,
            'per_subject': per_subject,
        }

    def _evaluate_dataset_polynomial(
        self,
        dataset: Sequence[Dict[str, object]],
        method: str,
        joint_type: str,
        side: str,
        coeffs: np.ndarray,
    ) -> Dict[str, object]:
        per_subject: List[Dict[str, float]] = []
        all_pred = []
        all_true = []
        for sample in dataset:
            pred = self.compute_series(sample['mp_landmarks'], method, joint_type, side)
            pred = self._apply_polynomial_transform(pred, coeffs)
            true = np.asarray(sample['hospital_angles'], dtype=float)
            mask = ~np.isnan(pred) & ~np.isnan(true)
            if mask.sum() == 0:
                continue
            pred_masked = pred[mask]
            true_masked = true[mask]
            metrics = self._calculate_metrics(pred_masked, true_masked)
            per_subject.append(metrics.__dict__)
            all_pred.append(pred_masked)
            all_true.append(true_masked)
        if not all_pred:
            return {
                'overall': {'rmse': float('nan'), 'mae': float('nan'), 'correlation': float('nan'), 'icc': float('nan')},
                'per_subject': per_subject,
            }
        overall_metrics = self._calculate_metrics(np.concatenate(all_pred), np.concatenate(all_true))
        return {
            'overall': overall_metrics.__dict__,
            'per_subject': per_subject,
        }

    def _evaluate_dataset_piecewise(
        self,
        dataset: Sequence[Dict[str, object]],
        method: str,
        joint_type: str,
        side: str,
        segments: List[Tuple[float, float, float]],
    ) -> Dict[str, object]:
        per_subject: List[Dict[str, float]] = []
        all_pred = []
        all_true = []
        for sample in dataset:
            pred = self.compute_series(sample['mp_landmarks'], method, joint_type, side)
            pred = self._apply_piecewise_transform(pred, segments)
            true = np.asarray(sample['hospital_angles'], dtype=float)
            mask = ~np.isnan(pred) & ~np.isnan(true)
            if mask.sum() == 0:
                continue
            pred_masked = pred[mask]
            true_masked = true[mask]
            metrics = self._calculate_metrics(pred_masked, true_masked)
            per_subject.append(metrics.__dict__)
            all_pred.append(pred_masked)
            all_true.append(true_masked)
        if not all_pred:
            return {
                'overall': {'rmse': float('nan'), 'mae': float('nan'), 'correlation': float('nan'), 'icc': float('nan')},
                'per_subject': per_subject,
            }
        overall_metrics = self._calculate_metrics(np.concatenate(all_pred), np.concatenate(all_true))
        return {
            'overall': overall_metrics.__dict__,
            'per_subject': per_subject,
        }

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------
    def _calculate_metrics(self, predicted: np.ndarray, target: np.ndarray) -> MethodMetrics:
        rmse = float(np.sqrt(np.mean((predicted - target) ** 2)))
        mae = float(np.mean(np.abs(predicted - target)))
        corr = float(pearsonr(predicted, target)[0]) if len(predicted) > 1 else float('nan')
        icc = float(self.calculate_icc(predicted, target))
        return MethodMetrics(rmse=rmse, mae=mae, correlation=corr, icc=icc)

    @staticmethod
    def calculate_icc(y1: np.ndarray, y2: np.ndarray) -> float:
        y1 = np.asarray(y1, dtype=float)
        y2 = np.asarray(y2, dtype=float)
        mask = ~np.isnan(y1) & ~np.isnan(y2)
        if mask.sum() < 2:
            return float('nan')
        Y = np.column_stack([y1[mask], y2[mask]])
        n, k = Y.shape
        subject_means = np.mean(Y, axis=1)
        grand_mean = np.mean(Y)
        bms = k * np.sum((subject_means - grand_mean) ** 2) / (n - 1)
        residuals = Y - subject_means[:, None]
        wms = np.sum(residuals ** 2) / (n * (k - 1))
        denominator = bms + (k - 1) * wms
        if denominator == 0:
            return float('nan')
        return float((bms - wms) / denominator)

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------
    def register_best_method(
        self,
        joint_type: str,
        side: str,
        method: str,
        conversion: str,
        params: Dict[str, object],
    ) -> None:
        joint_type = joint_type.lower()
        side = side.lower()
        self.conversion_params.setdefault(joint_type, {})[side] = {
            'method': method,
            'conversion': conversion,
            'params': params,
        }

    def save_parameters(self, output_path: str) -> None:
        Path(output_path).write_text(json.dumps(self.conversion_params, indent=2), encoding='utf-8')

    def load_parameters(self, path: str) -> None:
        self.conversion_params = json.loads(Path(path).read_text(encoding='utf-8'))


def main() -> None:
    print("AngleConverter is a library module; see documentation for usage with cross-validation pipelines.")


if __name__ == "__main__":
    main()
