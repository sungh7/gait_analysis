"""
Dynamic Time Warping utilities for gait cycle alignment between MediaPipe and hospital datasets.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
from fastdtw import fastdtw
from scipy.interpolate import interp1d
from scipy.spatial.distance import euclidean
from scipy.stats import pearsonr


@dataclass
class AlignmentResult:
    aligned_angles: np.ndarray
    dtw_distance: float
    rmse: float
    correlation: float
    icc: float


class DTWAligner:
    """Align gait cycles using FastDTW with configurable radius."""

    def __init__(self, radius: int = 10):
        self.radius = radius

    def align_single_cycle(self, mp_angles: np.ndarray, hospital_angles: np.ndarray) -> Tuple[np.ndarray, float]:
        mp_angles = self._sanitize_series(mp_angles)
        hosp_angles = self._sanitize_series(hospital_angles)

        mp_norm = self._zscore(mp_angles)
        hosp_norm = self._zscore(hosp_angles)

        distance, path = fastdtw(mp_norm.reshape(-1, 1), hosp_norm.reshape(-1, 1), radius=self.radius, dist=euclidean)

        mp_indices = np.array([p[0] for p in path])
        hosp_indices = np.array([p[1] for p in path])

        aligned = np.full_like(hospital_angles, np.nan, dtype=float)
        for hosp_idx in range(len(hospital_angles)):
            matching = mp_angles[mp_indices[hosp_indices == hosp_idx]]
            if matching.size:
                aligned[hosp_idx] = np.nanmean(matching)

        if np.isnan(aligned).any():
            valid = ~np.isnan(aligned)
            if valid.sum() >= 2:
                interp = interp1d(np.where(valid)[0], aligned[valid], kind='linear', fill_value='extrapolate')
                aligned[np.isnan(aligned)] = interp(np.where(np.isnan(aligned))[0])
            else:
                aligned = hospital_angles.copy()

        return aligned, float(distance)

    def align_and_validate(self, mp_angles: np.ndarray, hospital_angles: np.ndarray) -> AlignmentResult:
        aligned, distance = self.align_single_cycle(mp_angles, hospital_angles)
        metrics = self._calculate_metrics(aligned, hospital_angles)
        return AlignmentResult(
            aligned_angles=aligned,
            dtw_distance=distance,
            rmse=metrics['rmse'],
            correlation=metrics['correlation'],
            icc=metrics['icc'],
        )

    def compare_before_after(self, mp_original: np.ndarray, mp_aligned: np.ndarray, hospital_angles: np.ndarray) -> Dict[str, Dict[str, float]]:
        before = self._calculate_metrics(mp_original, hospital_angles)
        after = self._calculate_metrics(mp_aligned, hospital_angles)
        improvement = {
            'rmse_delta': before['rmse'] - after['rmse'],
            'correlation_delta': after['correlation'] - before['correlation'],
            'icc_delta': after['icc'] - before['icc'],
        }
        return {'before': before, 'after': after, 'improvement': improvement}

    @staticmethod
    def _zscore(series: np.ndarray) -> np.ndarray:
        mean = np.nanmean(series)
        std = np.nanstd(series)
        if std == 0 or np.isnan(std):
            return np.zeros_like(series)
        return (series - mean) / std

    @staticmethod
    def _sanitize_series(series: np.ndarray) -> np.ndarray:
        series = np.asarray(series, dtype=float)
        if series.ndim != 1:
            raise ValueError("Series must be 1D")
        if np.isnan(series).all():
            raise ValueError("Series contains only NaNs")
        if np.isnan(series).any():
            valid = ~np.isnan(series)
            interp = interp1d(np.where(valid)[0], series[valid], kind='linear', fill_value='extrapolate')
            series = interp(np.arange(len(series)))
        return series

    @staticmethod
    def _calculate_metrics(predicted: np.ndarray, target: np.ndarray) -> Dict[str, float]:
        mask = ~np.isnan(predicted) & ~np.isnan(target)
        if mask.sum() < 2:
            return {'rmse': float('nan'), 'correlation': float('nan'), 'icc': float('nan')}
        pred = predicted[mask]
        true = target[mask]
        rmse = float(np.sqrt(np.mean((pred - true) ** 2)))
        corr = float(pearsonr(pred, true)[0])
        icc = float(DTWAligner._calculate_icc(pred, true))
        return {'rmse': rmse, 'correlation': corr, 'icc': icc}

    @staticmethod
    def _calculate_icc(y1: np.ndarray, y2: np.ndarray) -> float:
        Y = np.column_stack([y1, y2])
        n, k = Y.shape
        subject_means = np.mean(Y, axis=1)
        grand_mean = np.mean(Y)
        bms = k * np.sum((subject_means - grand_mean) ** 2) / (n - 1)
        residuals = Y - subject_means[:, None]
        wms = np.sum(residuals ** 2) / (n * (k - 1))
        denominator = bms + (k - 1) * wms
        if denominator == 0:
            return float('nan')
        return (bms - wms) / denominator
