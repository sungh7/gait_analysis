"""
Preprocessing pipeline for MediaPipe landmark data.

This module standardizes preprocessing across subjects:
  - Visibility-based filtering
  - Outlier suppression via z-score thresholding
  - Temporal resampling to a target frame rate
  - Savitzky-Golay smoothing with consistent parameters
All steps record structured log entries so downstream validation
pipelines can audit the exact transformations applied.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter


@dataclass
class ProcessingLogEntry:
    """Structured record of a preprocessing step."""

    step: str
    details: Dict[str, float]


@dataclass
class PreprocessingResult:
    """Container for processed landmarks and corresponding log."""

    dataframe: pd.DataFrame
    fps: float
    log: List[ProcessingLogEntry] = field(default_factory=list)


class PreprocessingPipeline:
    """Standard preprocessing pipeline for MediaPipe landmark time-series."""

    def __init__(
        self,
        visibility_threshold: float = 0.5,
        zscore_threshold: float = 3.0,
        smoothing_window: int = 11,
        smoothing_polyorder: int = 3,
        target_fps: float = 30.0,
        critical_landmarks: Optional[Sequence[str]] = None,
    ) -> None:
        self.visibility_threshold = visibility_threshold
        self.zscore_threshold = zscore_threshold
        self.smoothing_window = smoothing_window
        self.smoothing_polyorder = smoothing_polyorder
        self.target_fps = target_fps
        self.critical_landmarks = (
            list(critical_landmarks)
            if critical_landmarks is not None
            else [
                "left_hip",
                "right_hip",
                "left_knee",
                "right_knee",
                "left_ankle",
                "right_ankle",
                "left_heel",
                "right_heel",
                "left_foot_index",
                "right_foot_index",
            ]
        )

    def process(self, df: pd.DataFrame, fps: float) -> PreprocessingResult:
        """Execute the preprocessing routine on a copy of the input data."""
        df_processed = df.copy()
        log: List[ProcessingLogEntry] = []

        if "frame" not in df_processed.columns:
            raise ValueError("Input DataFrame must contain a 'frame' column.")

        coordinate_cols = [col for col in df_processed.columns if col.startswith(("x_", "y_", "z_"))]
        visibility_cols = [col for col in df_processed.columns if col.startswith("visibility_")]

        # 1. Visibility-based gating
        if visibility_cols:
            vis_mask = np.ones(len(df_processed), dtype=bool)
            for landmark in self.critical_landmarks:
                col_name = f"visibility_{landmark}"
                if col_name in df_processed.columns:
                    vis_mask &= df_processed[col_name].values >= self.visibility_threshold

            if not np.all(vis_mask):
                frames_flagged = int(len(df_processed) - np.sum(vis_mask))
                for col in coordinate_cols:
                    df_processed.loc[~vis_mask, col] = np.nan

                log.append(
                    ProcessingLogEntry(
                        step="visibility_filter",
                        details={
                            "frames_flagged": frames_flagged,
                            "total_frames": float(len(df_processed)),
                            "threshold": self.visibility_threshold,
                        },
                    )
                )

        # 2. Z-score outlier suppression
        outlier_counts = 0
        for col in coordinate_cols:
            series = df_processed[col].astype(float)
            mask = series.notna()
            if mask.sum() < 5:
                continue

            values = series[mask].values
            std = values.std(ddof=1)
            if std == 0:
                continue

            mean = values.mean()
            zscores = (values - mean) / std
            outlier_idx = np.abs(zscores) > self.zscore_threshold
            if np.any(outlier_idx):
                flagged = mask[mask].index[outlier_idx]
                series.loc[flagged] = np.nan
                outlier_counts += int(outlier_idx.sum())
                df_processed[col] = series

        if outlier_counts > 0:
            log.append(
                ProcessingLogEntry(
                    step="zscore_outlier_suppression",
                    details={
                        "zscore_threshold": self.zscore_threshold,
                        "samples_flagged": outlier_counts,
                        "coordinate_dimensions": float(len(coordinate_cols)),
                    },
                )
            )

        # 3. Interpolate missing values (bidirectional linear)
        interpolated_cells = 0
        for col in coordinate_cols + visibility_cols:
            if col not in df_processed.columns:
                continue

            series = df_processed[col].astype(float)
            missing_before = series.isna().sum()
            if missing_before == 0:
                continue

            series = series.interpolate(method="linear", limit_direction="both", limit_area=None)
            missing_after = series.isna().sum()
            interpolated_cells += int(missing_before - missing_after)
            df_processed[col] = series

        if interpolated_cells > 0:
            log.append(
                ProcessingLogEntry(
                    step="linear_interpolation",
                    details={"cells_interpolated": interpolated_cells},
                )
            )

        # 4. Resample to target FPS if necessary
        effective_fps = float(fps)
        if effective_fps <= 0 or not np.isfinite(effective_fps):
            raise ValueError(f"Invalid FPS provided: {fps}")

        if abs(effective_fps - self.target_fps) > 1e-6:
            df_processed, effective_fps = self._resample_to_target_fps(df_processed, effective_fps)
            log.append(
                ProcessingLogEntry(
                    step="temporal_resampling",
                    details={
                        "original_fps": float(fps),
                        "target_fps": self.target_fps,
                        "resampled_fps": effective_fps,
                        "frames_after_resample": float(len(df_processed)),
                    },
                )
            )

        # 5. Savitzky-Golay smoothing
        window_length = self._valid_window_length(len(df_processed))
        smoothed_cols = 0
        if window_length is not None:
            for col in coordinate_cols:
                series = df_processed[col].astype(float)
                if series.isna().sum() >= len(series) - 2:
                    continue

                filled = series.interpolate(method="linear", limit_direction="both", limit_area=None)
                if filled.isna().sum() > 0:
                    continue

                smoothed = savgol_filter(
                    filled.values,
                    window_length=window_length,
                    polyorder=min(self.smoothing_polyorder, window_length - 1),
                    mode="interp",
                )
                df_processed[col] = smoothed
                smoothed_cols += 1

        if smoothed_cols > 0:
            log.append(
                ProcessingLogEntry(
                    step="savgol_smoothing",
                    details={
                        "window_length": window_length,
                        "polyorder": min(self.smoothing_polyorder, window_length - 1),
                        "columns_smoothed": smoothed_cols,
                    },
                )
            )

        return PreprocessingResult(dataframe=df_processed, fps=effective_fps, log=log)

    def _resample_to_target_fps(self, df: pd.DataFrame, original_fps: float) -> Tuple[pd.DataFrame, float]:
        time = df["frame"].values.astype(float) / original_fps
        if len(time) < 2:
            return df, original_fps

        total_duration = time[-1]
        target_interval = 1.0 / self.target_fps
        target_times = np.arange(0.0, total_duration + target_interval / 2.0, target_interval)

        resampled_data: Dict[str, np.ndarray] = {"frame": np.arange(len(target_times))}
        for col in df.columns:
            if col == "frame":
                continue

            series = df[col].astype(float)
            mask = series.notna()
            if mask.sum() < 2:
                resampled_data[col] = np.full_like(target_times, np.nan, dtype=float)
                continue

            resampled_values = np.interp(
                target_times,
                time[mask],
                series[mask],
                left=np.nan,
                right=np.nan,
            )
            resampled_data[col] = resampled_values

        resampled_df = pd.DataFrame(resampled_data)
        resampled_df = resampled_df.interpolate(method="linear", limit_direction="both", limit_area=None)
        return resampled_df, self.target_fps

    def _valid_window_length(self, sequence_length: int) -> Optional[int]:
        if sequence_length < 3:
            return None

        window = min(self.smoothing_window, sequence_length)
        if window % 2 == 0:
            window -= 1
        if window < self.smoothing_polyorder + 2:
            window = self.smoothing_polyorder + 2
            if window % 2 == 0:
                window += 1

        return window if window <= sequence_length else None
