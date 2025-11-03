"""
Tiered gait evaluation pipeline.

Implements three analysis layers:
1. Temporal-spatial parameters (ICC against hospital info.json)
2. Joint angle waveform comparison (DTW/ICC/correlation + SPM across subjects)
3. Direction-sensitive gait segmentation (outbound vs inbound with turn exclusion)
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter, find_peaks
from scipy.stats import pearsonr

from dtw_alignment import DTWAligner
from mediapipe_csv_processor import MediaPipeCSVProcessor
from spm_analysis import SPMAnalyzer


# ---------------------------------------------------------------------------
# Utility dataclasses
# ---------------------------------------------------------------------------

@dataclass
class MetricSeries:
    """Holds paired ground-truth/prediction values for ICC and error stats."""

    name: str
    ground_truth: List[float] = field(default_factory=list)
    prediction: List[float] = field(default_factory=list)

    def add(self, gt: Optional[float], pred: Optional[float]) -> None:
        if gt is None or pred is None:
            return
        if np.isnan(gt) or np.isnan(pred):
            return
        self.ground_truth.append(float(gt))
        self.prediction.append(float(pred))

    def as_arrays(self) -> Tuple[np.ndarray, np.ndarray]:
        if not self.ground_truth or not self.prediction:
            return np.array([]), np.array([])
        return np.array(self.ground_truth, dtype=float), np.array(self.prediction, dtype=float)

    def icc(self) -> float:
        gt, pred = self.as_arrays()
        if gt.size < 2:
            return float("nan")
        return float(calculate_icc(gt, pred))

    def rmse(self) -> float:
        gt, pred = self.as_arrays()
        if gt.size == 0:
            return float("nan")
        return float(np.sqrt(np.mean((gt - pred) ** 2)))

    def mae(self) -> float:
        gt, pred = self.as_arrays()
        if gt.size == 0:
            return float("nan")
        return float(np.mean(np.abs(gt - pred)))

    def correlation(self) -> float:
        gt, pred = self.as_arrays()
        if gt.size < 2:
            return float("nan")
        return float(pearsonr(gt, pred)[0])


@dataclass
class TemporalCycle:
    """Stores per-cycle temporal metrics for a single foot."""

    start_frame: int
    end_frame: int
    stride_duration: float
    stance_percent: float
    swing_percent: float
    ids_percent: Optional[float]
    ss_percent: Optional[float]
    direction: Optional[str] = None


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def calculate_icc(y_true: Iterable[float], y_pred: Iterable[float]) -> float:
    """ICC(2,1) absolute agreement."""
    y_true = np.asarray(list(y_true), dtype=float)
    y_pred = np.asarray(list(y_pred), dtype=float)
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    if mask.sum() < 2:
        return float("nan")
    Y = np.column_stack([y_true[mask], y_pred[mask]])
    n, k = Y.shape
    subject_means = np.mean(Y, axis=1)
    grand_mean = np.mean(Y)
    bms = k * np.sum((subject_means - grand_mean) ** 2) / (n - 1)
    residuals = Y - subject_means[:, None]
    wms = np.sum(residuals ** 2) / (n * (k - 1))
    denominator = bms + (k - 1) * wms
    if denominator == 0:
        return float("nan")
    return float((bms - wms) / denominator)


def extract_fps_from_name(csv_path: Path, default_fps: float = 30.0) -> float:
    name = csv_path.stem
    if "fps" not in name.lower():
        return default_fps
    lower = name.lower()
    try:
        segment = lower.split("fps", 1)[1]
        digits = ""
        for char in segment:
            if char.isdigit():
                digits += char
            else:
                break
        if digits:
            return float(digits)
    except Exception:
        pass
    return default_fps


# ---------------------------------------------------------------------------
# Core evaluator
# ---------------------------------------------------------------------------

class TieredGaitEvaluator:
    """End-to-end gait evaluation orchestrator."""

    def __init__(
        self,
        data_root: Path = Path("data"),
        processed_root: Path = Path("data/processed"),
        walkway_distance_m: float = 7.25,
        turn_buffer_frames: int = 15,
        dtw_radius: int = 10,
    ) -> None:
        self.data_root = data_root
        self.processed_root = processed_root
        self.walkway_distance_m = walkway_distance_m
        self.turn_buffer_frames = turn_buffer_frames

        self.processor = MediaPipeCSVProcessor()
        self.dtw_aligner = DTWAligner(radius=dtw_radius)
        self.spm_analyzer = SPMAnalyzer(alpha=0.05, smooth_sigma=1.0)

        # Aggregation registries
        temporal_metrics = [
            "strides_left",
            "strides_right",
            "cadence_left",
            "cadence_right",
            "cadence_average",
            "stance_percent_left",
            "stance_percent_right",
            "ids_percent_left",
            "ids_percent_right",
            "ids_ss_percent_left",
            "ids_ss_percent_right",
            "ss_percent_left",
            "ss_percent_right",
        ]
        self.temporal_registry: Dict[str, MetricSeries] = {
            key: MetricSeries(name=key) for key in temporal_metrics
        }

        self.waveform_registry: Dict[str, Dict[str, List[np.ndarray]]] = {}
        self.direction_registry: Dict[str, Dict[str, List[float]]] = {
            "stance_percent_left": {"outbound": [], "inbound": []},
            "stance_percent_right": {"outbound": [], "inbound": []},
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate(self, subjects: Optional[Iterable[str]] = None) -> Dict[str, object]:
        subject_ids = list(subjects) if subjects is not None else self._discover_subjects()
        per_subject: Dict[str, Dict[str, object]] = {}

        for subject_id in subject_ids:
            try:
                result = self._evaluate_subject(subject_id)
                if result is not None:
                    per_subject[subject_id] = result
            except Exception as exc:
                per_subject[subject_id] = {"error": str(exc)}

        aggregate = self._build_aggregate_summary()
        return {"subjects": per_subject, "aggregate": aggregate}

    # ------------------------------------------------------------------
    # Subject-level evaluation
    # ------------------------------------------------------------------

    def _evaluate_subject(self, subject_id: str) -> Optional[Dict[str, object]]:
        info = self._load_info(subject_id)
        csv_path = self._find_mediapipe_csv(subject_id)
        gait_csv = self._load_processed_gait_csv(subject_id)

        if info is None or csv_path is None or gait_csv is None:
            return None

        fps = extract_fps_from_name(csv_path)
        df_wide = self.processor.load_csv(csv_path)
        df_angles = self.processor.calculate_joint_angles(df_wide)

        events = self._detect_events(df_angles, fps)
        left_cycles = self._compute_cycles(df_angles, events, "left")
        right_cycles = self._compute_cycles(df_angles, events, "right")

        direction_summary = self._analyze_directions(
            subject_id, df_angles, events, left_cycles, right_cycles
        )

        temporal_summary = self._summarize_temporal_metrics(
            subject_id, info, events, left_cycles, right_cycles, direction_summary
        )

        waveform_summary = self._compare_waveforms(
            subject_id, df_angles, gait_csv, events
        )

        return {
            "temporal": temporal_summary,
            "waveforms": waveform_summary,
            "direction": direction_summary,
        }

    # ------------------------------------------------------------------
    # Discovery / loading helpers
    # ------------------------------------------------------------------

    def _discover_subjects(self) -> List[str]:
        subjects = []
        for path in sorted(self.processed_root.glob("S1_*_info.json")):
            subjects.append(path.stem.replace("_info", ""))
        return subjects

    def _load_info(self, subject_id: str) -> Optional[Dict[str, object]]:
        path = self.processed_root / f"{subject_id}_info.json"
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def _find_mediapipe_csv(self, subject_id: str) -> Optional[Path]:
        subj_num = self._subject_number(subject_id)
        if subj_num is None:
            return None
        root = self.data_root / str(subj_num)
        if not root.exists():
            return None
        candidates = sorted(root.glob("*side_pose_fps*.csv"))
        return candidates[0] if candidates else None

    def _load_processed_gait_csv(self, subject_id: str) -> Optional[pd.DataFrame]:
        path = self.processed_root / f"{subject_id}_gait_long.csv"
        if not path.exists():
            return None
        return pd.read_csv(path)

    @staticmethod
    def _subject_number(subject_id: str) -> Optional[int]:
        try:
            return int(subject_id.replace("S1_", ""))
        except ValueError:
            return None

    # ------------------------------------------------------------------
    # Event detection
    # ------------------------------------------------------------------

    def _detect_events(self, df_angles: pd.DataFrame, fps: float) -> Dict[str, Dict[str, np.ndarray]]:
        events: Dict[str, Dict[str, np.ndarray]] = {}
        for side in ("left", "right"):
            heel = self.processor.detect_heel_strikes_fusion(df_angles, side=side, fps=fps)
            toe = self._detect_toe_offs(df_angles, side=side, fps=fps)
            events[side] = {"heel_strikes": heel, "toe_offs": toe}
        events["fps"] = fps
        return events

    def _detect_toe_offs(self, df_angles: pd.DataFrame, side: str, fps: float) -> np.ndarray:
        heel_col = f"y_{side}_heel"
        ankle_col = f"y_{side}_ankle"
        toe_col = f"y_{side}_foot_index"
        required = [heel_col, ankle_col, toe_col]
        if any(col not in df_angles.columns for col in required):
            return np.array([], dtype=int)

        signal = (
            0.6 * df_angles[heel_col].to_numpy(float)
            + 0.3 * df_angles[ankle_col].to_numpy(float)
            + 0.1 * df_angles[toe_col].to_numpy(float)
        )

        valid_mask = ~np.isnan(signal)
        frames = df_angles["frame"].to_numpy(dtype=int)
        frames = frames[valid_mask]
        signal = signal[valid_mask]

        if signal.size < fps:
            return np.array([], dtype=int)

        window = int(min(max(int(fps // 2) * 2 + 1, 5), 31))
        smooth = savgol_filter(signal, window_length=window, polyorder=2)
        prominence = np.std(smooth) * 0.2 if np.std(smooth) > 0 else 0.01
        min_distance = max(int(fps * 0.35), 10)

        peaks, _ = find_peaks(smooth, distance=min_distance, prominence=prominence)
        return frames[peaks] if peaks.size else np.array([], dtype=int)

    # ------------------------------------------------------------------
    # Temporal metrics
    # ------------------------------------------------------------------

    def _compute_cycles(
        self,
        df_angles: pd.DataFrame,
        events: Dict[str, Dict[str, np.ndarray]],
        side: str,
    ) -> List[TemporalCycle]:
        heel = events[side]["heel_strikes"]
        toe = events[side]["toe_offs"]
        fps = events["fps"]

        if heel.size < 2:
            return []

        cycles: List[TemporalCycle] = []
        toe_iter = iter(sorted(toe))
        toe_current = next(toe_iter, None)

        for idx in range(len(heel) - 1):
            start = int(heel[idx])
            end = int(heel[idx + 1])
            stride_duration = (end - start) / fps if end > start else 0.0
            if stride_duration <= 0:
                continue

            # Find toe-off for this stride
            toe_off_frame = None
            while toe_current is not None and toe_current <= start:
                toe_current = next(toe_iter, None)
            if toe_current is not None and start < toe_current < end:
                toe_off_frame = toe_current
                toe_current = next(toe_iter, None)

            stance_percent = None
            swing_percent = None
            if toe_off_frame is not None:
                stance_percent = (toe_off_frame - start) / (end - start) * 100.0
                swing_percent = 100.0 - stance_percent

            cycles.append(
                TemporalCycle(
                    start_frame=start,
                    end_frame=end,
                    stride_duration=stride_duration,
                    stance_percent=stance_percent if stance_percent is not None else float("nan"),
                    swing_percent=swing_percent if swing_percent is not None else float("nan"),
                    ids_percent=None,
                    ss_percent=None,
                )
            )

        return cycles

    def _summarize_temporal_metrics(
        self,
        subject_id: str,
        info: Dict[str, object],
        events: Dict[str, Dict[str, np.ndarray]],
        left_cycles: List[TemporalCycle],
        right_cycles: List[TemporalCycle],
        direction_summary: Dict[str, object],
    ) -> Dict[str, object]:
        fps = events["fps"]
        duration_frames = int(events["right"]["heel_strikes"][-1] - events["right"]["heel_strikes"][0]) if events["right"]["heel_strikes"].size >= 2 else int(events["left"]["heel_strikes"][-1] - events["left"]["heel_strikes"][0]) if events["left"]["heel_strikes"].size >= 2 else 0
        duration_minutes = (duration_frames / fps) / 60.0 if fps > 0 and duration_frames > 0 else float("nan")

        valid_left = [cycle for cycle in left_cycles if getattr(cycle, "direction", None) in ("outbound", "inbound")]
        valid_right = [cycle for cycle in right_cycles if getattr(cycle, "direction", None) in ("outbound", "inbound")]
        counts_left = self._direction_counts(left_cycles)
        counts_right = self._direction_counts(right_cycles)

        predicted_strides_left = self._directional_stride_estimate(counts_left, len(left_cycles))
        predicted_strides_right = self._directional_stride_estimate(counts_right, len(right_cycles))

        predicted = {
            "strides": {
                "left": predicted_strides_left,
                "right": predicted_strides_right,
            },
            "cadence": {
                "left": self._cadence_from_cycles(valid_left),
                "right": self._cadence_from_cycles(valid_right),
                "average": self._cadence_overall(valid_left, valid_right),
            },
            "gait_cycle_timing": self._estimate_phase_percentages(events, valid_left, valid_right),
            "directional_counts": {
                "left": counts_left,
                "right": counts_right,
            },
        }

        ground_truth = {
            "strides": info.get("strides", {}),
            "cadence": info.get("cadence", {}),
            "gait_cycle_timing": info.get("gait_cycle_timing", {}),
        }

        self._update_temporal_registry(ground_truth, predicted)

        return {
            "ground_truth": ground_truth,
            "prediction": predicted,
            "duration_minutes": duration_minutes,
        }

    def _cadence_from_cycles(self, cycles: List[TemporalCycle]) -> float:
        if not cycles:
            return float("nan")

        per_direction = []
        for direction in ("outbound", "inbound"):
            dir_cycles = [cycle for cycle in cycles if getattr(cycle, "direction", None) == direction]
            if not dir_cycles:
                continue
            total_time = sum(c.stride_duration for c in dir_cycles)
            if total_time > 0:
                per_direction.append(len(dir_cycles) / (total_time / 60.0))

        if per_direction:
            return float(np.mean(per_direction))

        total_time = sum(c.stride_duration for c in cycles)
        if total_time > 0:
            return float(len(cycles) / (total_time / 60.0))
        return float("nan")

    def _cadence_overall(self, left_cycles: List[TemporalCycle], right_cycles: List[TemporalCycle]) -> float:
        combined = [cycle for cycle in left_cycles + right_cycles if getattr(cycle, "direction", None) in ("outbound", "inbound")]
        if not combined:
            combined = left_cycles + right_cycles
        if not combined:
            return float("nan")
        total_time = sum(c.stride_duration for c in combined)
        if total_time <= 0:
            return float("nan")
        return float(len(combined) / (total_time / 60.0))

    def _estimate_phase_percentages(
        self,
        events: Dict[str, Dict[str, np.ndarray]],
        left_cycles: List[TemporalCycle],
        right_cycles: List[TemporalCycle],
    ) -> Dict[str, Dict[str, Optional[float]]]:
        fps = events["fps"]
        result = {"left": {"ids": None, "ss": None, "ids_ss": None, "sls": None}, "right": {"ids": None, "ss": None, "ids_ss": None, "sls": None}}
        if not left_cycles and not right_cycles:
            return result

        def compute_for_side(
            side_cycles: List[TemporalCycle],
            side: str,
            contralateral_cycles: List[TemporalCycle],
            contralateral_heel: np.ndarray,
            contralateral_toe: np.ndarray,
        ) -> Dict[str, Optional[float]]:
            if not side_cycles:
                return {"ids": None, "ss": None, "ids_ss": None, "sls": None}

            values_ids = []
            values_ss = []
            values_sls = []

            heel = events[side]["heel_strikes"]
            contralateral_heel_frames = contralateral_heel
            contralateral_toe_frames = contralateral_toe

            for idx, cycle in enumerate(side_cycles):
                start = cycle.start_frame
                end = cycle.end_frame
                stride_frames = end - start
                if stride_frames <= 0:
                    continue

                toe_off_same = self._nearest_event_between(events[side]["toe_offs"], start, end)
                toe_off_contra = self._first_event_after(contralateral_toe_frames, start, end)
                heel_contra = self._first_event_after(contralateral_heel_frames, start, end)

                if toe_off_same is not None:
                    stance_percent = (toe_off_same - start) / stride_frames * 100.0
                    values_sls.append(stance_percent)

                if toe_off_contra is not None and heel_contra is not None and heel_contra > toe_off_contra:
                    ids_percent = (toe_off_contra - start) / stride_frames * 100.0
                    ss_percent = (heel_contra - toe_off_contra) / stride_frames * 100.0
                    values_ids.append(ids_percent)
                    values_ss.append(ss_percent)

            def safe_mean(sequence: List[float]) -> Optional[float]:
                if not sequence:
                    return None
                return float(np.mean(sequence))

            ids_mean = safe_mean(values_ids)
            ss_mean = safe_mean(values_ss)
            sls_mean = safe_mean(values_sls)
            ids_ss_mean = None
            if ids_mean is not None and ss_mean is not None:
                ids_ss_mean = ids_mean + ss_mean

            return {"ids": ids_mean, "ss": ss_mean, "ids_ss": ids_ss_mean, "sls": sls_mean}

        result["left"] = compute_for_side(
            left_cycles,
            "left",
            right_cycles,
            events["right"]["heel_strikes"],
            events["right"]["toe_offs"],
        )
        result["right"] = compute_for_side(
            right_cycles,
            "right",
            left_cycles,
            events["left"]["heel_strikes"],
            events["left"]["toe_offs"],
        )
        return result

    @staticmethod
    def _nearest_event_between(events: np.ndarray, start: int, end: int) -> Optional[int]:
        filtered = events[(events > start) & (events < end)]
        if filtered.size == 0:
            return None
        return int(filtered[0])

    @staticmethod
    def _first_event_after(events: np.ndarray, start: int, end: int) -> Optional[int]:
        filtered = events[events > start]
        if filtered.size == 0:
            return None
        candidate = int(filtered[0])
        if candidate >= end:
            return None
        return candidate

    def _update_temporal_registry(
        self,
        ground_truth: Dict[str, Dict[str, Optional[float]]],
        predicted: Dict[str, Dict[str, Optional[float]]],
    ) -> None:
        strides_gt = ground_truth.get("strides", {})
        strides_pred = predicted.get("strides", {})
        self.temporal_registry["strides_left"].add(
            strides_gt.get("left"), strides_pred.get("left")
        )
        self.temporal_registry["strides_right"].add(
            strides_gt.get("right"), strides_pred.get("right")
        )

        cadence_gt = ground_truth.get("cadence", {})
        cadence_pred = predicted.get("cadence", {})
        self.temporal_registry["cadence_left"].add(
            cadence_gt.get("left"), cadence_pred.get("left")
        )
        self.temporal_registry["cadence_right"].add(
            cadence_gt.get("right"), cadence_pred.get("right")
        )
        self.temporal_registry["cadence_average"].add(
            cadence_gt.get("average"), cadence_pred.get("average")
        )

        timing_gt = ground_truth.get("gait_cycle_timing", {})
        timing_pred = predicted.get("gait_cycle_timing", {})

        for side, prefix in (("left", "left"), ("right", "right")):
            gt_side = timing_gt.get(side, {})
            pred_side = timing_pred.get(side, {})
            self.temporal_registry[f"stance_percent_{prefix}"].add(
                gt_side.get("sls"), pred_side.get("sls")
            )
            self.temporal_registry[f"ids_percent_{prefix}"].add(
                gt_side.get("ids"), pred_side.get("ids")
            )
            self.temporal_registry[f"ids_ss_percent_{prefix}"].add(
                gt_side.get("ids_ss"), pred_side.get("ids_ss")
            )
            self.temporal_registry[f"ss_percent_{prefix}"].add(
                gt_side.get("ss"), pred_side.get("ss")
            )

    # ------------------------------------------------------------------
    # Waveform comparison
    # ------------------------------------------------------------------

    def _compare_waveforms(
        self,
        subject_id: str,
        df_angles: pd.DataFrame,
        gait_csv: pd.DataFrame,
        events: Dict[str, Dict[str, np.ndarray]],
    ) -> Dict[str, Dict[str, float]]:
        waveforms = self._compute_mediapipe_waveforms(df_angles, events)
        hospital_series = self._extract_hospital_waveforms(gait_csv)

        joints = ["l.hi.angle", "l.kn.angle", "l.an.angle", "r.hi.angle", "r.kn.angle", "r.an.angle"]
        summary: Dict[str, Dict[str, float]] = {}

        for joint_code in joints:
            mp_series = waveforms.get(joint_code)
            hosp_series = hospital_series.get(joint_code)
            if mp_series is None or hosp_series is None:
                continue

            mp_values = mp_series.copy()
            hosp_values = hosp_series.copy()
            if len(mp_values) != len(hosp_values):
                continue

            dtw_result = self.dtw_aligner.align_and_validate(mp_values, hosp_values)
            metrics = self.dtw_aligner.compare_before_after(
                mp_values, dtw_result.aligned_angles, hosp_values
            )

            summary[joint_code] = {
                "rmse_before": metrics["before"]["rmse"],
                "rmse_after": metrics["after"]["rmse"],
                "rmse_delta": metrics["improvement"]["rmse_delta"],
                "correlation_before": metrics["before"]["correlation"],
                "correlation_after": metrics["after"]["correlation"],
                "correlation_delta": metrics["improvement"]["correlation_delta"],
                "icc_before": metrics["before"]["icc"],
                "icc_after": metrics["after"]["icc"],
                "icc_delta": metrics["improvement"]["icc_delta"],
                "dtw_distance": dtw_result.dtw_distance,
            }

            registry = self.waveform_registry.setdefault(joint_code, {"mediapipe": [], "hospital": []})
            registry["mediapipe"].append(mp_values)
            registry["hospital"].append(hosp_values)

        return summary

    def _compute_mediapipe_waveforms(
        self,
        df_angles: pd.DataFrame,
        events: Dict[str, Dict[str, np.ndarray]],
    ) -> Dict[str, np.ndarray]:
        fps = events["fps"]
        waveforms: Dict[str, np.ndarray] = {}

        for side in ("left", "right"):
            gait_cycles = self.processor.segment_gait_cycles(df_angles, side=side, fps=fps)
            averaged = self.processor.average_gait_cycles(gait_cycles, side=side)
            if averaged is None or averaged.empty:
                continue

            for joint_code in averaged["joint"].unique():
                joint_df = averaged[averaged["joint"] == joint_code]
                joint_df = joint_df.sort_values("gait_cycle")
                waveforms[joint_code] = joint_df["angle_mean"].to_numpy(dtype=float)

        return waveforms

    def _extract_hospital_waveforms(self, gait_df: pd.DataFrame) -> Dict[str, np.ndarray]:
        hospital: Dict[str, np.ndarray] = {}
        filtered = gait_df[gait_df["plane"] == "y"]
        for joint_code, group in filtered.groupby("joint"):
            sorted_group = group.sort_values("gait_cycle")
            hospital[joint_code] = sorted_group["condition1_avg"].to_numpy(dtype=float)
        return hospital

    # ------------------------------------------------------------------
    # Directional analysis
    # ------------------------------------------------------------------

    def _analyze_directions(
        self,
        subject_id: str,
        df_angles: pd.DataFrame,
        events: Dict[str, Dict[str, np.ndarray]],
        left_cycles: List[TemporalCycle],
        right_cycles: List[TemporalCycle],
    ) -> Dict[str, object]:
        axis, positions = self._walk_axis(df_angles)
        if axis is None or positions.size == 0:
            return {"axis": None, "segments": []}

        direction_by_cycle = self._label_cycle_directions(positions, left_cycles, right_cycles)
        segments_summary = []

        for side, cycles in (("left", left_cycles), ("right", right_cycles)):
            for cycle in cycles:
                cycle.direction = direction_by_cycle.get(cycle.start_frame)

        for direction in ("outbound", "inbound"):
            left_values = [cycle.stance_percent for cycle in left_cycles if cycle.direction == direction and np.isfinite(cycle.stance_percent)]
            right_values = [cycle.stance_percent for cycle in right_cycles if cycle.direction == direction and np.isfinite(cycle.stance_percent)]
            if left_values:
                self.direction_registry["stance_percent_left"][direction].extend(left_values)
            if right_values:
                self.direction_registry["stance_percent_right"][direction].extend(right_values)

        segments_summary.append(
            {
                "axis": axis,
                "direction_counts": {
                    "left": self._direction_counts(left_cycles),
                    "right": self._direction_counts(right_cycles),
                },
            }
        )

        return {"axis": axis, "segments": segments_summary}

    def _walk_axis(self, df_angles: pd.DataFrame) -> Tuple[Optional[str], np.ndarray]:
        axes = ["x", "z"]
        hip_series = {}

        for axis in axes:
            left_col = f"{axis}_left_hip"
            right_col = f"{axis}_right_hip"
            if left_col not in df_angles.columns or right_col not in df_angles.columns:
                continue
            left = df_angles[left_col].to_numpy(dtype=float)
            right = df_angles[right_col].to_numpy(dtype=float)
            hip_series[axis] = np.nanmean(np.vstack([left, right]), axis=0)

        if not hip_series:
            return None, np.array([])

        ranges = {axis: np.nanmax(values) - np.nanmin(values) for axis, values in hip_series.items()}
        axis = max(ranges.items(), key=lambda item: item[1])[0]
        return axis, hip_series[axis]

    def _label_cycle_directions(
        self,
        hip_positions: np.ndarray,
        left_cycles: List[TemporalCycle],
        right_cycles: List[TemporalCycle],
    ) -> Dict[int, str]:
        valid = np.isfinite(hip_positions)
        if not valid.any():
            return {}

        smooth_window = min(len(hip_positions) // 4 * 2 + 1, 201)
        if smooth_window < 7:
            smooth_window = 7
        smooth = savgol_filter(hip_positions, window_length=smooth_window, polyorder=2)

        distance = max(self.turn_buffer_frames * 2, 20)
        peaks, _ = find_peaks(smooth, distance=distance)
        troughs, _ = find_peaks(-smooth, distance=distance)
        turn_points = np.sort(np.concatenate(([0], peaks, troughs, [len(smooth) - 1])))
        turn_points = np.unique(turn_points)

        axis_range = np.nanmax(smooth) - np.nanmin(smooth)
        threshold = axis_range * 0.01 if axis_range > 0 else 0.01
        buffer = self.turn_buffer_frames

        intervals: List[Tuple[int, int, str]] = []
        for i in range(len(turn_points) - 1):
            start = int(turn_points[i])
            end = int(turn_points[i + 1])
            if end <= start:
                continue
            displacement = smooth[end] - smooth[start]
            if abs(displacement) < threshold:
                direction = "turn"
            elif displacement > 0:
                direction = "outbound"
            else:
                direction = "inbound"
            intervals.append((start, end, direction))

        direction_map: Dict[int, str] = {}
        for cycles in (left_cycles, right_cycles):
            for cycle in cycles:
                start = max(0, cycle.start_frame)
                end = min(len(smooth) - 1, cycle.end_frame)
                label = "turn"
                for seg_start, seg_end, direction in intervals:
                    if end <= seg_start or start >= seg_end:
                        continue
                    if start < seg_start + buffer or end > seg_end - buffer:
                        label = "turn"
                        break
                    label = direction
                    break
                direction_map[cycle.start_frame] = label

        return direction_map

    @staticmethod
    def _direction_counts(cycles: List[TemporalCycle]) -> Dict[str, int]:
        counts = {"outbound": 0, "inbound": 0, "turn": 0}
        for cycle in cycles:
            if cycle.direction in counts:
                counts[cycle.direction] += 1
        return counts

    @staticmethod
    def _directional_stride_estimate(counts: Dict[str, int], fallback: int) -> float:
        outbound = counts.get("outbound", 0)
        inbound = counts.get("inbound", 0)
        if outbound and inbound:
            return float((outbound + inbound) / 2.0)
        if outbound or inbound:
            return float(outbound or inbound)
        return float(fallback)

    # ------------------------------------------------------------------
    # Aggregate summary
    # ------------------------------------------------------------------

    def _build_aggregate_summary(self) -> Dict[str, object]:
        temporal_summary = {
            key: {
                "icc": series.icc(),
                "rmse": series.rmse(),
                "mae": series.mae(),
                "correlation": series.correlation(),
                "n": len(series.ground_truth),
            }
            for key, series in self.temporal_registry.items()
        }

        spm_summary = {}
        for joint, registry in self.waveform_registry.items():
            mp_array = np.array(registry["mediapipe"], dtype=float)
            hosp_array = np.array(registry["hospital"], dtype=float)
            if mp_array.shape != hosp_array.shape or mp_array.ndim != 2 or mp_array.shape[0] < 2:
                continue
            spm_result = self.spm_analyzer.paired_ttest_spm(mp_array, hosp_array, use_permutation=True, n_permutations=5000)
            spm_summary[joint] = self.spm_analyzer.spm_summary(spm_result)

        symmetry_summary = {
            metric: {
                direction: {
                    "mean": float(np.mean(values)) if values else float("nan"),
                    "std": float(np.std(values)) if values else float("nan"),
                    "n": len(values),
                }
                for direction, values in per_direction.items()
            }
            for metric, per_direction in self.direction_registry.items()
        }

        return {
            "temporal_metrics": temporal_summary,
            "spm": spm_summary,
            "directional_metrics": symmetry_summary,
        }


if __name__ == "__main__":
    evaluator = TieredGaitEvaluator()
    report = evaluator.evaluate()
    output_path = Path("tiered_evaluation_report.json")
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved report to {output_path}")
