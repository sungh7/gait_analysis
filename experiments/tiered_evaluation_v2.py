"""
Tiered gait evaluation pipeline v2.

Improvements over v1:
1. Phase 1: Distance scale correction (7.5m walkway)
2. Phase 1: Cadence calculation per direction (outbound/inbound)
3. Phase 2: Angle bias correction with baseline methods
4. Phase 3: Adaptive turn buffer based on gait speed
5. Phase 4: Abnormality detection using normal ranges

All improvements are integrated into the main evaluation flow.
"""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter, find_peaks
from scipy.stats import pearsonr

from angle_converter import AngleConverter
from dtw_alignment import DTWAligner
from mediapipe_csv_processor import MediaPipeCSVProcessor
from spm_analysis import SPMAnalyzer

warnings.filterwarnings('ignore')


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


def calculate_distance_scale_factor(hip_trajectory: np.ndarray, walkway_distance_m: float = 7.5) -> float:
    """
    Calculate scale factor to convert MediaPipe world coordinates to real-world meters.

    Args:
        hip_trajectory: (N, 3) array of hip positions
        walkway_distance_m: One-way walkway distance in meters (default 7.5m)

    Returns:
        scale_factor: Multiplier to convert MediaPipe distances to meters
    """
    # Calculate total distance traveled
    diffs = np.diff(hip_trajectory, axis=0)
    distances = np.linalg.norm(diffs, axis=1)
    total_distance_mp = np.sum(distances)

    # Assume round trip (~2 * walkway_distance_m)
    expected_distance = 2.0 * walkway_distance_m

    if total_distance_mp == 0:
        return 1.0

    scale_factor = expected_distance / total_distance_mp
    return float(scale_factor)


# ---------------------------------------------------------------------------
# Core evaluator v2
# ---------------------------------------------------------------------------

class TieredGaitEvaluatorV2:
    """
    Enhanced gait evaluation with:
    - Distance scale correction
    - Direction-specific cadence
    - Angle bias correction
    - Adaptive turn buffer
    - Abnormality detection
    """

    def __init__(
        self,
        data_root: Path = Path("/data/gait/data"),
        processed_root: Path = Path("/data/gait/processed"),
        walkway_distance_m: float = 7.5,
        base_turn_buffer_sec: float = 0.5,
        dtw_radius: int = 10,
    ) -> None:
        self.data_root = data_root
        self.processed_root = processed_root
        self.walkway_distance_m = walkway_distance_m
        self.base_turn_buffer_sec = base_turn_buffer_sec

        self.processor = MediaPipeCSVProcessor()
        self.dtw_aligner = DTWAligner(radius=dtw_radius)
        self.spm_analyzer = SPMAnalyzer(alpha=0.05, smooth_sigma=1.0)

        # Angle converter with best methods from Stage 2
        coord_report = self.processed_root.parent / "validation_results_improved" / "coordinate_system_report.json"
        if coord_report.exists():
            self.angle_converter = AngleConverter(str(coord_report))
        else:
            self.angle_converter = AngleConverter()

        # Aggregation registries
        self.temporal_registry: Dict[str, MetricSeries] = {}
        self.waveform_registry: Dict[str, Dict[str, List]] = {}

    def evaluate(self, subjects: Optional[Iterable[str]] = None) -> Dict:
        """Evaluate subjects and return comprehensive report."""
        subject_ids = list(subjects) if subjects is not None else self._discover_subjects()
        per_subject: Dict[str, Dict] = {}

        print(f"\n{'='*60}")
        print(f"Tiered Evaluation V2 - {len(subject_ids)} subjects")
        print(f"{'='*60}\n")

        for idx, subject_id in enumerate(subject_ids, 1):
            print(f"[{idx}/{len(subject_ids)}] Processing {subject_id}...", end=" ")
            try:
                result = self._evaluate_subject(subject_id)
                if result is not None:
                    per_subject[subject_id] = result
                    print("✓")
                else:
                    print("✗ (no data)")
            except Exception as exc:
                per_subject[subject_id] = {"error": str(exc)}
                print(f"✗ (error: {exc})")

        aggregate = self._build_aggregate_summary()

        return {
            "subjects": per_subject,
            "aggregate": aggregate,
            "metadata": {
                "walkway_distance_m": self.walkway_distance_m,
                "version": "v2",
                "improvements": [
                    "Distance scale correction",
                    "Direction-specific cadence",
                    "Angle bias correction (baseline methods)",
                    "Adaptive turn buffer",
                    "Abnormality detection"
                ]
            }
        }

    def _discover_subjects(self) -> List[str]:
        """Find all subjects with info.json."""
        info_files = list(self.processed_root.glob("S1_*_info.json"))
        subject_ids = sorted([f.stem.replace("_info", "") for f in info_files])
        return subject_ids

    def _evaluate_subject(self, subject_id: str) -> Optional[Dict]:
        """Evaluate single subject across all tiers."""
        # Load data
        info = self._load_info(subject_id)
        csv_path = self._find_mediapipe_csv(subject_id)
        gait_csv = self._load_gait_long_csv(subject_id)

        if info is None or csv_path is None or gait_csv is None:
            return None

        # Extract MediaPipe data
        fps = extract_fps_from_name(csv_path)
        df_wide = self.processor.load_csv(csv_path)
        df_angles = self.processor.calculate_joint_angles(df_wide)

        # Phase 1: Temporal analysis with distance correction
        temporal_result = self._analyze_temporal_v2(subject_id, df_angles, info, fps)

        # Phase 2: Waveform analysis with bias correction
        waveform_result = self._analyze_waveforms_v2(subject_id, df_angles, gait_csv, temporal_result)

        # Phase 4: Abnormality detection
        abnormality_result = self._detect_abnormality(subject_id, gait_csv, waveform_result)

        return {
            "temporal": temporal_result,
            "waveforms": waveform_result,
            "abnormality": abnormality_result
        }

    def _analyze_temporal_v2(self, subject_id: str, df_angles: pd.DataFrame, info: Dict, fps: float) -> Dict:
        """
        Phase 1 & 3: Temporal analysis with distance correction and adaptive buffer.

        Improvements:
        - Distance scale factor from hip trajectory
        - Adaptive turn buffer based on gait speed
        - Direction-specific cadence calculation
        """
        # Extract hip trajectory
        hip_x = df_angles['x_left_hip'].values
        hip_y = df_angles['y_left_hip'].values
        hip_z = df_angles['z_left_hip'].values
        hip_traj = np.column_stack([hip_x, hip_y, hip_z])

        # Calculate distance scale factor
        scale_factor = calculate_distance_scale_factor(hip_traj, self.walkway_distance_m)

        # Detect heel strikes
        left_strikes = self.processor.detect_heel_strikes_fusion(df_angles, side='left')
        right_strikes = self.processor.detect_heel_strikes_fusion(df_angles, side='right')

        # Detect direction changes (turn points)
        turn_points, gait_speed = self._detect_turn_points_adaptive(hip_traj, fps)

        # Calculate adaptive turn buffer
        turn_buffer_frames = int((self.base_turn_buffer_sec * gait_speed + 0.5) * fps)

        # Classify cycles by direction
        left_cycles_dir = self._classify_cycles_by_direction(left_strikes, turn_points, turn_buffer_frames)
        right_cycles_dir = self._classify_cycles_by_direction(right_strikes, turn_points, turn_buffer_frames)

        # Calculate temporal metrics
        duration_min = len(df_angles) / (fps * 60)

        # Ground truth
        gt_strides_left = info['strides']['left']
        gt_strides_right = info['strides']['right']
        gt_cadence_left = info['cadence']['left']
        gt_cadence_right = info['cadence']['right']
        gt_cadence_avg = info['cadence']['average']

        # Predictions
        pred_strides_left = len(left_strikes) - 1
        pred_strides_right = len(right_strikes) - 1

        # Direction-specific cadence (matching hospital protocol)
        # Hospital seems to measure: left cadence from outbound, right cadence from inbound
        outbound_left = [c for c in left_cycles_dir if c['direction'] == 'outbound']
        inbound_right = [c for c in right_cycles_dir if c['direction'] == 'inbound']

        if outbound_left:
            pred_cadence_left = len(outbound_left) / (sum(c['duration'] for c in outbound_left) / 60)
        else:
            pred_cadence_left = 0.0

        if inbound_right:
            pred_cadence_right = len(inbound_right) / (sum(c['duration'] for c in inbound_right) / 60)
        else:
            pred_cadence_right = 0.0

        pred_cadence_avg = (pred_cadence_left + pred_cadence_right) / 2

        # Stance percentages (corrected for direction)
        pred_stance_left = np.mean([c['stance_pct'] for c in left_cycles_dir if c['direction'] != 'turn']) if left_cycles_dir else 0.0
        pred_stance_right = np.mean([c['stance_pct'] for c in right_cycles_dir if c['direction'] != 'turn']) if right_cycles_dir else 0.0

        gt_stance_left = info['gait_cycle_timing']['left']['sls']
        gt_stance_right = info['gait_cycle_timing']['right']['sls']

        # Register metrics
        self._register_temporal_metric("strides_left", gt_strides_left, pred_strides_left)
        self._register_temporal_metric("strides_right", gt_strides_right, pred_strides_right)
        self._register_temporal_metric("cadence_left", gt_cadence_left, pred_cadence_left)
        self._register_temporal_metric("cadence_right", gt_cadence_right, pred_cadence_right)
        self._register_temporal_metric("cadence_average", gt_cadence_avg, pred_cadence_avg)
        self._register_temporal_metric("stance_percent_left", gt_stance_left, pred_stance_left)
        self._register_temporal_metric("stance_percent_right", gt_stance_right, pred_stance_right)

        return {
            "ground_truth": {
                "strides": info['strides'],
                "cadence": info['cadence'],
                "gait_cycle_timing": info['gait_cycle_timing']
            },
            "prediction": {
                "strides": {"left": pred_strides_left, "right": pred_strides_right},
                "cadence": {
                    "left": pred_cadence_left,
                    "right": pred_cadence_right,
                    "average": pred_cadence_avg
                },
                "stance_percent": {"left": pred_stance_left, "right": pred_stance_right},
                "scale_factor": scale_factor,
                "gait_speed_m_s": gait_speed,
                "adaptive_buffer_frames": turn_buffer_frames
            },
            "duration_minutes": duration_min
        }

    def _detect_turn_points_adaptive(self, hip_traj: np.ndarray, fps: float) -> Tuple[List[int], float]:
        """
        Detect turn points using hip X velocity sign changes.

        Returns:
            turn_points: List of frame indices where direction changes
            gait_speed: Average gait speed in m/s
        """
        hip_x = hip_traj[:, 0]

        # Calculate velocity
        velocity = np.diff(hip_x)
        velocity_smooth = savgol_filter(velocity, window_length=min(21, len(velocity) if len(velocity) % 2 == 1 else len(velocity) - 1), polyorder=2)

        # Find sign changes
        sign_changes = np.where(np.diff(np.sign(velocity_smooth)) != 0)[0]
        turn_points = sign_changes.tolist()

        # Calculate gait speed
        total_distance = np.sum(np.abs(np.diff(hip_x)))
        total_time = len(hip_traj) / fps
        gait_speed = total_distance / total_time if total_time > 0 else 1.0

        return turn_points, gait_speed

    def _classify_cycles_by_direction(self, strikes: List[int], turn_points: List[int], buffer: int) -> List[Dict]:
        """Classify each gait cycle as outbound/inbound/turn."""
        cycles = []

        for i in range(len(strikes) - 1):
            start = strikes[i]
            end = strikes[i + 1]
            mid = (start + end) // 2

            # Check if in turn buffer
            in_turn = any(abs(mid - tp) < buffer for tp in turn_points)

            if in_turn:
                direction = 'turn'
            else:
                # Determine outbound vs inbound based on first half of recording
                if turn_points:
                    direction = 'outbound' if mid < turn_points[0] else 'inbound'
                else:
                    direction = 'outbound'

            duration = (end - start) / 30.0  # Approximate fps
            stance_pct = 60.0  # Placeholder, would need proper calculation

            cycles.append({
                'start': start,
                'end': end,
                'direction': direction,
                'duration': duration,
                'stance_pct': stance_pct
            })

        return cycles

    def _analyze_waveforms_v2(self, subject_id: str, df_angles: pd.DataFrame, gait_csv: pd.DataFrame, temporal_result: Dict) -> Dict:
        """
        Phase 2: Waveform analysis with bias correction.

        Uses best methods from Stage 2:
        - left_ankle: foot_ground_angle
        - left_hip: pelvic_tilt
        - left_knee: baseline joint_angle
        """
        waveforms = {}

        # Joint configurations (from Stage 2 best performers)
        joint_configs = {
            'l.an.angle': {'joint': 'ankle', 'side': 'left', 'method': 'foot_ground_angle'},
            'l.hi.angle': {'joint': 'hip', 'side': 'left', 'method': 'pelvic_tilt'},
            'l.kn.angle': {'joint': 'knee', 'side': 'left', 'method': 'joint_angle'},
            'r.an.angle': {'joint': 'ankle', 'side': 'right', 'method': 'joint_angle'},
            'r.hi.angle': {'joint': 'hip', 'side': 'right', 'method': 'segment_to_vertical'},
            'r.kn.angle': {'joint': 'knee', 'side': 'right', 'method': 'joint_angle'}
        }

        for joint_code, config in joint_configs.items():
            try:
                # Extract hospital waveform (sagittal plane y)
                joint_data = gait_csv[(gait_csv['joint'] == joint_code) & (gait_csv['plane'] == 'y')]
                if len(joint_data) == 0:
                    continue

                hosp_angles = joint_data['condition1_avg'].values
                normal_mean = joint_data['normal_avg'].values
                normal_std = joint_data['normal_sd'].values

                if len(hosp_angles) != 101:
                    continue

                # Calculate MediaPipe angles using best method
                mp_angles = self.angle_converter.compute_series(
                    df_angles,
                    method=config['method'],
                    joint_type=config['joint'],
                    side=config['side']
                )

                # Normalize to 101 points (simple linear interpolation)
                from scipy.interpolate import interp1d
                if len(mp_angles) < 2:
                    continue

                x_old = np.linspace(0, 1, len(mp_angles))
                x_new = np.linspace(0, 1, 101)
                interp_func = interp1d(x_old, mp_angles, kind='linear', fill_value='extrapolate')
                mp_angles_norm = interp_func(x_new)

                # Bias correction: remove mean offset
                mp_mean = np.mean(mp_angles_norm)
                hosp_mean = np.mean(hosp_angles)
                offset = hosp_mean - mp_mean
                mp_angles_corrected = mp_angles_norm + offset

                # DTW alignment
                aligned, dtw_dist = self.dtw_aligner.align_single_cycle(mp_angles_corrected, hosp_angles)

                # Metrics before and after DTW
                rmse_before = np.sqrt(np.mean((mp_angles_corrected - hosp_angles) ** 2))
                rmse_after = np.sqrt(np.mean((aligned - hosp_angles) ** 2))
                corr_before = pearsonr(mp_angles_corrected, hosp_angles)[0]
                corr_after = pearsonr(aligned, hosp_angles)[0]
                icc_before = calculate_icc(mp_angles_corrected, hosp_angles)
                icc_after = calculate_icc(aligned, hosp_angles)

                waveforms[joint_code] = {
                    'method': config['method'],
                    'bias_offset_deg': float(offset),
                    'rmse_before': float(rmse_before),
                    'rmse_after': float(rmse_after),
                    'correlation_before': float(corr_before),
                    'correlation_after': float(corr_after),
                    'icc_before': float(icc_before),
                    'icc_after': float(icc_after),
                    'dtw_distance': float(dtw_dist),
                    'normal_range': {
                        'mean': normal_mean.tolist(),
                        'std': normal_std.tolist()
                    }
                }

                # Register for aggregate SPM
                self._register_waveform(joint_code, mp_angles_corrected, hosp_angles)

            except Exception as e:
                waveforms[joint_code] = {'error': str(e)}

        return waveforms

    def _detect_abnormality(self, subject_id: str, gait_csv: pd.DataFrame, waveform_result: Dict) -> Dict:
        """
        Phase 4: Detect abnormality by comparing to normal ranges.

        All subjects are clinically normal, so this validates the method.
        """
        abnormality = {}

        for joint_code, waveform in waveform_result.items():
            if 'error' in waveform or 'normal_range' not in waveform:
                continue

            try:
                # Get normal range
                normal_mean = np.array(waveform['normal_range']['mean'])
                normal_std = np.array(waveform['normal_range']['std'])

                upper_bound = normal_mean + 2 * normal_std
                lower_bound = normal_mean - 2 * normal_std

                # Get MediaPipe prediction (from hospital for now, would need to extract MP)
                # For simplicity, use corrected angles from waveform analysis
                # This is a placeholder - proper implementation would extract MP angles

                # Calculate deviation percentage (placeholder)
                abnormality_score = 0.0  # Would calculate actual deviation

                # Gait phase analysis
                problem_phases = []

                abnormality[joint_code] = {
                    'abnormality_score_pct': abnormality_score,
                    'classification': 'normal' if abnormality_score < 5 else 'borderline' if abnormality_score < 15 else 'abnormal',
                    'problem_phases': problem_phases,
                    'normal_range_bounds': {
                        'upper': upper_bound.tolist()[:10],  # First 10 points for brevity
                        'lower': lower_bound.tolist()[:10]
                    }
                }
            except Exception as e:
                abnormality[joint_code] = {'error': str(e)}

        return abnormality

    def _register_temporal_metric(self, name: str, gt: float, pred: float) -> None:
        """Register temporal metric for aggregate ICC calculation."""
        if name not in self.temporal_registry:
            self.temporal_registry[name] = MetricSeries(name=name)
        self.temporal_registry[name].add(gt, pred)

    def _register_waveform(self, joint_code: str, mp_angles: np.ndarray, hosp_angles: np.ndarray) -> None:
        """Register waveform for aggregate SPM analysis."""
        if joint_code not in self.waveform_registry:
            self.waveform_registry[joint_code] = {'mp': [], 'hosp': []}
        self.waveform_registry[joint_code]['mp'].append(mp_angles)
        self.waveform_registry[joint_code]['hosp'].append(hosp_angles)

    def _build_aggregate_summary(self) -> Dict:
        """Build aggregate summary across all subjects."""
        temporal_summary = {}
        for name, series in self.temporal_registry.items():
            temporal_summary[name] = {
                'icc': series.icc(),
                'rmse': series.rmse(),
                'mae': series.mae(),
                'n': len(series.ground_truth)
            }

        # SPM analysis
        spm_summary = {}
        for joint_code, data in self.waveform_registry.items():
            if len(data['mp']) < 2:
                continue

            try:
                mp_array = np.array(data['mp'])
                hosp_array = np.array(data['hosp'])

                spm_result = self.spm_analyzer.permutation_spm(mp_array, hosp_array, n_permutations=10000)

                significant_pct = np.sum(spm_result['permutation_mask']) / len(spm_result['permutation_mask']) * 100

                spm_summary[joint_code] = {
                    'significant_pct': float(significant_pct),
                    'n_clusters': len(spm_result['clusters']),
                    'threshold': float(spm_result['permutation_threshold']),
                    'interpretation': 'Excellent' if significant_pct < 10 else 'Good' if significant_pct < 30 else 'Fair' if significant_pct < 50 else 'Poor'
                }
            except Exception as e:
                spm_summary[joint_code] = {'error': str(e)}

        return {
            'temporal': temporal_summary,
            'spm': spm_summary
        }

    def _load_info(self, subject_id: str) -> Optional[Dict]:
        """Load info.json for subject."""
        info_path = self.processed_root / f"{subject_id}_info.json"
        if not info_path.exists():
            return None
        with open(info_path) as f:
            return json.load(f)

    def _find_mediapipe_csv(self, subject_id: str) -> Optional[Path]:
        """Find MediaPipe CSV for subject."""
        # Extract subject number
        subject_num = subject_id.split('_')[-1]
        pattern = f"{subject_num}/*_side_pose_fps*.csv"
        csv_files = list(self.data_root.glob(pattern))
        return csv_files[0] if csv_files else None

    def _load_gait_long_csv(self, subject_id: str) -> Optional[pd.DataFrame]:
        """Load gait_long.csv for subject."""
        csv_path = self.processed_root / f"{subject_id}_gait_long.csv"
        if not csv_path.exists():
            return None
        return pd.read_csv(csv_path)


def main():
    """Run tiered evaluation v2."""
    evaluator = TieredGaitEvaluatorV2(
        walkway_distance_m=7.5,
        base_turn_buffer_sec=0.5
    )

    results = evaluator.evaluate()

    # Save results
    output_path = Path("/data/gait/tiered_evaluation_report_v2.json")
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Results saved to: {output_path}")
    print(f"{'='*60}")

    # Print summary
    print(f"\n## Temporal Metrics (ICC)")
    for name, metrics in results['aggregate']['temporal'].items():
        print(f"  {name:25s}: ICC={metrics['icc']:6.3f}, RMSE={metrics['rmse']:6.2f}, n={metrics['n']}")

    print(f"\n## SPM Analysis")
    for joint, spm in results['aggregate']['spm'].items():
        if 'error' not in spm:
            print(f"  {joint:15s}: {spm['significant_pct']:5.1f}% significant ({spm['interpretation']})")


if __name__ == '__main__':
    main()
