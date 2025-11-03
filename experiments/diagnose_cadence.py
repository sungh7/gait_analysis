#!/usr/bin/env python3
"""
Diagnose cadence discrepancies between hospital ground truth and MediaPipe estimates.

Produces a JSON report summarising cadence calculations using two MediaPipe methods:
1. Total steps over total recording duration
2. Clean gait cycles only (excluding detected turns)

Example:
    python diagnose_cadence.py --subjects S1_01 --output cadence_diagnosis_report.json
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from tiered_evaluation_v3 import (
    TieredGaitEvaluatorV3,
    extract_fps_from_name,
    calculate_distance_scale_factor,
)


@dataclass
class CadenceDiagnostics:
    hospital_left: Optional[float]
    hospital_right: Optional[float]
    mp_method1_steps_total: Optional[float]
    mp_method2_clean_cycles_left: Optional[float]
    mp_method2_clean_cycles_right: Optional[float]
    mp_total_filtered: Optional[float]
    mp_stride_cadence_left: Optional[float]
    mp_stride_cadence_right: Optional[float]
    mp_directional_left: Optional[float]
    mp_directional_right: Optional[float]
    total_steps: int
    clean_steps: int
    duration_minutes: float
    clean_duration_minutes: float
    fps: float

    def to_dict(self) -> Dict:
        def safe(value: Optional[float]) -> Optional[float]:
            if value is None:
                return None
            if isinstance(value, (float, np.floating)):
                if not np.isfinite(value):
                    return None
                return float(value)
            return float(value)

        return {
            "hospital": {
                "cadence_left_steps_min": safe(self.hospital_left),
                "cadence_right_steps_min": safe(self.hospital_right),
                "cadence_average_steps_min": safe(
                    np.mean(
                        [v for v in [self.hospital_left, self.hospital_right] if v is not None]
                    )
                    if any(v is not None for v in [self.hospital_left, self.hospital_right])
                    else None
                ),
            },
            "mediapipe": {
                "total_steps_method": safe(self.mp_method1_steps_total),
                "clean_cycles_method_left": safe(self.mp_method2_clean_cycles_left),
                "clean_cycles_method_right": safe(self.mp_method2_clean_cycles_right),
                "clean_cycles_method_average": safe(
                    np.mean(
                        [
                            v
                            for v in [self.mp_method2_clean_cycles_left, self.mp_method2_clean_cycles_right]
                            if v is not None
                        ]
                    )
                    if any(
                        v is not None
                        for v in [self.mp_method2_clean_cycles_left, self.mp_method2_clean_cycles_right]
                    )
                    else None
                ),
                "filtered_total": safe(self.mp_total_filtered),
                "stride_based_left": safe(self.mp_stride_cadence_left),
                "stride_based_right": safe(self.mp_stride_cadence_right),
                "stride_based_average": safe(
                    np.mean(
                        [
                            v
                            for v in [self.mp_stride_cadence_left, self.mp_stride_cadence_right]
                            if v is not None
                        ]
                    )
                    if any(
                        v is not None
                        for v in [self.mp_stride_cadence_left, self.mp_stride_cadence_right]
                    )
                    else None
                ),
                "directional_left": safe(self.mp_directional_left),
                "directional_right": safe(self.mp_directional_right),
            },
            "counts": {
                "total_detected_steps": int(self.total_steps),
                "clean_detected_steps": int(self.clean_steps),
                "duration_minutes": safe(self.duration_minutes),
                "clean_duration_minutes": safe(self.clean_duration_minutes),
                "fps": safe(self.fps),
            },
            "differences": {
                "method1_minus_hospital_avg": safe(
                    (
                        self.mp_method1_steps_total
                        - np.mean(
                            [v for v in [self.hospital_left, self.hospital_right] if v is not None]
                        )
                    )
                    if (
                        self.mp_method1_steps_total is not None
                        and any(v is not None for v in [self.hospital_left, self.hospital_right])
                    )
                    else None
                ),
                "method2_minus_hospital_avg": safe(
                    (
                        np.mean(
                            [
                                v
                                for v in [
                                    self.mp_method2_clean_cycles_left,
                                    self.mp_method2_clean_cycles_right,
                                ]
                                if v is not None
                            ]
                        )
                        - np.mean(
                            [v for v in [self.hospital_left, self.hospital_right] if v is not None]
                        )
                    )
                    if (
                        any(
                            v is not None
                            for v in [
                                self.mp_method2_clean_cycles_left,
                                self.mp_method2_clean_cycles_right,
                            ]
                        )
                        and any(v is not None for v in [self.hospital_left, self.hospital_right])
                    )
                    else None
                ),
                "filtered_minus_hospital": safe(
                    (
                        self.mp_total_filtered
                        - np.mean(
                            [v for v in [self.hospital_left, self.hospital_right] if v is not None]
                        )
                    )
                    if (
                        self.mp_total_filtered is not None
                        and any(v is not None for v in [self.hospital_left, self.hospital_right])
                    )
                    else None
                ),
                "stride_minus_hospital": safe(
                    (
                        np.mean(
                            [
                                v
                                for v in [self.mp_stride_cadence_left, self.mp_stride_cadence_right]
                                if v is not None
                            ]
                        )
                        - np.mean(
                            [v for v in [self.hospital_left, self.hospital_right] if v is not None]
                        )
                    )
                    if (
                        any(
                            v is not None
                            for v in [self.mp_stride_cadence_left, self.mp_stride_cadence_right]
                        )
                        and any(v is not None for v in [self.hospital_left, self.hospital_right])
                    )
                    else None
                ),
            },
        }


def directional_cadence(cycles: List[Dict], fps: float, preferred_direction: str) -> Optional[float]:
    preferred = [c for c in cycles if c["direction"] == preferred_direction]
    usable = preferred if preferred else [c for c in cycles if c["direction"] != "turn"]
    if not usable or fps <= 0:
        return None
    total_minutes = sum((c["end"] - c["start"]) for c in usable) / (fps * 60.0)
    if total_minutes <= 0:
        return None
    return len(usable) / total_minutes


def diagnose_subject(evaluator: TieredGaitEvaluatorV3, subject_id: str) -> CadenceDiagnostics:
    info = evaluator._load_info(subject_id)
    if info is None:
        raise FileNotFoundError(f"Missing info.json for {subject_id}")

    csv_path = evaluator._find_mediapipe_csv(subject_id)
    if csv_path is None:
        raise FileNotFoundError(f"Missing MediaPipe CSV for {subject_id}")

    fps = extract_fps_from_name(csv_path)
    df_wide = evaluator.processor.load_csv(csv_path)
    df_angles = evaluator.processor.calculate_joint_angles(df_wide)

    # Hip trajectory for turn detection
    hip_traj = np.column_stack(
        [
            df_angles["x_left_hip"].values,
            df_angles["y_left_hip"].values,
            df_angles["z_left_hip"].values,
        ]
    )

    left_strikes = evaluator.processor.detect_heel_strikes_fusion(df_angles, side="left", fps=fps)
    right_strikes = evaluator.processor.detect_heel_strikes_fusion(df_angles, side="right", fps=fps)

    total_frames = len(df_angles)
    duration_minutes = total_frames / (fps * 60.0) if fps > 0 else 0.0
    total_steps = max(len(left_strikes) + len(right_strikes) - 2, 0)
    method1_cadence = (
        total_steps / duration_minutes if duration_minutes > 0 and total_steps > 0 else None
    )

    # Turn-aware cadence
    scale_factor = calculate_distance_scale_factor(hip_traj, evaluator.walkway_distance_m)
    hip_traj_scaled = hip_traj * scale_factor
    turn_points, gait_speed = evaluator._detect_turn_points_adaptive(hip_traj_scaled, fps)
    turn_buffer_frames = int((evaluator.base_turn_buffer_sec * gait_speed + 0.5) * fps)
    left_cycles = evaluator._classify_cycles_by_direction(left_strikes, turn_points, turn_buffer_frames, fps)
    right_cycles = evaluator._classify_cycles_by_direction(
        right_strikes, turn_points, turn_buffer_frames, fps
    )
    clean_cycles_left = [c for c in left_cycles if c["direction"] != "turn"]
    clean_cycles_right = [c for c in right_cycles if c["direction"] != "turn"]

    def cadence_from_cycles(cycles):
        if not cycles or fps <= 0:
            return None, 0.0
        total_minutes = sum((c["end"] - c["start"]) for c in cycles) / (fps * 60.0)
        if total_minutes <= 0:
            return None, total_minutes
        return len(cycles) / total_minutes, total_minutes

    cadence_left_clean, left_minutes = cadence_from_cycles(clean_cycles_left)
    cadence_right_clean, right_minutes = cadence_from_cycles(clean_cycles_right)

    clean_steps = len(clean_cycles_left) + len(clean_cycles_right)
    clean_duration_minutes = left_minutes + right_minutes

    filtered_total_cadence = evaluator._compute_total_cadence(
        left_strikes,
        right_strikes,
        turn_points,
        turn_buffer_frames,
        fps
    )

    stride_cadence_left = evaluator._cadence_from_strikes(left_strikes, fps)
    stride_cadence_right = evaluator._cadence_from_strikes(right_strikes, fps)

    directional_left = directional_cadence(left_cycles, fps, "outbound")
    directional_right = directional_cadence(right_cycles, fps, "inbound")

    patient = info.get("patient", {})
    patient_left = patient.get("left", {})
    patient_right = patient.get("right", {})
    hospital_left = patient_left.get("cadence_steps_min")
    hospital_right = patient_right.get("cadence_steps_min")

    print(f"\nSubject {subject_id}")
    print(f"  Hospital cadence: L={hospital_left}, R={hospital_right}")
    print(
        f"  MediaPipe cadence method 1 (total steps): "
        f"{method1_cadence:.2f} steps/min" if method1_cadence is not None else "  MediaPipe cadence method 1: n/a"
    )
    print(
        f"  MediaPipe cadence method 2 (clean cycles): "
        f"L={cadence_left_clean:.2f}, R={cadence_right_clean:.2f}"
        if cadence_left_clean is not None and cadence_right_clean is not None
        else "  MediaPipe cadence method 2: n/a"
    )
    print(
        f"  MediaPipe cadence filtered total: {filtered_total_cadence:.2f} steps/min"
        if filtered_total_cadence is not None and filtered_total_cadence > 0
        else "  MediaPipe cadence filtered total: n/a"
    )
    print(
        f"  MediaPipe cadence stride-based: L={stride_cadence_left:.2f}, R={stride_cadence_right:.2f}"
        if stride_cadence_left is not None and stride_cadence_right is not None and stride_cadence_left > 0 and stride_cadence_right > 0
        else "  MediaPipe cadence stride-based: n/a"
    )
    print(
        f"  Directional cadence (outbound/inbound): "
        f"L={directional_left:.2f}, R={directional_right:.2f}"
        if directional_left is not None and directional_right is not None
        else "  Directional cadence: n/a"
    )

    return CadenceDiagnostics(
        hospital_left=hospital_left,
        hospital_right=hospital_right,
        mp_method1_steps_total=method1_cadence,
        mp_method2_clean_cycles_left=cadence_left_clean,
        mp_method2_clean_cycles_right=cadence_right_clean,
        mp_total_filtered=filtered_total_cadence,
        mp_stride_cadence_left=stride_cadence_left,
        mp_stride_cadence_right=stride_cadence_right,
        mp_directional_left=directional_left,
        mp_directional_right=directional_right,
        total_steps=total_steps,
        clean_steps=clean_steps,
        duration_minutes=duration_minutes,
        clean_duration_minutes=clean_duration_minutes,
        fps=fps,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose MediaPipe cadence against hospital ground truth.")
    parser.add_argument(
        "--subjects",
        nargs="+",
        default=["S1_01"],
        help="Subject IDs to analyse (default: S1_01).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("cadence_diagnosis_report.json"),
        help="Where to store the JSON report.",
    )
    args = parser.parse_args()

    evaluator = TieredGaitEvaluatorV3()
    report: Dict[str, Dict] = {}

    for subject_id in args.subjects:
        diagnostics = diagnose_subject(evaluator, subject_id)
        report[subject_id] = diagnostics.to_dict()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\nCadence diagnosis report saved to {args.output}")


if __name__ == "__main__":
    main()
