"""
Phase 2 (retry): Cadence estimation with V5 template-based heel strikes.

Uses the DTW template detector (Phase 3B) to produce clean strike sequences
before applying the RANSAC cadence estimator. Results are compared against
the legacy percentile method and stored for documentation.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from fastdtw import fastdtw
from mediapipe_csv_processor import MediaPipeCSVProcessor
from P2_ransac_cadence import estimate_cadence_percentile, estimate_cadence_ransac
from scipy.signal import savgol_filter
from scipy.spatial.distance import euclidean
from tiered_evaluation_v5 import create_reference_template, normalize_signal


DATA_ROOT = Path("data")
PROCESSED_ROOT = Path("data/processed_new")
OUTPUT_JSON = Path("P2_ransac_v5_results.json")
OUTPUT_CSV = Path("P2_ransac_v5_diagnostics.csv")


@dataclass
class SubjectResult:
    subject_id: str
    gt_cadence_left: Optional[float]
    gt_cadence_right: Optional[float]
    gt_cadence_avg: Optional[float]
    strikes_left: int
    strikes_right: int
    cadence_ransac_left: Optional[float]
    cadence_ransac_right: Optional[float]
    cadence_ransac_avg: Optional[float]
    cadence_pct_left: Optional[float]
    cadence_pct_right: Optional[float]
    cadence_pct_avg: Optional[float]
    ransac_error: Optional[float]
    pct_error: Optional[float]
    detector_notes: str


def infer_fps_from_filename(csv_path: Path) -> float:
    """Infer frame rate from filename (fallback to 30 fps)."""
    match = re.search(r"fps(\d+)", csv_path.stem)
    if match:
        return float(match.group(1))
    return 30.0


def find_mediapipe_csv(subject_id: str) -> Optional[Path]:
    """Locate the sagittal MediaPipe CSV for the given subject."""
    subject_token = subject_id.split("_")[-1]

    patterns = [
        f"{subject_token}/*_side_pose_fps*.csv",
        f"{int(subject_token):02d}/*_side_pose_fps*.csv" if subject_token.isdigit() else None,
        f"{int(subject_token)}/*_side_pose_fps*.csv" if subject_token.isdigit() else None,
        f"{subject_id}/*_side_pose_fps*.csv",
    ]

    for pattern in filter(None, patterns):
        matches = sorted(DATA_ROOT.glob(pattern))
        if matches:
            return matches[0]

    return None


def load_subject_info(subject_id: str) -> Optional[Dict]:
    info_path = PROCESSED_ROOT / f"{subject_id}_info.json"
    if not info_path.exists():
        return None
    return json.loads(info_path.read_text())


def extract_gt_cadence(info: Dict) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    patient = info.get("patient", {})
    left = patient.get("left", {})
    right = patient.get("right", {})

    cadence_left = left.get("cadence_steps_min")
    cadence_right = right.get("cadence_steps_min")

    avg_values = [v for v in (cadence_left, cadence_right) if isinstance(v, (int, float))]
    cadence_avg = float(sum(avg_values) / len(avg_values)) if avg_values else None

    return cadence_left, cadence_right, cadence_avg


def extract_gt_stride_counts(info: Dict) -> Tuple[int, int]:
    demographics = info.get("demographics", {})
    left_strides = demographics.get("left_strides")
    right_strides = demographics.get("right_strides")

    # Fallback to legacy structure if needed
    if left_strides is None or right_strides is None:
        strides = info.get("strides", {})
        left_strides = left_strides or strides.get("left", 0)
        right_strides = right_strides or strides.get("right", 0)

    return int(left_strides or 0), int(right_strides or 0)


def _refine_with_fusion(
    template_hits: List[int],
    window_length: int,
    valid_frames: np.ndarray,
    fusion_candidates: List[int]
) -> List[int]:
    """Refine template window hits by snapping to fusion detections within the window."""
    if not template_hits:
        return []

    refined: List[int] = []
    used_candidates: set[int] = set()

    for start_idx in template_hits:
        start_idx_clamped = min(start_idx, len(valid_frames) - 1)
        end_idx_clamped = min(start_idx + window_length, len(valid_frames) - 1)

        start_frame = int(valid_frames[start_idx_clamped])
        end_frame = int(valid_frames[end_idx_clamped])

        # Primary search: fusion strike inside the window
        window_candidates = [
            c for c in fusion_candidates
            if start_frame <= c <= end_frame and c not in used_candidates
        ]

        # Secondary search: expand the window slightly if nothing found
        if not window_candidates:
            expand = max(window_length // 4, 1)
            lower = max(start_frame - expand, 0)
            upper = end_frame + expand
            window_candidates = [
                c for c in fusion_candidates
                if lower <= c <= upper and c not in used_candidates
            ]

        if window_candidates:
            center_idx = min(start_idx + window_length // 2, len(valid_frames) - 1)
            center_frame = int(valid_frames[center_idx])
            best_candidate = min(window_candidates, key=lambda c: abs(c - center_frame))
            refined.append(best_candidate)
            used_candidates.add(best_candidate)
        else:
            refined.append(start_frame)

    refined.sort()
    return refined


def detect_strikes_v5(
    processor: MediaPipeCSVProcessor,
    df_angles: pd.DataFrame,
    fps: float,
    gt_stride_left: int,
    gt_stride_right: int
) -> Tuple[List[int], List[int], str]:
    """Detect heel strikes with template matching; fall back to fusion if needed."""
    notes: List[str] = []

    # Fusion candidates provide precise timing
    fusion_left = processor.detect_heel_strikes_fusion(df_angles, "left", fps)
    fusion_right = processor.detect_heel_strikes_fusion(df_angles, "right", fps)

    left_window_override = _median_interval(fusion_left)
    right_window_override = _median_interval(fusion_right)

    left_template, left_meta = create_reference_template(
        df_angles,
        "left",
        _adjusted_stride_count(df_angles, left_window_override, gt_stride_left),
        fps,
    )
    right_template, right_meta = create_reference_template(
        df_angles,
        "right",
        _adjusted_stride_count(df_angles, right_window_override, gt_stride_right),
        fps,
    )

    if left_template is None:
        notes.append("left_fallback")
        left_strikes = fusion_left
    else:
        left_strikes = _detect_template_windows(
            df_angles,
            left_template,
            left_meta["expected_stride_frames"],
            "left",
            fps,
            fusion_left,
            window_override=left_window_override,
        )

    if right_template is None:
        notes.append("right_fallback")
        right_strikes = fusion_right
    else:
        right_strikes = _detect_template_windows(
            df_angles,
            right_template,
            right_meta["expected_stride_frames"],
            "right",
            fps,
            fusion_right,
            window_override=right_window_override,
        )

    detector_note = ",".join(notes) if notes else "template_v5"
    return left_strikes, right_strikes, detector_note


def _median_interval(strikes: List[int]) -> Optional[int]:
    if len(strikes) < 2:
        return None
    intervals = np.diff(strikes)
    positive = intervals[intervals > 0]
    if len(positive) == 0:
        return None
    return int(np.median(positive))


def _adjusted_stride_count(
    df_angles: pd.DataFrame,
    window_override: Optional[int],
    gt_stride_count: int
) -> int:
    if window_override and window_override > 0:
        estimated = int(len(df_angles) / window_override)
        if estimated > 0:
            return estimated
    return max(gt_stride_count, 1)


def _detect_template_windows(
    df_angles: pd.DataFrame,
    template: np.ndarray,
    expected_stride_frames: int,
    side: str,
    fps: float,
    fusion_candidates: List[int],
    similarity_threshold: float = 0.7,
    window_override: Optional[int] = None,
) -> List[int]:
    """Run template matching and snap accepted windows to fusion detections."""
    heel_y = df_angles[f"y_{side}_heel"].values
    ankle_y = df_angles[f"y_{side}_ankle"].values

    valid_idx = ~(np.isnan(heel_y) | np.isnan(ankle_y))
    if valid_idx.sum() < fps:
        return fusion_candidates

    heel_y = heel_y[valid_idx]
    ankle_y = ankle_y[valid_idx]
    valid_frames = df_angles["frame"].values[valid_idx]

    window_size = min(11, max(3, len(heel_y) // 3))
    if window_size % 2 == 0:
        window_size += 1

    heel_smooth = savgol_filter(heel_y, window_size, 2) if len(heel_y) >= window_size else heel_y
    ankle_smooth = savgol_filter(ankle_y, window_size, 2) if len(ankle_y) >= window_size else ankle_y

    composite = 0.7 * heel_smooth + 0.3 * ankle_smooth
    window_length = max(int(window_override or expected_stride_frames), 1)
    step_size = max(1, window_length // 4)

    template_hits: List[int] = []
    for start_idx in range(0, len(composite) - window_length + 1, step_size):
        end_idx = start_idx + window_length
        window = composite[start_idx:end_idx]

        window_resampled = np.interp(
            np.linspace(0, len(window) - 1, 101),
            np.arange(len(window)),
            window,
        )
        window_normalized = normalize_signal(window_resampled)

        distance, _ = fastdtw(template.reshape(-1, 1), window_normalized.reshape(-1, 1), dist=euclidean)
        similarity = 1.0 / (1.0 + (distance / len(template)))

        if similarity >= similarity_threshold:
            if not template_hits or (start_idx - template_hits[-1]) >= window_length * 0.6:
                template_hits.append(start_idx)

    refined = _refine_with_fusion(template_hits, window_length, valid_frames, fusion_candidates)
    return refined or fusion_candidates


def compute_cadence(
    strikes: List[int],
    fps: float
) -> Tuple[Optional[float], Optional[float], Dict, Dict]:
    """Convenience wrapper returning both RANSAC and percentile cadence estimates."""
    if len(strikes) < 3 or fps <= 0:
        return None, None, {"error": "insufficient_strikes"}, {"error": "insufficient_strikes"}

    cadence_ransac, diag_ransac = estimate_cadence_ransac(strikes, fps=fps)
    cadence_pct, diag_pct = estimate_cadence_percentile(strikes, fps=fps)

    return cadence_ransac, cadence_pct, diag_ransac, diag_pct


def safe_average(values: List[Optional[float]]) -> Optional[float]:
    valid = [v for v in values if isinstance(v, (int, float)) and not math.isnan(v)]
    return float(sum(valid) / len(valid)) if valid else None


def process_subject(subject_id: str, processor: MediaPipeCSVProcessor) -> Optional[SubjectResult]:
    info = load_subject_info(subject_id)
    if info is None:
        print(f"[WARN] Missing info for {subject_id}")
        return None

    csv_path = find_mediapipe_csv(subject_id)
    if csv_path is None or not csv_path.exists():
        print(f"[WARN] Missing MediaPipe CSV for {subject_id}")
        return None

    fps = infer_fps_from_filename(csv_path)
    df_wide = processor.load_csv(csv_path)
    df_angles = processor.calculate_joint_angles(df_wide)

    gt_stride_left, gt_stride_right = extract_gt_stride_counts(info)
    left_strikes, right_strikes, detector_note = detect_strikes_v5(
        processor,
        df_angles,
        fps,
        gt_stride_left,
        gt_stride_right,
    )

    cadence_ransac_left, cadence_pct_left, _, _ = compute_cadence(left_strikes, fps)
    cadence_ransac_right, cadence_pct_right, _, _ = compute_cadence(right_strikes, fps)

    cadence_ransac_avg = safe_average([cadence_ransac_left, cadence_ransac_right])
    cadence_pct_avg = safe_average([cadence_pct_left, cadence_pct_right])

    gt_left, gt_right, gt_avg = extract_gt_cadence(info)

    ransac_error = (cadence_ransac_avg - gt_avg) if (cadence_ransac_avg is not None and gt_avg is not None) else None
    pct_error = (cadence_pct_avg - gt_avg) if (cadence_pct_avg is not None and gt_avg is not None) else None

    return SubjectResult(
        subject_id=subject_id,
        gt_cadence_left=gt_left,
        gt_cadence_right=gt_right,
        gt_cadence_avg=gt_avg,
        strikes_left=len(left_strikes),
        strikes_right=len(right_strikes),
        cadence_ransac_left=cadence_ransac_left,
        cadence_ransac_right=cadence_ransac_right,
        cadence_ransac_avg=cadence_ransac_avg,
        cadence_pct_left=cadence_pct_left,
        cadence_pct_right=cadence_pct_right,
        cadence_pct_avg=cadence_pct_avg,
        ransac_error=ransac_error,
        pct_error=pct_error,
        detector_notes=detector_note,
    )


def select_subject_ids() -> List[str]:
    """Determine which subjects to process (intersects processed_new with available CSVs)."""
    summary_path = Path("data/processed_new/conversion_summary.json")
    if not summary_path.exists():
        return []

    summary = json.loads(summary_path.read_text())
    candidates = [entry["subject_id"] for entry in summary.get("subjects", [])]

    available = []
    for subject_id in candidates:
        if find_mediapipe_csv(subject_id):
            available.append(subject_id)

    return available


def summarize_results(results: List[SubjectResult]) -> Dict:
    def mae(values: List[Optional[float]]) -> Optional[float]:
        valid = [abs(v) for v in values if isinstance(v, (int, float)) and not math.isnan(v)]
        return float(np.mean(valid)) if valid else None

    ransac_mae = mae([res.ransac_error for res in results if res.ransac_error is not None])
    pct_mae = mae([res.pct_error for res in results if res.pct_error is not None])

    improvements = []
    for res in results:
        if res.ransac_error is None or res.pct_error is None:
            continue
        improvements.append(abs(res.pct_error) - abs(res.ransac_error))

    improvement_mean = float(np.mean(improvements)) if improvements else None

    return {
        "ransac_mae": ransac_mae,
        "percentile_mae": pct_mae,
        "mean_improvement": improvement_mean,
        "n_subjects": len(results),
    }


def write_outputs(results: List[SubjectResult]) -> None:
    timestamp = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    summary = summarize_results(results)

    payload = {
        "test_date": timestamp,
        "n_subjects": len(results),
        "results": [
            {
                "subject_id": res.subject_id,
                "gt_cadence_left": res.gt_cadence_left,
                "gt_cadence_right": res.gt_cadence_right,
                "gt_cadence_avg": res.gt_cadence_avg,
                "strikes_left": res.strikes_left,
                "strikes_right": res.strikes_right,
                "cadence_ransac_left": res.cadence_ransac_left,
                "cadence_ransac_right": res.cadence_ransac_right,
                "cadence_ransac_avg": res.cadence_ransac_avg,
                "cadence_percentile_left": res.cadence_pct_left,
                "cadence_percentile_right": res.cadence_pct_right,
                "cadence_percentile_avg": res.cadence_pct_avg,
                "ransac_error": res.ransac_error,
                "percentile_error": res.pct_error,
                "detector": res.detector_notes,
            }
            for res in results
        ],
        "summary": summary,
    }

    OUTPUT_JSON.write_text(json.dumps(payload, indent=2))

    df = pd.DataFrame(payload["results"])
    df.to_csv(OUTPUT_CSV, index=False)

    print(f"[INFO] Saved JSON results to {OUTPUT_JSON}")
    print(f"[INFO] Saved diagnostics CSV to {OUTPUT_CSV}")

    if summary["ransac_mae"] is not None and summary["percentile_mae"] is not None:
        print(
            f"[INFO] MAE (steps/min): RANSAC={summary['ransac_mae']:.2f}, "
            f"Percentile={summary['percentile_mae']:.2f}"
        )


def main() -> None:
    subject_ids = select_subject_ids()
    if not subject_ids:
        print("[ERROR] No subjects found for cadence evaluation.")
        return

    processor = MediaPipeCSVProcessor()
    results: List[SubjectResult] = []

    for subject_id in subject_ids:
        print(f"\n[PROCESS] {subject_id}")
        result = process_subject(subject_id, processor)
        if result is not None:
            results.append(result)

    if not results:
        print("[ERROR] No valid results produced.")
        return

    write_outputs(results)


if __name__ == "__main__":
    main()
