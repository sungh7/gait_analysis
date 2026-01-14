#!/usr/bin/env python3
"""
Summarize V5 tiered evaluation results.

Usage:
    python summarize_v5_report.py tiered_evaluation_report_v5.json --out summary.txt
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple


def to_float(value) -> float:
    if value is None:
        return float("nan")
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(value)
    except Exception:
        return float("nan")


def load_report(path: Path) -> Dict:
    data = json.loads(path.read_text())
    if "aggregate" not in data or "subjects" not in data:
        raise ValueError(f"{path} is not a valid V5 report.")
    return data


def compute_errors(report: Dict) -> Tuple[List[Tuple[str, float, float]], List[Tuple[str, float]]]:
    step_errors: List[Tuple[str, float, float]] = []
    cadence_errors: List[Tuple[str, float]] = []

    for sid, entry in report["subjects"].items():
        gt = entry["temporal"]["ground_truth"]
        pred = entry["temporal"]["prediction"]

        gt_step_left = to_float(gt["step_length_cm"]["left"])
        gt_step_right = to_float(gt["step_length_cm"]["right"])
        pred_step_left = to_float(pred["step_length_cm"]["left"])
        pred_step_right = to_float(pred["step_length_cm"]["right"])

        if all(x == x for x in [gt_step_left, pred_step_left]):
            step_errors.append((sid, abs(pred_step_left - gt_step_left), pred_step_left / gt_step_left if gt_step_left else float("nan")))
        if all(x == x for x in [gt_step_right, pred_step_right]):
            step_errors.append((sid, abs(pred_step_right - gt_step_right), pred_step_right / gt_step_right if gt_step_right else float("nan")))

        gt_cadence = to_float(gt["cadence_steps_min"]["average"])
        pred_cadence = to_float(pred["cadence_steps_min"]["average"])
        if gt_cadence == gt_cadence and pred_cadence == pred_cadence:
            cadence_errors.append((sid, abs(pred_cadence - gt_cadence)))

    return step_errors, cadence_errors


def detect_suspects(report: Dict) -> List[str]:
    suspects = []
    for sid, entry in report["subjects"].items():
        diag = entry["temporal"]["prediction"].get("scale_diagnostics", {})
        if diag.get("suspect_stride_data") or diag.get("method") != "stride_based":
            suspects.append(sid)
    return suspects


def summarize(report_path: Path, out_path: Path | None) -> None:
    report = load_report(report_path)
    aggregate = report["aggregate"]["temporal"]

    lines: List[str] = []
    lines.append("# V5 Evaluation Summary")
    lines.append(f"Report: {report_path}")
    lines.append("")

    lines.append("## Aggregate Metrics (temporal)")
    for key in ["step_length_left_cm", "step_length_right_cm", "forward_velocity_left_cm_s", "forward_velocity_right_cm_s", "cadence_left", "cadence_right", "cadence_average"]:
        if key in aggregate:
            metric = aggregate[key]
            lines.append(f"- {key}: RMSE={metric.get('rmse'):.3f}, MAE={metric.get('mae'):.3f}, ICC={metric.get('icc'):.3f}, n={metric.get('n')}")
    lines.append("")

    step_errors, cadence_errors = compute_errors(report)
    step_errors.sort(key=lambda x: x[1], reverse=True)
    cadence_errors.sort(key=lambda x: x[1], reverse=True)

    lines.append("## Largest Step Length Errors (Top 10)")
    for sid, err, ratio in step_errors[:10]:
        lines.append(f"- {sid}: abs error = {err:.2f} cm, ratio = {ratio:.2f}" if ratio == ratio else f"- {sid}: abs error = {err:.2f} cm")
    lines.append("")

    lines.append("## Largest Cadence Errors (Top 10)")
    for sid, err in cadence_errors[:10]:
        lines.append(f"- {sid}: abs error = {err:.2f} steps/min")
    lines.append("")

    suspects = detect_suspects(report)
    lines.append("## Suspect Stride Scaling")
    if suspects:
        lines.append(f"- Suspect subjects: {', '.join(sorted(set(suspects)))}")
    else:
        lines.append("- None detected")
    lines.append("")

    lines.append("## Strike Ratio Overview")
    ratios = []
    for sid, entry in report["subjects"].items():
        gt = entry["temporal"]["ground_truth"]["strides"]
        pred = entry["temporal"]["prediction"]["strides"]
        for side in ["left", "right"]:
            g = to_float(gt.get(side))
            p = to_float(pred.get(side))
            if g > 0 and p == p:
                ratios.append(p / g)
    if ratios:
        ratios_sorted = sorted(ratios)
        lines.append(f"- mean={sum(ratios)/len(ratios):.3f}, median={ratios_sorted[len(ratios)//2]:.3f}, >1.2 count={sum(r>1.2 for r in ratios)}, <0.8 count={sum(r<0.8 for r in ratios)}")
    else:
        lines.append("- No valid ratios found")

    output = "\n".join(lines)
    print(output)
    if out_path:
        out_path.write_text(output)
        print(f"\nSummary written to {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize tiered evaluation V5 report.")
    parser.add_argument("report", type=Path, help="Path to tiered_evaluation_report_v5.json")
    parser.add_argument("--out", type=Path, help="Optional summary file path")
    args = parser.parse_args()

    summarize(args.report, args.out)


if __name__ == "__main__":
    main()
