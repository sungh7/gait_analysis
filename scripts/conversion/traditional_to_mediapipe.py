#!/usr/bin/env python3
"""
Utilities to reshape legacy gait lab Excel exports into tidy tables that line up
with MediaPipe 3D pose outputs (frame × landmark × axis).

Usage
-----

    python traditional_to_mediapipe.py data/excel_files/S1_01.xlsx \
        --output-dir processed --base-name S1_01

Three CSVs are produced:
    * <base>_traditional_tidy_long.csv    – fully long-format across all metrics.
    * <base>_traditional_condition.csv    – Condition 1 values pivoted wide (one row
      per variable × sample, columns per axis) for quick alignment with pose data.
    * <base>_traditional_normals.csv      – normal reference curves (average/SD
      bands) pivoted wide for comparison or visualization.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd


# Metrics we care about and the short keys we will emit.
METRIC_ALIASES: Dict[str, str] = {
    "Condition 1": "condition",
    "Condition 1 Upper StDev": "condition_upper_sd",
    "Condition 1 Lower StDev": "condition_lower_sd",
    "Conditon StDev": "condition_sd",
    "Condition 1.1": "condition_rep1",
    "Condition 1.2": "condition_rep2",
    "Normal Average": "normal_average",
    "Normal Upper SD": "normal_upper_sd",
    "Normal Lower SD": "normal_lower_sd",
    "Normal SD": "normal_sd",
    "Normal SDX2": "normal_sdx2",
}


@dataclass
class TraditionalData:
    """Container for the parsed Excel workbook."""

    dataframe: pd.DataFrame


def _forward_fill_level(values: Iterable) -> List:
    """Forward-fill the first level of headers so pandas can build a MultiIndex."""
    filled: List = []
    current = None
    for val in values:
        if pd.isna(val):
            filled.append(current)
        else:
            current = val
            filled.append(val)
    return filled


def _normalize_top_level(value) -> str:
    """Rename anonymous numeric header buckets to something more descriptive."""
    if isinstance(value, (int, float, np.integer, np.floating)):
        # The Excel exports use raw numbers for the second header row.
        if int(value) == 7:
            return "Sample"
        if int(value) == 0:
            return "Frame"
        if int(value) == 2:
            return "Frame2"
        return f"Level{int(value)}"
    return str(value)


def _normalize_bottom_level(value) -> str:
    if pd.isna(value):
        return ""
    normalized = str(value).strip()
    return normalized


def load_traditional_excel(path: Path) -> TraditionalData:
    """Load the two-row-header Excel sheet into a MultiIndex dataframe."""
    df = pd.read_excel(path, header=[0, 1])
    level0 = _forward_fill_level(df.columns.get_level_values(0))
    level1 = df.columns.get_level_values(1)
    top = [_normalize_top_level(v) for v in level0]
    bottom = [_normalize_bottom_level(v) for v in level1]
    df.columns = pd.MultiIndex.from_arrays([top, bottom])
    return TraditionalData(df)


def _normalize_axis(label: str) -> str:
    """Standardise axis labels (X, X.1 -> X_1, etc.)."""
    label = str(label).strip()
    if not label:
        return "scalar"
    label = label.replace(".", "_").replace(" ", "")
    return label.upper()


def _categorise_variable(var_name: str) -> str:
    """Lightweight categorisation to help downstream filtering."""
    if var_name.isupper():
        return "emg"
    if ".angle" in var_name:
        return "joint_angle"
    if ".moment" in var_name:
        return "joint_moment"
    if ".power" in var_name:
        return "joint_power"
    if ".force" in var_name:
        return "force"
    return "other"


def build_tidy_long(data: TraditionalData) -> pd.DataFrame:
    """Produce a long-format dataframe across all supported metrics."""
    df = data.dataframe
    required_cols = {
        "var_name": ("Version:", "VarName"),
        "sample": ("Sample", "Value1"),
        "zero": ("Sample", "Zero"),
    }
    missing = [alias for alias, key in required_cols.items() if key not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    tidy_frames: List[pd.DataFrame] = []
    base_payload = pd.DataFrame(
        {
            "var_name": df[required_cols["var_name"]],
            "sample": df[required_cols["sample"]],
            "zero": df[required_cols["zero"]],
        }
    )
    base_payload["category"] = base_payload["var_name"].map(_categorise_variable)
    base_payload["source"] = "traditional"

    for excel_name, short_name in METRIC_ALIASES.items():
        if excel_name not in df.columns.get_level_values(0):
            continue

        metric_block = df[excel_name].copy()
        metric_block = metric_block.dropna(axis=1, how="all")
        if metric_block.empty:
            continue

        renamed = {col: _normalize_axis(col) for col in metric_block.columns}
        metric_block = metric_block.rename(columns=renamed)

        metric_payload = pd.concat([base_payload.copy(), metric_block], axis=1)
        long = metric_payload.melt(
            id_vars=["var_name", "sample", "zero", "category", "source"],
            var_name="axis",
            value_name="value",
        )
        long["metric"] = short_name
        tidy_frames.append(long)

    if not tidy_frames:
        raise ValueError("No metrics were found in the workbook.")

    tidy = pd.concat(tidy_frames, ignore_index=True)
    tidy = tidy.dropna(subset=["value"])
    tidy["sample"] = tidy["sample"].astype(float)
    tidy["cycle_index"] = tidy["sample"].round().astype(int)
    # Normalise per variable to guard against exports that do not hit 100 samples.
    max_per_var = (
        tidy.groupby("var_name")["sample"].transform("max").replace(0, np.nan)
    )
    tidy["phase_fraction"] = tidy["sample"] / max_per_var
    tidy["phase_fraction"] = tidy["phase_fraction"].fillna(0.0)
    tidy["phase_percent"] = tidy["phase_fraction"] * 100.0
    tidy["zero"] = tidy["zero"].fillna(0).astype(float)
    tidy["axis"] = tidy["axis"].astype(str)
    tidy = tidy.sort_values(["var_name", "metric", "sample", "axis"]).reset_index(drop=True)
    return tidy


def build_condition_wide(tidy: pd.DataFrame) -> pd.DataFrame:
    """Pivot condition-only data so each axis becomes a dedicated column."""
    condition = tidy[tidy["metric"] == "condition"].copy()
    if condition.empty:
        raise ValueError("Condition metric empty; cannot build comparison table.")

    pivot = (
        condition.pivot_table(
            index=["var_name", "sample", "phase_percent", "category"],
            columns="axis",
            values="value",
        )
        .reset_index()
    )
    # Flatten axis column names to simple strings.
    pivot.columns = [str(col) if col != "" else "value" for col in pivot.columns]
    return pivot


def build_normals_wide(tidy: pd.DataFrame) -> pd.DataFrame:
    """Pivot normal reference metrics (joint angles only) so each metric-axis pair is a column."""
    normals = tidy[
        (tidy["metric"].str.startswith("normal_"))
        & (tidy["category"] == "joint_angle")
    ].copy()
    if normals.empty:
        raise ValueError(
            "Normal metrics empty for joint angles; cannot build normals table."
        )

    pivot = (
        normals.pivot_table(
            index=["var_name", "sample", "phase_percent", "category"],
            columns=["metric", "axis"],
            values="value",
        )
        .reset_index()
    )

    flattened_cols: List[str] = []
    for col in pivot.columns:
        if isinstance(col, tuple):
            metric, axis = col
            if metric in {"var_name", "sample", "phase_percent", "category"}:
                flattened_cols.append(metric)
                continue
            axis = axis.lower() if axis else "scalar"
            flattened_cols.append(f"{metric}__{axis}")
        else:
            flattened_cols.append(str(col))
    pivot.columns = flattened_cols
    rename_map = {
        "var_name": "var_name",
        "sample": "sample",
        "phase_percent": "phase_percent",
        "category": "category",
        "var_name__scalar": "var_name",
        "sample__scalar": "sample",
        "phase_percent__scalar": "phase_percent",
        "category__scalar": "category",
    }
    pivot = pivot.rename(columns=rename_map)
    ordering = ["var_name", "sample", "phase_percent", "category"]
    remaining = [col for col in pivot.columns if col not in ordering]
    pivot = pivot[ordering + remaining]
    pivot = pivot.dropna(axis=1, how="all")
    return pivot


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reshape legacy gait Excel export into MediaPipe-compatible tables."
    )
    parser.add_argument("excel_path", type=Path, help="Path to legacy Excel file.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("processed"),
        help="Directory to store generated CSVs.",
    )
    parser.add_argument(
        "--base-name",
        type=str,
        default=None,
        help="Optional prefix for the generated files; defaults to Excel stem.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    excel_path: Path = args.excel_path
    if not excel_path.exists():
        raise FileNotFoundError(excel_path)

    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    base_name = args.base_name or excel_path.stem

    traditional = load_traditional_excel(excel_path)
    tidy_long = build_tidy_long(traditional)
    condition_wide = build_condition_wide(tidy_long)
    normals_wide = build_normals_wide(tidy_long)

    long_path = output_dir / f"{base_name}_traditional_tidy_long.csv"
    condition_path = output_dir / f"{base_name}_traditional_condition.csv"
    normals_path = output_dir / f"{base_name}_traditional_normals.csv"

    tidy_long.to_csv(long_path, index=False)
    condition_wide.to_csv(condition_path, index=False)
    normals_wide.to_csv(normals_path, index=False)

    print(f"Saved tidy long table to {long_path}")
    print(f"Saved condition comparison table to {condition_path}")
    print(f"Saved normal reference table to {normals_path}")


if __name__ == "__main__":
    main()
