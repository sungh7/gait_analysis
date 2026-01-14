# Tiered Evaluation Summary (MediaPipe vs. Hospital)

## Overview
- Subjects processed: 21 (`tiered_evaluation.py` pipeline)
- Source data: `tiered_evaluation_report.json`
- Structure: Tier 1 temporal–spatial ICC, Tier 2 waveform DTW/SPM, Tier 3 directional symmetry

## Tier 1 – Temporal/Spatial Agreement (ICC vs. info.json)
| Metric | ICC | RMSE | MAE | n |
| --- | --- | --- | --- | --- |
| Strides (Left) | -0.630 | 8.39 | 7.76 | 21 |
| Strides (Right) | -0.578 | 8.08 | 7.57 | 21 |
| Cadence (Left) | -0.609 | 29.20 | 25.74 | 21 |
| Cadence (Right) | -0.941 | 49.56 | 48.79 | 21 |
| Cadence (Avg) | -0.472 | 14.11 | 12.29 | 21 |
| Stance % (Left) | -0.582 | 15.45 | 14.42 | 21 |
| Stance % (Right) | -0.666 | 17.03 | 15.61 | 21 |
| IDS % (Left) | 0.034 | 8.31 | 6.36 | 20 |
| IDS % (Right) | 0.033 | 15.16 | 8.85 | 16 |
| IDS+SS % (Left) | -0.363 | 17.52 | 14.76 | 20 |
| IDS+SS % (Right) | 0.012 | 14.57 | 10.51 | 16 |
| SS % (Left) | -0.238 | 13.58 | 11.60 | 20 |
| SS % (Right) | -0.036 | 10.45 | 8.82 | 16 |

**Observations**
- Widespread negative ICC indicates large disagreement with info.json cadence/phase metrics.
- Small positive ICC values for `IDS %` suggest partial alignment when comparing relative phase ratios rather than absolute counts.
- Cadence predictions remain inflated (~70 steps/min) versus hospital ground truth (~49), reflecting whole-video timing versus clinic protocol windows.

## Tier 2 – Waveform Comparison (DTW + SPM)
- All six sagittal joints show ≥85% significant points in SPM permutation tests → **"Poor agreement"** classification.
- Example (aggregate FDR % significant):
  - `l.hi.angle`: 100%
  - `l.kn.angle`: 85.1%
  - `r.hi.angle`: 100%
  - `r.kn.angle`: 86.1%
- Large SPM t-statistics and Cohen's d highlight systematic bias instead of pure phase offsets.
- DTW improves correlation/ICC per subject but cannot overcome scale mismatch without further angle calibration.

## Tier 3 – Direction-Sensitive Symmetry
| Metric | Outbound Mean ± SD | Inbound Mean ± SD | Samples |
| --- | --- | --- | --- |
| Stance % (Left) | 46.24 ± 17.68 | 53.03 ± 17.24 | 106 / 105 |
| Stance % (Right) | 43.31 ± 17.22 | 49.70 ± 15.27 | 116 / 101 |

**Observations**
- Inbound (return) legs exhibit longer stance phases (~6–7% increase) relative to outbound, suggesting speed reduction or cautious steps during turn-back segment.
- Turn detection uses hip extrema; cycles inside buffer windows are excluded from cadence/phase averages.

## Subject Snapshot – S1_01
| Metric | MediaPipe Prediction | Hospital Ground Truth |
| --- | --- | --- |
| Strides (Left / Right) | 16.0 / 15.5 | 11 / 15 |
| Cadence (Left / Right / Avg) | 70.39 / 70.92 / 70.39 | 38 / 12 / 49 |
| Stance % (Left / Right) | 56.84 / 43.01 | 62.42 / 62.11 |
| IDS % (Left / Right) | 13.67 / 7.04 | 13.03 / 11.89 |
| SS % (Left / Right) | 53.34 / 59.78 | 37.89 / 37.58 |
| Direction counts (Left / Right) | Outbound 19 / 21, Inbound 13 / 10, Turn 30 / 31 | — |

**Notes**
- Stride counts align better after outbound/inbound averaging but still exceed hospital logs, implying extra cycles captured near turnaround.
- IDS timing matches left side closely, while right side shows substantial deviation due to toe-off detection lag.
- Direction segmentation distributes cycles roughly evenly between outbound/inbound with ~30 frames flagged as turn-buffer on each side.

## Recommendations
1. Reconcile cadence measurement windows with hospital protocol to avoid inflated MediaPipe cadence values.
2. Introduce spatial scaling (camera-to-world) so stride length/time metrics reflect the 7–7.5 m walkway rather than pixel distances.
3. Refit joint-angle conversions (e.g., using `angle_converter.py`) or apply bias correction before SPM to reduce 100% significance outcomes.
4. Tune turn-buffer dynamically per subject (based on speed variance) to further isolate clean outbound/inbound cycles.

