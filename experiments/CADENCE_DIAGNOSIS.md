# MediaPipe Cadence Diagnosis

## Dataset & Scripts
- Ground truth: `data/processed_new/S1_xx_info.json`
- MediaPipe source: corresponding `*_side_pose_fps*.csv`
- Diagnostic tool: `diagnose_cadence.py`
  ```bash
  python diagnose_cadence.py \
    --subjects S1_01 S1_02 S1_03 ... S1_30 \
    --output cadence_diagnosis_report.json
  ```

## Key Findings (S1_01 exemplar)
| Metric | Value |
| --- | --- |
| Hospital cadence (left/right) | 115.40 / 114.80 steps/min |
| MP method 1 – total steps | **136.01** steps/min |
| MP method 2 – clean cycles (per foot) | 75.00 / 68.66 steps/min |
| MP stride-based cadence | **109.09 / 112.50** steps/min |
| MP filtered total cadence | 78.52 steps/min |
| MP directional cadence (outbound/inbound) | 75.00 / 68.66 steps/min |
| Total detected steps | 124 (91 retained after turn filter) |
| Clean cycles considered | 41 left / 40 right (sum duration ≈ 0.56 min of 0.91 total) |

`cadence_diagnosis_report.json` provides the same breakdown for all 21 analysed subjects.

## Interpretation
1. **Stride-based estimator** – Using percentile-trimmed stride intervals keeps S1_01 within ~5% of hospital cadence while cutting negative ICCs across the cohort.
2. **Turn sensitivity** – Filtering out strikes within ~0.6 s of detected turns removes 33 of 124 steps for S1_01; buffer tuning still materially affects cadence estimates.
3. **Directional drift** – Outbound vs inbound cadence remains imbalanced across subjects, but storing directional, filtered, and stride-based values in the reports helps track where the discrepancy originates.

## Recommended Actions
1. Compute a cohort-wide bias (currently ≈1.10× for stride cadence) and formalise it in the evaluator so cadence predictions stay near hospital values without per-subject tuning.
2. Continue improving cycle labelling to isolate steady walking windows instead of fixed buffers; reassess directional cadence once segmentation stabilises.
3. Validate heel-strike counts against hospital stride totals (`demographics.left_strides/right_strides`) to catch systematic over-detection.
4. Re-run `diagnose_cadence.py` whenever the cadence estimator changes to monitor filtered, directional, and stride-based agreement.
