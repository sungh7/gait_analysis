# Improved Sagittal Validation Pipeline Summary

## Phase Overview
- **Phase 0** – Coordinate system validation via `coordinate_system_validator.py` confirmed MediaPipe axis orientation and emitted `validation_results_improved/coordinate_system_report.json` for downstream transforms.
- **Phase 1** – Hospital angle reverse engineering (`angle_reverse_engineering.py`) captured joint ranges, inter-subject correlations, and embedded coordinate guidance in `hospital_angle_analysis.json`.
- **Phase 2** – Standardised preprocessing (`preprocessing_pipeline.py`) plus fusion heel-strike detection (`mediapipe_csv_processor.py`) produced consistent gait cycles; regression harness verifies invariants.
- **Phase 3** – `angle_converter.py` now cross-validates multiple joint definitions with NNLS scaling while respecting coordinate flips.
- **Phase 4** – DTW alignment (`dtw_alignment.py`, `sagittal_validation_metrics.py`) and SPM analysis (`spm_analysis.py`) provide temporal alignment metrics and statistical agreement checks.
- **Phase 5** – `run_improved_validation.py` orchestrates the pipeline, writes `improved_validation_report.json`, generates ICC/RMSE plots, and emits regression results.

## Key Artifacts
- **Improved report**: `validation_results_improved/improved_validation_report.json`
- **Baseline vs improved plots**: `validation_results_improved/icc_comparison.png`, `validation_results_improved/rmse_comparison.png`
- **Regression log**: `validation_results_improved/regression_report.json`

## Metric Snapshot (Test Split)
| Joint       | Baseline ICC | Improved ICC | ΔICC |
|-------------|--------------|--------------|------|
| right_knee  | 0.025        | 0.018        | -0.008 |
| left_knee   | 0.126        | 0.344        | +0.218 |
| right_hip   | -0.977       | -0.003       | +0.974 |
| left_hip    | -0.978       | -0.016       | +0.962 |
| right_ankle | -0.982       | -0.008       | +0.973 |
| left_ankle  | -0.951       | -0.071       | +0.880 |

> **Note:** Negative ICCs highlight persistent variance mismatches—investigate subject-specific angle conversions and heel-strike detection tuning, especially for ankles.

## Regression Suite Summary
- `python regression_test.py`
  - Confirms preprocessing log steps, coordinate recommendations, angle conversion, DTW alignment outputs, and SPM diagnostics.

## Next Actions
1. Inspect poor right-side ICCs—review converted angles vs hospital references and revisit fusion weights.
2. Tune angle converter method/weight search (e.g., broader grid, subject-specific offset regularisation).
3. Expand SPM diagnostics with permutation tests when residual diagnostics flag non-normality.
4. Integrate regression suite into CI or pre-commit workflow for continuous guardrails.
