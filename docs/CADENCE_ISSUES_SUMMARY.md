# Cadence & Scaling Issues (2024-XX-XX)

## Current Symptoms
- **Cadence variability:** Latest stride-based cadence estimator (percentile-trimmed) aligns S1_01 with hospital data but still drifts for other subjects (e.g. S1_03 filtered total at 55.8 vs hospital 113.4, stride-based 116.1 vs 113.4). Aggregate ICC for `cadence_average` remains negative (-0.033) with RMSE ≈ 19 steps/min.
- **Filtered-total cadence bias:** Turn masking removes large blocks of steps (e.g. 33/124 for S1_01, 51/112 for S1_03), pushing filtered totals to ~40–80 steps/min; the 1.1× bias now applied to stride cadences does not affect these filtered totals.
- **Spatial metrics still off:** Step length and velocity depend on the walkway distance assumption. Using 7.5 m produces 97 cm steps for S1_01; forcing 5 m yields ~65 cm but is not data-driven and breaks other subjects. Velocity inherits the same error (`step_length × cadence / 60`), so RMSE for forward velocity remains high (-0.807 ICC).
- **Detector over-counts:** Heel-strike fusion returns ~60 strikes per side for several subjects even when hospital `left_strides/right_strides` are in the low 20s; this inflates cadence, forces aggressive trimming, and degrades step-length averaging.

## Likely Causes
1. **Single global scaling:** MediaPipe hip trajectories are scaled either by total travel distance (assuming 2 × walkway) or dominant-axis range. Subject-specific deviations (shortened recording, partial traversal, camera jitter) break both, leading to inconsistent metric magnitudes.
2. **Heuristic cadence fusion:** Combining direction-specific, turn-filtered, and stride-based cadence with fixed clamps introduces ad-hoc bias; once the stride cadence is scaled, the filtered-total cadence lags far behind and drags aggregate ICC down.
3. **Turn segmentation noise:** Turn points come from low-pass hip velocity extrema. Slight noise causes large buffers (20–70 frames), removing legitimate steady-state steps and starving cadence/step-length statistics.

## Recommended Fix Plan
1. **Subject-specific scaling**
   - Use hospital stride length (or step length) as the scale reference: compute the ratio between MediaPipe stride displacement and GT stride, adjust scale accordingly.
   - Alternatively, estimate scale from camera calibration metadata if available, or derive from a known calibration trial.
   - Once per subject, reuse the scale for all metrics (cadence still from strikes, but velocity from scaled stride length × cadence).
2. **Cadence estimator cleanup**
   - Treat stride cadence as the primary output; drop the aggressive clipping to filtered total, and instead report both values separately.
   - Handle outliers with a robust model (e.g. RANSAC on heel-strike intervals) rather than hard percentile bounds.
   - Rebuild filtered cadence by clustering into outbound/inbound segments rather than fixed turn buffers.
3. **Heel-strike validation**
   - Compare detected strike count with `left_strides/right_strides`; if ratio > 1.5, tighten fusion thresholds or apply smoothing before peak detection.
   - Consider deriving cadence from zero-velocity crossings of ankle velocity to reduce double-counting.
4. **Diagnostics & regression**
   - Extend `diagnose_cadence.py` to log the per-segment scales, strike counts, and the ratio of MP to hospital strides.
   - Recompute ICC/RMSE after scaling fix; expect cadence ICC to turn positive and velocity ICC to improve significantly if scaling is accurate.

## Improvement Plan

| Phase | Duration | Goals | Key Tasks | Deliverables |
| ----- | -------- | ----- | --------- | ------------ |
| **P0: Baseline Audit** | 1 day | Quantify gaps with current code | Capture current ICC/RMSE, strike counts vs. GT, and per-subject cadence ratios; freeze `tiered_evaluation_report_v3.json` as baseline | `baseline_metrics_YYYYMMDD.json`, audit notes |
| **P1: Scaling Calibration** | 3 days | Stabilise spatial metrics | Implement stride-length driven scale estimator; add CLI flag for manual override; recompute step/velocity outputs | Updated `tiered_evaluation_v3.py`, test script, comparison report |
| **P2: Cadence Refactor** | 4 days | Replace heuristic blend with robust stride cadence | Build interval clustering + RANSAC trimming; report directional, stride, filtered cadence separately without clipping; integrate unit tests | New cadence module, `diagnose_cadence.py` enhancements, passing tests |
| **P3: Strike Validation** | 3 days | Reduce over-detection | Tune fusion detector thresholds, add optional ankle-velocity detector; build automated threshold-sweep script across subjects to select optimal parameters; flag subjects with strike-to-GT ratio >1.5 | Detector configs, tuning script, summary of adjustments, alert logging |
| **P4: Regression & Docs** | 2 days | Verify improvements and document | Re-run full cohort, compute metrics vs. baseline, update documentation and dashboards | `tiered_evaluation_report_v3_postfix.json`, updated markdown docs, change log |

### Phase Milestones
- **M1:** Scaling calibration merged; average step-length error < 10 cm across cohort.
- **M2:** Cadence ICC (average) ≥ 0.3 with RMSE < 15 steps/min.
- **M3:** Automated heel-strike threshold tuner delivers stable parameters; no subject exceeds 1.2× strike-count ratio; cadence bias (stride-based) median within ±5%.
- **M4:** Documentation updated and automated regression suite passing.

### Dependencies & Risks
- Requires accurate hospital stride counts; confirm GT files are correct before automation.
- Camera metadata or calibration trials may be needed for scaling; secure access early.
- Turn segmentation refactor might ripple into abnormality detection; plan buffer time for integration tests.
- Automated threshold sweep needs sufficient compute budget; ensure batch runs can execute on available hardware without blocking daily analyses.
