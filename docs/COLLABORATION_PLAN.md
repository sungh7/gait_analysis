# ICC Improvement Implementation: Collaboration Plan

**Date**: 2025-11-07

**Team**: Claude (Analysis & Decision) + Codex (Implementation & Automation)

**Timeline**: 5-6 weeks (Phase 1-4 + External Validation)

---

## Role Assignment

### 🤖 Codex Responsibilities

#### **Phase 1 (Week 1): Core Calibration & Filtering Pipeline**

##### Day 1-3: λ Estimation & Deming Regression
**Deliverables**:
- [ ] `estimate_lambda_from_repeatability.py`
  - Input: Repeatability CSV (5 subjects × 3 trials)
  - Output: `lambda_estimates.json`
  - Validation: Within-subject variance, 95% CI for λ

- [ ] `improve_calibration_deming.py` (enhance existing)
  - Input: GT angles, MP angles, lambda_estimates.json
  - Output: `calibration_parameters_deming.json`
  - Validation: Equivalence test (slope CI contains 1.0, intercept CI contains 0.0)

- [ ] `apply_deming_calibration.py`
  - Apply calibration to all 17 subjects
  - Output: Calibrated MP angles
  - Validation: Mean(GT - MP_calibrated) < 0.5°

**Quality Checks**:
```python
# Auto-validation in scripts
assert lambda_ratio > 0.8 and lambda_ratio < 2.0, "λ out of reasonable range"
assert abs(mean_difference) < 0.5, "Calibration bias too large"
assert slope_ci[0] < 1.0 < slope_ci[1], "Proportional bias not corrected"
```

##### Day 4-5: Signal Stabilization
**Deliverables**:
- [ ] `apply_butterworth_filter_batch.py`
  - Joint-specific cutoffs (6 Hz hip/knee, 4 Hz ankle)
  - Bidirectional filtering (zero-phase)
  - Validation: corr(raw, filtered) > 0.95 for each subject

- [ ] `extract_all_cycles_averaging.py`
  - Detect all HS events per subject
  - Weighted multi-cycle averaging
  - Output: Representative cycles with reduced variance

- [ ] `validate_filtering_quality.py`
  - Auto-check correlation, RMSE, peak preservation
  - Generate validation report per joint
  - Flag subjects failing validation

**Output**:
- `processed/phase1_filtered_angles/` (17 subjects × 3 joints)
- `processed/phase1_filter_validation.csv`
- `processed/phase1_multicycle_summary.json`

##### Day 6-7: Phase 1 Validation Report
**Deliverable**:
- [ ] `generate_phase1_report.py`
  - Compute ICC(2,1) with 95% CI (using pingouin)
  - Centered RMSE, correlation
  - Pass/fail criteria check
  - Output: `phase1_validation_report.json`

**Success Criteria** (auto-checked):
```python
criteria = {
    'icc_target': 0.48,
    'icc_ci_lower_min': 0.40,
    'rmse_reduction_min': 0.15,  # 15% reduction
    'correlation_improvement_min': 0.05
}
```

---

#### **Phase 2 (Week 2): Temporal Alignment & Agreement Analysis**

##### Day 8-9: Gait Event Detection
**Deliverables**:
- [ ] `detect_gait_events.py`
  - GT: Force plate threshold (10N)
  - MP: Heel height local minima
  - Validation: Manual check on 9 cycles (3 subjects × 3)
  - Output: `processed/gait_events.json`

**Validation Protocol**:
```python
# Manual validation on sample
sample_subjects = ['S1_01', 'S1_02', 'S1_03']
for subj in sample_subjects:
    gt_hs, mp_hs = load_events(subj)
    timing_diff = abs(gt_hs - mp_hs)
    assert timing_diff.mean() < 0.05 * cycle_length, "HS timing error > 5%"
```

##### Day 10-11: Multi-Stage DTW
**Deliverables**:
- [ ] `multistage_dtw_alignment.py`
  - Stage 1: Coarse event-based alignment
  - Stage 2: Derivative DTW (radius=10)
  - Stage 3: Savitzky-Golay smoothing (window=11)
  - Output: Aligned MP cycles + metadata

- [ ] `validate_dtw_quality.py`
  - Correlation improvement Δr
  - DTW distance vs signal variance
  - Visual inspection plots (sample subjects)

**Success Criteria**:
```python
assert delta_r > 0.15, "Correlation improvement < 0.15"
assert dtw_distance / signal_variance < 0.05, "DTW distance too high"
```

##### Day 12: Bland-Altman Analysis
**Deliverables**:
- [ ] `bland_altman_comprehensive.py`
  - Generate plots for 3 joints
  - Compute mean difference, 95% LoA
  - Proportional bias test (correlation)
  - Output: `processed/bland_altman_phase2.csv` + PNG plots

**Target Limits**:
```python
target_loa = {
    'ankle': 5.0,  # ±5°
    'hip': 12.0,   # ±12°
    'knee': 8.0    # ±8°
}
```

##### Day 13-14: Phase 2 Validation Report
**Deliverable**:
- [ ] `generate_phase2_report.py`
  - ICC(2,1) ≥ 0.63 check
  - Correlation r > 0.55 check
  - Bland-Altman summary
  - Output: `phase2_validation_report.json`

---

### 🧑‍🔬 Claude Responsibilities

#### **Phase 1 (Week 1): Validation & Decision**

##### Day 6-7: Phase 1 Analysis & Go/No-Go
**Tasks**:
- [ ] Review `phase1_validation_report.json`
- [ ] Interpret ICC 95% CI
- [ ] Analyze failure cases (if any)
- [ ] **Go/No-Go Decision**:
  - ✅ GO if ICC ≥ 0.48 → Proceed to Phase 2
  - ❌ NO-GO if ICC < 0.48 → Troubleshoot
    - Check Deming parameters reasonable?
    - Review subjects failing filter validation
    - Decide: Re-tune or exclude outliers?

**Deliverable**:
- `PHASE1_DECISION_MEMO.md`
- Updated plan if troubleshooting needed

---

#### **Phase 2 (Week 2): Validation & Decision**

##### Day 13-14: Phase 2 Analysis & Go/No-Go
**Tasks**:
- [ ] Review `phase2_validation_report.json`
- [ ] Analyze Bland-Altman results
- [ ] Check for proportional bias
- [ ] **Go/No-Go Decision**:
  - ✅ GO if ICC ≥ 0.63 → Proceed to Phase 3
  - ❌ NO-GO if ICC < 0.63 → Troubleshoot
    - Event detection accuracy sufficient?
    - DTW distance reasonable?
    - Re-tune DTW radius?

**Deliverable**:
- `PHASE2_DECISION_MEMO.md`

---

#### **Phase 3 (Week 3): Joint-Specific Tuning & LOOCV**

Claude takes lead on Phase 3 (implementation + analysis):

##### Day 15-16: Joint-Specific Pipelines
**Tasks**:
- [ ] Design joint-specific processing logic
- [ ] Implement `joint_specific_processing.py`:
  - Ankle: 4 Hz filter
  - Knee: Event-based alignment (peak flexion)
  - Hip: Pelvis-referenced angles (if feasible)

##### Day 17-18: Hyperparameter Optimization
**Tasks**:
- [ ] Implement `optimize_hyperparameters.py`
- [ ] Grid search:
  - Filter cutoffs: [3, 4, 5, 6, 7] Hz
  - DTW radius: [5, 8, 10, 12, 15]
  - Smooth window: [7, 9, 11, 13, 15]
- [ ] Select optimal per joint

##### Day 19-20: LOOCV
**Tasks**:
- [ ] Implement `cross_validate_icc.py`
- [ ] Leave-one-subject-out (n=17 folds)
- [ ] Overfitting detection: LOOCV ICC < Full ICC - 0.05

##### Day 21: Phase 3 Validation & CRITICAL MILESTONE
**Tasks**:
- [ ] Compute final ICC with optimized pipeline
- [ ] **Success**: ICC ≥ 0.73 (GOOD agreement!)
- [ ] **Decision**: Paper ready for submission vs continue to Phase 4

**Deliverable**:
- `phase3_validation_report.json`
- `PHASE3_DECISION_MEMO.md`
- **If ICC ≥ 0.73**: Draft Results section for paper

---

#### **Phase 4 (Week 4): Quality Stratification**

Claude takes lead:

##### Day 22-23: Quality Scoring
**Tasks**:
- [ ] Implement `compute_quality_scores.py`
- [ ] 4-component scoring:
  1. Landmark confidence (from Codex's MP processing)
  2. Video quality (resolution, lighting)
  3. Cycle consistency (CV from GT)
  4. Alignment quality (correlation, DTW distance)

##### Day 24-25: Stratified Analysis
**Tasks**:
- [ ] Implement `stratified_icc_analysis.py`
- [ ] Compute ICC for:
  - All subjects (n=17)
  - High quality (n≈10)
  - Very high quality (n≈5)
- [ ] Target: High-quality ICC ≥ 0.80

##### Day 26-27: Measurement Protocol
**Tasks**:
- [ ] Write `MEASUREMENT_PROTOCOL.md`
- [ ] Video quality guidelines
- [ ] Minimum cycles recommendation

##### Day 28: Phase 4 Final Report
**Deliverable**:
- `phase4_stratified_results.json`
- `MEASUREMENT_PROTOCOL.md`

---

#### **Week 5-6: External Validation (GAVD)**

Claude takes lead, Codex assists with batch processing:

##### Week 5: GAVD Processing
**Codex Tasks**:
- [ ] Apply Phase 1-3 pipeline to 137 GAVD samples
- [ ] Output: `processed/gavd_processed_angles/`

**Claude Tasks**:
- [ ] Design external validation protocol
- [ ] Compare GAVD distribution to GT reference
- [ ] Pseudo-ICC computation

##### Week 6: Analysis & Paper Revision
**Claude Tasks**:
- [ ] Implement `gavd_external_validation.py`
- [ ] Check ICC degradation < 0.10
- [ ] Check 90% samples within GT ±2SD
- [ ] **Final Decision**: External validation successful?

**Deliverable**:
- `gavd_external_validation_report.json`
- **Updated paper Results section** with external validation

---

## Communication Protocol

### Daily Standups (Async)

**Codex Reports** (end of day):
```markdown
## Day X Progress
- Completed: [script names]
- Validation results: [pass/fail summary]
- Blockers: [if any]
- Tomorrow: [next tasks]
```

**Claude Reviews** (next morning):
```markdown
## Review of Day X
- Validation approved: ✅ / ⚠️ / ❌
- Issues to address: [if any]
- Go ahead with Day X+1: Yes / No
```

### Weekly Validation Gates

**Phase Completion**:
1. Codex generates validation report JSON
2. Claude reviews within 24 hours
3. Go/No-Go decision documented in memo
4. If GO → Next phase starts
5. If NO-GO → Troubleshooting plan documented

---

## File Naming Conventions

### Scripts
```
action_target_details.py

Examples:
- estimate_lambda_from_repeatability.py
- apply_butterworth_filter_batch.py
- generate_phase1_report.py
```

### Output Data
```
processed/phase{N}_{description}.{ext}

Examples:
- processed/phase1_filtered_angles/S1_01_ankle.csv
- processed/phase1_validation_report.json
- processed/phase2_bland_altman.csv
```

### Reports
```
PHASE{N}_{TYPE}.md

Examples:
- PHASE1_DECISION_MEMO.md
- PHASE3_DECISION_MEMO.md
- MEASUREMENT_PROTOCOL.md
```

---

## Success Metrics Tracking

### Phase 1 Target: ICC 0.48+
| Metric | Baseline | Target | Codex Computes | Claude Interprets |
|--------|----------|--------|----------------|-------------------|
| ICC(2,1) | 0.35 | 0.48 | ✅ | ✅ |
| 95% CI | - | Lower > 0.40 | ✅ | ✅ |
| Centered RMSE | 5.41/25.4/10.86° | 15% reduction | ✅ | ✅ |
| Correlation | 0.25 | +0.05 improvement | ✅ | ✅ |

### Phase 2 Target: ICC 0.63+
| Metric | Baseline | Target | Codex Computes | Claude Interprets |
|--------|----------|--------|----------------|-------------------|
| ICC(2,1) | 0.48 | 0.63 | ✅ | ✅ |
| Correlation | 0.30 | 0.55 | ✅ | ✅ |
| Δr (DTW improvement) | - | >0.15 | ✅ | ✅ |
| Bland-Altman LoA | - | Within target | ✅ | ✅ |

### Phase 3 Target: ICC 0.73+ (CRITICAL)
| Metric | Baseline | Target | Computed By | Interpreted By |
|--------|----------|--------|-------------|----------------|
| ICC(2,1) | 0.63 | 0.73 | Claude | Claude |
| LOOCV ICC | - | Within 0.03 | Claude | Claude |
| Centered RMSE | - | <4/15/8° | Claude | Claude |

### Phase 4 Target: ICC 0.80+ (High-Quality)
| Metric | Baseline | Target | Computed By | Interpreted By |
|--------|----------|--------|-------------|----------------|
| ICC (all) | 0.73 | Maintain | Claude | Claude |
| ICC (high-Q) | - | 0.80+ | Claude | Claude |

---

## Contingency Plans

### If Phase 1 Fails (ICC < 0.48)
**Troubleshooting Owner**: Claude

**Actions**:
1. Review Deming parameters (slope, intercept reasonable?)
2. Check subjects with low calibration quality
3. Options:
   - Re-tune λ estimation (check repeatability data)
   - Exclude 1-2 outlier subjects
   - Adjust filter cutoffs (too aggressive?)

**Decision Timeline**: 1-2 days max

---

### If Phase 2 Fails (ICC < 0.63)
**Troubleshooting Owner**: Claude

**Actions**:
1. Manual validation of event detection (accuracy?)
2. Check DTW distance distribution
3. Options:
   - Increase DTW radius (10 → 15)
   - Adjust event detection parameters
   - Exclude subjects with persistent alignment failures

**Decision Timeline**: 1-2 days max

---

### If Phase 3 Fails (ICC < 0.73)
**Critical Decision Point**

**Options**:
1. **Option A**: Paper with ICC 0.65-0.70 ("moderate-good")
   - Still publishable in applied journals
   - Emphasize high-quality subset (may reach 0.75+)

2. **Option B**: Extended troubleshooting (1 week)
   - Deep dive into outlier subjects
   - Consider excluding 2-3 lowest-quality subjects
   - Re-optimize entire pipeline

**Decision**: Claude consults with team

---

## Immediate Next Steps

### Week 1 Day 1 (Tomorrow): Codex Starts

**Task 1**: Create repeatability data structure
```csv
# Example: repeatability_study.csv
subject_id,trial,gt_ankle,mp_ankle,gt_hip,mp_hip,gt_knee,mp_knee
1,1,10.5,9.8,30.2,28.5,55.3,54.1
1,2,10.3,9.9,30.5,28.8,55.1,54.3
1,3,10.4,9.7,30.3,28.6,55.2,54.0
...
```

**Task 2**: Run `estimate_lambda_from_repeatability.py`
- Input: repeatability_study.csv
- Output: lambda_estimates.json
- Expected: λ ≈ 1.0-1.5 for all joints

**Task 3**: Implement Deming regression with λ
- Input: GT angles (17 subjects), MP angles, λ
- Output: calibration_parameters_deming.json

**Estimated Time**: Day 1-2 (2 days for λ + Deming + validation)

---

## Questions for Codex Before Starting

1. **Data availability**: Do we have repeatability study data (5 subjects × 3 trials)?
   - If NO: Should we simulate or skip λ estimation (use λ=1.0 default)?

2. **GT and MP angle data format**: What's the current structure?
   - JSON: `{subject_id: {joint_name: [101 points]}}`?
   - Or different format?

3. **Dependencies**: Any missing Python packages?
   - scipy (for ODR)
   - pingouin (for ICC)
   - pandas, numpy, matplotlib

4. **Computation environment**: Local or server?
   - Any GPU available (not needed, but asking)?

---

## Success Definition

### Phase 1-2 Success (Codex's scope)
- ✅ All scripts run without errors
- ✅ Validation reports generated automatically
- ✅ Pass/fail criteria clearly marked
- ✅ ICC improvement trajectory on track (0.35 → 0.50 → 0.65)

### Phase 3-4 Success (Claude's scope)
- ✅ ICC ≥ 0.73 achieved (GOOD agreement)
- ✅ LOOCV confirms no overfitting
- ✅ High-quality subset ICC ≥ 0.80
- ✅ Paper Results section drafted

### External Validation Success (Joint effort)
- ✅ GAVD pipeline applied (137 samples)
- ✅ ICC degradation < 0.10
- ✅ 90% samples within GT ±2SD
- ✅ Paper revised with external validation

---

**Document Status**: Collaboration plan finalized

**Start Date**: Week 1 Day 1 (Codex begins)

**Review Frequency**: Daily progress updates, Weekly validation gates

**Primary Goal**: ICC 0.75+ by Week 3 Day 21 → Paper submission ready
