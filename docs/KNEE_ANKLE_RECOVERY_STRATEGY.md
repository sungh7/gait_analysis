# Knee & Ankle Recovery Strategy

**Date**: 2025-11-09
**Goal**: Improve Knee and Ankle ICC to >0.60 (Good agreement)

---

## Current Status

| Joint | Current ICC | ROM Issue | Root Cause | Target ICC |
|-------|-------------|-----------|------------|------------|
| **Knee** | 0.096 | **Underestimation** (1/2-1/3) | Coordinate frame scale | >0.60 |
| **Ankle** | -0.023 | **Overestimation** (2-3x) | Coordinate frame scale | >0.50 |

---

## Problem Analysis

### Knee: ROM Underestimation

**Observed**:
```
MP ROM:   8-30° (too small!)
GT ROM:  58-67° (normal)
Ratio:   ~1/2 to 1/3

Example (S1_01):
  MP: -48.0° ± 7.0°,  ROM = 28.2°
  GT:  19.0° ± 20.6°, ROM = 61.9°
```

**Deming regression failure**:
```
Slope:     10.115  (abnormally high!)
Intercept: 540.7°  (nonsensical)
```

This indicates **fundamental scale mismatch**, not a calibration issue.

### Ankle: ROM Overestimation

**Observed**:
```
MP ROM:  45-85° (too large!)
GT ROM:  15-20° (normal)
Ratio:   2-3x

Example (S1_01):
  MP: 31.1° ± 20.4°, ROM = 78.6°
  GT:  1.5° ± 8.5°,  ROM = 17.0°
```

**Deming regression failure**:
```
Slope:     0.025  (nearly zero!)
Intercept: -0.6°  (reasonable)
```

Slope near zero means MP and GT have **opposite scaling issues**.

---

## Strategy Options

### Option A: Coordinate Frame Deep Dive (Rigorous)

**Approach**: Systematically validate every step of angle calculation against Vicon PiG standard.

#### Phase 1: Theoretical Validation (2-3 days)

**Knee**:
1. Review Vicon PiG femur/tibia frame definitions
2. Compare with current implementation
3. Check for missing transformations or normalizations
4. Validate YXZ Cardan angle extraction formulas

**Ankle**:
1. Review YZX extraction implementation
2. Check foot frame axis scaling
3. Validate heel-toe-ankle geometry
4. Test with synthetic data (known angles)

#### Phase 2: Implementation Fix (2-3 days)

**Test-driven approach**:
1. Create unit tests with known rotation matrices
2. Implement fixes based on Phase 1 findings
3. Validate on 1-2 test subjects
4. Apply to full cohort

#### Phase 3: Validation (1 day)

1. Re-run Deming + DTW
2. Check ICC improvement
3. Validate waveform shapes

**Total time**: 5-7 days
**Success probability**: 70-80%
**Expected ICC**: Knee 0.60-0.70, Ankle 0.50-0.60

---

### Option B: Empirical Scaling Factors (Quick Fix)

**Approach**: Add empirical correction factors to fix ROM scale.

#### Implementation (1 day)

**Knee correction**:
```python
def _calculate_knee_angle(self, landmarks, side='left'):
    # ... existing calculation ...
    theta_y, _, _ = self._cardan_yxz(rel)

    # Empirical ROM correction
    KNEE_SCALE_FACTOR = 2.3  # Average GT_ROM / MP_ROM
    theta_y_corrected = theta_y * KNEE_SCALE_FACTOR

    return theta_y_corrected
```

**Ankle correction**:
```python
def _calculate_ankle_angle(self, landmarks, side='left'):
    # ... existing calculation ...
    theta_y, _, _ = self._cardan_yzx(rel)

    # Empirical ROM correction
    ANKLE_SCALE_FACTOR = 0.35  # Average GT_ROM / MP_ROM
    theta_y_corrected = theta_y * ANKLE_SCALE_FACTOR

    return -theta_y_corrected
```

#### Validation (0.5 days)

1. Calculate optimal scale factors from training data
2. Apply to full cohort
3. Re-run Deming + DTW
4. Check ICC

**Total time**: 1.5 days
**Success probability**: 50-60% (may not address waveform shape issues)
**Expected ICC**: Knee 0.40-0.50, Ankle 0.30-0.40

**Advantages**:
- ✅ Fast implementation
- ✅ Immediate results
- ✅ Can publish quickly

**Disadvantages**:
- ❌ Not theoretically sound
- ❌ May not generalize to new subjects
- ❌ Reviewers may question methodology

---

### Option C: Hybrid Approach (Recommended)

**Approach**: Combine theoretical validation with empirical tuning.

#### Phase 1: Quick Empirical Fix (1 day)

1. Calculate subject-specific scale factors
2. Identify patterns (e.g., which subjects need more/less correction)
3. Apply average correction to get baseline ICC

**Goal**: Get ICC to 0.40-0.50 range as starting point.

#### Phase 2: Targeted Theoretical Fix (3-4 days)

1. Focus on subjects with worst scaling issues
2. Identify common failure modes
3. Fix specific coordinate frame issues
4. Validate improvements

**Goal**: Improve ICC from 0.40-0.50 to 0.60+ range.

#### Phase 3: Refinement (1-2 days)

1. Optimize parameters
2. Cross-validate
3. Document methodology

**Total time**: 5-7 days
**Success probability**: 80-90%
**Expected ICC**: Knee 0.60-0.70, Ankle 0.50-0.65

---

## Detailed Action Plan (Option C - Recommended)

### Week 1: Quick Wins

#### Day 1-2: Empirical Scale Factor Analysis

**Morning (Day 1)**:
```python
# Calculate subject-specific scale factors
for subject in subjects:
    mp_rom = calculate_rom(mp_data[subject])
    gt_rom = calculate_rom(gt_data[subject])
    scale_factor = gt_rom / mp_rom

# Analyze distribution
knee_factors = [...]  # e.g., [2.1, 2.3, 2.5, 1.9, ...]
ankle_factors = [...] # e.g., [0.3, 0.35, 0.4, 0.28, ...]

# Compute statistics
mean_knee_scale = np.mean(knee_factors)
std_knee_scale = np.std(knee_factors)
```

**Afternoon (Day 1)**:
- Apply average scale factors
- Regenerate MP data
- Run Deming + DTW
- **Target**: Knee ICC > 0.40, Ankle ICC > 0.30

**Day 2**:
- Analyze per-subject results
- Identify outliers (subjects where scale factor doesn't help)
- Look for patterns (e.g., certain subjects consistently need different factors)

**Deliverable**: Baseline ICC improvement + list of problematic subjects

---

#### Day 3-5: Targeted Coordinate Frame Fixes

**Focus areas**:

**Knee**:
1. Check femur frame Y-axis direction
   - Current: knee → hip (proximal)
   - Vicon: Should point along femur shaft

2. Check tibia frame Y-axis direction
   - Current: knee → ankle (distal)
   - Vicon: Should point along tibia shaft

3. Validate relative rotation calculation
   ```python
   rel = R_femur^T × R_tibia
   ```

4. Check for missing Gram-Schmidt orthogonalization

**Ankle**:
1. Re-examine foot frame scaling
   - Current: Uses heel-ankle distance for vertical
   - Issue: May amplify noise in MediaPipe landmarks

2. Check tibia frame consistency
   - Same frame used for knee and ankle
   - Errors propagate from knee

3. Validate YZX extraction against test cases
   ```python
   # Test with known rotation matrix
   R_test = rotation_matrix_from_euler([10, 20, 5], 'YZX')
   angles = _cardan_yzx(R_test)
   assert angles == (10, 20, 5)  # Should recover exact values
   ```

**Deliverable**: Fixed coordinate frames for 2-3 test subjects, validated against GT

---

#### Day 6-7: Full Cohort Validation

**Morning**:
- Apply fixes to all 17 subjects
- Regenerate MP data
- Compute representative cycles

**Afternoon**:
- Run Deming + DTW calibration
- Calculate ICC for knee and ankle
- Run LOSO cross-validation

**Evening**:
- Generate comprehensive report
- Compare before/after waveforms
- Document methodology

**Deliverable**: Final ICC results, validation report

---

### Week 2: Refinement & Documentation

#### Day 8-9: Parameter Optimization

- Fine-tune any remaining scale factors
- Optimize Butterworth filter cutoffs
- Test sensitivity to landmark noise

#### Day 10: Cross-validation

- LOSO CV for knee and ankle
- Bland-Altman analysis
- Check generalization

#### Day 11-12: Documentation & Manuscript

- Update methods section
- Generate figures (waveform comparisons, Bland-Altman plots)
- Write results section

**Final deliverable**: Complete 3-joint validation paper

---

## Success Criteria

### Minimum Acceptable Results (MVP)

| Joint | ICC | Interpretation |
|-------|-----|----------------|
| Hip | 0.813 | Excellent (already achieved) ✅ |
| Knee | **≥0.50** | Moderate (acceptable) |
| Ankle | **≥0.40** | Fair (acceptable) |

**Justification**:
- Knee/Ankle are more challenging than hip in markerless systems
- Literature shows lower ICC for these joints
- Fair-to-moderate agreement still clinically useful

### Target Results (Ideal)

| Joint | ICC | Interpretation |
|-------|-----|----------------|
| Hip | 0.813 | Excellent ✅ |
| Knee | **≥0.60** | Good |
| Ankle | **≥0.50** | Moderate |

**This would be state-of-the-art for markerless 3-joint validation.**

---

## Risk Mitigation

### Risk 1: Scale factors don't improve ICC

**Probability**: 30%

**Mitigation**:
- Fall back to hip-only publication
- Document knee/ankle as "ongoing research"
- Include preliminary results in supplementary materials

### Risk 2: Coordinate frame fixes introduce new issues

**Probability**: 20%

**Mitigation**:
- Maintain version control (git)
- Keep backup of working hip implementation
- Test incrementally (1 subject at a time)

### Risk 3: Time overrun (>2 weeks)

**Probability**: 40%

**Mitigation**:
- Set hard deadline: 2 weeks max
- If not solved by then, publish hip-only
- Continue knee/ankle as separate follow-up study

---

## Resource Requirements

### Time Commitment

**Option A (Rigorous)**: 7-10 days full-time
**Option B (Quick fix)**: 1.5 days
**Option C (Hybrid)**: 5-7 days

### Technical Requirements

- ✅ Already have all data (17 subjects, GT data)
- ✅ Already have infrastructure (Deming, DTW, ICC)
- ✅ Already have partial fixes (ankle YZX)

**Additional needs**:
- Access to Vicon PiG documentation (already have)
- Debugging time (main constraint)
- Possibly: Biomechanics expertise (optional)

---

## Decision Matrix

| Criterion | Option A (Rigorous) | Option B (Quick) | Option C (Hybrid) |
|-----------|---------------------|------------------|-------------------|
| Time | 7-10 days | 1.5 days | 5-7 days |
| Success probability | 70-80% | 50-60% | 80-90% |
| Expected ICC (Knee) | 0.60-0.70 | 0.40-0.50 | 0.60-0.70 |
| Expected ICC (Ankle) | 0.50-0.60 | 0.30-0.40 | 0.50-0.65 |
| Scientific rigor | ✅✅✅ | ⚠️ | ✅✅ |
| Publication quality | Excellent | Fair | Good to Excellent |
| Generalization | Excellent | Poor | Good |

**Recommendation**: **Option C (Hybrid)**

---

## Immediate Next Steps (If Proceeding)

### Tomorrow Morning

1. **Calculate scale factors** for all 17 subjects
   ```bash
   python calculate_scale_factors.py
   ```

2. **Apply average corrections** and regenerate data
   ```bash
   python apply_scale_correction.py --knee-factor 2.3 --ankle-factor 0.35
   python generate_mediapipe_cycle_dataset.py --subjects 1 2 3 ... --workers 4
   ```

3. **Quick validation**
   ```bash
   python knee_deming_calibration.py
   python ankle_deming_calibration.py
   ```

**Expected time**: 3-4 hours
**Expected outcome**: Baseline ICC improvement

### Tomorrow Afternoon

4. **Analyze problematic subjects**
   - Which subjects have worst scale factor deviation?
   - Any patterns (e.g., age, height, gait speed)?

5. **Review coordinate frame code** for top 3 worst subjects
   - Step through calculation manually
   - Compare intermediate values with GT

6. **Identify specific issues** (e.g., axis inversion, missing normalization)

**Expected time**: 4-5 hours
**Expected outcome**: List of specific fixes needed

---

## Conclusion

**Recommended approach**: **Option C (Hybrid)**

**Timeline**: 5-7 days
**Expected results**: Knee ICC 0.60-0.70, Ankle ICC 0.50-0.65
**Risk**: Medium (but acceptable given fallback to hip-only publication)

**Go/No-Go decision point**: After Day 2
- If ICC improves to >0.40 (knee) and >0.30 (ankle) → Continue
- If no improvement → Stop, publish hip-only

**Final deadline**: 2 weeks maximum
- If not solved by then → Publish hip-only, document lessons learned

---

**Next action**: Create `calculate_scale_factors.py` script and begin Day 1 analysis.

**Approval needed**: Should we proceed with Option C?
