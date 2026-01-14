# Deming Regression Data Mismatch - Investigation Report

## Executive Summary

**Status**: 🚨 **CRITICAL BLOCKER IDENTIFIED AND DIAGNOSED**

Codex's Phase 1 Day 1-3 Deming regression implementation (`improve_calibration_deming.py`) produced abnormally small slopes (0.013-0.062 instead of expected 0.8-1.2) due to **data format mismatch** between MediaPipe and Ground Truth sources.

**Answer to user's question "제대로 일한 거 맞음?" (Did it work properly?)**: **NO** - the implementation has critical data loading bugs that invalidate all calibration results.

---

## Problem Identification

### Abnormal Deming Regression Results

```json
{
  "ankle_dorsi_plantarflexion": {
    "slope": 0.051,  // ❌ Should be ~1.0
    "intercept": 1.75,
    "equivalence_slope": false,  // ❌ FAILED
    "equivalence_intercept": false  // ❌ FAILED
  },
  "knee_flexion_extension": {
    "slope": 0.062,  // ❌ Should be ~1.0
    "intercept": -20.98
  },
  "hip_flexion_extension": {
    "slope": 0.014,  // ❌ Should be ~1.0 (70x too small!)
    "intercept": 0.29
  }
}
```

**Diagnosis**: All slopes < 0.1 indicates severe data scale mismatch, NOT a calibration issue.

---

## Root Cause Analysis

### 1. Wrong CSV Column Used for GT Data

**File**: `improve_calibration_deming.py:165`

```python
# CURRENT (WRONG):
subject_curves[joint] = subset["Z"].to_numpy(dtype=float)

# SHOULD BE:
subject_curves[joint] = subset["X"].to_numpy(dtype=float)
```

**Evidence**: `extract_gt_normal_reference.py:190` correctly uses column 'X':
```python
waveform = df_var['X'].values  # ✅ Correct reference implementation
```

### 2. Data Format Mismatch: Absolute vs Centered Angles

#### MediaPipe Data (Absolute Angles)
```
ANKLE: -29.6° to -11.8° (mean = -23.0°)
HIP: 48.8° to 138.7° (mean = 91.0°)
KNEE: 17.8° to 48.6° (mean = 30.5°)
✅ Matches gait biomechanics literature ranges
```

#### GT Data Column X (Mean-Centered Deviations)
```
ANKLE: -5.5° to 5.3° (mean = -2.0°)
HIP: -9.8° to 4.9° (mean = -1.0°)
KNEE: -1.1° to 4.2° (mean = 1.7°)
❌ NOT absolute angles - these are deviations from population mean
```

#### GT Data Column Z (Unknown Normalization)
```
ANKLE: 5.0° to 23.2° (mean = 9.0°)
HIP: -8.0° to 4.4° (mean = -0.7°)
KNEE: -27.6° to -16.2° (mean = -21.1°)
❌ Different scaling - possibly Z-scores or other normalization
```

### 3. Missing Mean-Centering in Preprocessing

Current implementation stacks raw MP (absolute) vs GT (centered) directly:
```python
# Line 195 in improve_calibration_deming.py
return np.concatenate(mp_values), np.concatenate(gt_values), len(subjects_used)
# ❌ No centering applied!
```

This creates artificial scale mismatch where:
- GT variability: ~10° range (centered)
- MP variability: ~90° range (hip, absolute)
- Ratio: 0.01-0.06 ← **This explains the abnormal slopes!**

---

## Required Fixes

### Fix #1: Change GT Column from 'Z' to 'X'

**File**: `improve_calibration_deming.py`
**Line**: 165

```python
# Before:
subject_curves[joint] = subset["Z"].to_numpy(dtype=float)

# After:
subject_curves[joint] = subset["X"].to_numpy(dtype=float)
```

### Fix #2: Center Both MP and GT Data Before Regression

**File**: `improve_calibration_deming.py`
**Function**: `stack_joint_arrays` (lines 171-195)

```python
def stack_joint_arrays(
    mp_results: List[Dict], gt_curves: Dict[int, Dict[str, np.ndarray]], joint: str
) -> Tuple[np.ndarray, np.ndarray, int]:
    """Stack joint angle arrays from all subjects with per-curve centering."""
    mp_values: List[np.ndarray] = []
    gt_values: List[np.ndarray] = []
    subjects_used: set[int] = set()

    for entry in mp_results:
        sid = int(entry["subject_id"])
        gt_subject = gt_curves.get(sid, {})
        if joint not in gt_subject:
            continue
        joint_payload = entry.get("joints", {}).get(joint)
        if not joint_payload:
            continue

        mp_curve = np.asarray(joint_payload["mean_curve"], dtype=float)
        gt_curve = gt_subject[joint]
        length = min(len(mp_curve), len(gt_curve))
        if length == 0:
            continue

        mp_curve = mp_curve[:length]
        gt_curve = gt_curve[:length]

        # 🔧 NEW: Center each curve to zero mean
        mp_curve_centered = mp_curve - mp_curve.mean()
        gt_curve_centered = gt_curve - gt_curve.mean()

        mp_values.append(mp_curve_centered)
        gt_values.append(gt_curve_centered)
        subjects_used.add(sid)

    if not mp_values:
        raise ValueError(f"No overlapping GT/MP data for joint '{joint}'.")

    return np.concatenate(mp_values), np.concatenate(gt_values), len(subjects_used)
```

**Rationale**:
- Deming regression on centered data calibrates **amplitude/scale differences** (slope)
- Intercept should be ~0 since both datasets have mean=0
- This is the standard approach for waveform comparison (same as ICC centered RMSE)

### Fix #3: Document Calibration Scope

Add to `calibration_parameters_deming.json`:

```json
{
  "calibration_metadata": {
    "data_format": "mean_centered_per_cycle",
    "mp_source": "absolute_angles_then_centered",
    "gt_source": "traditional_condition_column_X_already_centered",
    "calibration_scope": "amplitude_scale_only",
    "note": "Calibration applies to CENTERED waveforms. For absolute angle predictions, add back subject-specific or population mean."
  }
}
```

---

## Expected Outcomes After Fixes

### Deming Regression Parameters
```
ANKLE:
  Slope: 0.8-1.2 (currently 0.051)
  Intercept: -0.5° to +0.5° (currently 1.75°)
  Equivalence tests: PASS ✅

KNEE:
  Slope: 0.8-1.2 (currently 0.062)
  Intercept: -0.5° to +0.5° (currently -20.98°)
  Equivalence tests: PASS ✅

HIP:
  Slope: 0.8-1.2 (currently 0.014)
  Intercept: -0.5° to +0.5° (currently 0.29°)
  Equivalence tests: PASS ✅
```

### ICC Improvement Trajectory
```
Current (broken calibration): ICC ~0.35
After fixes: ICC ~0.50-0.55
After Phase 1 complete (filtering + averaging): ICC ~0.50-0.60
Phase 2 target (DTW + Bland-Altman): ICC ~0.65-0.70
Phase 3 target (optimization): ICC ~0.73+  ← CRITICAL MILESTONE
Phase 4 target (quality stratification): ICC ~0.80+
```

---

## Implementation Priority

### IMMEDIATE (Day 1 - BLOCKER)
1. ✅ **Diagnosis complete** - documented in this report
2. 🔧 **Apply Fix #1**: Change column Z → X (1 line change)
3. 🔧 **Apply Fix #2**: Add per-curve centering (10 lines)
4. ✅ **Verify**: Re-run `improve_calibration_deming.py` and check slopes are 0.8-1.2

### SHORT-TERM (Day 2-3)
5. Generate new `calibration_parameters_deming.json` with corrected values
6. Update `ICC_IMPROVEMENT_STRATEGY.md` with actual baseline ICC post-fix
7. Create validation report comparing fixed vs broken calibration

### MEDIUM-TERM (Week 2)
8. Continue Phase 1 Day 4-5: Butterworth filtering + multi-cycle averaging
9. Phase 1 validation gate: Go/No-Go decision based on ICC improvement

---

## Lessons Learned

### Code Review Recommendations
1. **Consistency checks**: When multiple scripts use same data source, verify column names match
2. **Data range validation**: Add assertions checking expected angle ranges (e.g., hip 0-140°)
3. **Unit tests**: Mock data with known slope=1.0, verify Deming recovers it
4. **Diagnostic logging**: Print data ranges before regression for sanity checks

### Documentation Gaps
1. `traditional_condition.csv` column definitions not documented
2. No specification of whether angles are absolute vs. centered
3. Missing data flow diagram showing MP → GT comparison path

---

## Validation Checklist

Before proceeding to Phase 1 Day 4-5, verify:

- [ ] `calibration_parameters_deming.json` ankle slope is 0.8-1.2
- [ ] `calibration_parameters_deming.json` hip slope is 0.8-1.2
- [ ] `calibration_parameters_deming.json` knee slope is 0.8-1.2
- [ ] All equivalence_slope tests are `true`
- [ ] All equivalence_intercept tests are `true`
- [ ] Calibrated RMSE improved vs. uncalibrated baseline
- [ ] Calibrated correlation improved vs. uncalibrated baseline

**Go/No-Go Decision**: Only proceed to filtering (Day 4-5) if ALL checks above pass.

---

## References

- **Bug discovery**: `/data/gait/diagnose_deming_data_mismatch.py`
- **Diagnostic run output**: See above diagnostic script output
- **Correct reference implementation**: `/data/gait/extract_gt_normal_reference.py:190`
- **Buggy implementation**: `/data/gait/improve_calibration_deming.py:165`
- **Research basis**: EVIDENCE_BASED_ICC_IMPROVEMENT_STRATEGY.md lines 133-599

---

**Report Generated**: 2025-11-07
**Status**: CRITICAL BLOCKER - Fixes required before Phase 1 continuation
