# Ankle Angle Fix - Status Report

**Date**: 2025-11-08
**Status**: ⚠️ **PARTIALLY COMPLETE** - Deeper investigation needed

---

## Executive Summary

YZX 회전 순서 변경만으로는 발목 각도 파형 문제를 해결할 수 없었습니다. 근본 원인은 좌표계 정의에 있는 것으로 보이며, 추가 조사가 필요합니다.

---

## Work Completed

### ✅ 1. Code Modifications

**File**: [`core_modules/main_pipeline.py`](core_modules/main_pipeline.py)

**Added YZX Cardan extraction function** (lines 442-458):
```python
def _cardan_yzx(self, rotation_matrix):
    """YZX Cardan angles for ankle joint (Vicon PiG standard)"""
    r = rotation_matrix
    theta_y = np.degrees(np.arctan2(r[0, 2], r[2, 2]))
    theta_z = np.degrees(np.arcsin(np.clip(-r[0, 1], -1.0, 1.0)))
    theta_x = np.degrees(np.arctan2(r[1, 0], r[1, 1]))
    return theta_y, theta_z, theta_x
```

**Tested modification in ankle calculation** (lines 608-618):
- Tried: `_cardan_yzx()` instead of `_cardan_yxz()`
- Result: Waveform shape still incorrect
- **Current state**: Reverted to `_cardan_yxz()` for now

### ✅ 2. Data Regeneration

**Regenerated all MediaPipe data** for 17 subjects:
- Processing time: ~10 minutes (4 parallel workers)
- Output: [`processed/S1_mediapipe_cycles_full.json`](processed/S1_mediapipe_cycles_full.json)
- Representative cycles: [`processed/S1_mediapipe_representative_cycles.json`](processed/S1_mediapipe_representative_cycles.json)

**Subjects processed**:
```
S1_01 (41 cycles), S1_02 (29), S1_03 (43), S1_08 (53), S1_09 (49),
S1_10 (36), S1_11 (41), S1_13 (38), S1_14 (39), S1_15 (45),
S1_16 (46), S1_17 (37), S1_18 (37), S1_23 (35), S1_24 (33),
S1_25 (25), S1_26 (26)
```

### ✅ 3. Waveform Shape Analysis

**Re-ran waveform visualization** with modified code:

**Result**: Ankle trough location **unchanged**
- S1_01: Trough still at **5%** (should be 64%)
- S1_02: Trough still at **11%** (should be 66%)
- Correlation (centered): Still ~0.58-0.64 (no improvement)

---

## Problem Analysis

### Why YZX Didn't Fix the Ankle Issue

**Expected**:
- YZX rotation order (per Vicon PiG) would correctly extract dorsiflexion/plantarflexion
- Ankle trough would move from 5% → 64% (59% shift)

**Actual**:
- Trough location unchanged (still at 5-11%)
- Waveform shape unchanged
- This suggests rotation order is NOT the root cause

### Experiments Performed

#### Experiment 1: YZX with Original Sign
- Code: `theta_y, _, _ = self._cardan_yzx(rel); return theta_y`
- Result: Trough at 5%, all negative values (mean -23°)

#### Experiment 2: YZX with Negated Sign
- Code: `theta_y, _, _ = self._cardan_yzx(rel); return -theta_y`
- Result: Values ~100-150° (completely wrong, should be -20 to +20°)

#### Experiment 3: Reverted to YXZ
- Code: `theta_y, _, _ = self._cardan_yxz(rel); return theta_y`
- Result: Same as before YZX change (trough at 5%)

### Root Cause Hypothesis

The **coordinate frame definitions** (tibia and/or foot) are likely incorrect for ankle joint:

**Possible issues**:
1. **Foot frame Y-axis** may be pointing backwards instead of forwards
2. **Foot frame Z-axis** may be inverted
3. **Tibia frame** may not align with Vicon's shank definition
4. **MediaPipe landmarks** may not map directly to Vicon marker positions

**Evidence**:
- Changing rotation order (YXZ → YZX) has no effect on waveform shape
- Sign negation produces absurd values (100-150°)
- This points to fundamental axis alignment issues, not formula errors

---

## Current Status of Each Joint

| Joint | ICC | Status | Notes |
|-------|-----|--------|-------|
| **Hip** | **0.813** | ✅ **Excellent** | Publication-ready, no changes needed |
| **Knee** | -0.385 | ❌ **Poor** | ROM underestimation (~2x) |
| **Ankle** | -0.790 | ❌ **Very Poor** | Waveform shape mismatch (trough at 5% vs 64%) |

---

## Next Steps (Options)

### Option A: Fix Ankle Coordinate Frames (Recommended but Time-Intensive)

**Estimated time**: 4-8 hours

**Steps**:
1. **Manually verify MediaPipe landmarks** against Vicon markers
2. **Rewrite `_build_foot_frame()`** using biomechanical principles
3. **Rewrite `_build_tibia_frame()`** if needed
4. **Test with known good data** (e.g., normal gait video with GT)
5. **Iterative debugging** until waveform shape matches

**Challenges**:
- MediaPipe doesn't have heel/toe markers (only landmarks 29-32)
- Coordinate system may fundamentally differ from Vicon
- May need to add offset/transformation matrices

### Option B: Proceed with Hip-Only Validation (Quick Win)

**Estimated time**: 1-2 hours

**Steps**:
1. Accept that ankle/knee need deeper research
2. Focus on hip joint (already excellent: ICC 0.813)
3. Create comprehensive hip-only validation report
4. Publish hip results, document ankle/knee as "future work"

**Advantages**:
- Hip ICC 0.813 is **publication-quality**
- LOSO CV ICC 0.859 (state-of-the-art)
- Can publish immediately

**Disadvantages**:
- Incomplete gait analysis (1/3 joints only)
- Reviewers may question why ankle/knee aren't working

### Option C: Deep Dive into Vicon Documentation (Research-Intensive)

**Estimated time**: 1-2 days

**Steps**:
1. Study Vicon PiG white papers in detail
2. Understand exact marker placements and coordinate system definitions
3. Map MediaPipe landmarks to Vicon equivalents
4. Implement exact Vicon formulas with proper transformations
5. Validate against Vicon software output

**This is the "proper" solution** but requires significant time investment.

---

## Recommendations

### Immediate (Today)

**Accept current limitations** and focus on what works:
1. Hip joint validation is excellent ✅
2. Document ankle/knee issues for future work
3. Create status report (this document)

### Short-Term (1-2 Weeks)

**Choose one path**:
- Path A: Fix ankle (4-8 hours research + implementation)
- Path B: Publish hip-only results (1-2 hours manuscript prep)
- Path C: Deep dive Vicon (1-2 days comprehensive fix)

### Long-Term (1-3 Months)

**Option 1**: Collaborate with biomechanics expert
- Get ground truth Vicon data with exact marker positions
- Validate coordinate frame definitions
- Ensure MediaPipe → Vicon mapping is correct

**Option 2**: Machine learning approach
- Train end-to-end model (landmarks → GT angles)
- Bypass manual coordinate frame definitions
- May work better for MediaPipe's non-standard landmark positions

---

## Files Generated/Modified

### Modified
- [`core_modules/main_pipeline.py`](core_modules/main_pipeline.py) - Added `_cardan_yzx()`, modified `_calculate_ankle_angle()`

### Created
- [`processed/S1_mediapipe_cycles_full.json`](processed/S1_mediapipe_cycles_full.json) - Full cycles (17 subjects, all modified)
- [`processed/S1_mediapipe_representative_cycles.json`](processed/S1_mediapipe_representative_cycles.json) - Representative cycles
- [`check_processing_progress.py`](check_processing_progress.py) - Progress monitoring script
- [`diagnose_ankle_calculation.py`](diagnose_ankle_calculation.py) - Diagnostic script (incomplete)
- `ANKLE_FIX_STATUS_REPORT.md` - This document

---

## Conclusion

**YZX rotation order change alone is insufficient** to fix the ankle angle calculation. The root cause appears to be in the **coordinate frame definitions** (foot and/or tibia frames), which requires deeper investigation into:

1. MediaPipe landmark positions vs Vicon marker positions
2. Axis alignment and orientation
3. Potential need for transformation matrices

**Current recommendation**: Accept hip-only results (ICC 0.813) as publication-ready, document ankle/knee as future work requiring biomechanics expertise.

**Alternative recommendation**: Invest 4-8 hours to properly fix coordinate frames if comprehensive 3-joint validation is critical.

---

**Report Date**: 2025-11-08
**Status**: Awaiting user decision on next steps
