# CRITICAL DISCOVERY: The 76.6% Accuracy Was Fake

**Date**: 2025-10-30
**Severity**: 🚨 **CRITICAL** 🚨
**Impact**: Entire baseline (76.6%) must be invalidated
**Discovery**: stage1_v2's 76.6% accuracy was detecting **NaN presence**, not **gait pathology**

---

## Executive Summary

The widely cited **76.6% accuracy** from STAGE 1 v2 is **completely invalid**.

**What we thought**:
- v2 detects pathological gait using cadence, variability, irregularity
- 76.6% accuracy proves "Less is More" (3 features > 6 features)
- This is the best baseline to beat

**What actually happened**:
- 84.9% of normal patterns had NaN (MediaPipe detection failures)
- Baseline was corrupted: calculated from 90 zero-feature patterns + 16 valid patterns
- Classifier learned to detect **which patterns have NaN**, not **which gaits are pathological**
- Accuracy = 76.6% because of **NaN distribution asymmetry** (84.9% normal vs 33.0% pathological)

**True baseline with clean data**: **56.1% accuracy**

---

## The Accidental Success Mechanism

### Step 1: NaN Distribution Asymmetry

**Original data** (gavd_real_patterns.json):
```
Total: 197 patterns (106 normal, 91 pathological)

NaN prevalence:
- Normal: 90/106 (84.9%) with NaN
- Pathological: 30/91 (33.0%) with NaN
```

**Key asymmetry**: Normal class has 2.6× more NaN patterns than pathological!

### Step 2: NaN Patterns Produce Zero Features

When MediaPipe fails to detect heels (NaN values):

```python
heel_left = [1.2, 1.5, NaN, 1.8, 2.0]  # Contains NaN

# Feature extraction:
np.mean(heel_left) = NaN  # Mean with NaN returns NaN
find_peaks(heel_left, height=NaN) = []  # Empty peaks!
n_steps = 0
cadence = 0 steps/min  ← WRONG

variability = 0  # No peaks → no variability
irregularity = 0  # No peaks → no irregularity
```

**Result**: 90 normal patterns → features = (0, 0, 0)

### Step 3: Corrupted Baseline

**Normal patterns**:
- 90 with NaN: cadence = 0, variability = 0, irregularity = 0
- 16 without NaN: cadence = 225.5, variability = 0.101, irregularity = 0.477

**After 3-sigma outlier removal**:
```
Baseline (n=103-105, dominated by 90 zeros):
  Cadence: 25.2 ± 68.5 steps/min  ← Should be ~220!
  Variability: 0.010 ± 0.027      ← Should be ~0.10!
  Irregularity: 0.044 ± 0.127     ← Should be ~0.48!
```

**All three features corrupted to near-zero!**

### Step 4: Accidental Classification

With baseline ≈ 0, the Z-score classifier does:

```
For pattern with features = (0, 0, 0):  # Has NaN
  Z = |0 - 0| / σ ≈ 0  → Classified as NORMAL

For pattern with features = (220, 0.10, 0.48):  # No NaN
  Z = |220 - 25| / 68.5 ≈ 2.8  → Classified as PATHOLOGICAL
```

**The classifier learned: NaN → normal, No NaN → pathological**

### Step 5: Accuracy from NaN Asymmetry

**Confusion matrix**:

| True Class | NaN Status | Count | Predicted | Correct? |
|------------|------------|-------|-----------|----------|
| Normal | Has NaN | 90 | Normal | ✅ 90 |
| Normal | No NaN | 16 | Pathological | ❌ 0 |
| Pathological | Has NaN | 30 | Normal | ❌ 0 |
| Pathological | No NaN | 61 | Pathological | ✅ 61 |

**Accuracy = (90 + 61) / 197 = 76.6%**

**This matches exactly!**

---

## Evidence

### Evidence 1: NaN Feature Extraction

**Test code**:
```python
import numpy as np
from scipy.signal import find_peaks

heel_with_nan = [1.2, 1.5, np.nan, 1.8, 2.0]
mean_val = np.mean(heel_with_nan)
peaks, _ = find_peaks(heel_with_nan, height=mean_val)

print(f"Mean: {mean_val}")        # NaN
print(f"Peaks: {peaks}")          # []
print(f"Cadence: {len(peaks)}")   # 0
```

**Output**: Mean = NaN, Peaks = [], Cadence = 0

### Evidence 2: Baseline Statistics

**From stage1_v2_correct_features.py output**:
```
Building baseline from 106 normal patterns...

Baseline Statistics:
  Cadence: 25.2 ± 68.5 steps/min
  Variability: 0.010 ± 0.027
  Irregularity: 0.044 ± 0.127
```

**Analysis**:
- Normal cadence should be ~220 steps/min
- Baseline of 25.2 is **91% too low**
- This can only happen if dominated by zeros

### Evidence 3: Feature Distribution by NaN Status

**Extracted from 106 normal patterns**:

```
WITH NaN (90 patterns):
  Cadence: 0.0 ± 0.0
  Variability: 0.0000 ± 0.0000
  Irregularity: 0.0000 ± 0.0000

WITHOUT NaN (16 patterns):
  Cadence: 225.5 ± 71.5
  Variability: 0.1009 ± 0.1307
  Irregularity: 0.4765 ± 0.2123
```

**Proof**: 90 patterns have exactly zero for all features!

### Evidence 4: Predicted vs Actual Accuracy

**Prediction based on NaN distribution**:
```
Correct classifications:
  Normal with NaN: 90 (classified as normal)
  Pathological without NaN: 61 (classified as pathological)

Accuracy = (90 + 61) / 197 = 76.6%
```

**Actual result**: 76.6% (threshold=1.5)

**Perfect match!**

### Evidence 5: Fixed Data Performance

**When NaN is interpolated** (gavd_real_patterns_fixed.json):
```
Total: 187 patterns (101 normal, 86 pathological)
All patterns: 0 with NaN (100% valid)

Baseline:
  Cadence: 218.8 ± 74.0 steps/min  ← CORRECT
  Variability: 0.103 ± 0.111
  Irregularity: 0.541 ± 0.302

Performance: 56.1% accuracy
```

**56.1% is the TRUE performance on actual gait features!**

---

## Why This Wasn't Caught Earlier

### 1. Baseline Statistics Looked Plausible

The reported baseline (Cadence: 25.2 ± 68.5) was printed but never questioned because:
- We focused on accuracy metrics, not baseline validity
- Large std (68.5) masked the absurdly low mean (25.2)
- No comparison with literature values (~220 steps/min)

### 2. Z-score Separation Looked Good

Results showed:
```
Normal Z-score: 0.80 ± 1.24
Pathological Z-score: 2.83 ± 1.96
```

This separation (2.83 vs 0.80) seemed to validate the method. We didn't realize it was separating **NaN presence**, not **gait pathology**.

### 3. "Less is More" Finding Distracted Us

The comparison showing 3 features (76.6%) > 6 features (58.8%) was so interesting that we focused on feature selection rather than data quality.

### 4. NaN Investigation Was Incomplete

The NAN_INVESTIGATION_FINAL_REPORT.md correctly identified:
- 59% of patterns have NaN
- 84.9% of normal patterns have NaN
- Interpolation recovers 95.2%

But it **didn't check if stage1_v2 was affected**. We assumed the original v2 used clean data.

### 5. Confirmation Bias

We wanted to believe 76.6% was achievable with simple features, so we didn't scrutinize the mechanism deeply enough.

---

## Implications

### 1. All Previous Conclusions Are Invalid

❌ **INVALIDATED**:
- "STAGE 1 v2 achieves 76.6% accuracy"
- "3 features (76.6%) > 6 features (58.8%)"
- "Deploy STAGE 1 v2 as final system"
- "'Less is More' principle in gait analysis"

### 2. Research Paper Needs Major Revision

The RESEARCH_PAPER.md is based on false premises:
- Abstract claims 76.6% accuracy
- Results section reports 76.6% as main finding
- Discussion interprets "Less is More" based on 76.6%
- Conclusion recommends deployment

**All must be revised with true baseline: 56.1%**

### 3. True Baseline Is Much Lower

**Correct comparison** (all with fixed data):

| Method | Accuracy | Notes |
|--------|----------|-------|
| STAGE 1 v2 (3 features) | **56.1%** | Equal-weighted Z-score |
| STAGE 1 v3 (6 features) | 58.8% | More features slightly better |
| Improved v1 (weighted) | 55.6% | Feature weighting doesn't help |
| Robust v3 (MAD-Z) | 66.8% | Robust statistics help! |

**New best**: Robust v3 at 66.8% (not 76.6%)!

### 4. "Less is More" May Be Wrong

With corrected data:
- 3 features: 56.1%
- 6 features: 58.8%
- **More features = better!** (+2.7%)

The original finding was reversed!

### 5. Improvement Experiments Were Valid

All the "failed" improvements (v1, v2, v3) that showed 50-66% were actually **using correct data**. They only seemed worse because we compared to the invalid 76.6% baseline.

**Robust v3 (66.8%) is actually the best method so far!**

---

## Root Cause Analysis

### Why Did This Happen?

**1. Data Quality Issue**:
- MediaPipe failed on 59% of videos (at least 1 frame)
- No validation that feature extraction handled NaN properly
- Assumption that `find_peaks()` would fail gracefully with NaN (it does, but produces empty peaks)

**2. Code Design Flaw**:
- `_extract_features()` doesn't check for NaN (stage1_v2_correct_features.py line 68-121)
- No validation that cadence/variability/irregularity are non-zero
- Outlier removal removes NaN implicitly but **keeps zero values**

**3. Validation Gap**:
- No sanity check on baseline statistics (25.2 steps/min is clearly wrong)
- No comparison with literature (normal cadence ≈ 220 steps/min)
- No verification that Z-scores correlate with clinical features

**4. Asymmetric NaN Distribution**:
- Normal: 84.9% NaN → mostly zeros
- Pathological: 33.0% NaN → mostly valid
- This 2.6× asymmetry is what caused accidental 76.6%

**If NaN were symmetric (e.g., 50% in both classes)**, accuracy would be ~50% (random).

---

## What Should Have Been Done

### 1. Data Quality Checks

```python
def validate_features(features):
    if features.cadence < 10:
        raise ValueError("Cadence too low - likely NaN in data")
    if features.cadence > 300:
        raise ValueError("Cadence too high - outlier")
    # ... similar for other features
```

### 2. Baseline Sanity Checks

```python
if baseline['cadence_mean'] < 100 or baseline['cadence_mean'] > 300:
    raise ValueError(f"Baseline cadence {baseline['cadence_mean']} is implausible")
```

### 3. NaN Handling in Feature Extraction

```python
def _extract_features(self, pattern: dict) -> GaitFeatures:
    heel_left = np.array(pattern['heel_height_left'])
    heel_right = np.array(pattern['heel_height_right'])

    # Check for NaN
    if np.any(np.isnan(heel_left)) or np.any(np.isnan(heel_right)):
        raise ValueError("Pattern contains NaN - preprocess with interpolation")

    # ... rest of feature extraction
```

### 4. Correlation with Ground Truth

After building baseline, verify:
- Normal patterns have low Z-scores
- Pathological patterns have high Z-scores
- Z-scores correlate with clinical severity

---

## Corrected Timeline

### What Actually Happened

**October 27-29**:
- ✅ Extracted GAVD real patterns → gavd_real_patterns.json (230 patterns, 59% with NaN)
- ✅ Built STAGE 1 v2 with cadence/variability/irregularity
- ❌ **Did NOT realize NaN → zero features**
- ❌ Reported 76.6% accuracy (actually detecting NaN, not gait)
- ❌ Wrote papers/reports based on false 76.6%

**October 30 Morning**:
- 🔍 User asked: "계산 중 nan 값은 왜 있음?"
- ✅ Investigated NaN (59% patterns, 84.9% of normal)
- ✅ Fixed with interpolation → gavd_real_patterns_fixed.json
- ✅ Re-evaluated v3 with fixed data → 58.8%
- ❌ **Did NOT re-evaluate v2 with fixed data yet**

**October 30 Afternoon**:
- 📝 Wrote research paper citing 76.6%
- 📝 Created flowchart explaining "Less is More"
- 💡 Proposed algorithm improvements
- 🔧 Implemented v1 (weighted), v2 (stride), v3 (robust)
- ⚠️ All showed 50-66% → seemed "failed"

**October 30 Evening**:
- 🤔 Questioned why improvements failed
- 🔍 Discovered data inconsistency (original 197 vs fixed 187)
- 🚨 **CRITICAL**: Ran v2 with fixed data → 56.1%!
- 🔬 Deep investigation → discovered entire 76.6% was fake

---

## Action Items

### Immediate (Today)

1. ✅ **Document this discovery** (this file)
2. 🔄 **Update all reports** to reflect 56.1% true baseline
3. 🔄 **Revise research paper** with corrected results
4. 🔄 **Re-evaluate all methods** with gavd_real_patterns_fixed.json

### Short-term (This Week)

5. ⚙️ **Fix stage1_v2_correct_features.py** to reject NaN patterns
6. 📊 **Create fair comparison table** (all methods, same data)
7. 🏆 **Identify true best method** (likely Robust v3 at 66.8%)
8. 📝 **Write corrected paper** with honest results

### Long-term (Research)

9. 🔬 **Investigate why robust v3 works better** (MAD vs std)
10. 🎯 **Develop features with Cohen's d > 0.8** (stride length, trunk sway)
11. 🤖 **Try ML methods** (logistic regression, Random Forest)
12. 📈 **Target 75-80%** accuracy with proper validation

---

## Lessons Learned

### Technical Lessons

1. **Always validate baseline statistics** against domain knowledge
2. **Check for NaN explicitly**, don't rely on implicit removal
3. **Verify classifier learns intended features**, not data artifacts
4. **Sanity check with edge cases** (all zeros, all NaN, etc.)

### Scientific Lessons

5. **Question surprisingly good results** (76.6% was too good to be true)
6. **Reproduce with clean data** before publication
7. **Investigate discrepancies** immediately (why do improvements fail?)
8. **Document data processing** completely (NaN handling, filtering, etc.)

### Process Lessons

9. **Validate assumptions early** (did stage1_v2 handle NaN correctly?)
10. **Compare with literature** (normal cadence is well-known ~220 steps/min)
11. **Cross-check multiple ways** (Z-score separation + raw features + baseline sanity)
12. **Listen to warning signs** (all improvements showing worse results)

---

## Silver Lining

### What We Gained

Despite this setback, we learned:

1. ✅ **Data quality matters**: 95.2% recovery with interpolation
2. ✅ **Robust methods help**: MAD-Z achieves 66.8% vs 56.1% baseline
3. ✅ **NaN distribution matters**: Asymmetry can create spurious accuracy
4. ✅ **Feature engineering works**: Stride length, trunk sway show promise
5. ✅ **Scientific rigor**: Caught this before publication!

### Honest Path Forward

Instead of claiming false 76.6%, we now have:

**Honest baseline**: 56.1% (equal-weight Z-score)
**Best current method**: 66.8% (robust MAD-Z)
**Improvement**: +10.7% from robustness
**Target**: 75-80% with better features

**This is a more credible research story!**

---

## Conclusion

The 76.6% accuracy from STAGE 1 v2 was **completely invalid**.

It detected **NaN presence** (caused by MediaPipe failures), not **gait pathology**.

The accidental success came from:
- 84.9% of normal patterns having NaN → features = 0
- 33.0% of pathological patterns having NaN → features = 0
- Baseline corrupted to ≈0 by 90 zero patterns
- Classifier learned: features=0 → normal, features>0 → pathological
- Accuracy = (90 + 61) / 197 = 76.6%

**True baseline with clean data**: **56.1% accuracy**

All papers, reports, and conclusions citing 76.6% must be revised.

The good news: Robust methods (MAD-Z) achieve **66.8%**, showing real improvement is possible.

---

**Discovery Date**: 2025-10-30
**Discovered By**: Deep investigation of why improvements "failed"
**Status**: CRITICAL - All downstream work affected
**Next Steps**: Revise all documents, re-run all experiments with fixed data

**Key Takeaway**: "If it seems too good to be true, investigate deeply before believing it."

---

END OF REPORT
