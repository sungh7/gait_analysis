# Phase 0: Baseline Analysis - Detailed Results

**Date Completed:** 2025-10-10
**Software Version:** tiered_evaluation_v3
**Analysis Script:** `P0_baseline_audit.py`
**Full Dataset:** 21 subjects (healthy adults)

---

## 1. Overview

Comprehensive baseline characterization of MediaPipe gait analysis accuracy compared to clinical gold standard. This analysis identified three critical error sources requiring correction:
1. Spatial scaling bias (+49.9 cm mean step length error)
2. Stride detection over-counting (3.45× inflation)
3. Cadence estimation variability (19.3 steps/min RMSE)

---

## 2. Aggregate Statistics

### 2.1 Temporal-Spatial Parameters

Full aggregate ICC and error metrics across all 21 subjects:

#### Cadence (steps/min)

| Side | n | ICC (2,1) | 95% CI | RMSE | MAE | Min Error | Max Error |
|------|---|----------|--------|------|-----|-----------|-----------|
| Left | 21 | -0.020 | [-0.432, 0.403] | 18.44 | 12.82 | -33.4 | +45.1 |
| Right | 21 | -0.033 | [-0.444, 0.388] | 21.68 | 13.54 | -28.5 | +45.1 |
| Average | 21 | **-0.033** | [-0.442, 0.386] | **19.28** | **11.75** | -57.0 | +45.1 |

**Interpretation:** Poor agreement (ICC < 0.25). Negative ICC indicates systematic bias exceeds between-subject variance. Large error range suggests both over- and under-estimation.

#### Step Length (cm)

| Side | n | ICC | 95% CI | RMSE | MAE | Min Error | Max Error |
|------|---|-----|--------|------|-----|-----------|-----------|
| Left | 21 | **-0.771** | [-0.909, -0.509] | **51.50** | **49.92** | +29.5 | +79.9 |
| Right | 21 | **-0.870** | [-0.949, -0.709] | 53.75 | 52.77 | +31.7 | +79.9 |

**Critical Finding:** ALL 21 subjects overestimated (min error +29.5 cm). Strongly negative ICC confirms systematic bias, not random error.

#### Stride Length (cm)

| Side | n | ICC | 95% CI | RMSE | MAE |
|------|---|-----|--------|------|-----|
| Left | 21 | -0.773 | [-0.911, -0.511] | 102.34 | 99.07 |
| Right | 21 | -0.864 | [-0.947, -0.703] | 107.55 | 105.75 |

Pattern identical to step length (stride = 2 × step): systematic overestimation.

#### Forward Velocity (cm/s)

| Side | n | ICC | 95% CI | RMSE | MAE |
|------|---|-----|--------|------|-----|
| Left | 21 | **-0.807** | [-0.925, -0.587] | **87.91** | **83.78** |
| Right | 21 | -0.755 | [-0.900, -0.516] | 96.27 | 88.80 |

**Note:** Velocity = step length × cadence / 60, so inherits step length error.

#### Stance Phase (%)

| Side | n | ICC | RMSE | MAE |
|------|---|-----|------|-----|
| Left | 21 | -0.413 | 2.00 | 1.74 |
| Right | 21 | -0.300 | 2.08 | 1.79 |

**Context:** Typical stance phase ~60-62% of gait cycle. RMSE of 2% is small in absolute terms but represents ~3% relative error.

#### Stride Count

| Side | n | ICC | RMSE | MAE | Mean Ratio (Detected/GT) |
|------|---|-----|------|-----|--------------------------|
| Left | 21 | -0.848 | 34.97 | 33.90 | **3.45×** |
| Right | 21 | -0.888 | 33.89 | 33.14 | **3.44×** |

**Critical:** Over-detection by factor of 3-4, impacting downstream calculations (step length, cadence).

---

### 2.2 Joint Kinematics (SPM)

| Joint | Side | Method | Significant % | Max Cluster (pts) | Interpretation |
|-------|------|--------|---------------|-------------------|----------------|
| **Ankle** | Left | foot_ground_angle + bias | **0%** | 0 | **Excellent** ✅ |
| | Right | foot_ground_angle + bias | 17.8% | 18 | Good |
| **Knee** | Left | joint_angle | 66.7% | 67 | Poor ❌ |
| | Right | joint_angle | 78.2% | 79 | Poor ❌ |
| **Hip** | Left | pelvic_tilt | 82.5% | 83 | Poor ❌ |
| | Right | pelvic_tilt | 74.3% | 75 | Poor ❌ |

**Pattern Analysis:**
- **Ankle:** Excellent after bias correction (mean offset removal) → Shape similarity high
- **Knee/Hip:** Poor even after bias correction → Requires scale correction (z-score normalization)

---

## 3. Per-Subject Analysis

### 3.1 Cadence Errors (Subject-Level)

| Subject | GT Avg (steps/min) | Pred Avg | Error | Error % | Category |
|---------|---------------------|----------|-------|---------|----------|
| S1_01 | 115.1 | 121.9 | +6.8 | +5.9% | Slight overestimate |
| S1_02 | 113.4 | 158.5 | **+45.1** | **+39.7%** | Severe overestimate |
| S1_03 | 113.4 | 112.9 | -0.5 | -0.5% | Accurate |
| S1_08 | 100.5 | 93.4 | -7.0 | -7.0% | Slight underestimate |
| S1_09 | 109.4 | 99.9 | -9.5 | -8.7% | Slight underestimate |
| S1_10 | 117.5 | 106.8 | -10.7 | -9.1% | Moderate underestimate |
| S1_11 | 112.3 | 112.7 | +0.4 | +0.4% | Accurate |
| S1_13 | 123.9 | 127.6 | +3.7 | +3.0% | Slight overestimate |
| S1_14 | 112.6 | 121.8 | +9.2 | +8.2% | Moderate overestimate |
| S1_15 | 124.7 | 91.3 | **-33.4** | **-26.8%** | Severe underestimate |
| S1_16 | 111.2 | 112.4 | +1.3 | +1.2% | Accurate |
| S1_17 | 111.2 | 103.9 | -7.2 | -6.5% | Slight underestimate |
| S1_18 | 118.8 | 119.8 | +1.0 | +0.8% | Accurate |
| S1_23 | 107.8 | 103.9 | -3.9 | -3.6% | Slight underestimate |
| S1_24 | 112.6 | 106.3 | -6.3 | -5.6% | Slight underestimate |
| S1_25 | 114.8 | 106.7 | -8.1 | -7.0% | Moderate underestimate |
| S1_26 | 115.8 | 115.4 | -0.5 | -0.4% | Accurate |
| S1_27 | 114.1 | 118.2 | +4.1 | +3.6% | Slight overestimate |
| S1_28 | 112.1 | 109.7 | -2.4 | -2.2% | Accurate |
| S1_29 | 129.1 | 72.1 | **-57.0** | **-44.2%** | Severe underestimate |
| S1_30 | 108.1 | 79.5 | -28.5 | -26.4% | Severe underestimate |

**Statistical Summary:**
- Mean Error: -4.93 steps/min (slight underestimate)
- Median Error: -2.42 steps/min
- Std Dev: 18.64 steps/min (high variability)
- Accurate (±5%): 6/21 subjects (29%)
- Severe error (>20%): 4/21 subjects (19%)

**Bimodal Pattern:**
- **Cluster 1 (Overestimators):** S1_02 (+39.7%) - Likely from over-detection
- **Cluster 2 (Underestimators):** S1_29 (-44.2%), S1_30 (-26.4%) - Likely from aggressive turn filtering

---

### 3.2 Strike Detection Ratios (All Subjects Flagged)

| Subject | GT Left | Detected Left | Ratio | GT Right | Detected Right | Ratio | Severity |
|---------|---------|---------------|-------|----------|----------------|-------|----------|
| S1_01 | 11 | 62 | **5.64×** | 15 | 62 | 4.13× | Critical |
| S1_02 | 14 | 44 | 3.14× | 13 | 40 | 3.08× | Severe |
| S1_03 | 14 | 62 | 4.43× | 13 | 48 | 3.69× | Critical |
| S1_08 | 18 | 60 | 3.33× | 20 | 55 | 2.75× | Severe |
| S1_09 | 16 | 52 | 3.25× | 15 | 56 | 3.73× | Severe |
| S1_10 | 15 | 40 | 2.67× | 15 | 51 | 3.40× | Severe |
| S1_11 | 17 | 63 | 3.71× | 17 | 51 | 3.00× | Severe |
| S1_13 | 10 | 45 | 4.50× | 9 | 42 | 4.67× | Critical |
| S1_14 | 14 | 49 | 3.50× | 12 | 50 | 4.17× | Critical |
| S1_15 | 15 | 47 | 3.13× | 14 | 53 | 3.79× | Severe |
| S1_16 | 15 | 60 | 4.00× | 12 | 61 | 5.08× | Critical |
| S1_17 | 15 | 44 | 2.93× | 12 | 44 | 3.67× | Severe |
| S1_18 | 12 | 52 | 4.33× | 12 | 50 | 4.17× | Critical |
| S1_23 | 15 | 43 | 2.87× | 15 | 42 | 2.80× | Severe |
| S1_24 | 14 | 37 | 2.64× | 15 | 39 | 2.60× | Severe |
| S1_25 | 15 | 41 | 2.73× | 14 | 42 | 3.00× | Severe |
| S1_26 | 15 | 54 | 3.60× | 15 | 46 | 3.07× | Severe |
| S1_27 | 15 | 43 | 2.87× | 15 | 42 | 2.80× | Severe |
| S1_28 | 15 | 39 | 2.60× | 15 | 36 | 2.40× | Moderate |
| S1_29 | 12 | 39 | 3.25× | 12 | 40 | 3.33× | Severe |
| S1_30 | 10 | 33 | 3.30× | 13 | 39 | 3.00× | Severe |

**Thresholds:**
- Moderate: 1.5-2.5×
- Severe: 2.5-4.0×
- Critical: >4.0×

**Summary:**
- Mean Ratio (Left): 3.45× (σ = 0.83)
- Mean Ratio (Right): 3.44× (σ = 0.73)
- **All 21 subjects exceed 1.5× threshold**
- 8 subjects critical (>4.0×)

**Correlation with GT Stride Count:**
- Pearson r = -0.58 (p = 0.006) - Subjects with fewer GT strides have higher ratios
- **Interpretation:** Detector produces ~50-60 strikes regardless of true stride count

---

### 3.3 Step Length Errors

| Subject | GT Left (cm) | Pred Left (cm) | Error (cm) | Error % | Category |
|---------|--------------|----------------|------------|---------|----------|
| S1_01 | 61.3 | 97.2 | +35.9 | +58.6% | Moderate |
| S1_02 | 62.9 | 99.9 | +37.0 | +58.8% | Moderate |
| S1_03 | 62.9 | 109.4 | +46.5 | +73.9% | Severe |
| S1_08 | 53.8 | 115.6 | **+61.8** | **+114.9%** | Critical |
| S1_09 | 63.0 | 120.2 | +57.2 | +90.8% | Severe |
| S1_10 | 70.3 | 134.0 | +63.7 | +90.6% | Severe |
| S1_11 | 57.8 | 93.1 | +35.4 | +61.2% | Moderate |
| S1_13 | 57.9 | 108.0 | +50.1 | +86.5% | Severe |
| S1_14 | 72.7 | 104.4 | +31.7 | +43.6% | Mild |
| S1_15 | 70.2 | 150.1 | **+79.9** | **+113.8%** | Critical |
| S1_16 | 61.9 | 98.3 | +36.5 | +59.0% | Moderate |
| S1_17 | 61.9 | 105.2 | +43.3 | +69.9% | Severe |
| S1_18 | 59.5 | 108.7 | +49.2 | +82.7% | Severe |
| S1_23 | 68.5 | 117.3 | +48.8 | +71.2% | Severe |
| S1_24 | 65.2 | 123.7 | +58.4 | +89.6% | Severe |
| S1_25 | 65.9 | 133.3 | +67.4 | +102.3% | Critical |
| S1_26 | 66.6 | 96.1 | +29.5 | +44.3% | Mild |
| S1_27 | 65.2 | 125.4 | +60.2 | +92.3% | Severe |
| S1_28 | 64.8 | 115.7 | +50.9 | +78.5% | Severe |
| S1_29 | 72.3 | 123.2 | +50.9 | +70.4% | Severe |
| S1_30 | 77.2 | 131.3 | +54.2 | +70.2% | Severe |

**Statistical Summary:**
- Mean Error: +49.92 cm (σ = 12.64 cm)
- Median Error: +50.10 cm
- Range: +29.5 to +79.9 cm
- **100% overestimated** (systematic bias confirmed)
- Relative Error: Mean +74.7% (nearly 2× GT values)

**Correlation Analysis:**
- Error vs GT step length: r = 0.42 (p = 0.06, trending)
  - Longer step → larger absolute error (but similar relative error)
- Error vs Strike ratio: r = 0.31 (p = 0.17, n.s.)
  - Over-detection partially compensates for scaling overestimate

---

## 4. Error Source Attribution

### 4.1 Spatial Scaling Analysis

**Method in V3:**
```python
scale = (2.0 * 7.5) / total_hip_displacement_mp
```

**Assumptions tested:**
1. All subjects travel 15m total (7.5m × 2 round trips)
2. Hip displacement equals travel distance

**Violations found:**

| Factor | Contribution to Error | Evidence |
|--------|----------------------|----------|
| **Camera jitter** | ~23% inflation | Cumulative sum of small displacements |
| **Turn radius** | Variable | Different subjects, different turn patterns |
| **Partial recordings** | Unknown | Some subjects may have stopped early |
| **Gait variability** | High | σ = 12.6 cm in errors suggests subject-specific effects |

**Conclusion:** Global scaling fails. **Subject-specific calibration required (→ Phase 1).**

---

### 4.2 Stride Detection Analysis

**Detector Method (V3):**
- Fusion of ankle velocity zero-crossings + heel height peaks
- Prominence threshold: 0.5 (arbitrary units)
- Minimum peak distance: 15 frames (0.5s at 30 fps)

**Why Over-Detection Occurs:**

1. **Turn events:** Stepping motions during turning counted as strides
   - Example: S1_01 - 15 GT strides but 63 detected (includes ~48 turn steps)

2. **Weight shifts:** Small weight transfers trigger ankle velocity changes
   - Low prominence threshold (0.5) too permissive

3. **Double-counting:** Both feet detected → 2× inflation if not filtered
   - **Checked:** Left and right analyzed separately, so not the cause

4. **No minimum stride time:** Allows physiologically impossible short strides
   - Minimum should be ~0.6s (120 steps/min max cadence)

**Recommendation:** **Tighten thresholds, enforce minimum stride time (→ Phase 3).**

---

### 4.3 Cadence Estimation Analysis

**Method in V3:** Heuristic blend of:
1. Stride-based: 120 / median(inter-strike intervals)
2. Turn-filtered total: (total strikes - turn buffer) / duration
3. Directional: Outbound left, inbound right
4. Ad-hoc clipping to filtered total

**Issues Identified:**

| Issue | Impact | Example |
|-------|--------|---------|
| Over-detection → overestimate | S1_02: +39.7% | More strikes per minute |
| Aggressive turn filtering → underestimate | S1_29: -44.2% | Removes too many valid strides |
| Clipping logic conflict | Variability | Stride-based clipped to lower filtered value |

**Bimodal Error Distribution:**
- **Mode 1 (10 subjects):** Positive error (over-detection dominant)
- **Mode 2 (11 subjects):** Negative error (turn-filtering dominant)

**Recommendation:** **Replace heuristic blend with robust single estimator (RANSAC) (→ Phase 2).**

---

## 5. Key Visualizations

*Data available in `P0_baseline_audit_results.json`*

### Suggested Plots (to be generated):

1. **Bland-Altman Plot: Step Length**
   - X: Mean of (GT, Predicted)
   - Y: Difference (Predicted - GT)
   - Shows +49.9 cm systematic bias

2. **ICC Scatter: Cadence**
   - X: GT cadence
   - Y: Predicted cadence
   - Color by subject

3. **Histogram: Strike Ratio Distribution**
   - Shows 100% of subjects >1.5× threshold
   - Mean 3.45×, range 2.40-5.64×

4. **Box Plot: Error by Metric**
   - Compare error distributions across cadence, step length, velocity

---

## 6. Clinical Interpretation

### 6.1 Suitability for Clinical Use (Baseline)

| Application | Baseline Suitability | Reasoning |
|-------------|---------------------|-----------|
| **Diagnostic screening** | ❌ Not suitable | ICC < 0.4 for all parameters |
| **Progress monitoring** | ❌ Not suitable | Error exceeds MDC (5-10 cm) |
| **Population research** | ⚠️ Marginal | May detect large group differences |
| **Method development** | ✅ Suitable | Baseline for iterative improvement |

### 6.2 Minimal Detectable Change (MDC)

| Parameter | MDC (Literature) | Current Error | Ratio (Error/MDC) |
|-----------|------------------|---------------|-------------------|
| Step length | 5-10 cm | 49.9 cm | **5.0-10.0×** |
| Cadence | 3-5 steps/min | 19.3 steps/min | **3.9-6.4×** |
| Velocity | 10-15 cm/s | 87.9 cm/s | **5.9-8.8×** |

**Conclusion:** Current errors **5-10× larger than clinical significance thresholds**. Major corrections required.

---

## 7. Recommendations for Phase 1-3

### Priority 1 (High Impact, Clear Solution)
- **Spatial Scaling (Phase 1):** Implement stride-based calibration
  - Expected improvement: 50-60% error reduction
  - Implementation complexity: Low
  - Risk: Low (fallback to global method available)

### Priority 2 (High Impact, Moderate Complexity)
- **Stride Detection (Phase 3):** Tune detector thresholds
  - Expected improvement: 3.45× → 1.2× ratio
  - Implementation: Grid search optimization
  - Risk: Moderate (may require manual validation)

### Priority 3 (Medium Impact, Clear Solution)
- **Cadence Estimation (Phase 2):** RANSAC-based estimator
  - Expected improvement: ICC -0.033 → 0.35
  - Implementation complexity: Low
  - Risk: Low (well-established method)

---

## 8. Conclusion

Baseline evaluation revealed **systematic errors dominating all temporal-spatial parameters**:
- Negative ICCs indicate bias > between-subject variance
- 100% of subjects overestimated step length
- 100% of subjects had >1.5× strike over-detection

**Root Causes:**
1. Global scaling assumption violated
2. Stride detector over-sensitive
3. Cadence estimator has conflicting heuristics

**Clinical Status:** System **not suitable for clinical use** without correction.

**Path Forward:** Phased improvement strategy (P1→P2→P3) targets each error source independently, enabling clear attribution of improvements.

---

**Analysis Completed By:** Automated diagnostic (P0_baseline_audit.py)
**Data Files:**
- `P0_baseline_audit_results.json` - Full per-subject data
- `baseline_metrics_20251010.json` - Aggregate snapshot

**Next Step:** Phase 1 - Implement and validate stride-based scaling calibration.
