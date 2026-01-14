# Bland-Altman Analysis Findings Report

## Critical Discovery for Paper Revision

**Date:** 2025-11-13
**Analysis:** Comprehensive Bland-Altman agreement analysis
**Purpose:** Address PAPER_VALIDATION_REVIEW.md Required Revision #1

---

## Executive Summary

⚠️ **CRITICAL FINDING**: The hip joint, which showed "Good" correlation (|r| = 0.613) in waveform shape analysis, demonstrates **unacceptable clinical agreement** when assessed via Bland-Altman analysis.

This finding fundamentally changes the paper's conclusions and requires significant revision of the Results and Discussion sections.

---

## Key Distinction: Correlation vs Agreement

### What We Previously Reported
- **Hip correlation:** |r| = 0.613 (Good)
- **Interpretation:** "MediaPipe captures hip waveform shape well"
- **Grade:** 58.8% of subjects achieve Good+ similarity

### What Bland-Altman Reveals
- **Systematic bias:** +27.54° (MCID threshold: 10°)
- **95% LoA:** [-46.61°, +101.70°] (width: 148.32°)
- **Clinical acceptability:** ❌ Not Acceptable (0/17 subjects meet criteria)
- **Proportional bias:** Significant (p < 0.0001)

### Why the Discrepancy?

**Correlation measures SHAPE similarity:**
- High correlation = similar waveform patterns
- Can have high correlation even with large offset
- Example: Two curves with same shape but +30° offset → high correlation

**Bland-Altman measures VALUE agreement:**
- Bias = systematic error in measurement
- LoA = range of random error
- Essential for clinical decision-making
- Example: Same two curves → large bias, wide LoA → NOT acceptable

---

## Detailed Findings

### Hip Joint - Aggregate Statistics

| Metric | Value | Clinical Threshold | Status |
|--------|-------|-------------------|--------|
| Mean Bias | +27.54° | <10° | ❌ |
| 95% CI of Bias | [+25.75°, +29.33°] | - | - |
| SD of Differences | 37.84° | - | - |
| Lower LoA | -46.61° | - | - |
| Upper LoA | +101.70° | - | - |
| LoA Width | 148.32° | <20° | ❌ |
| LoA as % of ROM | 357.6% | <50% | ❌ |
| Points within LoA | 99.4% | ~95% | ✅ |
| Proportional Bias | p < 0.0001 | p ≥ 0.05 | ⚠️ |

**Clinical Assessment:**
- **Bias:** 2.75× larger than MCID (27.54° vs 10°)
- **LoA Width:** 7.4× larger than acceptable (148.32° vs 20°)
- **Overall Status:** NOT CLINICALLY ACCEPTABLE

### Proportional Bias

**Finding:** Highly significant proportional bias (p < 0.0001)

**Meaning:**
- The measurement error increases with larger angles
- Slope = +1.70: For every 1° increase in true angle, difference increases by 1.70°
- MediaPipe increasingly overestimates at higher flexion angles

**Clinical Impact:**
- Cannot apply a simple constant correction
- Would need magnitude-dependent calibration
- Limits clinical utility for ROM measurements

### Per-Subject Variability

**Mean Bias by Subject:**
- Mean: +27.54°
- SD: 16.53° (large between-subject variability)
- Range: [-3.16°, +51.88°] (55° range!)

**LoA Width by Subject:**
- Mean: 127.95°
- SD: 44.25°
- Range: [48.27°, 173.88°]

**Clinical Criteria Compliance:**
- Acceptable bias (<10°): 4/17 subjects (23.5%)
- Acceptable LoA (<20°): 0/17 subjects (0%)
- Both criteria met: 0/17 subjects (0%)

**Best Performer:**
- Subject: S1_24
- Bias: -1.51° (excellent)
- LoA: 48.27° (still 2.4× threshold)

**Worst Performer:**
- Subject: S1_02
- Bias: +34.82° (extreme)
- LoA: 173.88° (extreme)

---

## Implications for Paper

### 1. Title and Abstract

**Current positioning:** "Individual-Level Validation of MediaPipe for Gait Analysis"

**Problem:** The word "validation" implies clinical acceptability, which Bland-Altman shows is not met.

**Recommended revision:**
- "Individual-Level **Correlation** Analysis of MediaPipe..."
- "**Preliminary** Individual-Level Validation..."
- Emphasize relative/shape metrics, not absolute values

### 2. Results Section

**Must add:**
- New subsection: "3.X Agreement Analysis (Bland-Altman)"
- Report full B-A statistics
- Create Bland-Altman plot (Figure)
- Table comparing correlation vs agreement metrics

**Key table to add:**

| Joint | Correlation | Grade | Bland-Altman Bias | LoA Width | Clinical Acceptability |
|-------|-------------|-------|-------------------|-----------|----------------------|
| Hip | 0.613 (Good) | 58.8% Good+ | +27.54° | 148.32° | ❌ Not Acceptable |
| Ankle* | 0.678 (Good) | 70.6% Good+ | TBD | TBD | TBD |

*Requires DTW+Per-Subject improved curves

### 3. Discussion Section

**Must address:**

#### Shape vs Value Distinction
```
"While MediaPipe demonstrated Good correlation for hip joint angles (r = 0.613),
indicating acceptable waveform shape similarity, Bland-Altman analysis revealed
unacceptable systematic bias (+27.54°, 2.75× MCID) and wide limits of agreement
(148.32°, 7.4× threshold). This discrepancy highlights the critical distinction
between shape-based metrics (correlation, R²) and value-based agreement measures
(bias, LoA). High correlation indicates MediaPipe captures temporal patterns and
relative changes well, but systematic offset prevents accurate absolute angle
measurement."
```

#### Proportional Bias
```
"Significant proportional bias (slope = +1.70, p < 0.0001) indicates MediaPipe
increasingly overestimates hip flexion at larger angles. This magnitude-dependent
error cannot be corrected with simple constant calibration and would require
angle-specific or non-linear transformation methods."
```

#### Clinical Implications
```
"Given the large systematic bias and wide LoA, MediaPipe hip measurements are
NOT suitable for:
- Absolute ROM quantification
- Clinical decision thresholds (e.g., >45° flexion)
- Cross-subject comparisons of absolute values

However, the Good correlation suggests potential utility for:
- Within-subject relative change monitoring
- Temporal pattern analysis
- Gait cycle phase detection
- ML-based classification (shape-based features)
"
```

### 4. Limitations Section

**Must add:**
```
"Bland-Altman analysis revealed systematic bias and proportional error exceeding
clinical acceptability thresholds. While correlation-based metrics indicated Good
waveform similarity, absolute angle agreement was insufficient for clinical
measurement. This limits applicability to relative/shape-based analysis rather
than absolute value quantification."
```

### 5. Conclusions Section

**Current (problematic):**
> "MediaPipe achieves clinical-grade individual-level gait analysis..."

**Revised (accurate):**
> "MediaPipe achieves Good waveform shape similarity (correlation 0.613-0.678) for
> hip and ankle joints at the individual level. However, Bland-Altman analysis
> shows systematic bias and wide limits of agreement exceeding clinical thresholds,
> limiting absolute value measurement. The system is suitable for relative change
> monitoring and pattern-based analysis, with future work needed to achieve
> clinical-grade absolute angle quantification."

---

## Ankle Joint Analysis (Next Step)

### Status
⚠️ Ankle Bland-Altman analysis **not yet complete**

### Why Not?
The improved ankle curves (after DTW + Per-Subject Deming) are not saved by `improve_with_dtw_persubject.py` - only the metrics are saved.

### What's Needed
1. Update `improve_with_dtw_persubject.py` to save improved curves
2. Run Bland-Altman on improved ankle data
3. Compare to hip findings

### Expected Findings
- Ankle had better correlation (0.678 vs 0.613)
- DTW + Per-Subject calibration specifically targets alignment
- **Hypothesis:** Ankle may have smaller bias and narrower LoA than hip
- **Critical to verify:** Does DTW improve agreement, not just correlation?

---

## Comparison to Literature

### Typical Gait Analysis Agreement Studies

**Gold standard studies report:**
- Mean bias: < ±5°
- LoA width: 10-20° (Bland & Altman suggest <10° for clinical use)
- MCID thresholds: 5-10°

**Our findings:**
- Mean bias: +27.54° (5-6× worse)
- LoA width: 148.32° (7-15× worse)

**Interpretation:**
This positions MediaPipe as a **screening/monitoring tool** rather than a **clinical measurement device**.

---

## Required Actions

### Immediate (for Major Revision)

- [x] ✅ Complete hip Bland-Altman analysis
- [ ] Update `improve_with_dtw_persubject.py` to save improved curves
- [ ] Complete ankle Bland-Altman analysis
- [ ] Create Bland-Altman plots (scatter: mean vs difference)
- [ ] Add B-A section to Methods (2.5.3)
- [ ] Add B-A results to Results (3.X)
- [ ] Revise Discussion with shape vs value distinction
- [ ] Strengthen Limitations section
- [ ] Revise Conclusions to reflect agreement findings
- [ ] Update Abstract to be more conservative

### Medium-term (for Follow-up Study)

- [ ] Investigate source of systematic bias
  - Camera calibration?
  - MediaPipe depth estimation error?
  - Coordinate system transformation?
- [ ] Develop magnitude-dependent calibration
- [ ] Test alternative MediaPipe models
- [ ] Validate in pathological gait (different ROM ranges)

---

## Key Takeaway

**Correlation ≠ Agreement**

High correlation tells us MediaPipe captures temporal patterns well.
Bland-Altman tells us it cannot replace gold-standard measurement devices.

The paper remains valuable as:
1. First individual-level validation
2. Proof that DTW improves ankle performance
3. Demonstration of shape-based utility
4. Foundation for relative/within-subject analysis

But we must be honest about the limitations for absolute value clinical use.

---

## Files Generated

- `processed/bland_altman_hip_aggregate.csv` - Overall hip statistics
- `processed/bland_altman_hip_per_subject.csv` - Subject-level statistics
- `bland_altman_comprehensive.py` - Analysis script

---

## References for Paper

Key papers to cite:
1. Bland & Altman (1986) - Original B-A method
2. Bland & Altman (1999) - Measuring agreement in method comparison
3. Giavarina (2015) - Understanding B-A analysis (tutorial)
4. MCID values for hip/ankle ROM (clinical literature)

---

**Report Date:** 2025-11-13
**Analysis:** Bland-Altman Agreement Study
**Critical Finding:** Correlation-agreement discrepancy identified
**Action Required:** Major paper revision

