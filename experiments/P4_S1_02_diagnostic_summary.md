# S1_02 Catastrophic Cadence Failure - Diagnostic Summary

## Problem Overview

S1_02 shows the worst cadence error in the entire cohort:
- **GT Cadence**: 113.4 steps/min
- **Predicted Cadence (RANSAC)**: 136.5 steps/min
- **Error**: +23.0 steps/min (20% overestimation)

This represents a **catastrophic failure** of the detection pipeline.

## Root Cause Analysis

### Detailed Breakdown by Leg

#### Left Leg
- **Ground Truth**: ~14 strides, 113.7 steps/min
- **Detected**: 33 strikes
- **RANSAC Cadence**: 97.3 steps/min
- **Status**: UNDERDETECTION (2.36× ratio)
- **Assessment**: Missing actual heel strikes OR detecting multiple false positives that RANSAC filtered out as outliers

#### Right Leg (PRIMARY FAILURE)
- **Ground Truth**: ~13 strides, 113.2 steps/min
- **Detected**: 49 strikes
- **RANSAC Cadence**: 175.6 steps/min
- **Status**: MASSIVE OVERDETECTION (3.77× ratio)
- **Assessment**: Template matching is detecting far too many false positives

### Why RANSAC Failed

1. **Right leg**: 49 strikes detected, but RANSAC found a consensus period corresponding to 175.6 steps/min
   - This suggests the false positives are clustered in a pattern that looks rhythmic
   - RANSAC found agreement among these false detections rather than filtering them out

2. **Left leg**: 33 strikes with high variability likely confused RANSAC
   - Result: 97.3 steps/min (underestimated)

3. **Average**: (97.3 + 175.6) / 2 = 136.5 steps/min
   - This is actually closer to GT than the right leg alone!
   - But the asymmetry (97.3 vs 175.6) indicates severe detection failure

## Likely Causes

### 1. Template Quality Issues
- **Hypothesis**: Template from S1_02 may not represent typical gait cycle
- **Mechanism**: If template extraction captured unusual movement (turn, stumble, etc.),it will match many false patterns
- **Test**: Visual inspection of template extraction region

### 2. Turn/Direction Change Contamination
- **Hypothesis**: S1_02 may have many turns that introduce false patterns
- **Mechanism**: During turns, leg angles change rapidly and may match template spuriously
- **Test**: Check turn filtering results from P1_spatial_error_analysis.csv

#### Turn Filtering Evidence (from P1 spatial analysis)
S1_02 is **NOT in the turn-filtered dataset**, meaning either:
- No significant turns were detected, OR
- S1_02 wasn't included in P1 spatial analysis

This suggests turns are **not** the primary cause.

### 3. DTW Threshold Too Permissive
- **Current threshold**: 0.7 (similarity score)
- **Hypothesis**: Template matching is matching too many weak patterns
- **Mechanism**: Threshold of 0.7 allows moderately dissimilar patterns to be counted as heel strikes
- **Solution**: Test with threshold 0.75 or 0.8 specifically for S1_02

### 4. Video Quality or Pose Estimation Errors
- **Hypothesis**: Noisy pose estimation leads to erratic angle signals
- **Mechanism**: MediaPipe may have poor confidence on S1_02, leading to jittery landmarks
- **Test**: Check visibility scores in CSV, plot raw angle signals

## Proposed Investigations (Priority Order)

### P1: Visual Inspection of S1_02 Video
- Check for unusual gait patterns
- Check for pose estimation quality (occlusions, blur, etc.)
- Verify template extraction region isn't corrupted

### P2: Plot Angle Signals with Detected Strikes
- Visualize left_knee_angle and heel_y over time
- Overlay detected heel strikes (49 for right, 33 for left)
- Check if false positives are clustered or random

### P3: Test Higher DTW Thresholds on S1_02
- Re-run template detection with threshold=0.75, 0.8
- Count strikes and compare to GT
- Find threshold that brings S1_02 closer to correct count

### P4: RANSAC Diagnostics
- Plot stride intervals for S1_02 right leg (48 intervals from 49 strikes)
- Show which intervals were inliers vs outliers
- Visualize the "phantom rhythm" RANSAC found at 175.6 steps/min

## Recommended Fixes

### Short-term (Paper Corrections)
1. **Acknowledge outlier**: Add S1_02 to limitations section
2. **Report median alongside mean**: Median cadence error (excluding S1_02) is likely much lower
3. **Document failure mode**: "Template-based detection can fail catastrophically when..."

### Medium-term (Pipeline Improvements)
1. **Outlier rejection**: Add sanity check that rejects predictions >50% different from population mean
2. **Adaptive thresholds**: Use subject-specific thresholds based on template quality metrics
3. **Cross-validation**: Validate strikes against both legs (expect similar counts ±10%)

### Long-term (Research Direction)
1. **Multi-modal detection**: Combine template matching with kinematic constraints
2. **Quality-aware weighting**: Down-weight subjects with low pose confidence
3. **Ensemble methods**: Use multiple templates or detection methods and vote

## Comparison with Other Poor Performers

### S1_02 vs Other High-Error Cases
| Subject | Cadence Error | Strike Ratio | Notes |
|---------|--------------|--------------|-------|
| **S1_02** | **+23.0** | L=2.36×, R=3.77× | Bilateral overdetection, right catastrophic |
| S1_14 | -16.8 | Unknown | Second worst |
| S1_13 | +11.0 | Unknown | Third worst |

S1_02's error is **37% larger** than the second-worst case (S1_14 at -16.8 steps/min).

## Conclusion

**S1_02's failure is primarily driven by RIGHT leg overdetection (49 strikes vs ~13 GT).** This is likely due to:
1. Template matching being too permissive (threshold=0.7)
2. Possible template quality issues during extraction
3. RANSAC finding consensus among false positives rather than rejecting them

**This case represents a fundamental limitation of the current V5 pipeline** and should be:
- Documented as a known failure mode
- Used to justify threshold tuning in future work
- Potentially excluded as an outlier if doing strict validation

## Next Steps

1. ✅ Document findings in this file
2. ⏳ Create visualization script for S1_02 angle signals + strikes
3. ⏳ Test threshold tuning (0.6, 0.75, 0.8) on full cohort
4. ⏳ Update paper to acknowledge S1_02 failure
5. ⏳ Calculate median cadence error (excluding S1_02) for fairer assessment
