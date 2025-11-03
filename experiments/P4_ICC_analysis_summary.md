# ICC Analysis: Why V5 Scores Are So Low

## Current ICC Scores (V5)

| Metric | ICC | Target | Status |
|--------|-----|--------|--------|
| step_length_left_cm | 0.232 | >0.75 | ❌ Poor |
| step_length_right_cm | 0.050 | >0.75 | ❌ Unacceptable |
| cadence_left | 0.276 | >0.75 | ❌ Poor |
| cadence_right | 0.141 | >0.75 | ❌ Poor |
| cadence_average | 0.213 | >0.75 | ❌ Poor |
| forward_velocity_left | 0.443 | >0.75 | ⚠️ Fair |
| forward_velocity_right | 0.381 | >0.75 | ⚠️ Fair |

**None of the metrics meet clinical validation standards (ICC >0.75)**

## ICC Interpretation

ICC (Intraclass Correlation Coefficient) measures agreement between two measurement methods:
- **>0.75**: Excellent agreement (clinically valid)
- **0.60-0.75**: Good agreement
- **0.40-0.60**: Fair agreement
- **<0.40**: Poor agreement

V5's ICC scores (<0.45 for all temporal metrics) indicate **poor to very poor agreement** with hospital gold standard.

## Root Causes

### 1. Step Length ICC Catastrophically Low (0.050 right, 0.232 left)

#### Top 10 Worst Contributors (Step Length Error)
| Subject | Abs Error | Prediction/GT Ratio | Issue |
|---------|-----------|---------------------|-------|
| S1_30 | 28.11 cm | 0.63× | Severe underestimation |
| S1_28 | 27.99 cm | 0.56× | Severe underestimation |
| S1_27 | 23.98 cm | 0.63× | Severe underestimation |
| S1_23 | 21.60 cm | 0.68× | Underestimation |
| S1_16 | 20.64 cm | 0.67× | Underestimation |
| S1_14 | 18.77 cm | 0.74× | Underestimation |
| S1_09 | 15.80 cm | 0.77× | Underestimation |
| S1_13 | 13.84 cm | 0.76× | Underestimation |
| S1_29 | 13.39 cm | 0.81× | Slight underestimation |
| S1_28 | 13.11 cm | 0.80× | Slight underestimation |

**Pattern**: Systematic **underestimation** (all ratios <1.0)

#### Why Step Length Is Underestimated
1. **Scale factor issues**: Camera-to-world scaling may be consistently too small
2. **Turn contamination**: Even with filtering, some turn cycles may remain
3. **Heel strike underdetection**: Missing strikes → fewer samples → biased scale estimation

### 2. Cadence ICC Also Poor (0.213 average)

#### Top 5 Worst Contributors (Cadence Error)
| Subject | Abs Error | Notes |
|---------|-----------|-------|
| S1_02 | 60.05 steps/min | Catastrophic overdetection (right leg) |
| S1_14 | 16.76 steps/min | Moderate error |
| S1_13 | 10.96 steps/min | Moderate error |
| S1_03 | 10.26 steps/min | Moderate error |
| S1_01 | 8.65 steps/min | Moderate error |

**S1_02 alone accounts for massive ICC degradation**

Remove S1_02 → ICC likely improves to ~0.4-0.5 (still below target but much better)

### 3. Velocity ICC Best (But Still Fair at 0.38-0.44)

Velocity = step_length × cadence / time

Since both step length and cadence have errors, velocity compounds them.
However, velocity ICC is highest because:
- Some errors cancel out (underestimated length × underestimated cadence → closer velocity)
- Velocity is more stable metric (averaged over longer time window)

## Comparison with Baseline (V4)

Unfortunately, no V4 ICC scores were recorded. But qualitatively:
- V4 had overdetection (3.45× ratio) → likely *inflated* step length estimates
- V5 has underdetection (0.88× ratio) → *deflated* step length estimates

Both are bad for ICC, which measures *agreement* not just error magnitude.

## Why ICC Is So Sensitive

ICC penalizes:
1. **Systematic bias** (always too high or always too low)
2. **Inconsistent errors** (works for some subjects, fails for others)
3. **Outliers** (S1_02 catastrophic failure)

RMSE/MAE only measure magnitude of error, but ICC measures **correlation** between methods.

Example:
- If predictions are always 20% too low → RMSE may be ok, but ICC will be terrible
- V5 systematically underestimates → poor ICC despite moderate RMSE

## Improvement Strategies (Prioritized)

### P1: Fix Outliers (Quick Win)
**Action**: Implement outlier rejection for subjects with >30% error

**Expected gain**: ICC 0.21 → 0.40 (if S1_02 is filtered as outlier)

**Implementation**:
```python
if abs(predicted - population_mean) > 0.5 * population_mean:
    flag_as_outlier()
    use_population_median_instead()
```

### P2: Improve Scale Estimation (Medium Term)
**Action**:
- Use more stride cycles for scale (currently may use too few)
- Weight high-quality strides more (low CV, consistent intervals)
- Cross-validate left vs right leg scales (should be similar)

**Expected gain**: ICC 0.21 → 0.45-0.55

### P3: Fix Underdetection (Medium Term)
**Action**: Lower threshold to 0.65 OR use adaptive per-subject thresholds

**Expected gain**: Improve strike ratio 0.88× → 0.95×, which helps scale estimation

### P4: Multi-Method Ensemble (Long Term)
**Action**: Combine template matching with kinematic model

**Expected gain**: ICC 0.21 → 0.60-0.70 (speculative)

## Realistic Target Timeline

| Phase | Action | Expected ICC | Timeline |
|-------|--------|--------------|----------|
| **Current** | V5 baseline | 0.21 | Done |
| **Phase 1** | Outlier rejection | 0.35-0.45 | 1 week |
| **Phase 2** | Scale improvement | 0.45-0.55 | 2-3 weeks |
| **Phase 3** | Threshold tuning | 0.50-0.60 | 1 week |
| **Phase 4** | Ensemble methods | 0.60-0.75 | 2-3 months |

**Reaching ICC >0.75 will require fundamental architecture changes (Phase 4)**

## Conclusion

V5's low ICC (0.05-0.28) is primarily due to:
1. **Systematic underestimation** of step length (scale factor bias)
2. **High variance** across subjects (some work well, others catastrophically fail)
3. **S1_02 outlier** dragging down aggregate scores
4. **Underdetection** (0.88× strike ratio) → biased scale estimation

**Short-term fix**: Outlier rejection + scale refinement → ICC ~0.45
**Medium-term goal**: Threshold tuning + ensemble → ICC ~0.60
**Long-term goal**: New architecture + multi-modal fusion → ICC >0.75

**Current assessment**: V5 is **not yet clinically valid** (ICC <0.75) but shows promise for further improvement.
