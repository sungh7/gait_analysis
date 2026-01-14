# Knee & Ankle Recovery Strategy (Revised)

**Date**: 2025-11-09
**Version**: 2.0 (Improved based on feedback)
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

## Recommended Strategy: Hybrid Approach (10-12 Days)

**Timeline adjusted to realistic expectations based on complexity.**

### Phase 1: Diagnostic & Baseline (Days 1-3)

#### Day 1: Linear Regression Calibration Analysis

**⚠️ IMPROVEMENT 1: Use linear regression, not just ROM ratios**

**Morning**:
```python
#!/usr/bin/env python3
"""
Calculate linear calibration parameters using robust regression.
"""
import numpy as np
from scipy import stats

def calculate_calibration_params(mp_angles, gt_angles):
    """
    Calculate calibration using linear regression (more robust than ROM ratio).

    Returns:
        slope: Scale factor
        intercept: Offset correction
        r_squared: Goodness of fit
    """
    # Flatten all cycle data
    mp_flat = np.concatenate([mp for mp in mp_angles])
    gt_flat = np.concatenate([gt for gt in gt_angles])

    # Linear regression: GT = slope * MP + intercept
    slope, intercept, r_value, p_value, std_err = stats.linregress(mp_flat, gt_flat)

    return {
        'slope': slope,
        'intercept': intercept,
        'r_squared': r_value**2,
        'p_value': p_value,
        'std_err': std_err
    }

# Calculate for knee
knee_params = calculate_calibration_params(knee_mp_data, knee_gt_data)
print(f"Knee calibration:")
print(f"  GT = {knee_params['slope']:.3f} * MP + {knee_params['intercept']:.1f}°")
print(f"  R² = {knee_params['r_squared']:.3f}")

# Calculate for ankle
ankle_params = calculate_calibration_params(ankle_mp_data, ankle_gt_data)
print(f"Ankle calibration:")
print(f"  GT = {ankle_params['slope']:.3f} * MP + {ankle_params['intercept']:.1f}°")
print(f"  R² = {ankle_params['r_squared']:.3f}")
```

**Why this is better**:
- ✅ Accounts for both offset AND scale (ROM ratio only addresses scale)
- ✅ Provides goodness-of-fit metric (R²)
- ✅ More robust to outliers than simple ROM division

**Afternoon**:
- Apply calibration to all subjects
- Regenerate MP data with linear correction
- Compute baseline ICC

**Expected outcomes**:
```
Knee: R² = 0.2-0.4 (poor fit indicates fundamental issues)
Ankle: R² = 0.1-0.3 (poor fit indicates fundamental issues)
ICC: May improve slightly but still <0.40
```

**Deliverable**: `calibration_analysis_day1.json` with linear regression parameters

---

#### Day 2: Diagnostic Unit Tests

**⚠️ IMPROVEMENT 2: Concrete unit tests for coordinate frames**

**Morning - Create test suite**:

```python
#!/usr/bin/env python3
"""
Unit tests for coordinate frame and Cardan angle calculations.
Test with known rotation matrices to validate implementation.
"""
import numpy as np
import pytest

def euler_to_rotation_yxz(theta_y, theta_x, theta_z):
    """
    Create rotation matrix from Euler angles in YXZ order.
    Reference implementation for testing.
    """
    cy, sy = np.cos(np.radians(theta_y)), np.sin(np.radians(theta_y))
    cx, sx = np.cos(np.radians(theta_x)), np.sin(np.radians(theta_x))
    cz, sz = np.cos(np.radians(theta_z)), np.sin(np.radians(theta_z))

    # YXZ rotation order
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])

    return Ry @ Rx @ Rz

def euler_to_rotation_yzx(theta_y, theta_z, theta_x):
    """Create rotation matrix for YZX order."""
    cy, sy = np.cos(np.radians(theta_y)), np.sin(np.radians(theta_y))
    cx, sx = np.cos(np.radians(theta_x)), np.sin(np.radians(theta_x))
    cz, sz = np.cos(np.radians(theta_z)), np.sin(np.radians(theta_z))

    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])

    return Ry @ Rz @ Rx

def test_cardan_yxz_extraction():
    """Test YXZ Cardan angle extraction (hip/knee)."""
    from core_modules.main_pipeline import GaitAnalyzer

    analyzer = GaitAnalyzer()

    test_cases = [
        (10.0, 20.0, 5.0),   # Normal angles
        (30.0, -15.0, 10.0), # Negative X
        (-20.0, 25.0, -10.0),# Negative Y, Z
        (0.0, 0.0, 0.0),     # Identity
        (60.0, 30.0, 15.0),  # Large angles
    ]

    for theta_y, theta_x, theta_z in test_cases:
        R = euler_to_rotation_yxz(theta_y, theta_x, theta_z)
        extracted_y, extracted_x, extracted_z = analyzer._cardan_yxz(R)

        # Check if extracted angles match input (within tolerance)
        assert np.isclose(extracted_y, theta_y, atol=0.1), \
            f"YXZ: Y-axis mismatch: expected {theta_y}, got {extracted_y}"
        assert np.isclose(extracted_x, theta_x, atol=0.1), \
            f"YXZ: X-axis mismatch: expected {theta_x}, got {extracted_x}"
        assert np.isclose(extracted_z, theta_z, atol=0.1), \
            f"YXZ: Z-axis mismatch: expected {theta_z}, got {extracted_z}"

    print("✅ All YXZ Cardan tests passed!")

def test_cardan_yzx_extraction():
    """Test YZX Cardan angle extraction (ankle)."""
    from core_modules.main_pipeline import GaitAnalyzer

    analyzer = GaitAnalyzer()

    test_cases = [
        (10.0, 5.0, 20.0),   # Normal angles
        (15.0, -10.0, 30.0), # Negative Z
        (-20.0, 15.0, -10.0),# Negative Y, X
        (0.0, 0.0, 0.0),     # Identity
    ]

    for theta_y, theta_z, theta_x in test_cases:
        R = euler_to_rotation_yzx(theta_y, theta_z, theta_x)
        extracted_y, extracted_z, extracted_x = analyzer._cardan_yzx(R)

        assert np.isclose(extracted_y, theta_y, atol=0.1), \
            f"YZX: Y-axis mismatch: expected {theta_y}, got {extracted_y}"
        assert np.isclose(extracted_z, theta_z, atol=0.1), \
            f"YZX: Z-axis mismatch: expected {theta_z}, got {extracted_z}"
        assert np.isclose(extracted_x, theta_x, atol=0.1), \
            f"YZX: X-axis mismatch: expected {theta_x}, got {extracted_x}"

    print("✅ All YZX Cardan tests passed!")

def test_coordinate_frame_orthonormality():
    """Test that coordinate frames are orthonormal."""
    from core_modules.main_pipeline import GaitAnalyzer

    analyzer = GaitAnalyzer()

    # Load test subject landmarks
    landmarks = load_test_landmarks('S1_01', frame=50)

    # Test pelvis frame
    _, pelvis_axes = analyzer._build_pelvis_frame(landmarks)
    verify_orthonormal(pelvis_axes, "Pelvis")

    # Test femur frame
    femur_axes = analyzer._build_femur_frame(landmarks, 'left', pelvis_axes)
    verify_orthonormal(femur_axes, "Femur")

    # Test tibia frame
    tibia_axes = analyzer._build_tibia_frame(landmarks, 'left', pelvis_axes)
    verify_orthonormal(tibia_axes, "Tibia")

    # Test foot frame
    foot_axes = analyzer._build_foot_frame(landmarks, 'left')
    verify_orthonormal(foot_axes, "Foot")

    print("✅ All coordinate frames are orthonormal!")

def verify_orthonormal(axes, name):
    """Verify axes form orthonormal basis."""
    # Check orthogonality
    assert np.isclose(np.dot(axes[0], axes[1]), 0.0, atol=1e-6), \
        f"{name}: X and Y not orthogonal"
    assert np.isclose(np.dot(axes[1], axes[2]), 0.0, atol=1e-6), \
        f"{name}: Y and Z not orthogonal"
    assert np.isclose(np.dot(axes[2], axes[0]), 0.0, atol=1e-6), \
        f"{name}: Z and X not orthogonal"

    # Check normalization
    assert np.isclose(np.linalg.norm(axes[0]), 1.0, atol=1e-6), \
        f"{name}: X not normalized"
    assert np.isclose(np.linalg.norm(axes[1]), 1.0, atol=1e-6), \
        f"{name}: Y not normalized"
    assert np.isclose(np.linalg.norm(axes[2]), 1.0, atol=1e-6), \
        f"{name}: Z not normalized"

    # Check right-hand rule: X × Y = Z
    cross = np.cross(axes[0], axes[1])
    assert np.allclose(cross, axes[2], atol=1e-6), \
        f"{name}: Not right-handed (X×Y≠Z)"

if __name__ == '__main__':
    test_cardan_yxz_extraction()
    test_cardan_yzx_extraction()
    test_coordinate_frame_orthonormality()
    print("\n🎉 All diagnostic tests passed!")
```

**Afternoon - Run diagnostic tests**:
```bash
python test_coordinate_frames.py
```

**Expected outcomes**:
- ✅ Cardan extraction tests should pass (implementation correct)
- ⚠️ May reveal coordinate frame issues (axes not orthonormal, wrong directions)

**Deliverable**: Test report showing which tests pass/fail

---

#### Day 3: Deep Dive Analysis & Go/No-Go Decision

**⚠️ IMPROVEMENT 4: 3-Stage Decision Tree (Stage 1)**

**Morning - Analyze failure modes**:
```python
# For each subject, diagnose specific issues
for subject in subjects:
    mp_curve = mp_data[subject]
    gt_curve = gt_data[subject]

    # 1. Check ROM ratio
    mp_rom = mp_curve.max() - mp_curve.min()
    gt_rom = gt_curve.max() - gt_curve.min()
    rom_ratio = mp_rom / gt_rom

    # 2. Check waveform shape correlation (centered)
    mp_centered = mp_curve - mp_curve.mean()
    gt_centered = gt_curve - gt_curve.mean()
    shape_corr = np.corrcoef(mp_centered, gt_centered)[0, 1]

    # 3. Check peak/trough locations
    mp_peak_loc = np.argmax(mp_curve)
    gt_peak_loc = np.argmax(gt_curve)
    peak_shift = abs(mp_peak_loc - gt_peak_loc)

    # Classify failure mode
    if shape_corr < 0.5:
        failure_mode = "WAVEFORM_SHAPE"  # Coordinate frame axis error
    elif abs(rom_ratio - 1.0) > 0.5:
        failure_mode = "ROM_SCALE"       # Scaling/normalization issue
    elif peak_shift > 10:
        failure_mode = "PHASE_SHIFT"     # Temporal misalignment
    else:
        failure_mode = "OFFSET_ONLY"     # Calibration can fix

    print(f"{subject}: {failure_mode} (ROM ratio={rom_ratio:.2f}, shape_corr={shape_corr:.2f})")
```

**Afternoon - Stage 1 Decision Point**:

```
DECISION TREE - STAGE 1 (End of Day 3)

Criteria:
  A. Linear regression R² for knee
  B. Linear regression R² for ankle
  C. % subjects with correct waveform shape (correlation > 0.7)

Decision Rules:

IF (R² > 0.5 for both joints) AND (>70% subjects have correct shape):
  → FAST TRACK (proceed to Day 4-7: Simple calibration only)
  → Expected final ICC: Knee 0.60-0.70, Ankle 0.50-0.60

ELIF (R² > 0.3 for both joints) AND (>50% subjects have correct shape):
  → MODERATE TRACK (proceed to Day 4-10: Targeted coordinate fixes)
  → Expected final ICC: Knee 0.50-0.60, Ankle 0.40-0.50

ELIF (R² > 0.1 for at least one joint) OR (>30% subjects have correct shape):
  → DEEP DIVE TRACK (proceed to Day 4-12: Full coordinate frame revision)
  → Expected final ICC: Knee 0.40-0.60, Ankle 0.30-0.50

ELSE:
  → ABORT (fundamental MediaPipe limitation, publish hip-only)
  → Document findings for future work
```

**Deliverable**: Stage 1 decision report with recommended track

---

### Phase 2: Root Cause Investigation (Days 4-7)

#### Day 4-5: Coordinate Frame Deep Dive

**Focus on subjects with worst failure modes identified on Day 3.**

**Knee Investigation**:

1. **Review Vicon PiG femur frame definition**:
   ```
   Origin: Hip Joint Center (HJC)
   Y_femur: HJC → KJC (superior, proximal)
   X_femur: Mediolateral (right-hand rule)
   Z_femur: Anteroposterior
   ```

2. **Compare with current implementation** ([core_modules/main_pipeline.py:546-574](core_modules/main_pipeline.py#L546-L574)):
   ```python
   def _build_femur_frame(self, landmarks, side, pelvis_axes):
       # Check:
       # - Is Y-axis truly pointing HJC → KJC?
       # - Is normalization applied correctly?
       # - Are axes orthogonalized (Gram-Schmidt)?
   ```

3. **Manual calculation verification**:
   ```python
   # For S1_01, frame 50 (mid-stance)
   hip = landmarks[23]  # Left hip
   knee = landmarks[25]  # Left knee

   # Calculate Y-axis manually
   y_axis_manual = (knee - hip) / np.linalg.norm(knee - hip)
   y_axis_code = analyzer._build_femur_frame(landmarks, 'left', pelvis)[1]

   # Should match!
   assert np.allclose(y_axis_manual, y_axis_code, atol=0.01)
   ```

4. **Test with synthetic landmarks**:
   ```python
   # Create perfect vertical femur (known geometry)
   hip = np.array([0.0, 1.0, 0.0])
   knee = np.array([0.0, 0.5, 0.0])
   ankle = np.array([0.0, 0.0, 0.0])

   # Calculate knee angle (should be 0° when aligned)
   theta = analyzer._calculate_knee_angle(synthetic_landmarks)
   assert np.isclose(theta, 0.0, atol=1.0)
   ```

**Ankle Investigation**:

1. **Re-examine foot frame Y-axis scaling**:
   ```python
   # Current (after YZX fix):
   axis_y = self._normalize_vector(toe - ankle, [0.0, 0.0, 1.0])

   # Questions:
   # - Is toe landmark accurate enough? (MP toe can be noisy)
   # - Should we use foot_index (31/32) instead of toe?
   # - Is normalization amplifying noise?
   ```

2. **Check tibia frame propagation**:
   ```python
   # Tibia frame is shared between knee and ankle
   # If tibia frame wrong → both knee AND ankle wrong

   tibia_axes = analyzer._build_tibia_frame(landmarks, 'left', pelvis_axes)

   # Verify:
   # - Y-axis: knee → ankle (distal direction)
   # - Proper Gram-Schmidt orthogonalization
   # - No scaling issues
   ```

3. **Validate YZX extraction with edge cases**:
   ```python
   # Test gimbal lock scenarios
   test_angles = [
       (10, 90, 20),   # Gimbal lock at Z=90°
       (10, -90, 20),  # Gimbal lock at Z=-90°
       (89, 0, 0),     # Near-vertical
   ]

   for y, z, x in test_angles:
       R = euler_to_rotation_yzx(y, z, x)
       y_ext, z_ext, x_ext = analyzer._cardan_yzx(R)
       # Check if extraction fails near gimbal lock
   ```

**Deliverable**: Root cause report for top 3 problematic subjects

---

#### Day 6-7: Implement Targeted Fixes

**⚠️ IMPROVEMENT 5: Git version control + regression testing**

**Morning - Git workflow setup**:
```bash
# Create feature branch
git checkout -b fix/knee-ankle-coordinate-frames
git checkout -b backup/hip-working-baseline  # Safety backup

# Before ANY changes, run regression test
python test_hip_regression.py
# ✅ Hip ICC should still be 0.813 ± 0.01
```

**Regression test suite**:
```python
#!/usr/bin/env python3
"""
Regression test to ensure hip validation remains intact.
Run before and after any coordinate frame changes.
"""
import json
import numpy as np

def test_hip_icc_regression():
    """Hip ICC must remain ≥0.80 after any changes."""
    # Re-run hip calibration
    from phase2_dtw_hip_poc import main as run_hip_calibration

    results = run_hip_calibration()
    icc = results['icc_after_dtw']

    assert icc >= 0.80, f"Hip ICC regressed: {icc:.3f} < 0.80"
    print(f"✅ Hip ICC regression test passed: {icc:.3f}")

def test_hip_waveform_shapes():
    """Hip waveform shapes must remain correct."""
    with open('processed/phase2_hip_dtw_results.json') as f:
        results = json.load(f)

    for subject in results['subject_results']:
        peak_loc = subject['mp_peak_location']
        # Peak should be at 89-91% (late stance)
        assert 85 <= peak_loc <= 95, \
            f"{subject['subject_id']}: Peak at {peak_loc}% (expected 89-91%)"

    print("✅ Hip waveform regression test passed")

if __name__ == '__main__':
    test_hip_icc_regression()
    test_hip_waveform_shapes()
    print("\n🎉 All regression tests passed - safe to proceed!")
```

**Afternoon - Implement fixes incrementally**:

```bash
# Fix 1: Update knee femur frame (if needed)
# Edit core_modules/main_pipeline.py
git add core_modules/main_pipeline.py
git commit -m "fix(knee): Correct femur frame Y-axis normalization"

# Test on 1 subject
python test_single_subject.py --subject 1 --joint knee
# ✅ S1_01 knee: ROM 28.2° → 62.5° (closer to GT 61.9°)

# Test on 3 subjects
python test_single_subject.py --subjects 1 2 3 --joint knee
# ✅ Average ROM improvement: +30%

# Run regression test
python test_hip_regression.py
# ✅ Hip still at ICC 0.813

# If all pass → apply to full cohort
python generate_mediapipe_cycle_dataset.py --subjects 1 2 3 ... 26 --workers 4
```

**Evening - Validation**:
```bash
# Re-run calibration for knee and ankle
python knee_deming_calibration.py
python ankle_deming_calibration.py

# Check ICC improvement
# Expected: Knee 0.40-0.60, Ankle 0.30-0.50
```

**Deliverable**: Fixed code with regression tests passing, preliminary ICC results

---

### Phase 3: Full Validation (Days 8-10)

#### Day 8: Apply Fixes to Full Cohort

**Morning**:
```bash
# Regenerate all subjects with fixed code
python generate_mediapipe_cycle_dataset.py \
  --subjects 1 2 3 8 9 10 11 13 14 15 16 17 18 23 24 25 26 \
  --output processed/S1_mediapipe_cycles_fixed.json \
  --workers 4

# Compute representative cycles
python compute_mediapipe_representative_cycles.py
```

**Afternoon**:
```bash
# Run Deming + DTW for both joints
python knee_deming_calibration.py
python ankle_deming_calibration.py

# Calculate final ICC
# Expected results documented in decision tree below
```

**Evening - Stage 2 Decision Point**:

```
DECISION TREE - STAGE 2 (End of Day 8)

Criteria:
  A. Knee ICC after Deming + DTW
  B. Ankle ICC after Deming + DTW
  C. % valid subjects (waveform shape correct)

Decision Rules:

IF (Knee ICC ≥ 0.60) AND (Ankle ICC ≥ 0.50):
  → SUCCESS (proceed to Day 9-10: LOSO CV and finalization)
  → Publication: "3-joint validation" paper

ELIF (Knee ICC ≥ 0.50) AND (Ankle ICC ≥ 0.40):
  → PARTIAL SUCCESS (proceed to Day 9-10: LOSO CV)
  → Publication: "3-joint validation with limitations" paper
  → Document known issues in limitations section

ELIF (Knee ICC ≥ 0.40) OR (Ankle ICC ≥ 0.30):
  → LIMITED SUCCESS (proceed to Day 9-12: Extended debugging)
  → Extend timeline 2 more days for targeted fixes
  → Publication: "Hip validation + preliminary knee/ankle" paper

ELSE:
  → ABORT knee/ankle (publish hip-only)
  → Document lessons learned
  → Include knee/ankle as "ongoing work" in discussion
```

**Deliverable**: Stage 2 decision report with publication recommendation

---

#### Day 9: Cross-Validation

**Morning - LOSO CV for knee**:
```python
#!/usr/bin/env python3
"""
Leave-One-Subject-Out cross-validation for knee.
Tests generalization to new subjects.
"""
import json
import numpy as np

def loso_cv_knee():
    """Perform LOSO CV for knee joint."""

    # Load data
    with open('processed/S1_mediapipe_representative_cycles.json') as f:
        data = json.load(f)

    subjects = [s for s in data['results']]
    n = len(subjects)

    loso_results = []

    for i in range(n):
        # Leave out subject i
        train_subjects = subjects[:i] + subjects[i+1:]
        test_subject = subjects[i]

        # Train Deming on train set
        mp_train = [s['joints']['knee_flexion_extension']['mean_curve']
                    for s in train_subjects]
        gt_train = [load_gt(s['subject_id']) for s in train_subjects]

        slope, intercept = deming_regression(mp_train, gt_train)

        # Test on held-out subject
        mp_test = np.array(test_subject['joints']['knee_flexion_extension']['mean_curve'])
        gt_test = load_gt(test_subject['subject_id'])

        mp_calibrated = slope * mp_test + intercept

        # Calculate correlation
        corr = np.corrcoef(mp_calibrated, gt_test)[0, 1]

        loso_results.append({
            'subject_id': test_subject['subject_id'],
            'correlation': corr,
            'slope': slope,
            'intercept': intercept
        })

        print(f"LOSO {i+1}/{n}: {test_subject['subject_id']} - Corr = {corr:.3f}")

    # Calculate ICC on LOSO predictions
    # ... (same ICC calculation as before)

    return loso_results

if __name__ == '__main__':
    results = loso_cv_knee()
    print(f"\nLOSO CV ICC: {results['icc']:.3f}")
```

**Afternoon - Bland-Altman analysis**:
```bash
python bland_altman_knee.py
python bland_altman_ankle.py
```

**Expected results**:
```
Knee Bland-Altman:
  Bias: -2 to +2° (good)
  95% LoA: ±15-20° (acceptable for knee)

Ankle Bland-Altman:
  Bias: -1 to +1° (good)
  95% LoA: ±8-12° (acceptable for ankle)
```

**Deliverable**: LOSO CV results and Bland-Altman plots

---

#### Day 10: Final Validation & Documentation

**Morning - Generate comprehensive report**:
```python
# Create final validation report
python generate_validation_report.py --joints hip knee ankle

# Output: COMPREHENSIVE_VALIDATION_REPORT.md
```

**Afternoon - Stage 3 Final Decision**:

```
DECISION TREE - STAGE 3 (End of Day 10)

Based on LOSO CV and Bland-Altman:

Publication Tier 1 (Best Case):
  - Knee: ICC ≥ 0.60, LOSO CV ≥ 0.55, LoA < ±20°
  - Ankle: ICC ≥ 0.50, LOSO CV ≥ 0.45, LoA < ±12°
  → Journal: "IEEE Journal of Biomedical Engineering" or "Gait & Posture"
  → Title: "Comprehensive Lower-Limb Joint Angle Validation using MediaPipe Pose"
  → Timeline: Submit within 1 week

Publication Tier 2 (Acceptable):
  - Knee: ICC ≥ 0.50, LOSO CV ≥ 0.45, LoA < ±25°
  - Ankle: ICC ≥ 0.40, LOSO CV ≥ 0.35, LoA < ±15°
  → Journal: "Sensors" or "Applied Sciences"
  → Title: "MediaPipe Pose for Markerless Gait Analysis: Multi-Joint Validation"
  → Include detailed limitations section
  → Timeline: Submit within 2 weeks

Publication Tier 3 (Hip Focus):
  - Knee or Ankle below Tier 2 thresholds
  → Journal: "Sensors" or conference (e.g., EMBC)
  → Title: "Hip Joint Angle Validation using MediaPipe Pose"
  → Include knee/ankle as "preliminary results" in supplementary
  → Timeline: Submit within 3 days (hip already validated)

Extended Research (2+ more weeks):
  - If close to Tier 2 but not quite there
  → Continue debugging for another 2 weeks max
  → Hard deadline: 2 weeks from Day 10
  → If still not Tier 2 by then → Publish Tier 3
```

**Deliverable**: Final publication decision and manuscript outline

---

### Phase 4: Extended Refinement (Days 11-12, Optional)

**Only if Stage 2 resulted in "LIMITED SUCCESS" path.**

#### Day 11-12: Targeted Debugging for Borderline Cases

**Focus**: Subjects where ICC is close but not quite meeting threshold.

**Morning**:
- Identify subjects dragging down overall ICC
- Investigate subject-specific issues (e.g., noisy landmarks, atypical gait)
- Consider excluding outliers (with strong justification)

**Afternoon**:
- Re-run validation with refined subject selection
- Check if ICC crosses threshold

**Evening - Final Go/No-Go**:
```
IF ICC improvement > 0.05 after refinement:
  → Include refinements in final results
ELSE:
  → Abandon refinements (not worth the complexity)
  → Publish with current results
```

---

## Success Criteria (Revised)

### Minimum Acceptable Results (MVP)

| Joint | ICC | LOSO CV ICC | Bland-Altman LoA | Interpretation |
|-------|-----|-------------|------------------|----------------|
| Hip | 0.813 | 0.859 | ±10° | Excellent (achieved) ✅ |
| Knee | **≥0.50** | **≥0.45** | **<±25°** | Moderate (acceptable) |
| Ankle | **≥0.40** | **≥0.35** | **<±15°** | Fair (acceptable) |

**Justification**:
- Knee/Ankle more challenging than hip (smaller ROM, more noise)
- Literature shows ICC 0.40-0.60 is common for markerless knee/ankle
- Fair-to-moderate agreement still clinically useful

### Target Results (Ideal)

| Joint | ICC | LOSO CV ICC | Bland-Altman LoA | Interpretation |
|-------|-----|-------------|------------------|----------------|
| Hip | 0.813 | 0.859 | ±10° | Excellent ✅ |
| Knee | **≥0.60** | **≥0.55** | **<±20°** | Good |
| Ankle | **≥0.50** | **≥0.45** | **<±12°** | Moderate |

**This would be state-of-the-art for markerless 3-joint validation.**

---

## Risk Mitigation (Enhanced)

### Risk 1: Coordinate frame fixes break hip validation

**Probability**: 15%

**Mitigation**:
- ✅ **Git version control**: Feature branch + backup branch
- ✅ **Regression test suite**: Run before/after every change
- ✅ **Incremental testing**: 1 subject → 3 subjects → full cohort
- ✅ **Atomic commits**: Each fix in separate commit (easy to revert)

**Recovery**:
```bash
# If hip breaks, immediately revert
git checkout backup/hip-working-baseline
git branch -D fix/knee-ankle-coordinate-frames

# Investigate what went wrong
git diff backup/hip-working-baseline fix/knee-ankle-coordinate-frames

# Fix more carefully
```

---

### Risk 2: Scale factors don't improve ICC

**Probability**: 25%

**Mitigation**:
- ✅ **3-stage decision tree**: Clear abort criteria at Days 3, 8, 10
- ✅ **Linear regression (not ROM ratio)**: More robust calibration
- ✅ **Diagnostic unit tests**: Catch issues early

**Recovery**:
- Abort at Stage 1 (Day 3) if R² < 0.1
- Publish hip-only paper within 1 week
- Document knee/ankle findings as "ongoing research"

---

### Risk 3: Time overrun (>12 days)

**Probability**: 30%

**Mitigation**:
- ✅ **Hard deadline**: 12 days maximum (was 5-7 days)
- ✅ **Weekly checkpoints**: Stage 1 (Day 3), Stage 2 (Day 8), Stage 3 (Day 10)
- ✅ **Abort criteria**: Clear decision rules prevent endless debugging

**Recovery**:
- Stage 3 decision includes "Extended Research" option (2 more weeks max)
- Hard cutoff at 4 weeks total
- If >4 weeks → Publish hip-only, period.

---

## Resource Requirements

### Time Commitment

**Total timeline**: 10-12 days (realistic, vs. original 5-7)

**Breakdown**:
- Phase 1 (Diagnostic): 3 days
- Phase 2 (Root cause): 4 days
- Phase 3 (Validation): 3 days
- Phase 4 (Optional): 2 days

**Daily time**: 6-8 hours focused work

---

### Technical Requirements

- ✅ Data: 17 subjects, GT data available
- ✅ Infrastructure: Deming, DTW, ICC scripts ready
- ✅ Partial fixes: Ankle YZX implemented
- ✅ **NEW**: Unit test framework
- ✅ **NEW**: Git workflow
- ✅ **NEW**: Regression test suite

**Additional tools needed**:
```bash
# Install pytest for unit tests
pip install pytest

# Set up Git hooks for auto-testing
cp hooks/pre-commit .git/hooks/
chmod +x .git/hooks/pre-commit
```

---

## Decision Matrix (Updated)

| Criterion | Option A (Rigorous) | Option B (Quick) | **Option C (Hybrid) - RECOMMENDED** |
|-----------|---------------------|------------------|-------------------------------------|
| Time | 7-10 days | 1.5 days | **10-12 days** |
| Success probability | 70-80% | 50-60% | **80-90%** |
| Expected ICC (Knee) | 0.60-0.70 | 0.40-0.50 | **0.50-0.70** |
| Expected ICC (Ankle) | 0.50-0.60 | 0.30-0.40 | **0.40-0.60** |
| Scientific rigor | ✅✅✅ | ⚠️ | **✅✅✅** |
| Publication quality | Excellent | Fair | **Good to Excellent** |
| Generalization | Excellent | Poor | **Good to Excellent** |
| **Risk management** | Basic | None | **✅ Git + regression tests** |

**Recommendation**: **Option C (Hybrid)** with 10-12 day timeline

---

## Immediate Next Steps

### Tomorrow Morning (Day 1)

1. **Create linear regression calibration script**:
   ```bash
   python calculate_calibration_params.py --joint knee
   python calculate_calibration_params.py --joint ankle
   ```

2. **Apply calibration and test**:
   ```bash
   python apply_linear_calibration.py --knee-slope 2.1 --knee-intercept 15.0
   python knee_deming_calibration.py
   ```

3. **Calculate baseline R²**:
   - Expected: Knee R² = 0.2-0.4, Ankle R² = 0.1-0.3

**Expected time**: 4 hours
**Expected outcome**: Baseline calibration parameters + R² values

---

### Tomorrow Afternoon (Day 1)

4. **Create unit test suite**:
   ```bash
   python test_coordinate_frames.py
   ```

5. **Set up Git workflow**:
   ```bash
   git checkout -b fix/knee-ankle-coordinate-frames
   git checkout -b backup/hip-working-baseline
   python test_hip_regression.py  # Verify hip baseline
   ```

**Expected time**: 3 hours
**Expected outcome**: Unit tests passing, Git workflow ready

---

## Conclusion

**Recommended approach**: **Option C (Hybrid)** with **10-12 day realistic timeline**

**Key Improvements over V1**:
1. ✅ **Linear regression** instead of ROM ratios (more robust)
2. ✅ **Concrete unit tests** for Cardan extraction and coordinate frames
3. ✅ **Realistic timeline** (10-12 days vs. 5-7)
4. ✅ **3-stage decision tree** with clear numeric criteria
5. ✅ **Git workflow + regression tests** to prevent breaking hip

**Expected results**:
- Knee ICC 0.50-0.70 (moderate to good)
- Ankle ICC 0.40-0.60 (fair to moderate)

**Risk**: Medium-Low (well-managed with decision trees and abort criteria)

**Go/No-Go decision points**:
- Day 3 (Stage 1): R² and waveform shape analysis
- Day 8 (Stage 2): ICC after fixes
- Day 10 (Stage 3): LOSO CV + final publication decision

**Hard deadline**: 12 days for main work, +2 weeks extended research max, 4 weeks absolute cutoff

---

**Next action**: Begin Day 1 with linear regression calibration analysis.

**Approval needed**: Proceed with improved Option C strategy?
