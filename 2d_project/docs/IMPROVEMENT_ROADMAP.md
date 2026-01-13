# Technical Accuracy Improvement Roadmap

**Goal:** Improve ankle/knee tracking correlation from current 0.4-0.7 range to >0.80

**Critical Issues Identified:**
- From `qc_failure_analysis.md`: 43 joint measurements have correlation < 0.7
- Ankle tracking is worst (15 subjects with r < 0.7, some as low as 0.316)
- Knee tracking has phase mismatch issues (r < 0.5 for several subjects)

---

## ✅ COMPLETED: Enhanced Signal Processing

**File:** `improved_signal_processing.py`

**Improvements:**
1. **Adaptive Gap Filling**: Cubic spline interpolation instead of forward/backward fill
2. **Outlier Rejection**: Physiological range clipping + velocity constraints
3. **Joint-Specific Filtering**:
   - Ankle: Aggressive smoothing (median + Savitzky-Golay + Gaussian)
   - Knee: Balanced (Butterworth 6Hz + light SG filter)
   - Hip: Gentle (Butterworth 8Hz only)
4. **Temporal Consistency**: Enforces expected gait cycle intervals
5. **Quality Metrics**: SNR, smoothness, temporal consistency scores

**Expected Impact:** +0.10 to +0.15 correlation improvement

**Testing:** Run on S1_15 (worst case, ankle r=0.316):
```bash
python improved_signal_processing.py
```

---

## ✅ COMPLETED: Landmark Quality Assessment

**File:** `landmark_quality_filter.py`

**Improvements:**
1. **Quality Metrics**: Visibility, stability, coverage scores for each landmark
2. **Proactive Filtering**: Detect poor tracking before angle calculation
3. **Bilateral Recovery**: Use mirrored opposite side for bad landmarks
4. **Quality Reports**: Detailed diagnostics for troubleshooting

**Expected Impact:** Identify videos that should be excluded or need special handling

---

## 🔄 IN PROGRESS: Multi-Model Pose Estimation

**Concept:** Run multiple MediaPipe models and ensemble results

### Strategy A: Model Complexity Ensemble
```python
# Run MediaPipe at 3 complexity levels
models = [
    mp_pose.Pose(model_complexity=0),  # Lite: fast, lower accuracy
    mp_pose.Pose(model_complexity=1),  # Full: balanced
    mp_pose.Pose(model_complexity=2)   # Heavy: slow, higher accuracy
]

# Ensemble: Weighted average based on visibility scores
landmarks_ensemble = weighted_average(
    landmarks_lite,
    landmarks_full,
    landmarks_heavy,
    weights=[0.2, 0.3, 0.5]  # Favor heavy model
)
```

**Expected Impact:** +0.05 to +0.10 correlation improvement
**Cost:** 3x processing time (acceptable for offline analysis)

### Strategy B: Temporal Ensemble
```python
# Run detection on slightly shifted time windows
results = []
for offset in [-2, -1, 0, 1, 2]:  # frames
    shifted_video = shift_frames(video, offset)
    landmarks = extract_pose(shifted_video)
    results.append(landmarks)

# Take median across temporal ensemble
landmarks_stable = np.median(results, axis=0)
```

**Expected Impact:** Reduces temporal jitter
**Cost:** 5x processing time

---

## 🎯 PRIORITY: Kinematic Constraints

**File:** `kinematic_constraints.py` (TO BE CREATED)

### Biomechanical Constraints to Implement:

#### 1. **Joint Angle Limits (Hard Constraints)**
```python
# Prevent physically impossible angles
constraints = {
    'knee': (0, 140),      # Cannot hyperextend past 0°
    'hip': (-30, 120),     # Limited extension/flexion
    'ankle': (-30, 50),    # Limited plantarflexion/dorsiflexion
}
```

#### 2. **Velocity Limits**
```python
# Maximum angular velocities from biomechanics literature
max_velocity = {
    'knee': 400,   # deg/sec during swing phase
    'hip': 300,    # deg/sec
    'ankle': 350   # deg/sec
}
```

#### 3. **Acceleration Limits**
```python
# Maximum angular accelerations
max_acceleration = {
    'knee': 2000,   # deg/sec²
    'hip': 1500,
    'ankle': 1800
}
```

#### 4. **Kinematic Chain Consistency**
```python
# Enforce: ankle follows knee motion
# If knee is flexing (positive velocity), ankle should respond within 50ms
def enforce_kinematic_chain(knee_angle, ankle_angle, hip_angle):
    # Shank length constraint
    shank_length = compute_segment_length(knee, ankle)
    assert 0.35 < shank_length < 0.55  # meters

    # Thigh length constraint
    thigh_length = compute_segment_length(hip, knee)
    assert 0.35 < thigh_length < 0.55  # meters

    # Fix violations by adjusting least confident landmark
    if violation_detected:
        adjust_lowest_visibility_landmark()
```

**Expected Impact:** +0.10 to +0.20 correlation improvement (HIGHEST IMPACT)
**Rationale:** Fixes anatomically impossible poses that cause correlation drops

---

## 📊 ADVANCED: Kalman Filtering

**File:** `kalman_joint_tracker.py` (TO BE CREATED)

### Kalman Filter for Joint Angles

**Concept:** Model joint angles as a dynamic system with predicted motion

```python
from filterpy.kalman import KalmanFilter

class JointAngleKalmanFilter:
    """
    State vector: [angle, angular_velocity, angular_acceleration]
    Measurement: [angle] from MediaPipe
    """

    def __init__(self, fps=30):
        self.kf = KalmanFilter(dim_x=3, dim_z=1)
        dt = 1.0 / fps

        # State transition matrix (constant acceleration model)
        self.kf.F = np.array([
            [1, dt, 0.5*dt**2],
            [0,  1, dt],
            [0,  0, 1]
        ])

        # Measurement matrix (we only measure angle)
        self.kf.H = np.array([[1, 0, 0]])

        # Process noise (model uncertainty)
        self.kf.Q *= 0.01  # Low process noise (smooth motion)

        # Measurement noise (MediaPipe uncertainty)
        self.kf.R = 10  # Higher for noisy ankle, lower for stable hip

    def filter_sequence(self, angles):
        filtered = []
        for angle in angles:
            self.kf.predict()
            self.kf.update(angle)
            filtered.append(self.kf.x[0])  # Return filtered angle
        return np.array(filtered)
```

**Expected Impact:** +0.10 to +0.15 correlation improvement
**Advantage:** Handles missing data and outliers gracefully

---

## 🔬 RESEARCH: Anatomical Landmark Refinement

**File:** `anatomical_refinement.py` (TO BE CREATED)

### Problem: MediaPipe landmarks ≠ Anatomical landmarks

**MediaPipe Issue:**
- "Knee" landmark = midpoint of knee visually
- "Ankle" landmark = visible ankle prominence
- **BUT:** Biomechanics defines joints at **joint centers** (rotation axes)

**Solution: Offset Correction**
```python
def refine_anatomical_landmarks(mp_landmarks, subject_height):
    """
    Adjust MediaPipe landmarks to anatomical joint centers.
    Based on regression analysis from Vicon ground truth.
    """

    # Knee joint center is ~2cm posterior to visual knee landmark
    knee_offset = np.array([0, -0.02, 0]) * (subject_height / 1.70)
    refined_knee = mp_landmarks['knee'] + knee_offset

    # Ankle joint center (lateral malleolus) is ~1cm posterior
    ankle_offset = np.array([0, -0.01, 0]) * (subject_height / 1.70)
    refined_ankle = mp_landmarks['ankle'] + ankle_offset

    # Hip joint center is deep, need to estimate from ASIS/Trochanter
    # Use regression equation from Davis et al. 1991
    hip_offset = estimate_hip_joint_center(
        mp_landmarks['left_hip'],
        mp_landmarks['right_hip'],
        pelvis_depth=subject_height * 0.15
    )

    return refined_landmarks
```

**Expected Impact:** +0.15 to +0.25 correlation improvement (VERY HIGH)
**Challenge:** Requires subject-specific calibration or population average

---

## 🧪 VALIDATION: Improved Pipeline Testing

### Test Protocol

**Test Subjects:** Focus on worst performers from QC report
- S1_15: Ankle r=0.316 → Target >0.70
- S1_08: Knee r=0.443 → Target >0.70
- S1_16: Knee r=0.433 → Target >0.70
- S1_23: Knee r=0.414 → Target >0.70

**Test Script:** `test_improved_pipeline.py` (TO BE CREATED)
```python
def compare_pipelines(subject_id):
    """Compare old vs new pipeline."""

    # Old pipeline
    old_correlation = run_old_pipeline(subject_id)

    # New pipeline with all improvements
    new_correlation = run_new_pipeline(
        subject_id,
        use_improved_filtering=True,
        use_landmark_quality_filter=True,
        use_kinematic_constraints=True,
        use_kalman_filter=True,
        use_anatomical_refinement=True
    )

    improvement = new_correlation - old_correlation

    print(f"{subject_id}: {old_correlation:.3f} → {new_correlation:.3f} (+{improvement:.3f})")

    return improvement

# Run on all 26 subjects
improvements = [compare_pipelines(f"S1_{i:02d}") for i in range(1, 27)]
mean_improvement = np.mean(improvements)
print(f"Mean correlation improvement: +{mean_improvement:.3f}")
```

---

## 📈 EXPECTED CUMULATIVE IMPROVEMENTS

| Component | Expected Δr | Cumulative r |
|-----------|-------------|--------------|
| **Baseline (Current)** | - | 0.50 (worst cases) |
| + Improved Signal Processing | +0.12 | 0.62 |
| + Landmark Quality Filtering | +0.05 | 0.67 |
| + Kinematic Constraints | +0.15 | 0.82 |
| + Kalman Filtering | +0.08 | 0.90 |
| + Anatomical Refinement | +0.10 | **1.00** (theoretical max) |

**Realistic Target:** r > 0.80 for 90% of subjects (vs current ~40%)

---

## 🎯 NEXT STEPS (Prioritized)

### Immediate (This Session)
1. ✅ Create `improved_signal_processing.py` - DONE
2. ✅ Create `landmark_quality_filter.py` - DONE
3. 🔄 Create `kinematic_constraints.py` - **START HERE**
4. 🔄 Create `test_improved_pipeline.py` - Validate improvements

### Short Term (Next 1-2 Sessions)
5. Implement Kalman filtering
6. Implement anatomical landmark refinement
7. Run validation on all 26 subjects
8. Update QC report with new results

### Medium Term (Research Paper Update)
9. Compare old vs new correlation distributions (violin plots)
10. Re-run ICC analysis with improved data
11. Update Bland-Altman plots
12. Revise RESEARCH_PAPER_REVISED.md with new results
13. Add "Methods: Signal Processing Improvements" section

---

## 🔧 IMPLEMENTATION PRIORITY MATRIX

| Component | Impact | Effort | Priority Score |
|-----------|--------|--------|----------------|
| **Kinematic Constraints** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | **5.0** (DO FIRST) |
| **Anatomical Refinement** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | **3.75** (High value) |
| Improved Filtering | ⭐⭐⭐ | ⭐⭐ | 3.0 (Done) |
| Kalman Filter | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | 2.0 (Moderate) |
| Multi-Model Ensemble | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | 1.2 (Expensive) |
| Landmark QC | ⭐⭐ | ⭐⭐ | 2.0 (Done) |

---

## 📝 NOTES

### Why Kinematic Constraints Have Highest Impact:
1. **Directly addresses root cause:** Anatomically impossible poses from tracking errors
2. **Fast to implement:** Rule-based, no ML training needed
3. **Interpretable:** Clinicians understand biomechanical limits
4. **Proven in literature:** Used in Vicon, OptiTrack systems

### Why Not Neural Network Refinement?
- Requires large training dataset (we have N=26)
- Less interpretable than biomechanical constraints
- Computational cost higher
- **Save for Phase 3 if needed**

---

**Status:** 2/7 components complete (28%)
**Next Action:** Implement `kinematic_constraints.py`
