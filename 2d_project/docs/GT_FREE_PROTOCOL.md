# GT-Free Calibration Protocol

## Objective
To enable reliable, standalone gait analysis using 2D smartphone video without requiring concurrent Optical Motion Capture (OMCS) or external Ground Truth (GT) for calibration.

## Overview
The protocol consists of three phases:
1.  **Static Calibration:** Removes systematic sensor offsets (DC bias).
2.  **Functional Calibration:** Verifies anatomical reference frames and sign conventions.
3.  **Automated QC Gate:** Ensures signal quality before processing.

---

## Phase 1: Static Calibration (Standing)
**Duration:** 3-5 seconds
**Posture:** Upright standing, feet shoulder-width apart, facing the camera (Sagittal view measures from side).

### Procedure
1.  User initiates "Calibration Mode" in the app.
2.  Audio cue: "Stand still and look straight ahead."
3.  System records 3 seconds (approx. 90 frames).
4.  **Algorithm:**
    *   Detects keypoints (Hip, Knee, Ankle).
    *   Computes median Knee Flexion Angle over the window ($\theta_{stand}$).
    *   **Correction Factor:** $\delta = \theta_{stand} - 0^\circ$ (assuming neutral extension is 0°).
    *   Apply $\theta_{corrected}(t) = \theta_{raw}(t) - \delta$ to subsequent gait trials.

**Pass Criteria:** Standard deviation of knee angle during static phase < 2°.

---

## Phase 2: Functional Calibration
**Duration:** 10 seconds
**Movements:**
1.  **Squat (x2):**
    *   User performs two shallow squats.
    *   *System Check:* Verifies that Knee Flexion increases (Positive Slope). If slope is negative, invert the signal sign.
2.  **Toe Raise (x2):**
    *   User lifts heels while keeping toes on ground.
    *   *System Check:* Identifies distinct "Heel Rise" feature in trajectories to validate Ankle/Heel keypoint tracking.

---

## Phase 3: Automated Quality Control (QC)
Before analysis, the system evaluates the walking trial:

1.  **Periodicity Check:** Autocorrelation of Knee Angle must show a peak at lag 0.8-1.4s with prominence > 0.5.
2.  **Shape Consistency:** The extracted cycle template is compared to a general population prior (N=26 healthy dataset).
    *   DTW Distance < Threshold.
    *   Pearson Correlation > 0.6.

**Decision:**
*   **PASS:** Proceed to parameter extraction.
*   **FAIL:** Trigger "Reshoot" prompt.

## Protocol Validation
*   **Development:** Validated against Vicon (N=26).
*   **Deployment:** Bias correction reduces RMSE from 12° to <5°.
