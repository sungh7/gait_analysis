# Phase 2 Study Protocol: Calibration-Free Pathological Gait Screening

**Title:** Vision-Based Pathological Gait Screening via Smartphone Video: A Calibration-Free Full-Body Analysis

## 1. Objective
To develop and validate a **calibration-free** machine learning framework that detects and classifies pathological gait patterns (e.g., Hemiplegic, Parkinsonian) using **full-body kinematic features** extracted from uncalibrated smartphone video.

## 2. Rationale
*   **Limitation of Phase 1:** Phase 1 proved that accurate *absolute* joint angles require per-subject calibration (Ground Truth), which is a barrier to scalability.
*   **Opportunity:** Pathological gait is often characterized by **asymmetry**, **variability**, and **temporal disturbances**, which can be robustly measured without absolute calibration.
*   **Full-Body Approach:** Upper body features (arm swing, trunk sway) are critical biomarkers for conditions like Parkinson's Disease (reduced arm swing) and Ataxia (trunk instability), yet are often ignored in lower-limb-focused studies.

## 3. Methodology

### 3.1 Dataset
*   **Source:** GAVD (Gait Analysis Video Dataset) or similar clinical datasets.
*   **Cohorts:**
    1.  **Healthy Control (HC)**
    2.  **Hemiplegic Gait** (e.g., Stroke)
    3.  **Parkinsonian Gait** (e.g., PD)
    4.  **Neuropathic/Ataxic Gait**

### 3.2 Feature Engineering (The "Calibration-Free" Set)
Instead of absolute angles (which are prone to error), we extract **relative** and **temporal** features:

#### A. Temporal Parameters (Robust to camera angle)
*   **Cadence:** Steps per minute.
*   **Stance/Swing Ratio:** Proportion of gait cycle spent in stance phase.
*   **Step Time Asymmetry:** $|Time_{Left} - Time_{Right}| / Time_{Mean}$.

#### B. Full-Body Asymmetry Indices (Symmetry is independent of calibration)
*   **Arm Swing Asymmetry:** $|ROM_{LeftArm} - ROM_{RightArm}|$.
*   **Step Length Ratio:** $StepLen_{Left} / StepLen_{Right}$.
*   **Single Limb Support Asymmetry.**

#### C. Variability Metrics (Coefficient of Variation - CV)
*   **Stride Time Variability:** $SD(StrideTime) / Mean(StrideTime)$.
*   **Trunk Sway Variability:** Variability in lateral trunk displacement.

#### D. Kinematic Synergies
*   **Arm-Leg Coordination:** Phase relationship between contralateral arm and leg movement.

### 3.3 Machine Learning Pipeline
1.  **Input:** Vector of ~15-20 calibration-free features per subject.
2.  **Classifier:** Random Forest / XGBoost (for interpretability) or LSTM (for raw sequence modeling).
3.  **Output:** Probability of Pathology (Normal vs. Abnormal) or Multi-class Classification.

## 4. Proposed Paper Structure

### Introduction
*   The need for accessible screening (Telehealth, Elderly care).
*   Limitations of current "angle-based" approaches (calibration dependency).
*   The diagnostic value of full-body features (Arm swing, Trunk).

### Methods
*   **Data Processing:** MediaPipe Pose extraction (Uncalibrated).
*   **Feature Extraction:** Definition of the "Calibration-Free Feature Set".
*   **Model Training:** Cross-validation setup.

### Results (Expected)
*   **Classification Accuracy:** High accuracy (>90%) in distinguishing Pathological vs. Healthy.
*   **Feature Importance:** Ranking which features matter most (e.g., "Arm Swing Asymmetry was the #1 predictor for Parkinson's").
*   **Robustness:** Showing that performance holds even with different camera angles (since features are relative).

### Discussion
*   **Clinical Utility:** A "Red Flag" system for primary care or home monitoring.
*   **Advantage:** No Vicon, no calibration, just video.

## 5. Next Steps
1.  [ ] **Data Audit:** Confirm availability of pathological gait videos (GAVD or internal data).
2.  [ ] **Feature Scripting:** Write `extract_fullbody_features.py` to calculate asymmetry/variability.
3.  [ ] **Pilot Model:** Train a simple classifier to test feasibility.

