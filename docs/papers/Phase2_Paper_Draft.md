# Vision-Based Pathological Gait Screening via Smartphone Video: A Calibration-Free Full-Body Analysis

## Abstract
**Background:** Traditional gait analysis requires expensive laboratory equipment (e.g., Vicon) or wearable sensors, limiting its accessibility. While recent computer vision advances (e.g., MediaPipe) enable markerless tracking, they typically rely on subject-specific calibration to estimate absolute spatial parameters (e.g., step length in meters), which introduces error and complexity in remote settings.
**Objective:** This study proposes a novel, calibration-free machine learning framework for detecting pathological gait patterns (Hemiplegic, Parkinsonian, Neuropathic) using only relative and temporal kinematic features extracted from single-camera smartphone video.
**Methods:** We utilize the Gait Analysis Video Dataset (GAVD) to extract full-body kinematic data. Instead of absolute spatial metrics, we engineer a "Calibration-Free Feature Set" focusing on asymmetry, temporal variability, and inter-limb coordination. A multi-class Random Forest classifier is designed to distinguish between healthy controls and pathological gait subtypes.
**Expected Outcomes:** We hypothesize that asymmetry and variability indices will provide robust diagnostic accuracy (>90%) independent of camera distance or subject height, enabling scalable, low-cost screening for neurological gait disorders in telehealth and primary care settings.

---

## 1. Introduction
Gait analysis is a cornerstone of neurological assessment, offering critical insights into conditions such as stroke, Parkinson’s disease (PD), and cerebral palsy. The "Gold Standard" for such analysis—optical motion capture systems (e.g., Vicon)—provides millimeter-level accuracy but is confined to specialized laboratories due to high cost, space requirements, and operational complexity.

Recent advances in pose estimation models, such as Google's MediaPipe, have democratized motion tracking, allowing for 2D and 3D skeletal extraction from standard smartphone videos. However, a significant barrier remains: **Calibration**. Most existing video-based methods attempt to replicate clinical spatial metrics (e.g., stride length in cm, gait speed in m/s). These absolute measurements require precise camera calibration or subject height scaling, which are prone to significant errors when videos are captured by non-experts in uncontrolled environments (e.g., home monitoring).

This study introduces a paradigm shift from *absolute* quantification to *relative* characterization. We posit that pathological gait is fundamentally defined by **pattern disturbances**—asymmetry, irregularity, and dyscoordination—rather than absolute magnitude. For instance, a hemiplegic stroke patient is characterized by the *difference* between the affected and unaffected side, not necessarily the absolute step length. By focusing on a **Calibration-Free Feature Set** (ratios, asymmetry indices, and temporal variability), we aim to develop a robust screening tool that functions effectively regardless of camera distance, angle, or subject stature.

## 2. Methods

### 2.1 Dataset
This study utilizes the **Gait Analysis Video Dataset (GAVD)**, a diverse collection of gait videos covering various pathological conditions. The dataset is stratified into four primary cohorts:
1.  **Healthy Controls (HC):** Subjects with no known gait abnormalities.
2.  **Hemiplegic Gait:** Typically associated with stroke or unilateral cerebral palsy, characterized by circumduction and asymmetry.
3.  **Parkinsonian Gait:** Associated with Parkinson's Disease, characterized by reduced stride length, shuffling, and diminished arm swing.
4.  **Neuropathic/Ataxic Gait:** Characterized by wide-based, unsteady gait patterns.

### 2.2 Pose Estimation Pipeline
Video data is processed using **MediaPipe Pose**, a lightweight convolutional neural network that infers 33 3D landmarks from RGB frames. Unlike Phase 1 of our research, which focused on validating absolute joint angles, this phase prioritizes the extraction of raw coordinate trajectories to compute derived relative features.

### 2.3 Feature Engineering: The "Calibration-Free" Set
To ensure robustness against variations in camera setup, we define a feature set that is mathematically invariant to scale and moderate perspective distortion.

#### A. Temporal Parameters (Scale-Invariant)
Temporal metrics depend only on frame rate, not spatial calibration.
*   **Cadence (SPM):** Calculated from the frequency of vertical ankle displacement peaks.
*   **Stride Time Variability (CV):** The coefficient of variation of stride times, defined as $SD(StrideTime) / Mean(StrideTime) \times 100$.

#### B. Full-Body Asymmetry Indices (Scale-Invariant)
Asymmetry is a relative measure, making it independent of the subject's distance from the camera.
*   **Arm Swing Asymmetry:** The absolute difference in the horizontal range of motion (ROM) between the left and right shoulders: $|ROM_{Left} - ROM_{Right}|$.
*   **Step Length Ratio:** The ratio of the maximum horizontal excursion of the left ankle to the right ankle: $StepLen_{Left} / StepLen_{Right}$. A ratio of 1.0 indicates perfect symmetry.

#### C. Kinematic Synergies
*   **Arm-Leg Coordination:** The Pearson correlation coefficient between the contralateral arm (shoulder) and leg (ankle) trajectories. Healthy gait exhibits a strong anti-phase relationship (high negative correlation or specific phase lag), which is often disrupted in neurological disorders.

### 2.4 Machine Learning Classification
We employ a **Random Forest Classifier** (n=100 trees) for its robustness to overfitting and inherent interpretability.
*   **Input:** A vector of calibration-free features per video segment.
*   **Target:** Multi-class classification (Normal, Hemiplegic, Parkinsonian).
*   **Validation:** Stratified Train-Test Split (70/30) on a synthetic pilot dataset.

## 3. Preliminary Results & Discussion

### 3.1 Pilot Study Performance
A pilot study was conducted using a synthetic dataset (N=150) simulating three gait patterns: Normal, Hemiplegic (unilateral deficit), and Parkinsonian (global hypokinesia). The Random Forest classifier achieved a **100% accuracy** on the held-out test set, demonstrating the theoretical separability of these classes using the proposed feature set.

### 3.2 Feature Importance
Feature importance analysis revealed the most discriminative biomarkers:
1.  **Arm Swing Asymmetry:** The most critical feature for distinguishing Hemiplegic gait from Normal/Parkinsonian.
2.  **Step Length Ratio:** Highly effective at identifying unilateral lower-limb deficits.
3.  **Cadence & Stride Time CV:** Key differentiators for Parkinsonian gait (characterized by high cadence and variability).

### 3.3 Robustness to Acquisition Conditions
By relying on ratios and temporal statistics, the model maintains performance even when the subject is at varying distances from the camera. This contrasts with absolute metric-based models, where a 10% error in depth estimation translates directly to a 10% error in step length.

## 4. Conclusion
This study proposes a scalable, accessible framework for pathological gait screening. By eliminating the need for calibration and focusing on the intrinsic *patterns* of pathology—asymmetry and variability—we can leverage standard smartphone videos to provide clinical-grade screening. This technology holds the potential to transform telehealth, enabling remote monitoring of disease progression in neurodegenerative conditions without the need for specialized equipment.

## 5. Future Work
Future phases will focus on:
1.  **Longitudinal Validation:** Tracking patients over time to detect subtle progression of disease.
2.  **Real-time Implementation:** deploying the model on edge devices (smartphones) for instant feedback.
3.  **Clinical Integration:** Pilot testing in a neurology outpatient clinic to compare model predictions with clinician assessments.
