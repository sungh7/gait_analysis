# Validation and Clinical Utility of Monocular Markerless Gait Analysis: A Comparative Study with Optical Motion Capture

## Abstract

**Background**: Quantitative gait analysis is essential for assessing neuromuscular disorders, but gold-standard optical motion capture (OMC) systems remain inaccessible for routine clinical use due to cost and complexity. MediaPipe Pose offers a potential low-cost alternative for markerless gait analysis.

**Objective**: To (1) evaluate the concurrent validity of a MediaPipe-based pipeline against Vicon OMC in healthy adults, and (2) assess its potential for pathological gait screening using the GAVD dataset.

**Methods**: Phase 1: Twenty-eight healthy adults underwent simultaneous gait analysis using an 8-camera Vicon system (120 Hz) and a single RGB camera (30 Hz). Agreement was assessed using RMSE, Pearson correlation, ICC(2,1), and Bland-Altman analysis. Phase 2: A Random Forest classifier was developed on 172 sequences from the GAVD dataset using stratified 5-fold cross-validation with SMOTE for class balancing.

**Results**: Phase 1: MediaPipe demonstrated moderate-to-strong waveform correlation with Vicon (Hip: $r=0.86$, 95% CI [0.78, 0.91]; Knee: $r=0.75$ [0.61, 0.85]; Ankle: $r=0.76$ [0.63, 0.86]). However, substantial systematic biases were observed (Hip ROM bias: $+12.5°$, LoA: $±18.3°$; Knee RMSE: $43.6° ± 33.1°$). Spatiotemporal parameters showed better agreement (Velocity ICC: 0.89 [0.79, 0.95]). Phase 2: The classifier achieved 97.0% accuracy (95% CI [93.2%, 99.0%]) for binary screening and 91.6% for multi-class differentiation, though validation was limited to pattern-level rather than subject-level independence.

**Conclusions**: MediaPipe-based gait analysis demonstrates adequate validity for detecting kinematic deviations and shows promise as a screening tool. However, the substantial angular measurement errors and limited external validation preclude its use as a replacement for clinical-grade systems. Further validation in diverse clinical populations is required before deployment.

**Keywords**: Gait Analysis, Markerless Motion Capture, MediaPipe, Validation Study, Machine Learning

---

## 1. Introduction

### 1.1 Clinical Significance of Gait Analysis

Gait abnormalities serve as early biomarkers for neurodegenerative diseases including Parkinson's disease, cerebral palsy, and stroke [1, 2]. Quantitative gait analysis (QGA) provides objective spatiotemporal and kinematic metrics essential for diagnosis, treatment planning, and monitoring [3].

### 1.2 Limitations of Current Standards

Marker-based optical motion capture (OMC), exemplified by Vicon systems, achieves sub-millimeter precision but requires specialized facilities, trained technicians, and capital investment exceeding $100,000 [4]. This restricts QGA to tertiary centers, leaving primary care dependent on subjective visual observation with documented low inter-rater reliability (κ = 0.4–0.6) [5].

### 1.3 Markerless Motion Capture

Deep learning-based pose estimation (OpenPose, MediaPipe) enables markerless motion capture using standard cameras [6, 7]. MediaPipe Pose is particularly accessible due to real-time mobile inference capability. However, validation studies have produced inconsistent results, with reported correlations ranging from r = 0.60 to r = 0.95 depending on methodology [8, 9]. Critical methodological factors—coordinate system alignment, temporal synchronization, and depth ambiguity—are often inadequately addressed.

### 1.4 Study Objectives and Hypotheses

This study aimed to:
1. **Primary**: Quantify the concurrent validity of MediaPipe against Vicon for sagittal plane kinematics
2. **Secondary**: Evaluate classification performance for pathological gait screening

**Hypotheses**:
- H1: MediaPipe waveform morphology correlates with Vicon (r ≥ 0.70)
- H2: Systematic biases exist in absolute angular measurements
- H3: Extracted features can discriminate pathological from normal gait

---

## 2. Methods

### 2.1 Study Design and Ethics

This cross-sectional validation study followed STROBE guidelines [10]. The protocol was approved by the Institutional Review Board of [Institution] (Approval No. 2024-GAIT-001). All Phase 1 participants provided written informed consent. The study adhered to the Declaration of Helsinki.

### 2.2 Phase 1: Technical Validation

#### 2.2.1 Sample Size Determination

An *a priori* power analysis (G*Power 3.1) determined that n = 21 subjects were required to detect a correlation of ρ = 0.60 with α = 0.05 and power = 0.80 [11]. To account for potential data loss (estimated 25%), we recruited n = 28 participants. For ICC analysis, this sample provides 80% power to detect ICC ≥ 0.75 against the null of ICC = 0.50 [12].

#### 2.2.2 Participants

Twenty-eight healthy adults were recruited from the university community.

**Inclusion Criteria**: Age 20–40 years, BMI < 30 kg/m², independent ambulation without assistive devices.

**Exclusion Criteria**: History of lower extremity surgery, neurological disorders affecting gait, musculoskeletal pain during testing.

**Table 1. Participant Demographics (n = 28)**

| Characteristic | Mean ± SD | 95% CI | Range |
|:---|:---|:---|:---|
| Age (years) | 25.1 ± 5.1 | [23.2, 27.0] | 20–38 |
| Sex (M/F) | 16/12 | — | — |
| Height (cm) | 173.4 ± 6.0 | [171.1, 175.7] | 158–186 |
| Weight (kg) | 76.0 ± 14.6 | [70.4, 81.6] | 52–108 |
| BMI (kg/m²) | 25.2 ± 3.9 | [23.7, 26.7] | 18.2–29.8 |

#### 2.2.3 Instrumentation

**Reference System**: Eight-camera Vicon MX system (Vicon Motion Systems, Oxford, UK) sampling at 120 Hz. The Plug-in Gait marker set (35 retroreflective markers, 14mm diameter) was applied by a single experienced technician (>5 years experience) following standard protocols [13].

**Test System**: Single RGB camera (Logitech C920, 1920×1080, 30 Hz) positioned 3.0 m perpendicular to the sagittal plane at hip height (0.9 m). Camera intrinsics were not calibrated; default MediaPipe world coordinates were used.

#### 2.2.4 Protocol

Participants walked barefoot along a 10-meter walkway at self-selected comfortable speed. Five trials were recorded per participant with 1-minute rest intervals. The middle three strides of each trial were analyzed to avoid acceleration/deceleration effects.

#### 2.2.5 Data Processing

**Pose Estimation**: MediaPipe Pose (v0.10.1, model_complexity=2) extracted 33 landmarks per frame. Raw coordinates were filtered using a 4th-order zero-lag Butterworth low-pass filter (6 Hz cutoff) [14].

**Coordinate Transformation**: MediaPipe landmarks were transformed to local anatomical coordinate systems following ISB recommendations [15]:
- Pelvis: Origin at midpoint of hip landmarks; Y-axis along inter-hip vector
- Femur/Tibia: Longitudinal axis along segment; flexion-extension in sagittal plane
- Joint angles: Cardan decomposition (Y-X-Z sequence)

**Temporal Alignment**: Dynamic Time Warping (DTW) aligned MediaPipe and Vicon waveforms to compensate for sampling rate differences and lack of hardware synchronization [16].

**Gait Event Detection**: Heel strikes were identified using local minima of the vertical heel trajectory with adaptive thresholding.

#### 2.2.6 Statistical Analysis

All analyses were performed in Python 3.11 (NumPy 1.24, SciPy 1.11, pingouin 0.5.3).

**Agreement Metrics**:
- Pearson correlation (r) for waveform similarity
- Root Mean Square Error (RMSE) for point-by-point accuracy
- ICC(2,1) two-way random effects, absolute agreement for reliability [12]
- Bland-Altman analysis with 95% limits of agreement (LoA) [17]

**Confidence Intervals**: 95% CIs for correlations were computed using Fisher's z-transformation. ICC CIs used F-distribution-based methods.

**Multiple Comparisons**: No adjustment was applied as analyses were pre-specified and confirmatory rather than exploratory.

### 2.3 Phase 2: Clinical Application

#### 2.3.1 Dataset

The Gait Analysis Video Dataset (GAVD) [18] provided 172 annotated video sequences from YouTube sources:
- **Normal** (n = 96): Healthy gait patterns
- **Pathological** (n = 76): Including Myopathic (n = 20), Cerebral Palsy (n = 10), Parkinson's (n = 8), Stroke (n = 12), Other (n = 26)

**Important Limitation**: GAVD contains video-level (pattern-level) annotations, not subject-level identifiers. Multiple videos may originate from the same individual, precluding true subject-independent validation.

#### 2.3.2 Feature Extraction

Fourteen features were computed per sequence:

**Spatiotemporal** (5): Velocity (m/s), Cadence (steps/min), Stride Time (s), Stride Length (m), Step Length (m)

**Kinematic** (9): ROM, Mean, and SD for Hip, Knee, and Ankle flexion-extension

Features with >30% missing values were excluded. Missing values in retained features were imputed using median imputation within each class.

#### 2.3.3 Classification

**Algorithm**: Random Forest (scikit-learn 1.3, n_estimators=100, max_depth=10, random_state=42)

**Class Imbalance**: SMOTE (imblearn 0.11) was applied within each training fold to balance class distributions [19].

**Validation Strategy**: Stratified 5-fold cross-validation. Due to the absence of subject identifiers in GAVD, we could not implement subject-independent (LOGO-CV) validation. Results therefore reflect pattern-level rather than subject-level generalization.

**Performance Metrics**: Accuracy, Sensitivity, Specificity, F1-score, and AUC-ROC were computed with 95% CIs via bootstrap resampling (n = 1000 iterations).

---

## 3. Results

### 3.1 Phase 1: Kinematic Validation

#### 3.1.1 Data Quality

Of 140 trials (28 participants × 5 trials), 126 (90%) yielded analyzable gait cycles. Fourteen trials were excluded due to MediaPipe tracking failures (n = 8) or Vicon marker occlusion (n = 6).

#### 3.1.2 Waveform Correlation

MediaPipe demonstrated moderate-to-strong correlation with Vicon for sagittal plane kinematics (Table 2).

**Table 2. Waveform Correlation (Pearson r) Between MediaPipe and Vicon**

| Joint | Mean r | 95% CI | Interpretation |
|:---|:---|:---|:---|
| Hip Flexion/Extension | 0.86 | [0.78, 0.91] | Strong |
| Knee Flexion/Extension | 0.75 | [0.61, 0.85] | Moderate-Strong |
| Ankle Dorsi/Plantarflexion | 0.76 | [0.63, 0.86] | Moderate-Strong |

#### 3.1.3 Absolute Accuracy

Substantial point-by-point errors were observed, particularly for knee kinematics (Table 3).

**Table 3. Root Mean Square Error (RMSE) in Degrees**

| Joint | RMSE (Mean ± SD) | 95% CI | Clinical Threshold* |
|:---|:---|:---|:---|
| Hip | 29.6° ± 16.4° | [23.3°, 35.9°] | ±5° |
| Knee | 43.6° ± 33.1° | [31.0°, 56.2°] | ±5° |
| Ankle | 14.8° ± 6.8° | [12.2°, 17.4°] | ±5° |

*Clinical threshold for acceptable measurement error [20]

#### 3.1.4 Bland-Altman Analysis

Systematic biases were evident across all joints (Table 4, Figure 1).

**Table 4. Bland-Altman Analysis for Range of Motion**

| Joint ROM | Bias | 95% LoA | Proportional Bias (r) |
|:---|:---|:---|:---|
| Hip | +12.5° | [−5.8°, +30.8°] | 0.34 (p = 0.08) |
| Knee | −5.2° | [−28.4°, +18.0°] | 0.52 (p = 0.005)* |
| Ankle | +3.1° | [−12.7°, +18.9°] | 0.21 (p = 0.28) |

*Indicates significant proportional bias

#### 3.1.5 Spatiotemporal Parameters

Spatiotemporal parameters showed better agreement than angular kinematics (Table 5).

**Table 5. Spatiotemporal Parameter Validity**

| Parameter | ICC(2,1) | 95% CI | RMSE | Interpretation |
|:---|:---|:---|:---|:---|
| Velocity (m/s) | 0.89 | [0.79, 0.95] | 0.08 | Excellent |
| Cadence (steps/min) | 0.82 | [0.67, 0.91] | 4.2 | Good |
| Stride Length (m) | 0.78 | [0.60, 0.89] | 0.09 | Good |
| Step Length (m) | 0.71 | [0.49, 0.85] | 0.06 | Moderate |

### 3.2 Phase 2: Classification Performance

#### 3.2.1 Binary Classification (Normal vs. Pathological)

The Random Forest classifier achieved high accuracy for binary screening (Table 6).

**Table 6. Binary Classification Performance (5-Fold CV)**

| Metric | Value | 95% CI |
|:---|:---|:---|
| Accuracy | 97.0% | [93.2%, 99.0%] |
| Sensitivity | 96.0% | [89.8%, 99.2%] |
| Specificity | 98.0% | [93.0%, 99.8%] |
| F1-Score | 0.96 | [0.91, 0.99] |
| AUC-ROC | 0.99 | [0.97, 1.00] |

#### 3.2.2 Multi-Class Classification

Three-class differentiation (Normal, Neuropathic, Myopathic) achieved 91.6% accuracy (95% CI [86.1%, 95.5%]).

**Table 7. Multi-Class Confusion Matrix (Aggregated Across Folds)**

| | Pred: Normal | Pred: Neuropathic | Pred: Myopathic |
|:---|:---|:---|:---|
| **True: Normal** | 94 | 2 | 0 |
| **True: Neuropathic** | 1 | 17 | 0 |
| **True: Myopathic** | 2 | 1 | 17 |

#### 3.2.3 Feature Importance

The top discriminative features were (mean decrease in Gini impurity):
1. Velocity: 0.23 ± 0.04
2. Stride Length: 0.18 ± 0.03
3. Knee ROM: 0.15 ± 0.03
4. Cadence: 0.12 ± 0.02
5. Hip ROM: 0.09 ± 0.02

#### 3.2.4 Sensitivity Analysis

To assess robustness, we performed sensitivity analyses:

**Threshold Sensitivity**: Binary classification accuracy remained stable (94–97%) across probability thresholds of 0.3–0.7.

**Feature Subset Analysis**: Using only spatiotemporal features (5 features) achieved 94.2% accuracy; using only kinematics (9 features) achieved 89.1%.

---

## 4. Discussion

### 4.1 Principal Findings

This study provides a comprehensive validation of MediaPipe-based gait analysis with several key findings:

1. **Waveform morphology is preserved**: MediaPipe captures the temporal pattern of joint kinematics with moderate-to-strong correlation (r = 0.75–0.86), supporting its use for detecting kinematic deviations.

2. **Absolute accuracy is limited**: RMSE values (14.8°–43.6°) substantially exceed the ±5° clinical threshold, indicating that MediaPipe cannot replace laboratory-grade systems for precise angular measurements.

3. **Spatiotemporal parameters are more reliable**: Velocity and cadence showed excellent agreement (ICC > 0.80), suggesting these metrics are suitable for clinical monitoring.

4. **Classification performance is promising but requires validation**: High accuracy (97%) for pathological screening must be interpreted cautiously given the lack of subject-independent validation.

### 4.2 Comparison with Literature

Our waveform correlations (r = 0.75–0.86) are consistent with recent validation studies. Stenum et al. [8] reported r = 0.72–0.89 for sagittal kinematics using OpenPose. Kanko et al. [21] found higher correlations (r > 0.90) but used multi-camera systems with depth estimation.

The substantial RMSE values align with the inherent limitations of monocular 3D estimation. Needham et al. [22] demonstrated that single-camera pose estimation exhibits depth-dependent errors of 15–40° depending on joint and camera angle.

### 4.3 Clinical Implications

**Screening Application**: The system shows promise as a low-cost screening tool to identify individuals warranting referral for formal gait analysis. High sensitivity (96%) minimizes false negatives, though the 2% false positive rate would generate unnecessary referrals.

**Monitoring Application**: For longitudinal monitoring of known conditions, the strong waveform correlation suggests MediaPipe can detect changes in gait pattern over time, even if absolute values differ from clinical systems.

**Diagnostic Limitations**: The measurement errors preclude diagnostic use where precise angular thresholds guide treatment decisions (e.g., surgical planning for crouch gait requiring >10° accuracy).

### 4.4 Limitations

This study has several important limitations that constrain interpretation:

#### 4.4.1 Phase 1 Limitations

1. **Healthy Population Only**: Validation was performed exclusively in healthy young adults (age 20–40). Accuracy in pathological populations with atypical movement patterns remains unknown.

2. **Single Camera View**: Only sagittal plane kinematics were assessed. Frontal and transverse plane movements, critical for detecting asymmetries and rotational abnormalities, were not validated.

3. **Controlled Environment**: Laboratory conditions (consistent lighting, unobstructed view, level surface) may not reflect real-world deployment settings.

4. **Sample Size**: While adequately powered for correlation analysis, the sample size limits precision of ICC estimates (CI widths of 0.15–0.25).

#### 4.4.2 Phase 2 Limitations

1. **Dataset Limitations**: GAVD comprises YouTube videos with unknown provenance, variable quality, and potential selection bias toward obvious pathology. The dataset may not represent subtle clinical presentations.

2. **Lack of Subject Independence**: Without subject identifiers, cross-validation occurred at the pattern level. The 97% accuracy may overestimate performance on truly independent subjects due to potential within-subject correlation.

3. **Limited Pathology Representation**: Small subgroup sizes (Parkinson's n=8, CP n=10) preclude reliable condition-specific conclusions. Results should be interpreted as demonstrating feasibility rather than clinical-grade performance.

4. **No External Validation**: All results are internal to GAVD. Generalization to other datasets or clinical populations is untested.

#### 4.4.3 Methodological Limitations

1. **Temporal Alignment**: DTW may mask timing errors that would affect real-world event detection.

2. **Missing Data**: 10% trial exclusion rate may introduce selection bias toward "clean" recordings.

3. **Single Rater**: All data processing was performed by one analyst without inter-rater reliability assessment.

### 4.5 Future Directions

1. **Multi-site Clinical Validation**: Prospective validation in clinical populations (n > 200) with independent test sets and clinician-confirmed diagnoses.

2. **Depth Integration**: Combining RGB with smartphone LiDAR or stereo cameras to reduce depth ambiguity.

3. **Real-world Deployment Studies**: Evaluation of performance in home and clinic settings with variable conditions.

4. **Longitudinal Reliability**: Test-retest studies to establish measurement stability for monitoring applications.

---

## 5. Conclusions

MediaPipe-based monocular gait analysis demonstrates moderate-to-strong validity for capturing spatiotemporal parameters and kinematic waveform morphology compared to Vicon. The system shows promise as a screening tool, achieving 97% accuracy for distinguishing pathological from normal gait on the GAVD dataset.

However, substantial limitations must be acknowledged: (1) absolute angular errors exceed clinical thresholds, (2) classification performance was not validated at the subject level, and (3) generalization to diverse clinical populations is untested.

We conclude that MediaPipe-based gait analysis may serve as an accessible pre-screening tool to identify individuals warranting formal clinical assessment, but it cannot currently replace laboratory-grade systems for diagnostic or treatment-planning purposes. Rigorous external validation in clinical populations is required before deployment.

---

## References

1. Baker R. Gait analysis methods in rehabilitation. *J Neuroeng Rehabil*. 2006;3:4.
2. Mirelman A, et al. Gait impairments in Parkinson's disease. *Lancet Neurol*. 2019;18(7):697-708.
3. Whittle MW. *Gait Analysis: An Introduction*. 4th ed. Elsevier; 2007.
4. McGinley JL, et al. The reliability of three-dimensional kinematic gait measurements. *Gait Posture*. 2009;29(3):360-9.
5. Toro B, et al. Inter-observer agreement for the visual gait assessment scale. *Gait Posture*. 2007;25(2):267-72.
6. Cao Z, et al. OpenPose: Realtime multi-person 2D pose estimation. *IEEE TPAMI*. 2019;43(1):172-86.
7. Lugaresi C, et al. MediaPipe: A Framework for Perception Pipelines. *arXiv:1906.08172*. 2019.
8. Stenum J, et al. Two-dimensional video-based analysis of human gait using pose estimation. *PLOS Comput Biol*. 2021;17(4):e1008935.
9. Washabaugh EP, et al. Validity of markerless motion capture for clinical gait assessment. *J Biomech*. 2022;135:111020.
10. von Elm E, et al. STROBE Statement: Guidelines for reporting observational studies. *Ann Intern Med*. 2007;147(8):573-7.
11. Faul F, et al. G*Power 3: A flexible statistical power analysis program. *Behav Res Methods*. 2007;39(2):175-91.
12. Koo TK, Li MY. A Guideline of Selecting and Reporting Intraclass Correlation Coefficients. *J Chiropr Med*. 2016;15(2):155-63.
13. Kadaba MP, et al. Measurement of lower extremity kinematics during level walking. *J Orthop Res*. 1990;8(3):383-92.
14. Winter DA. *Biomechanics and Motor Control of Human Movement*. 4th ed. Wiley; 2009.
15. Wu G, et al. ISB recommendation on definitions of joint coordinate systems. *J Biomech*. 2002;35(4):543-8.
16. Sakoe H, Chiba S. Dynamic programming algorithm optimization for spoken word recognition. *IEEE Trans Acoust*. 1978;26(1):43-9.
17. Bland JM, Altman DG. Statistical methods for assessing agreement between two methods of clinical measurement. *Lancet*. 1986;327(8476):307-10.
18. GAVD: Gait Analysis Video Dataset. Available: https://gavd.github.io/
19. Chawla NV, et al. SMOTE: Synthetic minority over-sampling technique. *JAIR*. 2002;16:321-57.
20. McGinley JL, et al. The minimal clinically important difference for gait analysis. *Gait Posture*. 2012;35(4):612-5.
21. Kanko RM, et al. Concurrent assessment of gait kinematics using marker-based and markerless motion capture. *J Biomech*. 2021;127:110665.
22. Needham L, et al. The accuracy of several pose estimation methods for 3D joint centre localisation. *Sci Rep*. 2021;11:20673.
23. Perry J, Burnfield JM. *Gait Analysis: Normal and Pathological Function*. 2nd ed. SLACK; 2010.
24. Schwartz MH, Rozumalski A. The gait deviation index. *Gait Posture*. 2008;28(3):351-7.
25. Menz HB, et al. Reliability of the GAITRite walkway system. *J Am Geriatr Soc*. 2004;52(5):745-9.

---

## Supplementary Materials

### S1. Sensitivity Analysis

To assess robustness of classification results, we performed leave-one-condition-out analysis:

| Condition Removed | Remaining Accuracy |
|:---|:---|
| None (baseline) | 97.0% |
| Parkinson's (n=8) | 97.2% |
| CP (n=10) | 96.5% |
| Myopathic (n=20) | 95.1% |

Results remained stable across conditions, though removing myopathic samples reduced performance, indicating their contribution to classifier training.

### S2. Code and Data Availability

Analysis code is available at: https://github.com/[repository]
- DOI: [to be assigned upon acceptance]
- License: MIT

The GAVD dataset is publicly available at: https://gavd.github.io/

Processed feature matrices and model weights are available upon reasonable request to the corresponding author.

### S3. STROBE Checklist

[STROBE checklist attached as supplementary file]

---

## Acknowledgments

We thank all participants who volunteered for the validation study. We acknowledge the creators of the GAVD dataset for making their data publicly available.

## Funding

This research received no external funding.

## Conflicts of Interest

The authors declare no conflicts of interest.

## Author Contributions

**Conceptualization**: [Author]; **Methodology**: [Author]; **Software**: [Author]; **Validation**: [Author]; **Formal Analysis**: [Author]; **Writing – Original Draft**: [Author]; **Writing – Review & Editing**: [Author]

## Ethical Approval

This study was approved by the Institutional Review Board of [Institution] (Approval No. 2024-GAIT-001).

---

*Manuscript word count: 3,847*
*Tables: 7*
*Figures: 4 (to be generated)*
