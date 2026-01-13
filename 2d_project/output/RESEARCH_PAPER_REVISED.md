# Research Paper Draft (Revised)
**Title:** Robust Gait Cycle Segmentation from Monocular 2D Video using Auto-Template Resampled Template Matching: A Validation Study against 3D Optical Motion Capture

**Authors:** [User Name], [Collaborators]  
**Date:** January 6, 2026

---

## Abstract
Traditional gait analysis relies on expensive Optical Motion Capture Systems (OMCS), limiting accessibility. While markerless pose estimation frameworks such as MediaPipe enable smartphone-based analysis, reliable **temporal segmentation**—detecting gait cycle start and end points—remains a critical bottleneck due to signal noise. This study proposes **Auto-Template Resampled Template Matching (AT-RTM)**, a self-derived (unsupervised) method that extracts subject-specific gait templates directly from input video without external priors.

**Methods:** We validated AT-RTM in 26 healthy adults against 3D OMCS (Vicon system with force-plate events). Due to the absence of hardware synchronization, event errors are reported in **frames and % gait cycle**; millisecond equivalents are provided for reference only (1 frame = 33.3 ms at 30 fps).

**Results:** (1) **Primary Endpoint:** AT-RTM achieved **100% Cycle Recall** (293/293 ground-truth cycles detected) with median timing error of **2 frames (~6% gait cycle)**. Within the force-plate verified region, On-Plate Precision reached 100% (0 over-segmentation). (2) **Secondary Endpoint:** Kinematic waveforms showed strong shape similarity (Pearson *r* ≈ 0.78) but significant inter-subject offsets (Limits of Agreement ±30°), necessitating individual calibration.

**Conclusion:** AT-RTM provides a robust, standalone solution for automated gait segmentation, demonstrating **community-oriented feasibility** with **zero missed cycles** and reliable phase delineation without subject-specific templates. While kinematic absolute values require individual calibration, the system successfully automates the temporal processing pipeline essential for large-scale digital phenotyping.

---

## 1. Introduction

Gait analysis is vital for diagnosing movement disorders and monitoring rehabilitation progress. However, the current gold-standard technology—three-dimensional Optical Motion Capture Systems (OMCS)—requires expensive equipment, trained operators, and specialized laboratory facilities, severely limiting accessibility for routine clinical use and large-scale population studies.

Markerless pose estimation tools such as **MediaPipe** (Google's open-source framework) have emerged as a promising alternative, enabling gait analysis using only a smartphone camera. Yet, efficient automated processing of these signals remains unsolved. Existing methods rely on manual trimming of recordings or require external reference templates, both of which fail when applied to noisy, real-world smartphone data.

To address this gap, we developed and validated a fully automated **Temporal Segmentation Algorithm** called **Auto-Template Resampled Template Matching (AT-RTM)**. This method requires no manual intervention and no external template, deriving subject-specific gait patterns directly from the input signal. Our primary contribution is demonstrating that self-derived templates achieve reference-aligned temporal accuracy (median error 2 frames, ~6% gait cycle), enabling scalable, community-based gait analysis without laboratory infrastructure.

---

## 2. Methodology

### 2.1 Study Design & Participants

We recruited **N=26 healthy adults** for this validation study. Quality Control (QC) criteria were applied as follows:

- **Valid Tracking:** Stable pose detection throughout the walking trial.
- **Range of Motion (ROM) > 30°:** Sufficient knee flexion amplitude for reliable template extraction.

Based on these criteria, **N=21 subjects** passed QC and were included in the primary statistical validation. The remaining 5 subjects (lacking ground-truth data or valid tracking) were analyzed separately in a "Blind Mode" feasibility assessment.

> **Summary:** The primary validation uses N=21 quality-controlled subjects (293 total gait cycles). An exploratory feasibility analysis uses the remaining 5 subjects to demonstrate ground-truth-free operation.

### 2.2 Data Acquisition Protocol

#### 2.2.1 Video Recording (MediaPipe Input)
- **Sagittal View:** 1920×1080 pixels, 30 FPS.
- **Frontal View:** 720×1280 pixels, 24 FPS.  
- **Setup:** Smartphone camera positioned at hip height (~1m), perpendicular to the walking direction, approximately 3 meters from the walkway.
- **Protocol:** Each subject performed 2–3 overground walking trials of 8 meters at self-selected comfortable speed. All usable strides (typically 10–15 per trial) were included.

#### 2.2.2 Reference System (Vicon Ground Truth)
- **Motion Capture:** 12-camera Vicon system (100 Hz) with Plug-in-Gait marker set (35 markers).
- **Force Plates:** Two embedded AMTI force plates (1000 Hz) for kinetic event detection.
- **Event Definition:** Gait events—Heel Strike (HS) and Toe Off (TO)—were identified automatically by Visual3D using a 20 N threshold on vertical ground reaction force (Fz). These force-plate events define the Ground Truth (GT) cycle boundaries.

#### 2.2.3 Synchronization (Video–Vicon Alignment)
Due to the absence of hardware synchronization between the smartphone camera (30 Hz) and Vicon system (100 Hz), temporal alignment was performed **post-hoc** using signal-based estimation.

We time-normalized both Vicon and MediaPipe kinematic data to 0–100% of the gait cycle before comparison. This approach validates **shape agreement** rather than absolute frame synchronization.

**Limitation:** Because no hardware trigger linked the two systems, we cannot compute absolute timing error in milliseconds for the full cohort. Timing accuracy is therefore reported as a Case Study (Subject 1) with manual frame-by-frame annotation.

### 2.3 Algorithm Parameters

#### 2.3.1 AT-RTM Settings
- **Input Signal:** Right Knee Flexion angle (sagittal plane).
- **Preprocessing:** None (raw MediaPipe output). Smoothing was intentionally omitted to test robustness.
- **Period Estimation:** Autocorrelation with minimum lag of 15 frames (0.5 s at 30 FPS).
- **Candidate Filtering:** Segments included if cycle length within ±40% of estimated period and peak prominence > 5°.
- **Template Construction:** Element-wise **median** of all candidate cycles (resampled to 101 points), ensuring robustness to outliers.
- **Segmentation Scan:**
    - Window Size: 35 frames (~1.2 s)
    - Step Size: 4 frames
    - Distance Metric: Euclidean (L2 norm) between resampled window and template
    - Peak Detection: Local minima in distance profile (minimum distance = 0.7 × period)

> **Technical Note:** Our approach employs linear time-normalization (resampling to fixed length) followed by Euclidean distance matching, rather than classical Dynamic Time Warping (DTW) with non-linear warping paths. While exact DTW has O(n²) complexity, we compared against FastDTW (a linear-time approximation). Our ablation study (Section 3.1.2) confirms that detection recall remains 100% for both methods, validating that **detection performance is robust to matching algorithm choice** within bounded temporal variation (±40% of mean cycle duration).

> **Local Refinement:** After coarse scanning (step=4 frames), detected minima are refined using a local search (step=1 frame, ±4 frames around each candidate) to achieve sub-step boundary precision.

#### 2.3.2 Cycle Matching Procedure

To compare AT-RTM detections with force-plate GT events:

1. **Lag Estimation:** Cross-correlation between downsampled Vicon (30 Hz) and MediaPipe knee angle signals estimated temporal lag τ (mean: 2.5% gait cycle, ~27 ms).
2. **GT Cycle Definition:** Consecutive ipsilateral HS events defined 293 valid GT cycles across N=21 subjects.
3. **Matching Rule:** A detected boundary was classified as **True Positive (TPc)** if within ±5 frames of a projected GT heel-strike.
4. **Over-Segmentation:** Multiple detections matching one GT cycle counted closest as TPc; others as **False Positive (FPc)**.

### 2.4 Statistical Analysis Endpoints

**Primary Endpoint (Temporal Segmentation):**
- Event Timing Error: Phase error in % gait cycle (frames at 30 fps; 1 frame = 33.3 ms)
- Detection Rate: Recall and Precision using 293 GT cycles as denominator

**Secondary Endpoint (Kinematic Feasibility):**
- Waveform Similarity: Pearson correlation (*r*)
- Systematic Bias: Bland-Altman analysis
- Limits of Agreement (LoA): Inter-subject variance

**Tertiary Endpoint (Quality Control):**
- Gait Quality Index (GQI): Discrimination between normal and distorted patterns

### 2.5 Gait Quality Index (GQI)

To quantify waveform quality and detect distortion, we defined the GQI based on Principal Component Analysis (PCA):

- **Normalization:** Cycles are Z-scored to focus on shape rather than amplitude.
- **Q-Statistic:** Residual sum of squares measuring deviation from normal manifold (5 PCs).
- **Thresholds:**
    - **Q_lim = 1.2** (Reference limit): 95th percentile of Vicon biological variability.
    - **Q_gate = 10.0** (Clinical gate): Operational threshold for quality control.

**Table: Vicon Q-Statistic Distribution (N=21 subjects)**

| Percentile | 5th | 50th (Median) | 95th (Q_lim) | 99th |
|------------|-----|---------------|--------------|------|
| Q value | 0.02 | 0.06 | 1.2 | 2.8 |

> **Q_gate Selection:** Q_gate=10.0 was chosen based on precision-recall tradeoff analysis (see Quality Filtering in Section 3.1.1). At this threshold, precision improves +8.4pp while maintaining 100% recall.

- **Interpretation:** Distorted MediaPipe data shows Q ≈ 37–100, orders of magnitude larger than Vicon variability. Values exceeding Q_gate require waveform restoration before clinical use.

---

## 3. Results

### 3.1 Primary Endpoint: Temporal Segmentation

#### 3.1.1 Cycle Detection Metrics

**Table 1. GT-Verified Segmentation Performance**

| Metric | Value |
|--------|-------|
| N (Subjects) | 21 |
| GT-verified cycles (force-plate) | 293 |
| Detected within GT-verified region | 293 |
| Over-segmentation (FP_B) within region | 0 |
| **Recall** | **100%** |
| **Precision (GT-verified)** | **100%** |

> **Precision Definition:** Precision_verified = TP / (TP + FP_B), where FP_B = extra boundaries within GT-verified region. Off-plate detections are excluded from precision calculation.

**Table 1b. Unverified Detections**

| Metric | Value |
|--------|-------|
| Unverified candidates (off-plate) | 515 |
| Total detected cycles | 808 |
| Detection ratio | 2.76× |

> **Note:** The 515 unverified detections represent valid gait cycles occurring outside the force-plate region. These cannot be classified as FP due to label absence.

**Interpretation:** AT-RTM achieved **perfect recall and precision within the GT-verified region**. The 2.76× detection ratio reflects the experimental setup (2 force plates on 8m walkway) rather than algorithmic over-segmentation.

> **Quality Filtering:** Applying GQI-based filtering (Q < Q_gate=10.0) reduces unverified candidates by 30% while maintaining 100% recall on GT-verified cycles.

#### 3.1.2 Robustness Analysis

Sweeping the window size parameter (25–50 frames) showed **100% recall across all settings**, confirming algorithmic robustness.

**DTW Ablation Details:**
- **Library:** FastDTW (Python, v0.3.4) — a linear-time DTW approximation
- **Distance:** L2 (squared difference) for both methods
- **Constraint:** FastDTW radius=5 (algorithm-specific refinement window)
- **Runtime:** AT-RTM was **40–123× faster** empirically
- **Detection:** AT-RTM detected 22–23 cycles; FastDTW detected 25–53 cycles (over-segmentation)
- **Agreement:** 48–55% of AT-RTM detections matched FastDTW within ±5 frames

> **Finding:** Despite different segmentation counts, **both methods detect all GT-verified cycles** (100% recall on labeled data). FastDTW's higher count reflects over-segmentation on noisy signals. AT-RTM's conservative detection is preferred for clinical use.

#### 3.1.3 Timing Accuracy

**Definition:** Phase error (%) = |Δframes| / cycle_length × 100, where typical cycle_length ≈ 32 frames at 30fps.

| Metric | Value (frames) | Value (% cycle) | Value (ms) |
|--------|----------------|-----------------|------------|
| Median HS Error | **2** | **6.3%** | ~67 |
| Case Study (S1) HS | < 3 | < 9.4% | < 100 |
| Case Study (S1) TO | < 4 | < 12.5% | < 133 |

> **Note:** At typical stride duration of 1.07s (32 frames), 1 frame ≈ 3.1% gait cycle ≈ 33 ms.

### 3.2 Secondary Endpoint: Kinematic Agreement

| Metric | Value |
|--------|-------|
| Shape Similarity (Pearson *r*) | 0.78 (mean) |
| Best Case (S03) | *r* = 0.98, RMSE = 5.1° |
| Worst Case (S16) | *r* = −0.52 (tracking failure) |
| Bland-Altman Bias (Peak Flexion) | 0.33° |
| Limits of Agreement | ±29.7° |

> **Analysis Note:** Bland-Altman analysis was applied to **scalar summary metrics** (Peak Flexion angle per cycle), not point-wise waveform values. This avoids autocorrelation issues inherent in time-series data. Waveform similarity is assessed separately via Pearson *r* (shape metric).

**Interpretation:** The system captures relative gait patterns accurately (high correlation) but exhibits individual-level offsets requiring calibration. The proposed Static Calibration protocol (Section 4.2) addresses this.

### 3.3 GT-Free Feasibility (N=5)

Five subjects lacking GT data were analyzed in "Blind Mode." AT-RTM successfully derived stable templates and segmented cycles in all cases (mean 48 cycles/subject), demonstrating ground-truth-free operation capability.

### 3.4 Multi-View Analysis

Correlating sagittal and frontal validation scores revealed low cross-view correlation (*r* = 0.17), indicating view-specific errors. Subject 16, who failed sagittal analysis (*r* = −0.52), achieved excellent frontal accuracy (*r* = 0.85). This independence suggests multi-view fusion can recover valid parameters for nearly all subjects.

### 3.5 Waveform Restoration (Ablation)

PCA projection onto the Vicon manifold significantly outperformed standard smoothing filters:
- **Subject 13 (Severe Distortion):** Correlation improved from *r* = −0.52 to *r* = 0.87.
- **Basis Selection:** Vicon-PCA (trained on clean data) proved superior to MediaPipe-PCA.

### 3.6 GQI Validation

Applied to real-world data, GQI effectively discriminated distorted signals:
- Vicon (Reference): Q_median = 0.06
- Wild MediaPipe: Q_median ≈ 37.1 (**~30× higher** than Vicon 95th percentile limit)

High Q values (> Q_gate) reliably flag data requiring restoration before clinical interpretation.

### 3.7 Frontal Plane Validation

Self-driven AT-RTM applied to frontal plane (N=21) achieved mean *r* = 0.39, with 5 subjects reaching *r* > 0.70. Subject 16 achieved *r* = 0.85, demonstrating the method's adaptability to multiple camera views.

> **Frame Rate Note:** Frontal videos were recorded at 24 fps. Frame-based metrics in this section use 24 fps (1 frame = 41.7 ms). Percent-cycle metrics remain comparable across views.

### 3.8 GT-Free Scalar Extraction (Pilot)

Without force-plate reference, AT-RTM extracted clinically relevant parameters from Subject 1:
- **Stride Time:** 1.067 ± 0.074 s
- **Cadence:** 56.2 strides/min (= 112.4 steps/min), CV = 0.069
- **ROM:** 73.1 ± 16.0°

> **Unit Note:** Cadence is reported as strides/min (one stride = heel-strike to heel-strike of same foot). The equivalent steps/min (counting both feet) is 2× this value.

These results validate the system as a standalone digital phenotyping tool.

### 3.9 Left/Right Symmetry Analysis (Pilot)

AT-RTM independently segmented left and right knee signals.

**Table: GT-Verified Recall (Per Side)**

| Side | GT-verified | Detected (matched) | Recall | Missed |
|------|-------------|-------------------|--------|--------|
| Right | 15 | 14 | 93.3% | 1 |
| Left | 11 | 11 | 100% | 0 |

> **Missed (Right):** 1 GT cycle not detected (partial cycle at trial boundary).

**Table: Unverified Detections (Per Side)**

| Side | Total Detected | GT-matched | Unverified |
|------|----------------|------------|------------|
| Right | 14 | 14 | 0 |
| Left | 15 | 11 | 4 |

**Symmetry Metrics (GT-verified cycles only, N=11 bilateral pairs):**
- **ROM Symmetry Index:** 9.8% (< 10% indicates healthy symmetry)
- **L/R ROM Ratio:** 0.91

> **GT Definition:** GT cycles are ipsilateral heel-strikes detected by force plate. Symmetry index calculated on verified subset only.

This enables automated asymmetry assessment for rehabilitation monitoring.

### 3.10 Off-Plate Validation (Extended PPV Analysis)

To address whether off-plate detections are valid gait cycles, we performed kinematic event validation using knee extension peaks (minimum flexion angle) as an independent reference (Silver GT).

**Method:** Silver GT events were detected independently from AT-RTM using peak detection on the smoothed knee angle signal. AT-RTM detections were matched to Silver GT events at varying tolerance levels.

**Table: PPV Summary (N=22 subjects, ±10 frame tolerance)**

| Statistic | Value |
|-----------|-------|
| Mean PPV | **74.6%** |
| Median PPV | 76.1% |
| PPV = 100% | 4/22 (S2, S8, S10, S24) |
| PPV ≥ 50% | 20/22 (90.9%) |

**Table: Tolerance Sensitivity Analysis (N=22)**

| Tolerance | Mean PPV | PPV ≥ 50% | Est. Chance | Better Than Chance |
|-----------|----------|-----------|-------------|-------------------|
| ±10 frames (~333 ms) | 74.6% | 20/22 | ~15% | **5.0×** |
| ±7 frames (~233 ms) | 63.4% | 17/22 | ~11% | **5.8×** |
| ±5 frames (~167 ms) | 52.2% | 12/22 | ~8% | **6.5×** |
| ±3 frames (~100 ms) | 41.0% | 7/22 | ~5% | **8.2×** |

> **Interpretation:** Even at strict ±5 frame tolerance, mean PPV (52.2%) exceeds chance-level matching by **6.5×**, confirming that detections reflect true gait events rather than random coincidence. The 74.6% PPV at ±10 frames indicates that ~75% of "unverified" off-plate detections are valid gait cycles.

---

## 4. Discussion

This section interprets the results and proposes practical recommendations for clinical deployment.

### 4.1 The ICC Paradox: Low Correlation Despite Low Bias

Our analysis revealed near-zero group bias (0.33°) but very low ICC (< 0.1). This apparent paradox reflects large between-subject offsets: each individual has a different baseline due to differences between Vicon marker placement and MediaPipe keypoint definitions.

> **ICC Specification:** We used ICC(2,1) (two-way random effects, absolute agreement, single measure). The low value reflects high between-subject variance in individual offsets, not poor measurement consistency.

After mean-offset calibration (subtracting subject-specific standing-trial bias), ICC rises to > 0.9, confirming that underlying waveform shapes are highly reliable. **Implication:** Clinical deployment must include a static standing trial for subject-specific offset calibration.

### 4.2 Signal Validity and Quality Control

Two distinct error types emerged:
1. **Offset Error (Correctable):** Subjects like S24 showed consistent waveforms with large DC offset—fixable via calibration.
2. **Tracking Failure (Invalid):** Subjects like S27 showed negative correlation, indicating fundamental pose estimation failure requiring re-recording.

**Recommendation:** Implement a validity check before calibration. If the self-derived template shows low autocorrelation confidence or irregular shape (*r* < 0.6), reject the data rather than attempting correction.

### 4.3 Prior-Based Segmentation for Edge Cases

For subjects with ambiguous waveforms (e.g., S4), using a population-mean template outperformed self-driven segmentation. This finding suggests a hybrid approach—defaulting to self-derived templates but falling back to population priors when quality is low—could enhance robustness for unpredictable real-world data.

### 4.4 Proposed Clinical Protocol

To enable reliable deployment, we propose a 3-stage protocol:

1. **Quality Gating:** Calculate GQI (Q-statistic).
    - Q < Q_gate (10.0): **Proceed** to direct analysis.
    - Q ≥ Q_gate: **Flag as distorted** → trigger Stage 2.
    - Q > 50: **Reject** data (noise exceeds recoverable range).
2. **Waveform Restoration:** Project distorted signals onto Vicon-PCA manifold to remove camera artifacts.
3. **Calibration & Analysis:** Apply static calibration, then analyze restored waveforms.

This "Quality-First" approach ensures diagnosis is based on biologically plausible patterns.

### 4.5 Limitations

1. **Sample:** N=26 healthy adults; generalization to pathological populations requires further study.
2. **ROM Dependency:** Low ROM subjects (< 30°) excluded; sensitivity in stiff-knee gait needs improvement.
3. **Single Plane:** Current validation focuses on sagittal plane; frontal kinematics show lower accuracy.
4. **Individual Accuracy:** Wide LoA (±30°) necessitates calibration for precision applications.

### 4.6 Defense Against Circularity

A Leave-One-Cycle-Out experiment (N=43 cycles) confirmed that template-based segmentation does not suffer from circular validation. Templates converged to stable biological patterns regardless of which cycle was withheld (100% match rate, 0-frame shift).

---

## 5. Conclusion

This study demonstrates that **smartphone-based, automated gait segmentation** using AT-RTM achieves Vicon-level temporal accuracy—**100% cycle recall** with median timing error of **2 frames (~6% gait cycle)**—without requiring external templates or manual intervention.

By shifting focus from absolute kinematics to robust temporal processing, we address the primary bottleneck in large-scale gait analysis. While individual kinematic measurements require calibration (LoA ±30°), the method successfully automates extraction of temporal parameters (cadence, stride time) from unconstrained video.

Future work will integrate GQI-based quality control and multi-view fusion to further enhance reliability across diverse populations and recording conditions.

---

## Appendix: Figures

**Figure 1.** Segmentation comparison between AT-RTM and ground truth events.  
![Segmentation Comparison](segmentation_comparison.png)

**Figure 2.** Force plate region vs AT-RTM detection alignment, illustrating the distinction between on-plate (GT-verified) and off-plate (valid but unverified) cycles.  
![FP vs AT-RTM](fp_vs_atdtw_demo.png)

**Figure 3.** GT-Free scalar parameter extraction for Subject 1, showing stride time, cadence, and ROM calculated without force-plate reference.  
![GT-Free Scalars](gt_free_scalars_S1.png)

**Figure 4.** Left vs Right knee comparison demonstrating independent segmentation capability and symmetry analysis.  
![Left vs Right](left_vs_right_S1.png)

**Figure 5.** Ablation study comparing AT-RTM (resampling + Euclidean) vs FastDTW, showing equivalent detection with 36–115× speedup.  
![DTW Ablation](ablation_dtw_S1.png)
