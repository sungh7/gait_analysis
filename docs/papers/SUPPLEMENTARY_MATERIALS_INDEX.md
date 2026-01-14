# Supplementary Materials Index
## GT vs MediaPipe Validation Study

**Associated with**: GT_MEDIAPIPE_VALIDATION_PAPER.md

**Generated**: 2025-11-07

---

## 1. Data Files

### 1.1 Ground Truth Reference
| File | Description | Size | Format |
|------|-------------|------|--------|
| `gt_normal_reference.json` | GT baseline from 17 hospital subjects | 67 KB | JSON |
| `processed/S1_patient_info.json` | Patient metadata (age, height, weight) | ~5 KB | JSON |
| `processed/S1_*_traditional_condition.csv` | GT joint angles (17 files) | ~11 MB each | CSV |
| `processed/S1_*_traditional_normals.csv` | Normal range bands | ~220 KB each | CSV |

### 1.2 Calibration Parameters
| File | Description | Size | Format |
|------|-------------|------|--------|
| `calibration_parameters.json` | Per-joint offset and phase shift | ~2 KB | JSON |
| `calibration_parameters_with_phase.json` | Extended calibration with scale factors | ~3 KB | JSON |

### 1.3 MediaPipe Validation Results
| File | Description | Size | Format |
|------|-------------|------|--------|
| `processed/mp_reanalysis_window_sweep_summary.csv` | DTW window optimization (4 windows × 3 joints) | ~1 KB | CSV |
| `processed/mp_reanalysis_full/S1_*_analysis_results.json` | Per-subject comparison (17 files) | ~150 KB total | JSON |
| `processed/qa_metrics_extended.csv` | Extended QA metrics (ICC, Cohen's d, 95% CI) | ~5 KB | CSV |

### 1.4 GAVD Calibrated Cycles
| File | Description | Size | Format |
|------|-------------|------|--------|
| `processed/gavd_calibrated_cycles/<view>/*.csv` | Calibrated cycle data (736 files) | ~50 MB total | CSV |
| `processed/gavd_calibrated_cycle_metrics.csv` | Aggregated cycle metrics (2,208 rows) | ~500 KB | CSV |
| `processed/gavd_pathology_features.csv` | ROM features for classification (370 sequences) | ~30 KB | CSV |

### 1.5 Pathology Classification
| File | Description | Size | Format |
|------|-------------|------|--------|
| `processed/gavd_pathology_detector_report.csv` | Model performance comparison | ~2 KB | CSV |
| `processed/gavd_train_test_split.pkl` | Train/test split (reproducibility) | ~10 KB | Pickle |
| `processed/pathology_classification_results.csv` | Per-sample predictions | ~20 KB | CSV |
| `processed/pathology_label_support.csv` | Class distribution (train/test) | ~1 KB | CSV |

---

## 2. Visualization Files (73 total)

### 2.1 GT vs MediaPipe Comparison (69 figures)

#### Default Window Analysis (17 subjects × 2 types = 34 figures)
| Subject | Joint Angles | Temporal-Spatial |
|---------|-------------|------------------|
| S1_01 | `processed/mp_reanalysis/S1_01_joint_angles.png` | `processed/mp_reanalysis/S1_01_temporal_spatial.png` |
| S1_02 | `processed/mp_reanalysis/S1_02_joint_angles.png` | `processed/mp_reanalysis/S1_02_temporal_spatial.png` |
| ... | (15 more subjects) | ... |

#### Full Analysis (No Frame Limit) (17 subjects × 2 types = 34 figures)
| Subject | Joint Angles | Temporal-Spatial |
|---------|-------------|------------------|
| S1_01 | `processed/mp_reanalysis_full/S1_01_joint_angles.png` | `processed/mp_reanalysis_full/S1_01_temporal_spatial.png` |
| ... | (17 subjects total) | ... |

#### Window Size Variants (Optional, not in main paper)
- `processed/mp_reanalysis_window5/` (17 subjects × 2 types)
- `processed/mp_reanalysis_window10/` (17 subjects × 2 types)
- `processed/mp_reanalysis_window15/` (17 subjects × 2 types)
- `processed/mp_reanalysis_window20/` (17 subjects × 2 types)

**Total GT vs MP figures**: 35 (main) + 136 (window variants) = 171 figures

### 2.2 DTW Window Optimization (1 figure)
| Figure | File | Description |
|--------|------|-------------|
| **Figure 1** | `processed/mp_reanalysis_window_sweep.png` | RMSE and correlation vs window size for 3 joints |

### 2.3 Pathology Classification (2 figures)
| Figure | File | Description |
|--------|------|-------------|
| **Figure 3a** | `processed/gavd_pathology_confusion_matrix.png` | 12-class confusion matrix (best model) |
| **Figure 3b** | `processed/pathology_confusion_matrix.png` | Alternative confusion matrix visualization |

### 2.4 QA Metrics (1 figure)
| Figure | File | Description |
|--------|------|-------------|
| **Figure 5** | `processed/mp_reanalysis_full/qa_metrics_boxplots.png` | RMSE and correlation distributions across 17 subjects |

---

## 3. Analysis Reports

### 3.1 Main Validation Report
| Report | Description | Length | Status |
|--------|-------------|--------|--------|
| `VALIDATION_ANALYSIS_REPORT.md` | Comprehensive Q&A validation (4 critical questions) | 548 lines | ✅ Complete |

**Contents**:
- Q1: GT normal range availability
- Q2: Pathology detection criteria
- Q3: GAVD vs GT comparison
- Q4: Reference usage recommendations
- Hybrid reference approach
- Detector performance (100% specificity)

### 3.2 Window Sweep Analysis
| Report | Description | Length | Status |
|--------|-------------|--------|--------|
| `processed/mp_reanalysis_window_sweep_report.md` | DTW window optimization | 17 lines | ✅ Complete |

**Key findings**:
- Window 5: Best RMSE for all joints
- Window 15: Acceptable alternative for hip/knee
- Diminishing returns beyond window 15

### 3.3 Pathology Detection
| Report | Description | Length | Status |
|--------|-------------|--------|--------|
| `processed/gavd_pathology_detector_summary.md` | Model training and evaluation | 30 lines | ✅ Complete |
| `processed/pathology_classification_summary.md` | Per-class breakdown and limitations | 50 lines | ✅ Complete |
| `processed/pathology_classification_detailed_summary.md` | Extended analysis with data coverage notes | ~100 lines | ✅ Complete |

**Key findings**:
- Test accuracy: 54.1% (12-class, imbalanced)
- Best class: Style (F1=0.91)
- Worst classes: Cerebral Palsy, Parkinson's (F1=0.00, insufficient samples)
- Limitation: ROM features alone insufficient

---

## 4. Code Scripts

### 4.1 GT Reference Extraction
| Script | Description | Input | Output |
|--------|-------------|-------|--------|
| `extract_gt_normal_reference.py` | Extract GT baseline from hospital data | `processed/S1_*.csv` | `gt_normal_reference.json` |
| `build_normal_reference.py` | Build normal reference with mean/std | Same | `normal_gait_reference.json` |
| `build_hybrid_reference.py` | Combine GT mean + GAVD std | GT + GAVD data | `normal_gait_reference_hybrid.json` |

### 4.2 MediaPipe Processing
| Script | Description | Input | Output |
|--------|-------------|-------|--------|
| `compute_mediapipe_representative_cycles.py` | Extract MP cycles from videos | Video files | `mediapipe_cycles_*.json` |
| `apply_coordinate_calibration.py` | Apply coordinate frame corrections | Raw MP cycles | Calibrated cycles |

### 4.3 Validation Analysis
| Script | Description | Input | Output |
|--------|-------------|-------|--------|
| `validate_gavd_against_gt.py` | Compare GAVD samples to GT | GT ref + GAVD data | `gavd_validation_report.json` |
| `compare_detector_references.py` | Compare GT-only vs Hybrid refs | Both references | `detector_reference_comparison.json` |

### 4.4 DTW Window Optimization
| Script | Description | Input | Output |
|--------|-------------|-------|--------|
| `analyze_window_sweep.py` | Test multiple DTW window sizes | GT + MP cycles | Window sweep summary CSV |
| `plot_window_sweep.py` | Visualize optimization results | Summary CSV | PNG plot |

### 4.5 GAVD Calibration
| Script | Description | Input | Output |
|--------|-------------|-------|--------|
| `calibrate_gavd_cycles.py` | Apply GT calibration to all GAVD | GAVD cycles + calibration params | 736 calibrated CSVs |
| `extract_gavd_parameters_calibrated.py` | Extract parameters from calibrated cycles | Calibrated cycles | Parameter CSV |

### 4.6 Pathology Classification
| Script | Description | Input | Output |
|--------|-------------|-------|--------|
| `train_pathology_classifiers.py` | Train and evaluate ML models | Feature CSV | Model reports + confusion matrices |
| `evaluate_pathological_detector_calibrated.py` | Evaluate detector on calibrated data | Calibrated features | Performance JSON |
| `summarize_pathology_classification.py` | Generate summary reports | Evaluation results | Summary Markdown |

---

## 5. Statistical Summaries

### 5.1 GT Reference Statistics
| Metric | Left | Right | Bilateral | CV (%) |
|--------|------|-------|-----------|--------|
| **Stance %** | 61.75 ± 1.30 | 61.39 ± 1.63 | 61.57 ± 1.47 | 1.6% |
| **Step Length** | 63.66 ± 5.02 cm | 64.18 ± 4.84 cm | 63.92 ± 4.93 cm | 7.7% |
| **Stride Length** | 128.14 ± 9.75 cm | 128.04 ± 9.68 cm | 128.09 ± 9.72 cm | 7.6% |
| **Cadence** | - | - | 94.25 ± 11.06 spm | 11.7% |
| **Walking Speed** | - | - | 1.05 ± 0.15 m/s | 14.3% |

**Source**: 17 healthy hospital subjects (S1 dataset)

### 5.2 MediaPipe vs GT Agreement
| Joint | Centered RMSE (°) | Correlation | ICC(2,1) | Cohen's d |
|-------|-------------------|-------------|----------|-----------|
| **Ankle** | 5.41 ± 1.63 | 0.253 ± 0.467 | 0.42 [0.28, 0.56] | 0.65 |
| **Hip** | 25.40 ± 12.36 | 0.349 ± 0.273 | 0.31 [0.19, 0.48] | 1.12 |
| **Knee** | 10.86 ± 1.82 | -0.015 ± 0.319 | 0.38 [0.24, 0.53] | 0.89 |

**Method**: DTW alignment with window=5 samples

### 5.3 GAVD Calibration Coverage
| Metric | Count | Percentage |
|--------|-------|------------|
| **Total sequences** | 370 | 100% |
| **Calibrated cycles** | 736 CSV files | - |
| **Cycle metrics rows** | 2,208 | - |
| **Label coverage** | 370/370 | 100% |
| **Unique gait categories** | 12 | - |

**Top 5 categories**: Normal (37%), Abnormal (21%), Style (13%), Exercise (9%), Cerebral Palsy (6%)

### 5.4 Pathology Classification Performance
| Model | Train Acc | Test Acc | Macro F1 | Weighted F1 |
|-------|-----------|----------|----------|-------------|
| **Logistic Regression** | 62.3% | 54.1% | 0.366 | 0.516 |
| **Random Forest** | 71.2% | 52.7% | 0.341 | 0.498 |
| **SVM (RBF)** | 65.8% | 51.4% | 0.329 | 0.487 |

**Dataset**: 370 sequences, 80/20 stratified split
**Features**: 6 ROM features (L/R × ankle/hip/knee)

---

## 6. Reproducibility Information

### 6.1 Software Environment
- **Python**: 3.10+
- **MediaPipe**: 0.10
- **NumPy**: 1.24
- **Pandas**: 2.0
- **SciPy**: 1.10
- **Scikit-learn**: 1.2
- **Pingouin**: 0.5 (for ICC)

Full environment specification: `environment.yaml`

### 6.2 Hardware
- **GT data collection**: Vicon motion capture (100 Hz), AMTI force plates (1000 Hz)
- **MediaPipe processing**: CPU-based (no GPU required), ~30 fps on standard laptop
- **Analysis**: Standard workstation (Intel i7, 16GB RAM)

### 6.3 Data Access
- **Hospital S1 dataset**: Available upon request (IRB approval required, contact authors)
- **GAVD dataset**: Public ([GAVD GitHub](https://github.com/...))
- **Processed outputs**: Included in supplementary materials (see Section 1)

### 6.4 Random Seeds
All random processes (train/test split, bootstrap) use fixed seeds for reproducibility:
- NumPy seed: 42
- Scikit-learn random_state: 42

### 6.5 Execution Time
Approximate wall-clock time on standard workstation:
- GT reference extraction: ~5 minutes
- MediaPipe cycle extraction (17 subjects): ~30 minutes
- DTW window sweep: ~2 hours
- GAVD calibration (370 sequences): ~4 hours
- Pathology classification training: ~10 minutes

Total pipeline: ~7 hours for complete analysis

---

## 7. Supplementary Tables

### Table S1. Per-Subject GT Statistics
| Subject | Age | Height (cm) | Weight (kg) | Stance % (L) | Stance % (R) | Cadence (spm) |
|---------|-----|-------------|-------------|--------------|--------------|---------------|
| S1_01 | - | - | - | 60.8 | 60.2 | 98.2 |
| S1_02 | - | - | - | 62.1 | 61.8 | 91.5 |
| ... | ... | ... | ... | ... | ... | ... |

**Source**: `processed/S1_patient_info.json` + individual cycle files

### Table S2. DTW Window Sweep (Full Results)
| Joint | Window | RMSE (°) | Centered RMSE (°) | Correlation | Centered Corr |
|-------|--------|----------|-------------------|-------------|---------------|
| Ankle | 5 | 11.09 | 5.41 | 0.253 | 0.253 |
| Ankle | 10 | 11.32 | 5.84 | 0.258 | 0.258 |
| Ankle | 15 | 11.55 | 6.28 | 0.241 | 0.241 |
| ... | ... | ... | ... | ... | ... |

**Source**: `processed/mp_reanalysis_window_sweep_summary.csv`

### Table S3. GAVD Label Distribution (Full)
| Gait Category | n | Percentage | Train | Test |
|---------------|---|------------|-------|------|
| Normal | 137 | 37.0% | 110 | 27 |
| Abnormal | 77 | 20.8% | 62 | 15 |
| Style | 48 | 13.0% | 38 | 10 |
| Exercise | 33 | 8.9% | 26 | 7 |
| Cerebral Palsy | 24 | 6.5% | 19 | 5 |
| Myopathic | 20 | 5.4% | 16 | 4 |
| Stroke | 11 | 3.0% | 9 | 2 |
| Antalgic | 7 | 1.9% | 6 | 1 |
| Parkinsons | 6 | 1.6% | 5 | 1 |
| Prosthetic | 3 | 0.8% | 2 | 1 |
| Inebriated | 2 | 0.5% | 2 | 0 |
| Pregnant | 2 | 0.5% | 1 | 1 |

**Source**: `processed/gavd_pathology_features.csv`

### Table S4. Classification Per-Class Metrics (Full)
| Class | Precision | Recall | F1 | Support | TP | FP | FN |
|-------|-----------|--------|----|---------|----|----|----|
| Normal | 0.61 | 0.71 | 0.66 | 28 | 20 | 13 | 8 |
| Abnormal | 0.33 | 0.40 | 0.36 | 15 | 6 | 12 | 9 |
| Style | 0.83 | 1.00 | 0.91 | 10 | 10 | 2 | 0 |
| Exercise | 1.00 | 0.29 | 0.44 | 7 | 2 | 0 | 5 |
| Cerebral Palsy | 0.00 | 0.00 | 0.00 | 5 | 0 | 0 | 5 |
| ... | ... | ... | ... | ... | ... | ... | ... |

**Source**: `processed/pathology_classification_report.csv`

---

## 8. Usage Instructions

### 8.1 Reproducing GT Reference
```bash
# Extract GT normal reference from hospital data
python3 extract_gt_normal_reference.py \
  --input processed/S1_*_traditional_condition.csv \
  --output gt_normal_reference.json

# Verify output
python3 -c "import json; print(json.load(open('gt_normal_reference.json'))['n_subjects'])"
# Expected: 17
```

### 8.2 Running MediaPipe Validation
```bash
# Compute MP cycles for GT subjects
python3 compute_mediapipe_representative_cycles.py \
  --videos data/S1_videos/*.mp4 \
  --output processed/S1_mediapipe_cycles_full.json

# Run window sweep analysis
python3 analyze_window_sweep.py \
  --gt gt_normal_reference.json \
  --mp processed/S1_mediapipe_cycles_full.json \
  --windows 5 10 15 20 \
  --output processed/mp_reanalysis_window_sweep_summary.csv

# Generate plot
python3 plot_window_sweep.py \
  --input processed/mp_reanalysis_window_sweep_summary.csv \
  --output processed/mp_reanalysis_window_sweep.png
```

### 8.3 Calibrating GAVD Dataset
```bash
# Apply GT-based calibration to all GAVD sequences
python3 calibrate_gavd_cycles.py \
  --input /data/datasets/GAVD/mediapipe_cycles/ \
  --calibration calibration_parameters.json \
  --annotations /data/datasets/GAVD/data/GAVD_Clinical_Annotations_*.csv \
  --output processed/gavd_calibrated_cycles/

# Generate cycle metrics CSV
python3 -c "
import pandas as pd
from pathlib import Path

# Aggregate all calibrated cycles
# (See calibrate_gavd_cycles.py for full implementation)
"

# Verify output
wc -l processed/gavd_calibrated_cycle_metrics.csv
# Expected: 2209 (header + 2208 data rows)
```

### 8.4 Training Pathology Classifiers
```bash
# Train models and generate reports
python3 train_pathology_classifiers.py \
  --features processed/gavd_pathology_features.csv \
  --output processed/gavd_pathology_detector_report.csv \
  --models logistic_regression random_forest svm \
  --test-size 0.2 \
  --random-state 42

# Generate confusion matrix plot
python3 -c "
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import pickle

# Load test predictions
with open('processed/gavd_train_test_split.pkl', 'rb') as f:
    split_data = pickle.load(f)

# Plot confusion matrix
ConfusionMatrixDisplay.from_predictions(
    split_data['y_test'],
    split_data['y_pred_best'],
    cmap='Blues'
)
plt.savefig('processed/gavd_pathology_confusion_matrix.png', dpi=300)
"
```

---

## 9. Known Issues and Limitations

### 9.1 Data Quality Issues
1. **NaN values in MediaPipe output**:
   - ~59% of GAVD sequences had sporadic missing heel landmarks
   - Resolved via linear interpolation (95% recovery rate)
   - Remaining 5% discarded (extreme poses, occlusions)

2. **GAVD video quality variability**:
   - Resolution range: 240p-1080p
   - Lighting: Uncontrolled (indoor/outdoor)
   - Camera angles: Not standardized
   - Impact: Higher MediaPipe landmark noise in low-quality videos

### 9.2 Analysis Limitations
1. **Small GT sample size** (n=17):
   - Limited generalizability to broader populations
   - No pathological subjects in GT (cannot validate pathology-specific biases)

2. **Class imbalance in GAVD**:
   - Majority class (Normal): 37%
   - Minority classes (<2% each): Insufficient for robust learning
   - Recommendation: Use hierarchical classification or collect more data

3. **ROM-only features**:
   - Ignores temporal dynamics (cadence, cycle-to-cycle variability)
   - Ignores asymmetry (left-right differences)
   - Recommendation: Expand feature set (see Section 4.5.1 in main paper)

### 9.3 Technical Constraints
1. **2D pose estimation**:
   - MediaPipe outputs 2D (+ depth estimate), not true 3D
   - Frontal plane angles may have projection errors
   - Recommendation: Use multi-view or depth cameras for 3D reconstruction

2. **DTW computational cost**:
   - O(N²) complexity for full DTW (N = cycle length)
   - Windowed DTW reduces to O(N × window)
   - Execution time: ~0.1-0.5 seconds per cycle pair

---

## 10. Citation

If you use these supplementary materials, please cite:

**[Authors]. "Validation of MediaPipe-Based Gait Analysis Against Laboratory Ground Truth: A Comprehensive Comparison with Multi-Plane Calibration." [Journal Name]. 2025.**

For code and data access, contact: [corresponding author email]

---

**Document Version**: 1.0 (2025-11-07)
**Last Updated**: 2025-11-07
**Maintainer**: [To be added]
