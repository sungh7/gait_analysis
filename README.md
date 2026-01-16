# MediaPipe Gait Analysis System

**Version 8.1 - Corrected Methodology with Validated Performance**

A comprehensive markerless gait analysis system using MediaPipe 3D pose estimation with machine learning for automated pathological gait detection. This system provides accurate, accessible, and clinically-validated gait analysis achieving **88.8% accuracy** and **96.1% sensitivity** for pathological gait detection.

> ⚠️ **v8.1 Update**: Corrected data leakage in cross-validation using sklearn Pipeline. Results are slightly lower but represent honest real-world performance estimates.

## 🎯 Key Features (Version 8.1)

- **ML-Enhanced Detection**: 88.8% accuracy [86.2%, 91.5%], 96.1% sensitivity [92.8%, 99.4%]
- **Corrected Methodology**: sklearn Pipeline prevents data leakage in cross-validation
- **Bootstrap Confidence Intervals**: 1000-iteration bootstrap for uncertainty quantification
- **Clinical-Grade Performance**: Perfect detection (100%) for Parkinson's, cerebral palsy, antalgic gait
- **Pure 3D Biomechanics**: 10 interpretable features from MediaPipe world coordinates
- **Multi-pathology Validation**: 8 pathology types, 296 gait patterns (GAVD dataset)
- **Ablation Study**: Top 3 features achieve 89.2% accuracy (parsimonious model)
- **Subject-level Simulation**: Estimated generalization performance with clustering

## 🏗️ System Architecture

### Core Components

```
MediaPipe 3D Pose → V7 Feature Extraction → V8 ML Classifier → Clinical Report
                    (10 biomechanical       (Logistic           (Probability +
                     features)               Regression)         Interpretation)
```

1. **V7 Pure 3D Feature Extractor** ([core/extract_v7_features.py](core/extract_v7_features.py))
   - 10 biomechanical features from 3D world coordinates
   - No 2D projections or camera calibration required
   - Features: cadence, velocity, stride length, irregularity, trunk sway, etc.

2. **V8 ML-Enhanced Classifier** ([core/v8_ml_enhanced.py](core/v8_ml_enhanced.py))
   - Logistic Regression with balanced class weights
   - Feature importance analysis for clinical interpretation
   - Cross-validation for robust performance estimation
   - Probability scores for confidence assessment

3. **V9 Multi-view Fusion** ([core/v9_multiview_fusion.py](core/v9_multiview_fusion.py))
   - Ensemble learning across multiple camera views
   - Late fusion strategy for improved accuracy
   - Handles missing views gracefully

4. **MediaPipe Processing Pipeline** ([mediapipe_csv_processor.py](mediapipe_csv_processor.py))
   - Video → 3D pose landmarks extraction
   - Coordinate normalization and filtering
   - Gait cycle detection and segmentation

5. **iOS Native Integration** ([gait_analysis_mobile_app/](gait_analysis_mobile_app/))
   - Real-time MediaPipe on iOS devices
   - On-device ML inference
   - Clinical data export

## 📊 Performance Metrics (GAVD Dataset)

### V8.1 ML-Enhanced Results (Corrected Methodology)

| Metric | Value | 95% CI | Notes |
|--------|-------|--------|-------|
| **Accuracy** | 88.8% | [86.2%, 91.5%] | 5-fold stratified CV |
| **Sensitivity** | 96.1% | [92.8%, 99.4%] | High pathology detection |
| **Specificity** | 81.0% | [75.7%, 86.3%] | Room for improvement |
| **AUC-ROC** | 0.91 | [0.89, 0.94] | Good discrimination |
| **F1-Score** | 0.90 | [0.88, 0.92] | Balanced performance |

### Methodological Correction (v8.0 → v8.1)

| Metric | Before (v8.0) | After (v8.1) | Difference |
|--------|---------------|--------------|------------|
| Accuracy | 89.5% | 88.8% | -0.7% |
| Specificity | 82.4% | 81.0% | -1.4% |

> The v8.1 results use sklearn Pipeline to prevent data leakage, representing more honest estimates.

### Perfect Detection (100% Sensitivity)

- **Parkinson's Disease**: 6/6 patterns
- **Cerebral Palsy**: 24/24 patterns
- **Antalgic Gait**: 9/9 patterns

### Near-Perfect Performance

- **Generic Abnormal**: 79/80 (98.8%)
- **Myopathic Disorders**: 19/20 (95.0%)
- **Stroke/Hemiplegia**: 10/11 (90.9%)

## 🚀 Quick Start

### Installation

```bash
# Install dependencies
pip install mediapipe numpy pandas scikit-learn scipy tqdm

# Or use requirements file
pip install -r requirements.txt
```

**Required Python packages:**
- `mediapipe >= 0.10.0` - 3D pose estimation
- `scikit-learn >= 1.0.0` - ML classifier
- `numpy`, `pandas` - Data processing
- `scipy` - Signal processing for gait event detection

### Usage Examples

#### 1. Basic Pathological Gait Detection (V8 ML-Enhanced)

```python
from core.v8_ml_enhanced import V8_ML_Enhanced

# Initialize detector with trained model
detector = V8_ML_Enhanced(patterns_file="gavd_patterns_with_v7_features.json")

# Train the classifier
detector.train()

# Evaluate on test set
results = detector.evaluate()
print(f"Accuracy: {results['accuracy']:.1%}")
print(f"Sensitivity: {results['sensitivity']:.1%}")

# Analyze new video
prediction = detector.predict_pattern(your_pattern_dict)
print(f"Predicted: {prediction['class']}")
print(f"Probability: {prediction['probability']:.2%}")
print(f"Top features: {prediction['feature_importance'][:3]}")
```

#### 2. Extract V7 Features from Video

```python
from core.extract_v7_features import extract_v7_features
import mediapipe as mp

# First, process video with MediaPipe to get 3D pose CSV
# (see mediapipe_csv_processor.py for details)

# Then extract biomechanical features
pattern = {
    'csv_file': 'path/to/pose_3d.csv',
    'fps': 30
}

features = extract_v7_features(pattern)
print(f"Cadence: {features['cadence_3d']:.1f} steps/min")
print(f"Velocity: {features['velocity_3d']:.2f} m/s")
print(f"Stride Length: {features['stride_length_3d']:.2f} m")
print(f"Gait Irregularity: {features['gait_irregularity_3d']:.3f}")
```

#### 3. Multi-view Analysis (V9)

```python
from core.v9_multiview_fusion import V9_MultiView_Fusion

# Initialize multi-view detector
detector = V9_MultiView_Fusion()
detector.train()

# Analyze pattern from multiple views
pattern = {
    'front_csv': 'front_pose.csv',
    'left_csv': 'left_pose.csv',
    'right_csv': 'right_pose.csv',
    'fps': 30
}

result = detector.predict_multiview(pattern)
print(f"Ensemble prediction: {result['class']}")
print(f"Confidence: {result['confidence']:.2%}")
```

#### 4. iOS Real-time Analysis

See [gait_analysis_mobile_app/IOS_DEPLOYMENT_GUIDE.md](gait_analysis_mobile_app/IOS_DEPLOYMENT_GUIDE.md) for iOS integration guide.

## 📈 10 Biomechanical Features (V7 Pure 3D)

All features computed from MediaPipe world coordinates (metric space):

### Temporal Features
1. **Cadence (steps/min)**: Step frequency from heel peak detection
2. **Cycle Duration (sec)**: Average time between consecutive heel strikes
3. **Gait Irregularity**: Coefficient of variation of stride intervals

### Spatial Features
4. **Stride Length (m)**: 3D distance between consecutive same-foot heel strikes
5. **Step Width (m)**: Lateral separation between left/right foot trajectories
6. **Velocity (m/s)**: Average 3D movement speed
7. **Path Length (m)**: Total 3D trajectory length traveled

### Dynamic Features
8. **Jerkiness**: Rate of acceleration change (smoothness metric)
9. **Step Height Variability**: Coefficient of variation of peak heights
10. **Trunk Sway (m)**: Lateral hip displacement during gait

### Feature Importance (from V8 ML model)
Top discriminative features for pathology detection:
1. **Gait Irregularity** (coef: -1.63) - Most important
2. **Jerkiness** (coef: -1.24)
3. **Step Height Variability** (coef: -0.89)

## 🔬 Validation Studies

### GAVD Dataset Validation (v8.1)
- **Dataset**: 296 gait patterns (142 normal, 154 pathological)
- **Pathology types**: 8 categories including Parkinson's, stroke, cerebral palsy
- **Validation**: Stratified 5-fold CV with sklearn Pipeline (no data leakage)
- **Result**: 88.8% accuracy, 96.1% sensitivity, 81.0% specificity

### Extended Analyses
- **Multi-class Classification**: 71.6% accuracy across 4 pathology groups (AUC 0.89-0.93)
- **Ablation Study**: Top 3 features (Gait Irregularity, Cadence, Jerkiness) achieve 89.2% accuracy
- **Feature Stability**: 6/10 features show high stability across CV folds
- **Learning Curve**: Performance plateau reached; additional data unlikely to help

### Subject-level Simulation
Since GAVD lacks subject identifiers, we simulated subject-level validation using clustering:

| Validation Type | LR Accuracy | LR Sensitivity | LR Specificity |
|-----------------|-------------|----------------|----------------|
| Pattern-level (baseline) | 88.9% | 96.1% | 81.0% |
| Subject-level (simulated) | 88.5% | 96.1% | 80.3% |

> Logistic Regression shows robust generalization; Random Forest showed significant overfitting in subject-level simulation.

### Clinical Ground Truth Validation
- **Cohort**: 28 healthy subjects (Vicon validation)
- **Waveform Correlation**: r = 0.75-0.86 for sagittal kinematics
- **Spatiotemporal ICC**: 0.71-0.89 (moderate-excellent agreement)
- **Documentation**: See [docs/papers/RESEARCH_PAPER.md](docs/papers/RESEARCH_PAPER.md)

## 🏥 Clinical Applications

### Demonstrated Pathology Detection
- **Parkinson's Disease**: 100% detection (6/6)
- **Cerebral Palsy**: 100% detection (24/24)
- **Antalgic Gait**: 100% detection (9/9)
- **Stroke/Hemiplegia**: 90.9% detection (10/11)
- **Myopathic Disorders**: 95.0% detection (19/20)

### Use Cases
- **Primary Care Screening**: Accessible gait assessment without specialized equipment
- **Telehealth**: Remote monitoring of gait patterns
- **Rehabilitation**: Objective progress tracking during therapy
- **Fall Risk Assessment**: Identify high-risk gait patterns in elderly
- **Clinical Research**: Quantitative biomarkers for neurodegenerative diseases

## 📁 Project Structure

```
gait-analysis-system/
├── core/                          # Core algorithms
│   ├── extract_v7_features.py    # V7 Pure 3D feature extraction
│   ├── v8_ml_enhanced.py         # V8 ML classifier
│   └── v9_multiview_fusion.py    # V9 multi-view ensemble
├── gait_analysis_mobile_app/      # iOS native integration
│   ├── ios/                       # Swift/MediaPipe code
│   ├── lib/                       # Flutter/Dart services
│   └── IOS_DEPLOYMENT_GUIDE.md
├── docs/                          # Documentation
│   ├── RESEARCH_PAPER_DRAFT.md   # Journal submission draft
│   └── V8_ML_ENHANCED_RESULTS.md # V8 performance analysis
├── mediapipe_csv_processor.py     # Video → 3D pose CSV
├── preprocessing_pipeline.py      # Data preprocessing
├── angle_converter.py             # Joint angle calculations
├── RESEARCH_LOG.md                # Development log
└── README.md                      # This file
```

## 🔧 Configuration

### MediaPipe Settings
```python
import mediapipe as mp

# Configure MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=2,  # 0=Lite, 1=Full, 2=Heavy
    enable_segmentation=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
```

### V8 ML Classifier Settings
```python
from sklearn.linear_model import LogisticRegression

clf = LogisticRegression(
    random_state=42,
    max_iter=1000,
    class_weight='balanced',  # Handle class imbalance
    C=1.0,  # Regularization strength
    solver='lbfgs'
)
```

## 📊 Output Format

### Detection Results
```json
{
  "prediction": "pathological",
  "probability": 0.87,
  "confidence": "high",
  "features": {
    "cadence_3d": 98.5,
    "velocity_3d": 1.12,
    "gait_irregularity_3d": 0.23,
    "stride_length_3d": 1.35
  },
  "feature_importance": [
    {"name": "gait_irregularity_3d", "coefficient": -1.63},
    {"name": "jerkiness_3d", "coefficient": -1.24},
    {"name": "step_height_variability", "coefficient": -0.89}
  ],
  "interpretation": "Elevated gait irregularity suggests potential pathology"
}
```

## 🔧 Methodological Improvements (v8.1)

### Data Leakage Prevention
```python
# BEFORE (v8.0 - data leakage)
X_scaled = scaler.fit_transform(X)  # Fits on ALL data
cv_scores = cross_val_score(clf, X_scaled, y, cv=5)

# AFTER (v8.1 - corrected)
pipeline = Pipeline([
    ('scaler', StandardScaler()),  # Fit only on training fold
    ('clf', LogisticRegression(class_weight='balanced'))
])
cv_scores = cross_val_score(pipeline, X, y, cv=5)
```

### Analysis Scripts
- `scripts/analysis/multiclass_pathology_analysis.py` - 4-class classification
- `scripts/analysis/ablation_study.py` - Feature subset analysis
- `scripts/analysis/feature_stability_analysis.py` - Coefficient stability
- `scripts/analysis/learning_curve_analysis.py` - Sample size vs performance
- `scripts/analysis/subject_level_simulation.py` - Simulated LOGO-CV
- `scripts/analysis/improve_specificity.py` - Ensemble model comparison

## 🚨 Limitations and Considerations

### Technical Limitations
- **Occlusion sensitivity**: Requires clear view of body landmarks
- **Lighting dependency**: Poor lighting affects pose detection accuracy
- **Video quality**: Minimum 480p resolution, 30 fps recommended
- **Walking space**: Minimum 3-4 steps required for reliable analysis

### Clinical Considerations
- **Not a diagnostic tool**: Screening/monitoring purposes only
- **Requires validation**: Results should be interpreted by qualified professionals
- **Population bias**: Trained primarily on adult gait patterns
- **Pathology coverage**: Limited to 8 validated pathology types
- **Complement not replacement**: Use alongside clinical examination
- **Specificity limitation**: 81% specificity → 19% false positive rate
- **PPV in low prevalence**: At 10% prevalence, PPV = 36% (many false alarms)

### Methodological Limitations
- **Pattern-level validation**: Subject identifiers unavailable in GAVD dataset
- **YouTube data source**: Variable video quality and conditions
- **Single camera view**: Sagittal plane only; no frontal/transverse validation
- **Depth ambiguity**: Monocular 3D estimation has inherent depth errors

### Ethical Considerations
- Patient privacy and video data handling
- Informed consent for video recording
- Appropriate use in clinical decision-making

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📚 References

1. Topley, M., & Richards, J. G. (2020). A comparison of currently available optoelectronic motion capture systems. *Journal of Biomechanics*, 106, 109820.

2. Mentiplay, B. F., et al. (2013). Gait assessment using the Microsoft Xbox One Kinect: Concurrent validity and inter-day reliability of spatiotemporal and kinematic variables. *Journal of Biomechanics*, 46(15), 2166-2170.

3. Lugaresi, C., et al. (2019). MediaPipe: A framework for building perception pipelines. *arXiv preprint arXiv:1906.08172*.

4. Pataky, T. C. (2010). Generalized n-dimensional biomechanical field analysis using statistical parametric mapping. *Journal of Biomechanics*, 43(10), 1976-1982.

## 📞 Contact

For questions, issues, or collaboration opportunities, please open an issue on this repository or contact the research team.

---

**Note**: This system is designed for research purposes. Clinical applications should be validated according to relevant regulatory requirements and institutional guidelines.