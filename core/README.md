# Core Algorithms

This directory contains the core gait analysis algorithms.

## Modules

### 1. extract_v7_features.py - V7 Pure 3D Feature Extraction

**Purpose**: Extract 10 biomechanical features from MediaPipe 3D pose data.

**Key Function**:
```python
def extract_v7_features(pattern: dict) -> dict:
    """
    Extract V7 Pure 3D features from gait pattern.

    Args:
        pattern: Dictionary with keys:
            - csv_file: Path to 3D pose CSV
            - fps: Video frame rate (default: 30)

    Returns:
        Dictionary with 10 features or None if extraction fails
    """
```

**Features Extracted** (all from 3D world coordinates):
1. **cadence_3d** (steps/min) - Step frequency
2. **step_height_variability** (CV) - Consistency of step height
3. **gait_irregularity_3d** (CV) - Stride interval variability
4. **velocity_3d** (m/s) - Average 3D walking speed
5. **jerkiness_3d** (m/s³) - Smoothness of movement
6. **cycle_duration_3d** (s) - Average gait cycle time
7. **stride_length_3d** (m) - Distance between consecutive same-foot strikes
8. **trunk_sway** (m) - Lateral hip displacement
9. **path_length_3d** (m) - Total 3D trajectory length
10. **step_width_3d** (m) - Lateral foot separation

**Usage**:
```python
from core.extract_v7_features import extract_v7_features

pattern = {
    'csv_file': 'pose_landmarks.csv',
    'fps': 30
}

features = extract_v7_features(pattern)
print(f"Cadence: {features['cadence_3d']:.1f} steps/min")
```

**Requirements**:
- CSV with columns: frame, position, x, y, z
- Minimum 30 frames (~1 second at 30fps)
- Required landmarks: left_heel, right_heel, left_hip, right_hip

---

### 2. v8_ml_enhanced.py - V8 ML-Enhanced Classifier

**Purpose**: Machine learning classifier for pathological gait detection.

**Key Class**:
```python
class V8_ML_Enhanced:
    """ML-enhanced gait detector using Logistic Regression"""

    def __init__(self, patterns_file: str):
        """Initialize with GAVD patterns JSON file"""

    def train(self):
        """Train the classifier on loaded patterns"""

    def evaluate(self) -> dict:
        """Evaluate on test set, return metrics"""

    def predict_pattern(self, pattern: dict) -> dict:
        """Predict class for single pattern"""
```

**Performance** (GAVD dataset):
- **Accuracy**: 89.5%
- **Sensitivity**: 96.1%
- **Specificity**: 82.4%
- **Cross-validation**: 88.8% ± 3.0%

**Usage**:
```python
from core.v8_ml_enhanced import V8_ML_Enhanced

# Initialize and train
detector = V8_ML_Enhanced('gavd_patterns_with_v7_features.json')
detector.train()

# Evaluate
results = detector.evaluate()
print(f"Accuracy: {results['accuracy']:.1%}")

# Predict new pattern
prediction = detector.predict_pattern(your_pattern)
print(f"Class: {prediction['class']}")
print(f"Probability: {prediction['probability']:.2%}")
```

**Model Details**:
- **Algorithm**: Logistic Regression (scikit-learn)
- **Features**: 10 V7 features (standardized)
- **Class weighting**: Balanced (handles imbalance)
- **Regularization**: L2 with C=1.0
- **Solver**: lbfgs (quasi-Newton method)

**Feature Importance** (top 3):
1. **gait_irregularity_3d**: -1.63 (most discriminative)
2. **jerkiness_3d**: -1.24
3. **step_height_variability**: -0.89

---

### 3. v9_multiview_fusion.py - V9 Multi-view Ensemble

**Purpose**: Ensemble learning across multiple camera views.

**Key Class**:
```python
class V9_MultiView_Fusion:
    """Multi-view fusion using ensemble learning"""

    def __init__(self):
        """Initialize view-specific classifiers"""

    def train(self):
        """Train separate classifiers for each view"""

    def predict_multiview(self, pattern: dict) -> dict:
        """Predict using all available views"""
```

**Strategy**:
- Train separate V8 classifiers for front, left, right views
- Late fusion: Average probability scores
- Gracefully handles missing views
- Weights based on view reliability

**Usage**:
```python
from core.v9_multiview_fusion import V9_MultiView_Fusion

detector = V9_MultiView_Fusion()
detector.train()

# Predict with multiple views
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

**Improvements over V8**:
- More robust to occlusion
- Better handling of atypical gait patterns
- Higher confidence scores

---

## Pipeline Flow

```
1. Video → MediaPipe Pose
   ↓
2. 3D Pose CSV → extract_v7_features()
   ↓
3. 10 Features → V8_ML_Enhanced.predict()
   ↓
4. Classification Result (normal/pathological)
```

## Data Format

### Input: 3D Pose CSV
```csv
frame,position,x,y,z,visibility
0,left_heel,-0.1,0.05,0.2,0.95
0,right_heel,0.1,0.03,0.1,0.98
...
```

### Output: Features Dictionary
```python
{
    'cadence_3d': 105.3,
    'step_height_variability': 0.12,
    'gait_irregularity_3d': 0.08,
    'velocity_3d': 1.15,
    'jerkiness_3d': 2.3,
    'cycle_duration_3d': 1.14,
    'stride_length_3d': 1.32,
    'trunk_sway': 0.06,
    'path_length_3d': 9.5,
    'step_width_3d': 0.11
}
```

### Output: Classification Result
```python
{
    'prediction': 'pathological',
    'probability': 0.87,
    'confidence': 'high',
    'feature_importance': [
        {'name': 'gait_irregularity_3d', 'coefficient': -1.63},
        {'name': 'jerkiness_3d', 'coefficient': -1.24}
    ]
}
```

## Testing

Run unit tests:
```bash
python -m pytest tests/test_v7_features.py
python -m pytest tests/test_v8_classifier.py
```

## References

- **V7 Development**: Pure 3D biomechanical approach (Nov 2024)
- **V8 Development**: ML enhancement achieving 89.5% accuracy (Nov 2024)
- **V9 Development**: Multi-view fusion exploration (Nov 2024)

See [../docs/RESEARCH_PAPER_DRAFT.md](../docs/RESEARCH_PAPER_DRAFT.md) for detailed methodology.
