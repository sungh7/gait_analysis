# Gait Analysis System - Complete Documentation

## Table of Contents

1. [System Overview](#system-overview)
2. [Core Modules](#core-modules)
3. [Usage Guide](#usage-guide)
4. [API Reference](#api-reference)
5. [Development Guide](#development-guide)

---

## System Overview

### Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Input: Video File                        │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│            MediaPipe Pose Estimation                        │
│  • 33 body landmarks in 3D world coordinates               │
│  • Real-time tracking at 30 fps                            │
│  • Confidence scores per landmark                          │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│       mediapipe_csv_processor.py                           │
│  • Load 3D pose CSV                                        │
│  • Preprocess and filter coordinates                       │
│  • Calculate joint angles                                  │
│  • Detect gait events (heel strikes)                       │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│      core/extract_v7_features.py                           │
│  • Extract 10 biomechanical features                       │
│  • Pure 3D calculations (no 2D projections)                │
│  • Temporal, spatial, and dynamic features                 │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│         core/v8_ml_enhanced.py                             │
│  • Logistic Regression classifier                          │
│  • Feature importance analysis                             │
│  • Probability scores                                      │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│        Output: Classification Result                        │
│  • Normal vs Pathological                                  │
│  • Confidence score                                        │
│  • Feature contributions                                   │
└─────────────────────────────────────────────────────────────┘
```

---

## Core Modules

### 1. MediaPipe CSV Processor

**File**: [mediapipe_csv_processor.py](mediapipe_csv_processor.py)

**Purpose**: Process MediaPipe 3D pose data and calculate joint angles.

**Key Class**: `MediaPipeCSVProcessor`

**Main Methods**:
- `process_csv()` - Main processing pipeline
- `_calculate_joint_angles()` - Compute joint angles
- `_detect_heel_strikes()` - Identify gait events

**Configuration**:
```python
processor = MediaPipeCSVProcessor(
    filter_type='butterworth',
    cutoff_frequency={'ankle': 4.0, 'knee': 6.0, 'hip': 6.0},
    use_fusion_detection=False
)
```

### 2. Preprocessing Pipeline

**File**: [preprocessing_pipeline.py](preprocessing_pipeline.py)

**Purpose**: Filter and normalize 3D pose data.

**Key Class**: `PreprocessingPipeline`

**Features**:
- Butterworth low-pass filtering
- Savitzky-Golay smoothing
- Missing data interpolation
- Coordinate normalization

### 3. Angle Converter

**File**: [angle_converter.py](angle_converter.py)

**Purpose**: Convert MediaPipe joint angles to clinical reference frame.

**Key Class**: `AngleConverter`

**Methods**:
- `convert_angles()` - Apply calibration parameters
- `load_parameters()` - Load pre-trained conversion params

### 4. V7 Feature Extractor

**File**: [core/extract_v7_features.py](core/extract_v7_features.py)

**Purpose**: Extract 10 biomechanical features from 3D pose.

See [core/README.md](core/README.md) for detailed documentation.

### 5. V8 ML Classifier

**File**: [core/v8_ml_enhanced.py](core/v8_ml_enhanced.py)

**Purpose**: ML-enhanced pathological gait detection.

See [core/README.md](core/README.md) for detailed documentation.

---

## Usage Guide

### Basic Workflow

#### Step 1: Process Video with MediaPipe

```python
import mediapipe as mp

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Process video frames and save 3D coordinates to CSV
# (Implementation in mediapipe_csv_processor.py)
```

#### Step 2: Extract Features

```python
from core.extract_v7_features import extract_v7_features

pattern = {
    'csv_file': 'pose_3d.csv',
    'fps': 30
}

features = extract_v7_features(pattern)
```

#### Step 3: Classify Gait

```python
from core.v8_ml_enhanced import V8_ML_Enhanced

detector = V8_ML_Enhanced('gavd_patterns_with_v7_features.json')
detector.train()

result = detector.predict_pattern(your_pattern)
print(f"Classification: {result['class']}")
print(f"Probability: {result['probability']:.2%}")
```

### Advanced Usage

#### Custom Preprocessing

```python
from preprocessing_pipeline import PreprocessingPipeline
from mediapipe_csv_processor import MediaPipeCSVProcessor

# Custom preprocessing
preprocessor = PreprocessingPipeline(
    use_butterworth=True,
    butterworth_cutoff={'ankle': 3.5, 'knee': 5.5, 'hip': 5.5}
)

processor = MediaPipeCSVProcessor(preprocessor=preprocessor)
result = processor.process_csv('pose_data.csv', fps=30)
```

#### Multi-view Analysis

```python
from core.v9_multiview_fusion import V9_MultiView_Fusion

detector = V9_MultiView_Fusion()
detector.train()

result = detector.predict_multiview({
    'front_csv': 'front.csv',
    'left_csv': 'left.csv',
    'right_csv': 'right.csv',
    'fps': 30
})
```

---

## API Reference

### MediaPipeCSVProcessor

```python
class MediaPipeCSVProcessor:
    def __init__(
        self,
        preprocessor: Optional[PreprocessingPipeline] = None,
        use_fusion_detection: bool = False,
        filter_type: str = 'butterworth',
        cutoff_frequency: Union[float, Dict[str, float]] = {...},
        heel_strike_weights: Tuple[float, float, float] = (0.4, 0.4, 0.2)
    ):
        """Initialize processor with configuration."""

    def process_csv(
        self,
        csv_path: str,
        fps: int = 30
    ) -> ProcessingResult:
        """
        Process MediaPipe CSV file.

        Args:
            csv_path: Path to 3D pose CSV
            fps: Video frame rate

        Returns:
            ProcessingResult with joint angles and gait events
        """
```

### extract_v7_features

```python
def extract_v7_features(pattern: dict) -> Optional[dict]:
    """
    Extract 10 biomechanical features from gait pattern.

    Args:
        pattern: Dictionary with:
            - csv_file (str): Path to 3D pose CSV
            - fps (int): Frame rate (default: 30)

    Returns:
        Dictionary with 10 features:
        {
            'cadence_3d': float,
            'step_height_variability': float,
            'gait_irregularity_3d': float,
            'velocity_3d': float,
            'jerkiness_3d': float,
            'cycle_duration_3d': float,
            'stride_length_3d': float,
            'trunk_sway': float,
            'path_length_3d': float,
            'step_width_3d': float
        }

        Returns None if extraction fails.
    """
```

### V8_ML_Enhanced

```python
class V8_ML_Enhanced:
    def __init__(self, patterns_file: str):
        """Initialize with GAVD patterns JSON file."""

    def train(self):
        """Train logistic regression classifier."""

    def evaluate(self) -> dict:
        """
        Evaluate on test set.

        Returns:
            {
                'accuracy': float,
                'sensitivity': float,
                'specificity': float,
                'confusion_matrix': np.ndarray
            }
        """

    def predict_pattern(self, pattern: dict) -> dict:
        """
        Predict class for single pattern.

        Args:
            pattern: Dictionary with 10 V7 features

        Returns:
            {
                'class': str,  # 'normal' or 'pathological'
                'probability': float,  # 0.0 to 1.0
                'confidence': str,  # 'low', 'medium', 'high'
                'feature_importance': list
            }
        """
```

---

## Development Guide

### Project Structure

```
gait-analysis-system/
├── core/                      # Core algorithms (V7, V8, V9)
│   ├── extract_v7_features.py
│   ├── v8_ml_enhanced.py
│   ├── v9_multiview_fusion.py
│   └── README.md
├── tests/                     # Unit tests
│   ├── test_v7_features.py
│   ├── test_v8_classifier.py
│   └── test_mediapipe_processor.py
├── archive/                   # Legacy code
│   └── legacy_scripts/
├── docs/                      # Research documentation
│   └── RESEARCH_PAPER_DRAFT.md
├── gait_analysis_mobile_app/  # iOS integration
├── mediapipe_csv_processor.py # Main processor
├── preprocessing_pipeline.py  # Data preprocessing
├── angle_converter.py         # Angle conversion
└── README.md                  # Main documentation
```

### Adding New Features

1. **Create feature extraction function** in `core/extract_v7_features.py`
2. **Add to feature list** in `V8_ML_Enhanced.feature_names`
3. **Write unit tests** in `tests/test_v7_features.py`
4. **Retrain model** with new features
5. **Update documentation**

### Running Tests

```bash
# All tests
python -m pytest tests/ -v

# With coverage
python -m pytest tests/ --cov=core --cov=. --cov-report=html

# Specific test
python -m pytest tests/test_v7_features.py::TestV7Features::test_cadence
```

### Code Style

- Follow PEP 8 guidelines
- Use type hints for function signatures
- Write docstrings for all public functions
- Keep functions focused and modular

### Contributing

1. Create feature branch from `main`
2. Write tests for new functionality
3. Ensure all tests pass
4. Update documentation
5. Submit pull request with clear description

---

## Performance Benchmarks

### V8 ML-Enhanced (GAVD Dataset)

| Metric | Value |
|--------|-------|
| Overall Accuracy | 89.5% |
| Sensitivity | 96.1% |
| Specificity | 82.4% |
| Processing Time | <50ms |
| Memory Usage | ~100MB |

### Feature Extraction Speed

| Operation | Time |
|-----------|------|
| MediaPipe Pose (per frame) | ~20ms |
| V7 Feature Extraction | ~10ms |
| V8 Classification | <1ms |
| **Total (per video)** | ~30-50ms/frame |

---

## Troubleshooting

### Common Issues

**1. Import errors**
```bash
# Solution: Install dependencies
pip install mediapipe numpy pandas scikit-learn scipy
```

**2. CSV format errors**
```python
# Required CSV format:
# frame, position, x, y, z, visibility
```

**3. Low detection accuracy**
- Check video quality (min 480p, 30fps)
- Ensure clear view of subject
- Verify lighting conditions
- Check for occlusions

**4. Feature extraction returns None**
- Verify CSV has minimum 30 frames
- Check required landmarks present
- Ensure coordinates are valid numbers

---

## Additional Resources

- **Research Paper**: [docs/RESEARCH_PAPER_DRAFT.md](docs/RESEARCH_PAPER_DRAFT.md)
- **Research Log**: [RESEARCH_LOG.md](RESEARCH_LOG.md)
- **iOS Guide**: [gait_analysis_mobile_app/IOS_DEPLOYMENT_GUIDE.md](gait_analysis_mobile_app/IOS_DEPLOYMENT_GUIDE.md)
- **Tests**: [tests/README.md](tests/README.md)
- **Core Algorithms**: [core/README.md](core/README.md)

---

## Contact & Support

For questions, issues, or contributions:
- Open an issue on GitHub
- See [README.md](README.md) for project overview
- Check [tests/](tests/) for usage examples

---

**Last Updated**: 2024-11-24
**Version**: 8.0
