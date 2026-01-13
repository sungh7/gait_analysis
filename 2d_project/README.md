# Gait Analysis 2D Project

MediaPipe-based 2D gait analysis system with clinical-grade accuracy for pathological gait detection.

## Overview

This project implements a smartphone-based gait analysis system using MediaPipe pose estimation. It achieves **89.5% accuracy** in pathological gait detection with **100% sensitivity** for serious conditions like Parkinson's disease, cerebral palsy, and antalgic gait.

### Key Features

- **Auto-Template Resampled Template Matching (AT-RTM)**: Novel algorithm for automatic gait cycle segmentation with 100% recall
- **Joint-Specific Signal Processing**: Adaptive filtering strategies for knee, hip, and ankle
- **Kinematic Constraints**: Biomechanically-informed constraints to ensure anatomically valid poses
- **Production-Ready**: Validated against Vicon optical motion capture (gold standard)

## Installation

### Requirements

- Python 3.10+
- MediaPipe 0.10.1+
- See `requirements.txt` for full dependencies

### Setup

```bash
# Clone repository
git clone <repository-url>
cd 2d_project

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

## Project Structure

```
2d_project/
├── src/
│   ├── core/               # Core algorithms
│   │   ├── improved_signal_processing.py
│   │   ├── kinematic_constraints.py
│   │   ├── improved_extractor.py
│   │   └── landmark_quality_filter.py
│   ├── validation/         # Validation scripts
│   ├── analysis/           # Analysis tools
│   └── utils/              # Utilities (config, logging, exceptions)
├── tests/                  # Unit tests
├── docs/                   # Documentation
├── archive/                # Archived experimental scripts
├── config.yaml             # Configuration file
├── requirements.txt        # Python dependencies
└── README.md               # This file
```

## Quick Start

### 1. Process a Single Subject

```python
from src.core.improved_signal_processing import ImprovedSignalProcessor
from src.core.kinematic_constraints import KinematicConstraintEnforcer

# Initialize processors
signal_processor = ImprovedSignalProcessor(fps=30)
constraint_enforcer = KinematicConstraintEnforcer(fps=30, subject_height=1.70)

# Process joint angles
processed_knee, metrics = signal_processor.process_joint_angle(
    raw_knee_angles,
    joint_type='knee'
)

# Apply kinematic constraints
constrained_angles = constraint_enforcer.enforce_joint_angle_constraints(angles_df)
```

### 2. Run Batch Processing

```bash
python -m src.validation.batch_process_all_subjects
```

### 3. Run Tests

```bash
pytest tests/ -v
```

## Configuration

Edit `config.yaml` to customize parameters:

```yaml
processing:
  fps: 30
  subject_height: 1.70

signal_processing:
  knee:
    butterworth_cutoff: 6  # Hz
  ankle:
    enable_velocity_constraints: false  # CRITICAL for ankle
    angle_range: [0, 180]  # Geometric, not anatomical
```

### Critical Note: Ankle Processing

MediaPipe outputs **geometric angles (0-180°)** for the ankle, NOT anatomical angles (-30° to +50°). The configuration preserves this range to maintain ROM accuracy.

## Performance

### Validation Results

| Metric | Value |
|--------|-------|
| Overall Accuracy | 89.5% |
| Sensitivity | 96.1% |
| Specificity | 82.4% |
| Cycle Recall | 100% (293/293) |
| Median Timing Error | 2 frames (~67ms) |

### Pathology Detection

| Condition | Detection Rate |
|-----------|---------------|
| Parkinson's | 100% (6/6) |
| Cerebral Palsy | 100% (24/24) |
| Antalgic Gait | 100% (9/9) |
| Stroke/Hemiplegia | 90.9% (10/11) |

## Documentation

- [Research Paper](docs/RESEARCH_PAPER_REVISED.md) - Full methodology and results
- [Implementation Summary](docs/COMPLETE_IMPLEMENTATION_SUMMARY.md) - Technical details
- [Signal Processing Guide](docs/IMPROVEMENTS_SUMMARY.md) - Algorithm explanations
- [Ground-Truth-Free Protocol](docs/GT_FREE_PROTOCOL.md) - Calibration without lab equipment

## Testing

Run the test suite:

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_signal_processing.py -v

# Run with coverage
pytest tests/ --cov=src --cov-report=term-missing
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is part of ongoing research. Contact the authors for licensing information.

## Citation

If you use this work, please cite:

```bibtex
@article{gait2d2026,
  title={Smartphone-Based Gait Analysis Using MediaPipe:
         A Validated Approach for Clinical Assessment},
  author={Gait Analysis Team},
  journal={In Preparation},
  year={2026}
}
```

## Acknowledgments

- MediaPipe team for the pose estimation framework
- Biomechanics references: Winter (2009), Perry & Burnfield (2010)
- GAVD dataset for pathological gait validation
