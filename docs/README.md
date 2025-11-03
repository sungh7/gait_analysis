# Gait Analysis Project Documentation

## Core Documentation

### Research and Papers
- [RESEARCH_PAPER.md](RESEARCH_PAPER.md) - Main research paper
- [PAPER_FLOWCHART.md](PAPER_FLOWCHART.md) - Algorithm flowchart
- [PAPER_GRAPHICAL_ABSTRACT.md](PAPER_GRAPHICAL_ABSTRACT.md) - Visual summary

### Implementation
- [REALTIME_SYSTEM_COMPLETE.md](REALTIME_SYSTEM_COMPLETE.md) - Real-time mobile app integration

## Mobile App Deployment

See [gait_analysis_mobile_app/V7_DEPLOYMENT_GUIDE.md](../gait_analysis_mobile_app/V7_DEPLOYMENT_GUIDE.md)

## Core Algorithms

Located in `/data/gait/core/`:
- `v7_pure3d.py` - V7 Pure 3D algorithm (68.2% accuracy, 92.2% sensitivity, 98.6% clinical sensitivity)
- `baseline_v3.py` - Baseline V3 robust algorithm
- `gavd_extractor.py` - GAVD dataset feature extraction
- `evaluation_pipeline.py` - Complete evaluation pipeline

## Experimental Code

Located in `/data/gait/experiments/`:
- P0-P6 series: Research phase files
- Improved versions: Algorithm development iterations
- GAVD processors: Dataset processing tools
- Diagnostic tools: Analysis and validation scripts

## Key Performance Metrics

### V7 Pure 3D (296 GAVD patterns)
- Overall Accuracy: 68.2%
- Overall Sensitivity: 92.2%
- Clinical Pathology Sensitivity: 98.6%
- Parkinsons Detection: 100% (6/6)
- Stroke Detection: 100% (11/11)
- Cerebral Palsy Detection: 100% (24/24)
- Myopathic Detection: 100% (20/20)

## Dataset

- **GAVD**: 296 patterns (142 normal, 154 pathological)
- **Camera Views**: Front, Left Side, Right Side
- **MediaPipe CSVs**: 389 files
- **Features**: 10 pure 3D features from MediaPipe Pose

## Contact

For questions about the research or implementation, refer to the main project README.
