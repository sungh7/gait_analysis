# Gait Analysis Web Application

Streamlit-based web application for real-time gait analysis using MediaPipe Pose.

## Features

- **Video Upload Analysis**: Upload gait videos (MP4, AVI, MOV) for analysis
- **Real-time Streaming**: Analyze gait patterns from webcam in real-time
- **MediaPipe Integration**: 33 pose landmarks for accurate tracking
- **Comprehensive Metrics**:
  - Cadence (steps/min)
  - Stride length (meters)
  - Walking speed (m/s)
  - Joint angles (hip, knee, ankle)

## Architecture

```
Video/Camera Input
    ↓
MediaPipe Pose Detection
    ↓
33 Landmarks (x, y, z)
    ↓
GaitAnalysisTool
    ↓
Gait Parameters
```

## Installation

### Prerequisites
- Python 3.8+
- Webcam (for real-time mode)

### Setup

```bash
# Clone repository
git clone <repository-url>
cd webapp

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Run Web Application

```bash
streamlit run streamlit-gait-analysis-app.py
```

The app will open in your browser at `http://localhost:8501`

### Video Upload Mode

1. Select "비디오 업로드" (Video Upload)
2. Upload a gait video file
3. Wait for analysis to complete
4. View results for each detected segment

### Real-time Streaming Mode

1. Select "실시간 스트리밍" (Real-time Streaming)
2. Click "실시간 분석 시작" checkbox
3. Allow camera access
4. Walk in front of camera
5. View real-time analysis results

## Gait Analysis Tool

### GaitAnalysisTool Class

**Methods**:
- `add_frame(frame_data)`: Add single frame for real-time analysis
- `analyze_video(video_data)`: Analyze entire video
- `calculate_cadence()`: Steps per minute
- `calculate_stride_length()`: Distance between consecutive steps
- `calculate_walking_speed()`: Speed in m/s
- `calculate_joint_angles()`: Hip, knee, ankle angles

### Segment Detection

The tool automatically detects:
- Direction changes (turning around)
- Gait cycles (heel strikes)
- Valid walking segments (minimum 2 seconds)

### Analysis Parameters

- **FPS**: 30 (configurable)
- **Minimum segment duration**: 2 seconds
- **Minimum peak distance**: 15 frames (0.5 seconds)
- **Gaussian smoothing sigma**: 2

## Files

- `streamlit-gait-analysis-app.py`: Main Streamlit web application
- `gait-analysis-tool.py`: Core gait analysis algorithms
- `app.py`: Legacy Flask application (deprecated)
- `requirements.txt`: Python dependencies

## Dependencies

- `streamlit`: Web application framework
- `opencv-python`: Video processing
- `mediapipe`: Pose detection
- `numpy`: Numerical computations
- `scipy`: Signal processing (peak detection, filtering)
- `pandas`: Data handling

## Technical Details

### MediaPipe Landmark Indices

Key landmarks used:
- Hip: 23 (left), 24 (right)
- Knee: 25 (left), 26 (right)
- Ankle: 27 (left), 28 (right)
- Heel: 29 (left), 30 (right)
- Toe: 31 (left), 32 (right)

### Angle Calculation

Joint angles computed using vector dot products:
```python
angle = arccos(v1 · v2 / (|v1| * |v2|))
```

### Gait Cycle Detection

Uses `scipy.signal.find_peaks()` on ankle Y-coordinate:
- Higher Y = ankle lifted
- Peak detection = heel strike
- Distance between peaks = gait cycle

## Limitations

- Works best with side-view videos
- Requires clear view of full body
- Lighting conditions affect landmark detection
- Real-time mode requires stable camera position

## Performance

- Video processing: ~30 fps on modern hardware
- Real-time latency: < 100ms
- Accuracy depends on MediaPipe pose detection quality

## Future Enhancements

- [ ] Support for multiple views (front, side, back)
- [ ] Export results to CSV/JSON
- [ ] Visualization of gait patterns
- [ ] Comparison with normal gait reference
- [ ] Pathological gait detection
- [ ] Integration with V7 Pure 3D algorithm

## Related Projects

- Mobile App: Flutter app with V7 Pure 3D algorithm
- Research Pipeline: Python scripts for GAVD dataset analysis

## License

See main project LICENSE file.

## Contact

For questions or issues, please open an issue in the repository.
