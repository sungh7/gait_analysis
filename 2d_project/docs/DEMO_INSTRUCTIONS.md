# AT-RTM Demo Instructions

## 1. Video Abstract
The video abstract (`at_rtm_demo.mp4` or `.gif`) visualizes the core algorithmic process:
- **Top Panel**: Real-time signal streaming.
- **Bottom Left**: Automatic template matching in action.
- **Bottom Right**: Similarity score showing how peaks (gait cycles) are detected.

## 2. Interactive Demo App
We have created a Streamlit web application to demonstrate the algorithm interactively.

### How to Run
```bash
cd /data/gait/2d_project
streamlit run demo_app.py
```

### Features
1. **Inputs**: Upload any gait CSV file (or use built-in Subject 1 data).
2. **Analysis**: One-click execution of AT-RTM.
3. **Report**:
   - **Cadence & Stride Time**: Key temporal parameters.
   - **Gait Variability (CV)**: Stability metric.
   - **Visualization**: Segmentation boundaries and extracted template.

This demo proves that the code is not just a research script but a deployable software component.
