# OMCS Prediction Model Report

## Overview
We have developed a machine learning model to predict OMCS (Optical Motion Capture System) kinematic waveforms directly from MediaPipe (MP) data, without the need for subject-specific calibration.

## Methodology
- **Input**: Raw MediaPipe gait cycle waveforms (101 points, 0-100% gait cycle) + Joint Type + Side.
- **Output**: Ground Truth (OMCS) gait cycle waveforms (101 points).
- **Model**: Random Forest Regressor (100 trees).
- **Training Data**: Paired MP and OMCS data from ~20 subjects (S1_01 to S1_30).
- **Validation**: Leave-One-Group-Out Cross-Validation (Group = Subject).

## Results
The model achieves high accuracy in reconstructing OMCS waveforms from MP data:

- **Average RMSE**: 6.56 degrees
- **Average Correlation**: 0.964

This indicates that the model effectively learns the mapping from MP to OMCS, correcting for systematic biases and noise without requiring explicit calibration steps for new subjects.

### Model Comparison
We compared three different approaches to validate our choice of Random Forest:

| Model | Average RMSE (deg) | Average Correlation |
|-------|-------------------|---------------------|
| **Random Forest** | **6.56** | **0.964** |
| MLP (Neural Net) | 6.92 | 0.954 |
| Linear Regression | 17.76 | 0.827 |

- **Linear Regression** performed significantly worse, confirming that the relationship between MP and OMCS data is non-linear.
- **MLP (Multi-Layer Perceptron)** performed similarly to Random Forest but slightly worse, likely due to the small dataset size (~168 samples) which favors ensemble methods like Random Forest over Deep Learning.


## Files
- `extract_paired_data.py`: Script to extract paired MP and GT waveforms from the dataset.
- `train_correction_model.py`: Script to train the Random Forest model and evaluate it.
- `predict_omcs.py`: Script to predict OMCS waveforms for a new MediaPipe CSV file.
- `omcs_correction_model.joblib`: The trained model file.

## Usage
To predict OMCS results for a new MediaPipe CSV file:

```bash
python3 /data/gait/predict_omcs.py /path/to/your/mp_file.csv
```

This will generate:
1. `_predicted_omcs.csv`: The predicted kinematic waveforms.
2. `_prediction.png`: A plot comparing Raw MP vs Predicted OMCS.

## Conclusion
The Random Forest model provides a robust, calibration-free method to estimate OMCS-quality kinematics from MediaPipe video analysis.
