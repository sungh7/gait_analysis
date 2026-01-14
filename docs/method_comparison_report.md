# Method Comparison Report: Deming, DTW, and Random Forest

## Overview
We compared advanced calibration methods to determine if simpler, interpretable models (Deming Regression, DTW) could achieve similar performance to the Random Forest model for predicting OMCS waveforms from MediaPipe data.

## Methods Evaluated
1.  **Baseline (Raw)**: Raw MediaPipe angles vs Ground Truth.
2.  **Linear Regression**: Standard OLS regression (Global Scaling).
3.  **Deming Regression**: Orthogonal regression (Errors-in-variables model, $\lambda=1$).
4.  **DTW (Phase Only)**: Dynamic Time Warping alignment. Measures the error *after* perfect temporal alignment, assuming no amplitude correction.
5.  **Deming + DTW**: Theoretical limit of applying Deming Regression *after* perfect DTW alignment. This represents the best possible performance of a linear calibration model with perfect phase correction.
6.  **Random Forest**: Non-linear regression model (Current Best).

## Results

| Method | Average RMSE (deg) | Average Correlation |
|--------|-------------------|---------------------|
| **Random Forest** | **6.56** | **0.964** |
| Deming + DTW | 17.24 | 0.799 |
| Deming Regression | 18.43 | -0.039 |
| Linear Regression | 18.43 | -0.039 |
| DTW (Phase Only) | 26.22 | 0.799 |
| Baseline (Raw) | 37.18 | -0.039 |

## Analysis
1.  **Random Forest Superiority**: The Random Forest model significantly outperforms all other methods (RMSE 6.56 vs 17.24 for the next best). This confirms that the error in MediaPipe data is **highly non-linear** and cannot be corrected by simple phase alignment (DTW) and linear scaling (Deming).
2.  **Failure of Linear Models**: Deming and Linear Regression performed poorly (RMSE ~18.43), indicating that a single global scale factor is insufficient. The "negative" correlation suggests that without phase correction, the raw signals might be so out of phase or distorted that they negatively correlate in some windows, or simply that the linear fit fails to capture the waveform shape.
3.  **DTW Limitations**: Even with perfect phase alignment (DTW), the error remains high (RMSE 26.22). Adding Deming calibration to DTW (Deming + DTW) improves it to 17.24, but this is still nearly **3x worse** than Random Forest.

## Conclusion
**Random Forest is the necessary approach.** The complexity of the MediaPipe-to-OMCS mapping (likely due to soft tissue artifacts, camera perspective issues, and joint-dependent non-linearities) requires a non-linear model. Simple geometric or temporal corrections (Deming, DTW) are insufficient.
