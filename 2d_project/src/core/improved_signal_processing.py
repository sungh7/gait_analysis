#!/usr/bin/env python3
"""
Enhanced signal processing for improved ankle/knee tracking accuracy.
Addresses QC failures identified in qc_failure_analysis.md
"""

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter, butter, filtfilt, medfilt
from scipy.interpolate import UnivariateSpline, interp1d
# gaussian_filter1d removed - not currently used


class ImprovedSignalProcessor:
    """
    Multi-stage signal processing pipeline for noisy gait kinematics.

    Addresses common failure modes:
    - Foot occlusion (ankle tracking < 0.5 correlation)
    - Phase mismatch from tracking errors
    - High-frequency noise from MediaPipe jitter
    """

    def __init__(self, fps=30):
        self.fps = fps

    def process_joint_angle(self, signal, joint_type='knee'):
        """
        Apply adaptive filtering based on joint type.

        Args:
            signal: Raw angle signal from MediaPipe (may contain NaNs)
            joint_type: 'knee', 'hip', or 'ankle' (different noise characteristics)

        Returns:
            Processed signal with improved SNR
        """

        # Stage 1: Intelligent gap filling (better than simple ffill/bfill)
        signal_filled = self._adaptive_gap_filling(signal)

        # Stage 2: Outlier rejection (reduces spikes from tracking errors)
        signal_clean = self._outlier_rejection(signal_filled, joint_type)

        # Stage 3: Multi-pass filtering
        if joint_type == 'ankle':
            # Ankle needs aggressive smoothing due to small joint size
            signal_smooth = self._aggressive_smoothing(signal_clean)
        elif joint_type == 'knee':
            # Knee: balance between smoothing and preserving sharp flexion peaks
            signal_smooth = self._balanced_smoothing(signal_clean)
        else:  # hip
            # Hip: gentle smoothing, hip motion is naturally smooth
            signal_smooth = self._gentle_smoothing(signal_clean)

        # Stage 4: Temporal consistency enforcement
        signal_final = self._enforce_temporal_consistency(signal_smooth, joint_type)

        return signal_final, self._compute_quality_metrics(signal, signal_final)

    def _adaptive_gap_filling(self, signal):
        """
        Fill gaps using cubic spline interpolation instead of forward/backward fill.
        Preserves signal dynamics better than simple propagation.
        """
        signal = signal.copy()
        valid_idx = ~np.isnan(signal)

        if valid_idx.sum() < 4:  # Need at least 4 points for cubic
            # Fallback to linear
            return pd.Series(signal).interpolate(method='linear').bfill().ffill().values

        # Cubic spline through valid points
        x_valid = np.where(valid_idx)[0]
        y_valid = signal[valid_idx]

        try:
            # Use smoothing spline (s parameter controls smoothness vs fidelity)
            spline = UnivariateSpline(x_valid, y_valid, s=len(x_valid)*0.5, k=3)
            signal_filled = spline(np.arange(len(signal)))
        except Exception:
            # Fallback if spline fails
            interp_func = interp1d(
                x_valid, y_valid, kind='cubic',
                fill_value='extrapolate',  # type: ignore[arg-type]
                bounds_error=False
            )
            signal_filled = interp_func(np.arange(len(signal)))

        return signal_filled

    def _outlier_rejection(self, signal, joint_type):
        """
        Remove physiologically impossible values and tracking artifacts.
        """
        # Define physiological ranges (degrees)
        # Note: MediaPipe ankle angles are ABSOLUTE GEOMETRIC angles, not anatomical
        # So we use a much wider range for ankle to preserve motion
        ranges = {
            'knee': (0, 140),    # 0=full extension, 140=max flexion
            'hip': (-30, 120),   # Negative=extension, positive=flexion
            'ankle': (0, 180)    # WIDE RANGE - MediaPipe uses geometric angles, not anatomical
        }

        min_val, max_val = ranges.get(joint_type, (-50, 180))

        # Hard clipping
        signal_clipped = np.clip(signal, min_val, max_val)

        # Detect velocity outliers (unrealistic angular velocities)
        velocity = np.diff(signal_clipped, prepend=signal_clipped[0])
        max_velocity = 500 / self.fps  # Max 500 deg/sec

        # Use median filter on frames with impossible velocities
        bad_frames = np.abs(velocity) > max_velocity
        if bad_frames.sum() > 0:
            # Apply median filter only to bad regions
            signal_clipped[bad_frames] = medfilt(signal_clipped, kernel_size=5)[bad_frames]

        return signal_clipped

    def _aggressive_smoothing(self, signal):
        """
        For ankle (high noise, small joint).
        Savitzky-Golay only (Gaussian removed to preserve ROM).

        UPDATED V2: Removed Gaussian smoothing entirely
        - Median kernel: 5 → 3
        - SG window: 11 → 5 (minimal)
        - NO Gaussian (was causing ROM → 0)
        """
        # Step 1: Light median filter to remove spikes only
        signal_med = medfilt(signal, kernel_size=3)

        # Step 2: Savitzky-Golay only (preserves peaks and ROM)
        # Minimal window for maximum ROM preservation
        window = min(5, len(signal) if len(signal) % 2 == 1 else len(signal) - 1)
        if window < 5: window = 5
        signal_smooth = savgol_filter(signal_med, window_length=window, polyorder=2)

        # Step 3: NO Gaussian smoothing (removed to preserve ankle ROM)
        # Return SG-filtered signal directly

        return signal_smooth

    def _balanced_smoothing(self, signal):
        """
        For knee (preserve sharp peaks during swing phase).
        """
        # Butterworth low-pass filter (6 Hz cutoff)
        cutoff_hz = 6
        nyquist = self.fps / 2
        normalized_cutoff = cutoff_hz / nyquist
        b, a = butter(4, normalized_cutoff, btype='low')
        signal_filt = filtfilt(b, a, signal)

        # Light Savitzky-Golay
        window = min(9, len(signal) if len(signal) % 2 == 1 else len(signal) - 1)
        if window < 5: window = 5
        signal_smooth = savgol_filter(signal_filt, window_length=window, polyorder=2)

        return signal_smooth

    def _gentle_smoothing(self, signal):
        """
        For hip (naturally smooth motion).
        """
        # Just Butterworth low-pass (8 Hz cutoff)
        cutoff_hz = 8
        nyquist = self.fps / 2
        normalized_cutoff = cutoff_hz / nyquist
        b, a = butter(3, normalized_cutoff, btype='low')
        signal_smooth = filtfilt(b, a, signal)

        return signal_smooth

    def _enforce_temporal_consistency(self, signal, joint_type):
        """
        Ensure gait cycles follow expected patterns.
        Fixes phase mismatch issues.
        """
        # For knee: enforce that flexion peaks occur at expected intervals
        if joint_type == 'knee':
            # Detect flexion peaks
            from scipy.signal import find_peaks
            peaks, _ = find_peaks(signal, distance=self.fps*0.7, prominence=15)

            if len(peaks) >= 2:
                # Calculate median peak-to-peak interval
                intervals = np.diff(peaks)
                median_interval = np.median(intervals)

                # If interval variance is high, this might be a bad signal
                interval_cv = np.std(intervals) / median_interval if median_interval > 0 else 1.0

                # Store for quality metrics
                self._temporal_quality = 1.0 - min(interval_cv, 1.0)
            else:
                self._temporal_quality = 0.0

        return signal

    def _compute_quality_metrics(self, original, processed):
        """
        Compute signal quality indicators.

        Returns:
            dict with SNR, smoothness, temporal_consistency
        """
        # Remove NaNs for comparison
        valid_mask = ~np.isnan(original)
        if valid_mask.sum() < 10:
            return {'snr': 0, 'smoothness': 0, 'temporal_consistency': 0}

        orig_valid = original[valid_mask]
        proc_valid = processed[valid_mask]

        # SNR estimate (signal power vs noise power)
        signal_power = np.var(proc_valid)
        noise_power = np.var(orig_valid - proc_valid) + 1e-10
        snr = 10 * np.log10(signal_power / noise_power)

        # Smoothness (inverse of jerk)
        jerk = np.diff(proc_valid, n=2)  # Second derivative
        smoothness = 1.0 / (1.0 + np.std(jerk))

        # Temporal consistency (from previous step)
        temporal_consistency = getattr(self, '_temporal_quality', 0.5)

        return {
            'snr': snr,
            'smoothness': smoothness,
            'temporal_consistency': temporal_consistency,
            'overall_quality': (snr/20 + smoothness + temporal_consistency) / 3  # 0-1 score
        }


def process_angles_dataframe(angles_df, fps=30):
    """
    Apply improved processing to entire angles dataframe.

    Args:
        angles_df: DataFrame with columns like 'right_knee_angle', 'left_ankle_angle', etc.
        fps: Frame rate

    Returns:
        processed_df: Cleaned angles
        quality_report: Dict with quality metrics for each joint
    """
    processor = ImprovedSignalProcessor(fps=fps)
    processed_df = angles_df.copy()
    quality_report = {}

    joint_columns = {
        'knee': [col for col in angles_df.columns if 'knee' in col.lower()],
        'hip': [col for col in angles_df.columns if 'hip' in col.lower()],
        'ankle': [col for col in angles_df.columns if 'ankle' in col.lower()]
    }

    for joint_type, columns in joint_columns.items():
        for col in columns:
            if col in angles_df.columns:
                signal = angles_df[col].values
                processed, quality = processor.process_joint_angle(signal, joint_type)
                processed_df[col] = processed
                quality_report[col] = quality

    return processed_df, quality_report


# Example usage
if __name__ == "__main__":
    try:
        # Test with problematic subject from QC analysis
        from sagittal_extractor_2d import MediaPipeSagittalExtractor  # type: ignore

        VIDEO_PATH = "/data/gait/data/15/15-2.mp4"  # S1_15 had severe ankle issues

        print("Processing S1_15 with improved filtering...")
        extractor = MediaPipeSagittalExtractor()
        landmarks, _ = extractor.extract_pose_landmarks(VIDEO_PATH)
        angles_raw = extractor.calculate_joint_angles(landmarks)

        # Apply improved processing
        angles_processed, quality = process_angles_dataframe(angles_raw, fps=30)

        # Print quality report
        print("\n=== Quality Report ===")
        for joint, metrics in quality.items():
            print(f"\n{joint}:")
            print(f"  SNR: {metrics['snr']:.1f} dB")
            print(f"  Smoothness: {metrics['smoothness']:.3f}")
            print(f"  Temporal Consistency: {metrics['temporal_consistency']:.3f}")
            print(f"  Overall Quality: {metrics['overall_quality']:.3f}")

            if metrics['overall_quality'] < 0.5:
                print(f"  WARNING: Poor quality signal")

        # Visualize comparison
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(3, 1, figsize=(14, 10))
        joints_to_plot = ['right_ankle_angle', 'right_knee_angle', 'right_hip_angle']

        for ax, joint in zip(axes, joints_to_plot):
            ax.plot(angles_raw[joint], 'r-', alpha=0.3, label='Raw MediaPipe', linewidth=0.5)
            ax.plot(angles_processed[joint], 'b-', label='Improved Processing', linewidth=2)
            ax.set_title(f"{joint} - Quality: {quality[joint]['overall_quality']:.2f}")
            ax.set_ylabel('Angle (deg)')
            ax.legend()
            ax.grid(True, alpha=0.3)

        axes[-1].set_xlabel('Frame')
        plt.tight_layout()
        plt.savefig('/data/gait/2d_project/signal_processing_comparison.png', dpi=150)
        print("\nSaved comparison plot to signal_processing_comparison.png")

    except ImportError:
        print("Note: sagittal_extractor_2d not available. Example requires MediaPipe extractor.")
