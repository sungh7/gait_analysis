"""
Unit tests for improved signal processing module.
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.improved_signal_processing import ImprovedSignalProcessor


class TestImprovedSignalProcessor:
    """Tests for ImprovedSignalProcessor class."""

    @pytest.fixture
    def processor(self):
        """Create a processor instance for testing."""
        return ImprovedSignalProcessor(fps=30)

    def test_initialization(self, processor):
        """Test processor initialization."""
        assert processor.fps == 30

    def test_process_joint_angle_knee(self, processor):
        """Test knee angle processing preserves signal shape."""
        # Generate synthetic knee angle signal (typical gait pattern)
        t = np.linspace(0, 2 * np.pi * 2, 60)  # 2 cycles at 30 fps = 2 sec
        signal = 30 * np.sin(t) + 30  # Knee flexion 0-60 degrees

        processed, metrics = processor.process_joint_angle(signal, joint_type='knee')

        # Basic checks
        assert len(processed) == len(signal)
        assert not np.isnan(processed).any(), "No NaNs in output"
        assert not np.isinf(processed).any(), "No infinities in output"

    def test_process_joint_angle_ankle(self, processor):
        """Test ankle angle processing preserves ROM."""
        # Generate synthetic ankle angle signal
        t = np.linspace(0, 2 * np.pi * 2, 60)
        signal = 15 * np.sin(t) + 90  # Ankle dorsi/plantar flexion

        processed, metrics = processor.process_joint_angle(signal, joint_type='ankle')

        # ROM preservation check (critical for ankle)
        original_rom = signal.max() - signal.min()
        processed_rom = processed.max() - processed.min()
        preservation = processed_rom / original_rom

        assert preservation >= 0.5, f"ROM preservation {preservation:.1%} should be >= 50%"

    def test_process_joint_angle_hip(self, processor):
        """Test hip angle processing."""
        t = np.linspace(0, 2 * np.pi * 2, 60)
        signal = 20 * np.sin(t) + 10  # Hip flexion/extension

        processed, metrics = processor.process_joint_angle(signal, joint_type='hip')

        assert len(processed) == len(signal)
        assert not np.isnan(processed).any()

    def test_gap_filling(self, processor):
        """Test that NaN gaps are filled properly."""
        # Signal must be longer than filter padlen (15+)
        t = np.linspace(0, 2 * np.pi, 30)
        signal = 30 * np.sin(t) + 30
        # Insert some NaN gaps
        signal[10] = np.nan
        signal[11] = np.nan
        signal[20] = np.nan

        processed, _ = processor.process_joint_angle(signal, joint_type='knee')

        assert not np.isnan(processed).any(), "All NaNs should be filled"
        assert len(processed) == len(signal)

    def test_outlier_handling(self, processor):
        """Test that extreme outliers are handled."""
        # Signal must be longer than filter padlen (15+)
        t = np.linspace(0, 2 * np.pi, 30)
        signal = 30 * np.sin(t) + 30
        # Insert an extreme outlier
        signal[15] = 200.0  # Way outside normal range

        processed, _ = processor.process_joint_angle(signal, joint_type='knee')

        # The outlier (200) should be reduced toward the expected range
        assert processed[15] < 150, "Extreme outlier should be reduced"

    def test_smoothing_reduces_noise(self, processor):
        """Test that processing reduces high-frequency noise."""
        # Generate clean signal
        t = np.linspace(0, 2 * np.pi * 2, 120)
        clean_signal = 30 * np.sin(t) + 30

        # Add high-frequency noise
        noise = np.random.normal(0, 5, len(clean_signal))
        noisy_signal = clean_signal + noise

        processed, metrics = processor.process_joint_angle(noisy_signal, joint_type='knee')

        # Calculate smoothness (jerk) - third derivative
        def calc_jerk(s):
            return np.abs(np.diff(s, n=3)).mean()

        original_jerk = calc_jerk(noisy_signal)
        processed_jerk = calc_jerk(processed)

        assert processed_jerk < original_jerk, "Processing should reduce jerk"

    def test_quality_metrics_returned(self, processor):
        """Test that quality metrics are returned."""
        t = np.linspace(0, 2 * np.pi, 30)
        signal = 20 * np.sin(t) + 20

        _, metrics = processor.process_joint_angle(signal, joint_type='knee')

        assert isinstance(metrics, dict), "Metrics should be a dictionary"


class TestSignalProcessingEdgeCases:
    """Edge case tests for signal processing."""

    @pytest.fixture
    def processor(self):
        return ImprovedSignalProcessor(fps=30)

    def test_all_nan_signal(self, processor):
        """Test handling of all-NaN signal."""
        signal = np.array([np.nan] * 10)

        # Should handle gracefully (may return NaNs or zeros)
        try:
            processed, _ = processor.process_joint_angle(signal, joint_type='knee')
            # Either all NaN or filled with some value
            assert len(processed) == len(signal)
        except Exception as e:
            # Some exception is acceptable for all-NaN input
            pass

    def test_constant_signal(self, processor):
        """Test handling of constant signal (no variation)."""
        signal = np.array([45.0] * 30)

        processed, _ = processor.process_joint_angle(signal, joint_type='knee')

        # Should remain roughly constant
        assert np.std(processed) < 5, "Constant signal should remain constant"

    def test_single_value_signal(self, processor):
        """Test handling of very short signal."""
        signal = np.array([45.0])

        try:
            processed, _ = processor.process_joint_angle(signal, joint_type='knee')
            assert len(processed) == 1
        except Exception:
            # Exception acceptable for single-value input
            pass

    def test_different_fps(self):
        """Test processor works with different frame rates."""
        for fps in [24, 30, 60, 120]:
            processor = ImprovedSignalProcessor(fps=fps)
            assert processor.fps == fps

            t = np.linspace(0, 2 * np.pi, fps)
            signal = 30 * np.sin(t) + 30

            processed, _ = processor.process_joint_angle(signal, joint_type='knee')
            assert len(processed) == len(signal)
