"""
Unit tests for utility modules (config, logging, exceptions).
"""

import pytest
import sys
import tempfile
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.config import load_config, get_config
from src.utils.logging_config import setup_logging, get_logger
from src.utils.exceptions import (
    GaitAnalysisError,
    InvalidDataError,
    ProcessingError,
    QualityCheckError,
    ConfigurationError,
)


class TestConfig:
    """Tests for configuration module."""

    def test_load_config_exists(self):
        """Test loading existing config file."""
        # Should load from project root
        config = load_config()
        assert isinstance(config, dict)

    def test_config_has_required_sections(self):
        """Test config has all required sections."""
        config = get_config()

        required_sections = [
            'processing',
            'signal_processing',
            'kinematic_constraints',
            'quality_control',
            'paths',
        ]

        for section in required_sections:
            assert section in config, f"Missing config section: {section}"

    def test_signal_processing_config(self):
        """Test signal processing configuration."""
        config = get_config()
        sp = config['signal_processing']

        # Check joint configs exist
        assert 'knee' in sp
        assert 'hip' in sp
        assert 'ankle' in sp

        # Check ankle has geometric range flag
        ankle_config = sp['ankle']
        assert ankle_config.get('enable_velocity_constraints') == False

    def test_kinematic_constraints_config(self):
        """Test kinematic constraints configuration."""
        config = get_config()
        kc = config['kinematic_constraints']

        # Check knee limits
        assert kc['knee']['angle_range'] == [0, 140]

        # Check ankle uses geometric range
        assert kc['ankle']['angle_range'] == [0, 180]

    def test_get_config_caching(self):
        """Test that get_config returns cached instance."""
        config1 = get_config()
        config2 = get_config()

        # Should be same instance (cached)
        assert config1 is config2

    def test_get_config_reload(self):
        """Test config reload functionality."""
        config1 = get_config()
        config2 = get_config(reload=True)

        # Values should be equal even after reload
        assert config1 == config2


class TestLogging:
    """Tests for logging module."""

    def test_setup_logging_returns_logger(self):
        """Test that setup_logging returns a logger."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = setup_logging(
                log_level="INFO",
                log_dir=tmpdir,
                log_file=True,
                log_name="test_logger"
            )

            assert logger is not None
            assert logger.name == "test_logger"

    def test_get_logger_returns_logger(self):
        """Test that get_logger returns a logger."""
        logger = get_logger("test_module")
        assert logger is not None

    def test_log_levels(self):
        """Test different log levels work."""
        for level in ["DEBUG", "INFO", "WARNING", "ERROR"]:
            logger = setup_logging(
                log_level=level,
                log_file=False,
                log_name=f"test_{level}"
            )
            assert logger is not None

    def test_log_file_creation(self):
        """Test that log file is created."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = setup_logging(
                log_level="INFO",
                log_dir=tmpdir,
                log_file=True,
                log_name="file_test"
            )

            logger.info("Test message")

            # Check log file was created
            log_files = list(Path(tmpdir).glob("file_test_*.log"))
            assert len(log_files) >= 1


class TestExceptions:
    """Tests for custom exceptions."""

    def test_gait_analysis_error_base(self):
        """Test base exception."""
        with pytest.raises(GaitAnalysisError):
            raise GaitAnalysisError("Test error")

    def test_invalid_data_error(self):
        """Test InvalidDataError."""
        with pytest.raises(InvalidDataError) as exc_info:
            raise InvalidDataError("Bad data", details="Missing columns")

        assert "Bad data" in str(exc_info.value)
        assert exc_info.value.details == "Missing columns"

    def test_processing_error(self):
        """Test ProcessingError."""
        with pytest.raises(ProcessingError) as exc_info:
            raise ProcessingError("Filter failed", stage="smoothing")

        assert "Filter failed" in str(exc_info.value)
        assert exc_info.value.stage == "smoothing"

    def test_quality_check_error(self):
        """Test QualityCheckError."""
        with pytest.raises(QualityCheckError) as exc_info:
            raise QualityCheckError(
                check_name="rom_preservation",
                threshold=0.5,
                actual=0.3
            )

        assert "rom_preservation" in str(exc_info.value)
        assert exc_info.value.threshold == 0.5
        assert exc_info.value.actual == 0.3

    def test_configuration_error(self):
        """Test ConfigurationError."""
        with pytest.raises(ConfigurationError) as exc_info:
            raise ConfigurationError("Invalid value", key="fps")

        assert "fps" in str(exc_info.value)

    def test_exception_hierarchy(self):
        """Test that all exceptions inherit from GaitAnalysisError."""
        exceptions = [
            InvalidDataError("test"),
            ProcessingError("test"),
            QualityCheckError("test"),
            ConfigurationError("test"),
        ]

        for exc in exceptions:
            assert isinstance(exc, GaitAnalysisError)

    def test_catching_base_exception(self):
        """Test catching base exception catches all derived."""
        try:
            raise InvalidDataError("test")
        except GaitAnalysisError as e:
            assert True  # Correctly caught
        except Exception:
            pytest.fail("Should have been caught by GaitAnalysisError")
