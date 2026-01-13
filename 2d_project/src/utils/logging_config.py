"""
Logging configuration for gait analysis.

Provides standardized logging setup with console and file output.
"""

import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional

# Module-level logger cache
_loggers: dict = {}


def setup_logging(
    log_level: str = "INFO",
    log_dir: str = "./logs",
    log_file: bool = True,
    log_name: str = "gait_analysis"
) -> logging.Logger:
    """
    Configure logging for the gait analysis pipeline.

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_dir: Directory for log files
        log_file: Whether to write logs to file
        log_name: Name for the logger and log file prefix

    Returns:
        Configured logger instance
    """
    # Create logs directory if needed
    if log_file:
        Path(log_dir).mkdir(parents=True, exist_ok=True)

    # Configure format
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Get or create logger
    logger = logging.getLogger(log_name)

    # Clear existing handlers to avoid duplicates
    logger.handlers.clear()

    # Set level
    level = getattr(logging, log_level.upper(), logging.INFO)
    logger.setLevel(level)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(level)
    logger.addHandler(console_handler)

    # File handler (optional)
    if log_file:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_path = Path(log_dir) / f"{log_name}_{timestamp}.log"
        file_handler = logging.FileHandler(log_path, encoding='utf-8')
        file_handler.setFormatter(formatter)
        file_handler.setLevel(level)
        logger.addHandler(file_handler)
        logger.info(f"Logging to file: {log_path}")

    # Prevent propagation to root logger
    logger.propagate = False

    # Cache the logger
    _loggers[log_name] = logger

    return logger


def get_logger(name: str = "gait_analysis") -> logging.Logger:
    """
    Get a logger instance.

    If the logger hasn't been set up, creates one with default settings.

    Args:
        name: Logger name

    Returns:
        Logger instance
    """
    if name not in _loggers:
        # Create with default settings if not already set up
        return setup_logging(log_name=name, log_file=False)

    return _loggers[name]


class LoggerMixin:
    """
    Mixin class to provide logging capability to other classes.

    Usage:
        class MyClass(LoggerMixin):
            def my_method(self):
                self.logger.info("Doing something")
    """

    @property
    def logger(self) -> logging.Logger:
        """Get logger for this class."""
        name = f"gait_analysis.{self.__class__.__name__}"
        if name not in _loggers:
            logger = logging.getLogger(name)
            logger.setLevel(logging.INFO)
            if not logger.handlers:
                handler = logging.StreamHandler(sys.stdout)
                handler.setFormatter(logging.Formatter(
                    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
                ))
                logger.addHandler(handler)
            _loggers[name] = logger
        return _loggers[name]
