"""
Custom exceptions for gait analysis.

Provides a hierarchy of exceptions for different error types
encountered during gait analysis processing.
"""


class GaitAnalysisError(Exception):
    """
    Base exception for all gait analysis errors.

    All custom exceptions in this module inherit from this class,
    allowing for easy catching of any gait-analysis-related error.
    """
    pass


class FileNotFoundError(GaitAnalysisError):
    """
    Raised when a required input file is not found.

    Attributes:
        filepath: Path to the missing file
        message: Detailed error message
    """

    def __init__(self, filepath: str, message: str = None):
        self.filepath = filepath
        self.message = message or f"File not found: {filepath}"
        super().__init__(self.message)


class InvalidDataError(GaitAnalysisError):
    """
    Raised when input data is invalid or corrupted.

    This includes:
    - Wrong file format
    - Missing required columns
    - Invalid data types
    - Empty files

    Attributes:
        message: Description of the data issue
        details: Additional context about the error
    """

    def __init__(self, message: str, details: str = None):
        self.message = message
        self.details = details
        full_message = message
        if details:
            full_message = f"{message}. Details: {details}"
        super().__init__(full_message)


class ProcessingError(GaitAnalysisError):
    """
    Raised when signal processing fails.

    This includes:
    - Filter application failures
    - Interpolation errors
    - Numerical computation errors

    Attributes:
        stage: Processing stage where error occurred
        message: Description of the error
    """

    def __init__(self, message: str, stage: str = None):
        self.message = message
        self.stage = stage
        full_message = message
        if stage:
            full_message = f"[{stage}] {message}"
        super().__init__(full_message)


class QualityCheckError(GaitAnalysisError):
    """
    Raised when data fails quality control checks.

    This includes:
    - Low visibility scores
    - Excessive noise
    - ROM preservation below threshold
    - Too many missing frames

    Attributes:
        check_name: Name of the failed quality check
        threshold: Required threshold value
        actual: Actual value that failed the check
        message: Description of the failure
    """

    def __init__(
        self,
        check_name: str,
        threshold: float = None,
        actual: float = None,
        message: str = None
    ):
        self.check_name = check_name
        self.threshold = threshold
        self.actual = actual

        if message:
            self.message = message
        elif threshold is not None and actual is not None:
            self.message = (
                f"Quality check '{check_name}' failed: "
                f"expected >= {threshold}, got {actual}"
            )
        else:
            self.message = f"Quality check '{check_name}' failed"

        super().__init__(self.message)


class ConfigurationError(GaitAnalysisError):
    """
    Raised when there's an issue with configuration.

    This includes:
    - Missing configuration file
    - Invalid configuration values
    - Missing required configuration keys

    Attributes:
        key: Configuration key that caused the error
        message: Description of the configuration issue
    """

    def __init__(self, message: str, key: str = None):
        self.message = message
        self.key = key
        full_message = message
        if key:
            full_message = f"Configuration error for '{key}': {message}"
        super().__init__(full_message)


class CycleDetectionError(GaitAnalysisError):
    """
    Raised when gait cycle detection fails.

    This includes:
    - No cycles detected
    - Invalid cycle boundaries
    - Template creation failure

    Attributes:
        message: Description of the detection failure
        num_frames: Number of frames in the signal
    """

    def __init__(self, message: str, num_frames: int = None):
        self.message = message
        self.num_frames = num_frames
        full_message = message
        if num_frames is not None:
            full_message = f"{message} (signal length: {num_frames} frames)"
        super().__init__(full_message)
