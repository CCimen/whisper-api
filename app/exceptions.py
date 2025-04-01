"""
Custom exception classes for the Whisper Transcription API.
"""

class BaseApiException(Exception):
    """Base class for custom API exceptions."""
    def __init__(self, message: str):
        self.message = message
        super().__init__(self.message)

class ModelNotFoundError(BaseApiException):
    """Raised when a requested model is not found or configured."""
    pass

class TranscriptionError(BaseApiException):
    """Raised when an error occurs during the transcription process."""
    pass

class DiarizationError(BaseApiException):
    """Raised when an error occurs during the speaker diarization process."""
    pass

class ConfigurationError(BaseApiException):
    """Raised when there is a configuration problem preventing operation."""
    pass

class FileProcessingError(BaseApiException):
    """Raised when there is an error processing an uploaded file."""
    pass