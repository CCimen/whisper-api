"""
Custom exceptions for the Whisper Transcription API.
"""

class ModelNotFoundError(Exception):
    """Raised when a model is not found."""
    pass

class TranscriptionError(Exception):
    """Raised when transcription fails."""
    pass

class DiarizationError(Exception):
    """Raised when diarization fails."""
    pass

class ConfigurationError(Exception):
    """Raised when there is a configuration error."""
    pass