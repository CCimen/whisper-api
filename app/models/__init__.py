# This file makes the 'models' directory a Python package.

# Import models to ensure they are discoverable by the ModelRegistry
# when the package is imported.

try:
    # Import the base class first if needed elsewhere
    from app.services.model_registry import TranscriptionModel # Corrected import path
    # Import specific model implementations
    from .whisper_model import WhisperModel
except ImportError as e:
    # Log a warning if dependencies are missing but don't crash the app
    import logging
    logger = logging.getLogger(__name__)
    logger.warning(f"Could not import models, possibly due to missing dependencies: {e}")

# You can add imports for other model types here if you create them
# from .other_model import OtherModel