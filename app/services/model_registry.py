# app/services/model_registry.py
import logging
import importlib
import inspect
from typing import Dict, Any, Optional, List, Callable, Type, Union

# Configure logging
logger = logging.getLogger(__name__)

class TranscriptionModel:
    """Base class for all transcription models."""
    name: Optional[str] = None  # Should be set by subclasses or during registration

    def load(self, **kwargs):
        """Load the model into memory."""
        raise NotImplementedError("load method must be implemented by subclass")

    def unload(self):
        """Unload the model and free associated resources."""
        raise NotImplementedError("unload method must be implemented by subclass")

    def is_loaded(self) -> bool:
        """Check if the model is currently loaded."""
        raise NotImplementedError("is_loaded method must be implemented by subclass")

    def transcribe(self, audio_path: str, **kwargs) -> Dict[str, Any]:
        """Transcribe an audio file."""
        raise NotImplementedError("transcribe method must be implemented by subclass")

class ModelRegistry:
    """Registry for managing available transcription models."""
    _models: Dict[str, Type[TranscriptionModel]] = {} # Maps name -> class
    _instances: Dict[str, TranscriptionModel] = {}    # Maps name -> instance

    @classmethod
    def register(cls, model_class: Type[TranscriptionModel]):
        """
        Decorator to register a model class with the registry.
        The model class must have a 'name' attribute or it should be
        determinable (e.g., whisper-<size>).
        """
        # Determine the registration name
        model_name = getattr(model_class, 'name', None)
        if model_name:
             cls._models[model_name] = model_class
             logger.info(f"Registered model '{model_name}' from class {model_class.__name__}")
        else:
             # Handle cases like WhisperModel where name depends on size
             # We register the class itself and handle instantiation in get_model
             # Assume class name implies type for now if no name attr.
             base_name = model_class.__name__.lower().replace('model','') # e.g., 'whisper'
             if base_name == 'whisper':
                  # Special handling for whisper: register the class, size handled later
                  cls._models['whisper_base_class'] = model_class
                  logger.info(f"Registered base class {model_class.__name__}. Specific sizes will be handled by get_model.")
             else:
                  logger.warning(f"Model class {model_class.__name__} has no 'name' attribute and is not a known base type. Cannot register automatically.")

        return model_class # Return class for decorator usage

    @classmethod
    def get_model(cls, name: str) -> TranscriptionModel:
        """
        Get a model instance by name. Handles instantiation if needed.
        Supports names like 'whisper-large', 'whisper-kblab-large'.
        """
        if name in cls._instances:
            return cls._instances[name]

        # Handle Whisper model instantiation
        if name.startswith('whisper-'):
            size = name.split('-', 1)[1] # e.g., 'large', 'kblab-large'
            whisper_class = cls._models.get('whisper_base_class')
            if whisper_class:
                try:
                    logger.info(f"Instantiating WhisperModel for size: {size}")
                    instance = whisper_class(model_size=size)
                    instance.name = name # Assign the full name
                    cls._instances[name] = instance
                    return instance
                except Exception as e:
                    logger.exception(f"Failed to instantiate WhisperModel for size {size}: {e}")
                    raise ValueError(f"Could not create model instance for {name}")
            else:
                 raise ValueError("WhisperModel base class not registered.")

        # Handle other registered models (if any)
        elif name in cls._models:
            model_class = cls._models[name]
            try:
                logger.info(f"Instantiating model {name} from class {model_class.__name__}")
                instance = model_class()
                # Ensure name attribute is set if not already
                if not getattr(instance, 'name', None):
                     instance.name = name
                cls._instances[name] = instance
                return instance
            except Exception as e:
                logger.exception(f"Failed to instantiate model {name}: {e}")
                raise ValueError(f"Could not create model instance for {name}")

        else:
            available = cls.available_models() # Ensure discovery runs if needed
            logger.error(f"Model '{name}' not found in registry. Available: {available}")
            raise ModelNotFoundError(f"Model '{name}' not registered. Available models: {available}")

    @classmethod
    def discover_models(cls, module_path: str = "app.models"):
        """
        Auto-discover and register model implementations from a given module path.
        Models should use the @ModelRegistry.register decorator.
        """
        if 'whisper_base_class' in cls._models: # Avoid re-discovery if already done
            return

        logger.info(f"Discovering models in module: {module_path}")
        try:
            module = importlib.import_module(module_path)
            # The @register decorator handles registration when the module is imported
            logger.info(f"Successfully scanned module {module_path} for registered models.")

        except ImportError:
            logger.warning(f"Could not import model module: {module_path}")
        except Exception as e:
            logger.error(f"Error discovering models in {module_path}: {e}", exc_info=True)

        # Manually trigger registration for Whisper sizes based on config mapping
        try:
            from app.config import WHISPER_MODEL_MAPPING
            for size_key in WHISPER_MODEL_MAPPING.keys():
                model_key = f"whisper-{size_key}"
                # We don't need to store the class here again, get_model handles it
                # Just ensures these names are known implicitly
        except ImportError:
            logger.error("Could not import WHISPER_MODEL_MAPPING from config.")

        registered = list(cls._models.keys()) + [f"whisper-{s}" for s in WHISPER_MODEL_MAPPING.keys()]
        logger.info(f"Effectively available model keys (after discovery/implicit): {list(set(registered))}")


    @classmethod
    def available_models(cls) -> List[str]:
        """Get a list of available model names (including implicitly available whisper sizes)."""
        cls.discover_models() # Ensure discovery has run at least once

        # Combine explicitly registered models and implicitly available whisper models
        available = list(cls._models.keys())
        # Add whisper sizes from config
        try:
            from app.config import WHISPER_MODEL_MAPPING
            whisper_keys = [f"whisper-{size}" for size in WHISPER_MODEL_MAPPING.keys()]
            available.extend(whisper_keys)
        except ImportError:
            pass

        # Remove the base class key if present
        if 'whisper_base_class' in available:
            available.remove('whisper_base_class')

        return sorted(list(set(available))) # Return unique sorted list

    @classmethod
    def unload_all(cls) -> None:
        """Unload all currently loaded model instances."""
        if not cls._instances:
            return

        logger.info(f"Unloading all ({len(cls._instances)}) model instances...")
        # Iterate over a copy of keys as unload modifies the dict
        for name in list(cls._instances.keys()):
            instance = cls._instances.pop(name, None) # Remove from dict
            if instance:
                try:
                    instance.unload()
                except Exception as e:
                    logger.error(f"Error unloading model {name}: {e}", exc_info=True)
        logger.info("Finished unloading models.")
        # cls._instances.clear() # Should be empty now

    @classmethod
    def get_model_info(cls) -> Dict[str, Any]:
        """Get detailed information about available models."""
        info = {}
        available = cls.available_models()

        for name in available:
            model_info = {"name": name, "loaded": name in cls._instances}
            try:
                # If it's a whisper model, add size info
                if name.startswith('whisper-'):
                    size = name.split('-', 1)[1]
                    model_info["type"] = "WhisperModel"
                    model_info["size"] = size
                    # Add estimated memory (can refine this)
                    memory_gb = {
                        "tiny": 1, "small": 2, "medium": 5, "large": 10, "kblab-large": 10
                    }.get(size, 0)
                    model_info["estimated_memory_gb"] = memory_gb

                # Add instance-specific info if loaded
                if name in cls._instances:
                    instance = cls._instances[name]
                    model_info["is_loaded_check"] = instance.is_loaded() # Verify load status
                    if hasattr(instance, '_loaded_device') and instance._loaded_device:
                         model_info["loaded_device"] = str(instance._loaded_device)
                elif name in cls._models:
                     # Info from the class if not instantiated
                     model_info["type"] = cls._models[name].__name__

            except Exception as e:
                logger.warning(f"Error getting info for model {name}: {e}")
                model_info["error"] = str(e)
            info[name] = model_info

        return info