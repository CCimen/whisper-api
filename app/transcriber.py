import time
import os
import gc
import torch
import logging
import threading
import librosa
from typing import Dict, Any, Optional
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

from app.config import settings, WHISPER_MODELS
from app.exceptions import ModelNotFoundError, TranscriptionError

logger = logging.getLogger(__name__)

class ModelManager:
    """
    Manages Whisper model loading, caching, and GPU memory.
    """
    # Class-level lock for thread safety
    _model_lock = threading.RLock()
    
    # Model cache
    _model_cache = {}  # model_name -> (model, processor, pipeline)
    _current_model = None
    _model_load_timestamps = {}
    _model_usage_count = {}
    _initialized = False
    
    def __init__(self):
        """Initialize the model manager with optimal settings."""
        with ModelManager._model_lock:
            if ModelManager._initialized:
                logger.debug("ModelManager already initialized, reusing instance")
                return
            
            logger.info("Initializing ModelManager")
            
            # Apply PyTorch optimizations for faster inference
            if torch.cuda.is_available():
                logger.info("Setting up PyTorch optimization for GPU")
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.benchmark = True
            
            # Clean up GPU memory
            self._cleanup_gpu_memory()
            
            # Pre-load default model if configured
            if settings.PRELOAD_DEFAULT_MODEL:
                default_model = settings.DEFAULT_MODEL
                logger.info(f"Pre-loading default model: {default_model}")
                try:
                    self.get_pipeline(default_model)
                    logger.info(f"✓ Default model '{default_model}' successfully pre-loaded")
                except Exception as e:
                    logger.error(f"Failed to pre-load default model: {e}")
            
            ModelManager._initialized = True
    
    def _cleanup_gpu_memory(self):
        """Clean up GPU memory."""
        if torch.cuda.is_available():
            # Run garbage collection first
            gc.collect()
            
            # Empty GPU cache
            torch.cuda.empty_cache()
            
            if hasattr(torch.cuda, 'mem_get_info'):
                free_mem, total_mem = torch.cuda.mem_get_info(0)
                logger.info(f"GPU memory after cleanup: {free_mem / (1024**3):.2f}GB free of {total_mem / (1024**3):.2f}GB total")
    
    def _should_keep_model_in_memory(self, new_model_name):
        """
        Determine if we should keep the current model in memory.
        """
        # If multiple models in memory is disabled, always unload previous model
        if not settings.KEEP_MULTIPLE_MODELS_IN_MEMORY:
            return False
        
        # If the new model is already the current model, keep it
        if ModelManager._current_model == new_model_name:
            return True
        
        # Check if we have enough VRAM to load another model
        if torch.cuda.is_available() and hasattr(torch.cuda, 'mem_get_info'):
            free_mem, total_mem = torch.cuda.mem_get_info(0)
            free_gb = free_mem / (1024**3)
            
            # Check how many models are already loaded
            loaded_models = len(ModelManager._model_cache)
            
            # If we've reached the maximum number of models, unload one
            if loaded_models >= settings.MAX_MODELS_IN_MEMORY:
                logger.info(f"Reached maximum model count ({loaded_models}/{settings.MAX_MODELS_IN_MEMORY})")
                return False
            
            # If we don't have enough free memory, unload existing models
            if free_gb < settings.MIN_FREE_MEMORY_GB:
                logger.info(f"Low GPU memory ({free_gb:.1f}GB < {settings.MIN_FREE_MEMORY_GB}GB)")
                return False
            
            # We have enough memory and haven't reached the model limit
            return True
        
        # If CUDA is not available, don't keep multiple models in memory
        return False
    
    def _unload_model(self, model_name):
        """Unload a model and free GPU memory."""
        logger.info(f"Unloading model '{model_name}' to free memory")
        
        if model_name not in ModelManager._model_cache:
            logger.warning(f"Model {model_name} not found in cache for unloading")
            return
        
        try:
            # Get the model, processor, and pipeline
            _, _, pipeline_obj = ModelManager._model_cache[model_name]
            
            # Move model to CPU before deleting
            if torch.cuda.is_available():
                try:
                    pipeline_obj.model.to(torch.device("cpu"))
                    logger.info(f"Model {model_name} moved to CPU before unloading")
                except Exception as e:
                    logger.warning(f"Error moving model to CPU: {e}")
            
            # Delete from cache
            del ModelManager._model_cache[model_name]
            
            # Clean up
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            logger.info(f"✓ Model '{model_name}' unloaded and memory freed")
            return True
        except Exception as e:
            logger.warning(f"Error unloading model {model_name}: {e}")
            return False
    
    def _unload_least_used_model(self):
        """Unload the least recently used model."""
        if not ModelManager._model_cache:
            return
        
        # Find least used model, excluding the current model
        least_used_model = None
        lowest_usage = float('inf')
        
        for model_name, usage_count in ModelManager._model_usage_count.items():
            # Skip current model
            if model_name == ModelManager._current_model:
                continue
            
            # Find the model with lowest usage
            if model_name in ModelManager._model_cache and usage_count < lowest_usage:
                least_used_model = model_name
                lowest_usage = usage_count
        
        if least_used_model:
            logger.info(f"Unloading least used model '{least_used_model}' (usage count: {lowest_usage})")
            return self._unload_model(least_used_model)
        
        return False
    
    def get_pipeline(self, model_size):
        """
        Get a pipeline for the specified model size.
        If the model is not in cache, load it.
        """
        with ModelManager._model_lock:
            # Validate model size
            if model_size not in WHISPER_MODELS:
                available_models = list(WHISPER_MODELS.keys())
                logger.error(f"Model '{model_size}' not found. Available models: {available_models}")
                raise ModelNotFoundError(f"Model '{model_size}' not found. Available models: {available_models}")
            
            # Check if model is in cache
            if model_size in ModelManager._model_cache:
                logger.info(f"Using cached model: {model_size}")
                model, processor, pipeline_obj = ModelManager._model_cache[model_size]
                
                # Update usage statistics
                ModelManager._current_model = model_size
                if model_size in ModelManager._model_usage_count:
                    ModelManager._model_usage_count[model_size] += 1
                else:
                    ModelManager._model_usage_count[model_size] = 1
                
                return pipeline_obj
            
            # Model not in cache, need to load it
            logger.info(f"Loading model: {model_size}")
            
            # Check if we should unload other models
            if not self._should_keep_model_in_memory(model_size) and ModelManager._current_model is not None:
                # Unload current model if it's not the one we want
                if ModelManager._current_model != model_size:
                    self._unload_model(ModelManager._current_model)
            
            # If we're low on memory, unload least used model
            if torch.cuda.is_available() and hasattr(torch.cuda, 'mem_get_info'):
                free_mem, total_mem = torch.cuda.mem_get_info(0)
                free_gb = free_mem / (1024**3)
                if free_gb < settings.MIN_FREE_MEMORY_GB:
                    self._unload_least_used_model()
            
            # Load the model
            try:
                start_time = time.time()
                
                # Get the device and dtype
                device = "cuda:0" if torch.cuda.is_available() else "cpu"
                torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
                
                # Get the model ID
                model_id = WHISPER_MODELS[model_size]
                
                # Clean up memory before loading
                self._cleanup_gpu_memory()
                
                # Load model
                model = AutoModelForSpeechSeq2Seq.from_pretrained(
                    model_id, 
                    torch_dtype=torch_dtype, 
                    use_safetensors=True,
                    cache_dir=settings.MODELS_CACHE_DIR
                )
                model.to(device)
                
                # Load processor
                processor = AutoProcessor.from_pretrained(model_id)
                
                # Create pipeline
                pipeline_obj = pipeline(
                    "automatic-speech-recognition",
                    model=model,
                    tokenizer=processor.tokenizer,
                    feature_extractor=processor.feature_extractor,
                    torch_dtype=torch_dtype,
                    device=device,
                )
                
                # Cache the model, processor, and pipeline
                ModelManager._model_cache[model_size] = (model, processor, pipeline_obj)
                ModelManager._current_model = model_size
                ModelManager._model_load_timestamps[model_size] = time.time()
                ModelManager._model_usage_count[model_size] = 1
                
                load_time = time.time() - start_time
                logger.info(f"✓ Model '{model_size}' loaded in {load_time:.2f}s")
                
                return pipeline_obj
            
            except Exception as e:
                logger.exception(f"Error loading model '{model_size}': {e}")
                raise TranscriptionError(f"Failed to load model: {e}")


def check_gpu():
    """Verify GPU is available and print info"""
    if not torch.cuda.is_available():
        return {
            "available": False,
            "message": "CUDA not available. This service requires GPU."
        }
    
    device_count = torch.cuda.device_count()
    devices = []
    
    for i in range(device_count):
        device_info = {
            "id": i,
            "name": torch.cuda.get_device_name(i)
        }
        if hasattr(torch.cuda, 'get_device_properties'):
            prop = torch.cuda.get_device_properties(i)
            device_info["memory"] = f"{prop.total_memory / 1024**3:.2f} GB"
        devices.append(device_info)
    
    return {
        "available": True,
        "device_count": device_count,
        "active_device": torch.cuda.current_device(),
        "devices": devices
    }

# Create a global model manager instance
model_manager = ModelManager()

async def transcribe_audio(file_path, language="sv", model_size="medium"):
    """Transcribe audio file using Whisper on GPU with model caching"""
    start_time = time.time()
    
    try:
        # Verify GPU is available
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA not available. This service requires GPU.")
        
        logger.info(f"Transcribing file: {file_path} with model: {model_size}, language: {language}")
        
        # Get the pipeline from model manager (uses caching)
        pipe = model_manager.get_pipeline(model_size)
        
        # Set up generation kwargs
        generate_kwargs = {
            "task": "transcribe", 
            "language": language
        }
        
        # Get file info
        file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
        logger.info(f"File size: {file_size_mb:.2f} MB")
        
        # Run transcription with torch.inference_mode() for better memory usage
        with torch.inference_mode():
            result = pipe(
                file_path,
                chunk_length_s=30,      # Process in 30-second chunks
                batch_size=8,           # Adjust based on GPU memory
                return_timestamps=True, # Enable timestamps in output
                generate_kwargs=generate_kwargs
            )
        
        # Get audio duration
        duration = librosa.get_duration(path=file_path)
        
        # Extract text and timestamps
        full_text = result["text"]
        segments = []
        
        if "chunks" in result:
            for chunk in result["chunks"]:
                start, end = chunk["timestamp"]
                segments.append({
                    "start": start,
                    "end": end,
                    "text": chunk["text"]
                })
        
        processing_time = time.time() - start_time
        
        # Log stats
        logger.info(f"Transcription completed in {processing_time:.2f}s for {duration:.2f}s audio")
        logger.info(f"Realtime factor: {duration/processing_time:.2f}x")
        logger.info(f"Output text length: {len(full_text)} characters")
        
        return {
            "text": full_text,
            "segments": segments,
            "duration": duration,
            "processing_time": processing_time
        }
    except Exception as e:
        logger.exception(f"Transcription failed: {e}")
        raise TranscriptionError(f"Transcription failed: {e}")