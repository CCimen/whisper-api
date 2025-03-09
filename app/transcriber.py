"""
Transcription service with optimized model loading, caching, and GPU utilization.

This module provides functionality to transcribe audio using Whisper models with
optimized GPU usage, memory management, and caching strategies for performance.
"""

import time
import os
import gc
import torch
import logging
import threading
import librosa
import numpy as np
import concurrent.futures
import subprocess
from typing import Dict, Any, Optional, Callable, Tuple, List
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

from app.config import settings, WHISPER_MODELS, MODEL_CACHE_CONFIG
from app.exceptions import ModelNotFoundError, TranscriptionError

logger = logging.getLogger(__name__)

class ModelManager:
    """
    Manages Whisper model loading, caching, and GPU memory.
    
    This class optimizes model handling by:
    - Caching models based on size and usage
    - Intelligently managing GPU memory
    - Pre-loading frequently used models
    - Applying optimal inference settings
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
                
                # Set optimal thread settings
                torch.set_num_threads(min(8, os.cpu_count() or 4))
                if hasattr(torch, 'set_num_interop_threads'):
                    torch.set_num_interop_threads(min(8, os.cpu_count() or 4))
            
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
        """Clean up GPU memory with optimized strategies."""
        if torch.cuda.is_available():
            # Run garbage collection first
            gc.collect()
            
            # Empty GPU cache
            torch.cuda.empty_cache()
            
            # Report memory status
            if hasattr(torch.cuda, 'mem_get_info'):
                free_mem, total_mem = torch.cuda.mem_get_info(0)
                logger.info(f"GPU memory after cleanup: {free_mem / (1024**3):.2f}GB free of {total_mem / (1024**3):.2f}GB total")
    
    def _should_keep_model_in_memory(self, new_model_name):
        """
        Determine if we should keep the current model in memory using an optimized strategy.
        
        This looks at:
        - Config settings
        - Available GPU memory
        - Model usage patterns
        - Model sizes and relationships
        """
        # If multiple models is disabled, always unload previous model
        if not settings.KEEP_MULTIPLE_MODELS_IN_MEMORY:
            return False
        
        # If the new model is already the current model, keep it
        if ModelManager._current_model == new_model_name:
            return True
        
        # Check if the model we're loading is smaller than current
        if ModelManager._current_model and new_model_name in MODEL_CACHE_CONFIG and ModelManager._current_model in MODEL_CACHE_CONFIG:
            current_size = MODEL_CACHE_CONFIG[ModelManager._current_model]["max_memory_gb"]
            new_size = MODEL_CACHE_CONFIG[new_model_name]["max_memory_gb"]
            
            # If new model is much larger, unload current
            if new_size > current_size * 2:
                logger.info(f"New model {new_model_name} is significantly larger than current model, unloading current")
                return False
        
        # Check GPU memory
        if torch.cuda.is_available() and hasattr(torch.cuda, 'mem_get_info'):
            free_mem, total_mem = torch.cuda.mem_get_info(0)
            free_gb = free_mem / (1024**3)
            
            # Calculate model size estimate
            model_size_estimate = MODEL_CACHE_CONFIG.get(new_model_name, {}).get("max_memory_gb", 4.0)
            
            # Check if we have enough memory for this specific model
            if free_gb < model_size_estimate + settings.MIN_FREE_MEMORY_GB:
                logger.info(f"Insufficient memory for model {new_model_name}, need to unload current model")
                return False
            
            # Check how many models are already loaded
            loaded_models = len(ModelManager._model_cache)
            
            # If we've reached the maximum number of models, unload one
            if loaded_models >= settings.MAX_MODELS_IN_MEMORY:
                logger.info(f"Reached maximum model count ({loaded_models}/{settings.MAX_MODELS_IN_MEMORY})")
                return False
            
            # We have enough memory and haven't reached the model limit
            return True
        
        # If CUDA is not available, don't keep multiple models in memory
        return False
    
    def _unload_model(self, model_name):
        """Unload a model with optimal memory cleanup."""
        logger.info(f"Unloading model '{model_name}' to free memory")
        
        if model_name not in ModelManager._model_cache:
            logger.warning(f"Model {model_name} not found in cache for unloading")
            return
        
        try:
            # Get model components
            model, processor, pipeline_obj = ModelManager._model_cache[model_name]
            
            # Move model to CPU before deleting - helps prevent CUDA OOM errors
            if torch.cuda.is_available():
                try:
                    pipeline_obj.model.to(torch.device("cpu"))
                    logger.info(f"Model {model_name} moved to CPU before unloading")
                except Exception as e:
                    logger.warning(f"Error moving model to CPU: {e}")
            
            # Delete from cache and clear references
            del ModelManager._model_cache[model_name]
            
            # More aggressive memory cleanup
            for attr in ['model', 'tokenizer', 'feature_extractor', 'processor']:
                if hasattr(pipeline_obj, attr):
                    setattr(pipeline_obj, attr, None)
            
            # Explicitly delete pipeline to help garbage collection
            pipeline_obj = None
            model = None
            processor = None
            
            # Force Python garbage collection
            gc.collect()
            
            # Empty CUDA cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                if hasattr(torch.cuda, 'mem_reset_peak_memory_stats'):
                    torch.cuda.reset_peak_memory_stats()
            
            logger.info(f"✓ Model '{model_name}' unloaded and memory freed")
            return True
        except Exception as e:
            logger.warning(f"Error unloading model {model_name}: {e}")
            return False
    
    def _unload_least_used_model(self):
        """
        Unload the least valuable model using a scoring system that considers:
        - Usage count
        - Last used timestamp
        - Model size
        """
        if not ModelManager._model_cache:
            return
        
        # Skip if we only have the current model loaded
        if len(ModelManager._model_cache) == 1 and ModelManager._current_model in ModelManager._model_cache:
            return
        
        # Calculate a score for each model that balances usage and recency
        current_time = time.time()
        model_scores = {}
        
        for model_name in ModelManager._model_cache:
            # Skip current model
            if model_name == ModelManager._current_model:
                continue
            
            # Get usage metrics
            usage_count = ModelManager._model_usage_count.get(model_name, 0)
            last_used = ModelManager._model_load_timestamps.get(model_name, 0)
            time_since_used = current_time - last_used
            
            # Calculate score (lower is more likely to be unloaded)
            # This formula prioritizes:
            # - More frequently used models (higher usage_count)
            # - Recently used models (lower time_since_used)
            model_scores[model_name] = usage_count / (1 + time_since_used/3600)  # Time factor in hours
        
        if not model_scores:
            return False
            
        # Find model with lowest score
        model_to_unload = min(model_scores, key=model_scores.get)
        logger.info(f"Unloading least valuable model '{model_to_unload}' (score: {model_scores[model_to_unload]:.2f})")
        return self._unload_model(model_to_unload)
    
    def get_pipeline(self, model_size):
        """
        Get an optimized pipeline for the specified model size.
        If the model is not in cache, load it with optimal settings.
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
                
                # Update timestamp
                ModelManager._model_load_timestamps[model_size] = time.time()
                
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
            
            # Load the model with optimized settings
            try:
                start_time = time.time()
                
                # Get the device and dtype
                device = "cuda:0" if torch.cuda.is_available() else "cpu"
                torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
                
                # Get the model ID
                model_id = WHISPER_MODELS[model_size]
                
                # Clean up memory before loading
                self._cleanup_gpu_memory()
                
                # Load model with optimized settings
                model = AutoModelForSpeechSeq2Seq.from_pretrained(
                    model_id, 
                    torch_dtype=torch_dtype, 
                    use_safetensors=True,
                    low_cpu_mem_usage=True,
                    cache_dir=settings.MODELS_CACHE_DIR
                )
                
                # Optimize transformer model
                if torch.cuda.is_available():
                    # Move to GPU with optimal settings
                    model = model.to(device)
                    
                    # Apply compile if available (PyTorch 2.0+)
                    if hasattr(torch, 'compile'):
                        try:
                            model = torch.compile(model)
                            logger.info("Applied torch.compile() optimization")
                        except Exception as e:
                            logger.warning(f"Could not apply torch.compile(): {e}")
                
                # Load processor (much lighter memory-wise)
                processor = AutoProcessor.from_pretrained(
                    model_id,
                    cache_dir=settings.MODELS_CACHE_DIR
                )
                
                # Get optimal batch size based on model size
                if model_size == "tiny": 
                    batch_size = 16
                elif model_size == "small":
                    batch_size = 12
                elif model_size == "medium":
                    batch_size = 8
                else:
                    batch_size = 4  # Conservative for large
                
                # Tune chunk size based on model
                if model_size == "tiny":
                    chunk_size = 30
                elif model_size == "small":
                    chunk_size = 30
                elif model_size == "medium":
                    chunk_size = 25
                else:
                    chunk_size = 20
                
                # Create pipeline with optimized settings
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
                
                # Store optimal settings for this model in the pipeline object
                pipeline_obj.optimal_batch_size = batch_size
                pipeline_obj.optimal_chunk_size = chunk_size
                
                load_time = time.time() - start_time
                logger.info(f"✓ Model '{model_size}' loaded in {load_time:.2f}s")
                
                return pipeline_obj
            
            except Exception as e:
                logger.exception(f"Error loading model '{model_size}': {e}")
                raise TranscriptionError(f"Failed to load model: {e}")


def check_gpu():
    """
    Verify GPU availability and provide detailed information about capabilities.
    Returns a dictionary with comprehensive GPU status.
    """
    if not torch.cuda.is_available():
        return {
            "available": False,
            "message": "CUDA not available. This service requires GPU."
        }
    
    device_count = torch.cuda.device_count()
    devices = []
    
    # Get total system memory if possible
    system_memory_gb = None
    try:
        if os.name == 'posix':
            with open('/proc/meminfo', 'r') as f:
                for line in f:
                    if 'MemTotal' in line:
                        system_memory_gb = int(line.split()[1]) / (1024 * 1024)
                        break
    except Exception:
        pass
    
    # Collect detailed information about each GPU
    for i in range(device_count):
        device_info = {
            "id": i,
            "name": torch.cuda.get_device_name(i)
        }
        
        if hasattr(torch.cuda, 'get_device_properties'):
            prop = torch.cuda.get_device_properties(i)
            device_info.update({
                "memory": f"{prop.total_memory / 1024**3:.2f} GB",
                "memory_bytes": int(prop.total_memory),
                "compute_capability": f"{prop.major}.{prop.minor}",
                "multi_processor_count": prop.multi_processor_count
            })
            
        if hasattr(torch.cuda, 'mem_get_info'):
            free_mem, total_mem = torch.cuda.mem_get_info(i)
            device_info.update({
                "free_memory": f"{free_mem / 1024**3:.2f} GB",
                "free_memory_bytes": int(free_mem),
                "utilization": f"{(1 - free_mem/total_mem) * 100:.1f}%"
            })
            
        devices.append(device_info)
    
    # Gather CUDA version information
    cuda_version = torch.version.cuda if hasattr(torch.version, 'cuda') else "Unknown"
    
    return {
        "available": True,
        "device_count": device_count,
        "active_device": torch.cuda.current_device(),
        "devices": devices,
        "cuda_version": cuda_version,
        "torch_version": torch.__version__,
        "system_memory_gb": system_memory_gb
    }

async def estimate_audio_duration(file_path: str) -> float:
    """
    Get audio duration quickly without loading the entire file.
    Uses ffprobe for efficiency.
    """
    try:
        cmd = [
            "ffprobe",
            "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            file_path
        ]
        duration = float(subprocess.check_output(cmd).decode('utf-8').strip())
        return duration
    except Exception as e:
        logger.warning(f"Could not estimate audio duration: {e}")
        # Fallback: try to get duration from file size (very rough estimate)
        try:
            file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
            # Rough estimate: 1MB ≈ 1 minute of audio
            return file_size_mb * 60
        except:
            return 0

# Create a global model manager instance
model_manager = ModelManager()

async def transcribe_audio(
    file_path: str, 
    language: str = "sv", 
    model_size: str = "medium",
    progress_callback: Optional[Callable[[float], None]] = None
):
    """
    Transcribe audio file using Whisper on GPU with model caching and progress tracking.
    
    Args:
        file_path: Path to the audio file
        language: Language code
        model_size: Size of the whisper model to use
        progress_callback: Function to call with progress updates (0.0-1.0)
        
    Returns:
        Dictionary with transcription results
    """
    start_time = time.time()
    
    try:
        # Verify GPU is available
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA not available. This service requires GPU.")
        
        logger.info(f"Transcribing file: {file_path} with model: {model_size}, language: {language}")
        
        # Initial progress update
        if progress_callback:
            progress_callback(0.05)
        
        # Get audio duration for progress tracking
        try:
            audio_duration = await estimate_audio_duration(file_path)
            logger.info(f"Estimated audio duration: {audio_duration:.2f} seconds")
        except Exception as e:
            logger.warning(f"Could not estimate audio duration: {e}")
            audio_duration = None
        
        # Get the pipeline from model manager (uses caching)
        pipe = model_manager.get_pipeline(model_size)
        
        # Update progress after model loading
        if progress_callback:
            progress_callback(0.1)
        
        # Set up generation kwargs
        generate_kwargs = {
            "task": "transcribe", 
            "language": language
        }
        
        # Get file info
        try:
            file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
            logger.info(f"File size: {file_size_mb:.2f} MB")
        except Exception as e:
            logger.warning(f"Could not get file size: {e}")
        
        # Get optimal settings from the pipeline object
        batch_size = getattr(pipe, "optimal_batch_size", 8)
        chunk_length_s = getattr(pipe, "optimal_chunk_size", 30)
        
        # Run transcription with torch.inference_mode() for better memory usage
        with torch.inference_mode():
            # Pin memory for better CPU->GPU transfer performance
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            # Process audio
            result = pipe(
                file_path,
                chunk_length_s=chunk_length_s,
                batch_size=batch_size,
                return_timestamps=True,
                generate_kwargs=generate_kwargs
                # Removed chunk_callback parameter as it's not supported
            )
            
            # Update progress after processing
            if progress_callback:
                progress_callback(0.9)
        
        # Get audio duration
        if audio_duration is None:
            try:
                audio_duration = librosa.get_duration(path=file_path)
            except Exception as e:
                logger.warning(f"Could not get audio duration with librosa: {e}")
                # Try to infer from results
                if "chunks" in result and result["chunks"]:
                    audio_duration = result["chunks"][-1]["timestamp"][1]
                else:
                    audio_duration = 0
        
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
        
        # Update progress to almost complete
        if progress_callback:
            progress_callback(0.95)
        
        # Log stats
        logger.info(f"Transcription completed in {processing_time:.2f}s for {audio_duration:.2f}s audio")
        logger.info(f"Realtime factor: {audio_duration/processing_time:.2f}x")
        logger.info(f"Output text length: {len(full_text)} characters")
        
        # Final progress update
        if progress_callback:
            progress_callback(1.0)
        
        return {
            "text": full_text,
            "segments": segments,
            "duration": audio_duration,
            "processing_time": processing_time
        }
        
    except Exception as e:
        logger.exception(f"Transcription failed: {e}")
        raise TranscriptionError(f"Transcription failed: {e}")