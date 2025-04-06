"""
Whisper transcription model implementation using transformers pipeline.
"""
import os
import gc
import time
import logging
import weakref
import threading
from typing import Dict, Any, Optional, Callable, Tuple, List

import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from transformers.pipelines.audio_utils import ffmpeg_read

from app.config import settings, WHISPER_MODEL_MAPPING
from app.exceptions import ModelNotFoundError, TranscriptionError
from app.services.model_registry import TranscriptionModel, ModelRegistry

logger = logging.getLogger(__name__)

DEFAULT_COMPUTE_TYPES = {
    "cuda": torch.float16,
    "cpu": torch.float32,
    "mps": torch.float32 # Mac Silicon
}

@ModelRegistry.register
class WhisperModel(TranscriptionModel):
    """
    Whisper transcription model implementation using Hugging Face Transformers.
    Handles model loading, unloading, and transcription execution.
    """
    _model_lock = threading.RLock()
    
    def __init__(self, model_size: str):
        if model_size not in WHISPER_MODEL_MAPPING:
            available_keys = list(WHISPER_MODEL_MAPPING.keys())
            raise ModelNotFoundError(f"Invalid model size '{model_size}'. Available configured sizes: {available_keys}")

        self.model_size = model_size
        self.model_id = WHISPER_MODEL_MAPPING[model_size]
        self.name = f"whisper-{model_size}"

        self._model = None
        self._processor = None
        self._pipeline = None
        self._loaded_device = None
        self._loaded_dtype = None
        self._loaded = False
        logger.info(f"[MODEL:{self.name}] Initialized wrapper (ID: {self.model_id})")

    def _get_compute_settings(self, device_str: str) -> Tuple[torch.device, torch.dtype]:
        """Determine optimal compute type (dtype) based on config and device."""
        if not torch:
            logger.error("PyTorch is not available. Cannot determine compute settings.")
            return torch.device("cpu"), torch.float32

        device = torch.device(device_str)
        torch_dtype = DEFAULT_COMPUTE_TYPES.get(device.type, torch.float32)

        compute_type_setting = settings.COMPUTE_TYPE.lower()

        if compute_type_setting == "auto":
            if device.type == "cuda":
                 if hasattr(torch.cuda, 'is_bf16_supported') and torch.cuda.is_bf16_supported():
                      torch_dtype = torch.bfloat16
                      logger.info(f"[MODEL:{self.name}] Using auto compute type: bfloat16 (bf16) on CUDA.")
                 else:
                      torch_dtype = torch.float16
                      logger.info(f"[MODEL:{self.name}] Using auto compute type: float16 (fp16) on CUDA.")
            else: # CPU (or other non-CUDA devices)
                 torch_dtype = torch.float32
                 logger.info(f"[MODEL:{self.name}] Using auto compute type: float32 on CPU.")
        elif compute_type_setting == "float16":
             if device.type == "cuda":
                  torch_dtype = torch.float16
                  logger.info(f"[MODEL:{self.name}] Forcing compute type: float16 (fp16).")
             else:
                  logger.warning(f"[MODEL:{self.name}] float16 requested but device is not CUDA. Using float32.")
                  torch_dtype = torch.float32
        elif compute_type_setting == "bfloat16":
             if device.type == "cuda" and hasattr(torch.cuda, 'is_bf16_supported') and torch.cuda.is_bf16_supported():
                  torch_dtype = torch.bfloat16
                  logger.info(f"[MODEL:{self.name}] Forcing compute type: bfloat16 (bf16).")
             else:
                  logger.warning(f"[MODEL:{self.name}] bfloat16 requested but not supported on this device. Using defaults.")
                  torch_dtype = DEFAULT_COMPUTE_TYPES.get(device.type, torch.float32)
        elif compute_type_setting == "float32":
             torch_dtype = torch.float32
             logger.info(f"[MODEL:{self.name}] Forcing compute type: float32.")
        else:
             logger.warning(f"[MODEL:{self.name}] Unknown COMPUTE_TYPE '{settings.COMPUTE_TYPE}'. Using defaults.")

        return device, torch_dtype

    def load(self, device: Optional[str] = None):
        """Load the model and processor into memory."""
        with self._model_lock:
            if self._loaded and self.is_loaded():
                logger.info(f"[MODEL:{self.name}] Already loaded on {self._loaded_device}.")
                return
                
            self._loaded = False

            if not torch:
                 raise TranscriptionError("PyTorch not available, cannot load model.")

            logger.info(f"[MODEL:{self.name}] Loading model (ID: {self.model_id})...")
            load_start_time = time.time()

            # Determine device and compute type
            target_device_str = device or ("cuda" if settings.USE_CUDA and torch.cuda.is_available() else "cpu")
            target_device, torch_dtype = self._get_compute_settings(target_device_str)

            try:
                if target_device.type == "cuda":
                    torch.cuda.empty_cache()

                logger.info(f"[MODEL:{self.name}] Downloading/loading model weights for {self.model_id}...")
                model = AutoModelForSpeechSeq2Seq.from_pretrained(
                    self.model_id,
                    torch_dtype=torch_dtype,
                    low_cpu_mem_usage=True,
                    use_safetensors=True,
                    cache_dir=settings.MODELS_CACHE_DIR,
                    # attn_implementation="flash_attention_2" # Optional: Requires flash-attn
                )
                logger.info(f"[MODEL:{self.name}] Model weights loaded. Moving to device {target_device}...")
                model.to(target_device)
                model.eval()

                logger.info(f"[MODEL:{self.name}] Downloading/loading processor for {self.model_id}...")
                processor = AutoProcessor.from_pretrained(
                    self.model_id,
                    cache_dir=settings.MODELS_CACHE_DIR
                )
                logger.info(f"[MODEL:{self.name}] Processor loaded.")

                logger.info(f"[MODEL:{self.name}] Creating ASR pipeline...")
                whisper_pipeline = pipeline(
                    "automatic-speech-recognition",
                    model=model,
                    tokenizer=processor.tokenizer,
                    feature_extractor=processor.feature_extractor,
                    torch_dtype=torch_dtype,
                    device=target_device,
                )
                logger.info(f"[MODEL:{self.name}] ASR pipeline created.")

                self._model = model
                self._processor = processor
                self._pipeline = whisper_pipeline
                self._loaded_device = target_device
                self._loaded_dtype = torch_dtype
                
                self._loaded = True

                if not self.is_loaded():
                    raise TranscriptionError(f"[MODEL:{self.name}] References were immediately garbage collected after loading. Memory may be too constrained.")

                load_time = time.time() - load_start_time
                logger.info(f"[MODEL:{self.name}] ✓ Loaded successfully to {target_device} (dtype: {torch_dtype}) in {load_time:.2f}s.")

            except Exception as e:
                logger.exception(f"[MODEL:{self.name}] Error loading model: {e}")
                self.unload()
                raise TranscriptionError(f"[MODEL:{self.name}] Failed to load model: {e}")

    def unload(self):
        """Unload the model and free resources."""
        with self._model_lock:
            if not self._loaded:
                return

            logger.info(f"[MODEL:{self.name}] Unloading model...")
            unload_start_time = time.time()

            # Clear strong references
            pipeline_obj = self._pipeline
            model_obj = self._model
            processor_obj = self._processor

            self._pipeline = None
            self._model = None
            self._processor = None
            self._loaded = False

            # Hint to GC
            del pipeline_obj
            del model_obj
            del processor_obj

            # Explicit GC and cache clearing might be added back if memory issues persist

            self._loaded_device = None
            self._loaded_dtype = None

            unload_time = time.time() - unload_start_time
            logger.info(f"[MODEL:{self.name}] ✓ Unloaded in {unload_time:.2f}s.")

    def is_loaded(self) -> bool:
        """Check if the model is currently loaded and all components are available."""
        with self._model_lock:
            if not self._loaded:
                return False
            
            model_valid = self._model is not None
            processor_valid = self._processor is not None
            pipeline_valid = self._pipeline is not None

            if not (model_valid and processor_valid and pipeline_valid):
                if self._loaded:
                     logger.warning(f"[MODEL:{self.name}] Marked as loaded but internal references are missing. Resetting loaded state.")
                self._loaded = False
                self._model = None
                self._processor = None
                self._pipeline = None
                return False
            
            return True

    def transcribe(
        self,
        audio_path: str,
        language: Optional[str] = None,
        task: str = "transcribe",
        progress_callback: Optional[Callable[[float], None]] = None # Note: Pipeline doesn't support fine-grained progress
    ) -> Dict[str, Any]:
        """
        Transcribe audio using the loaded Whisper model pipeline.

        Args:
            audio_path: Path to the audio file.
            language: Language code (e.g., 'en', 'sv'). If None, model detects.
            task: 'transcribe' or 'translate'.
            progress_callback: Function to report progress (0.0 to 1.0). Called at start/end.

        Returns:
            Dictionary containing transcription results ('text', 'segments', 'language', etc.).
        """
        with self._model_lock:
            if not self.is_loaded():
                 raise TranscriptionError(f"[MODEL:{self.name}] Not loaded. Call load() first.")
            
            pipe = self._pipeline
            if pipe is None:
                 # Defensive check: This shouldn't happen if is_loaded() passed
                 self._loaded = False
                 raise TranscriptionError(f"[MODEL:{self.name}] Pipeline object missing unexpectedly. Call load() first.")
                
        if not torch:
             raise TranscriptionError("PyTorch is not available. Cannot transcribe.")

        logger.info(f"[MODEL:{self.name}] Starting transcription for {os.path.basename(audio_path)}...")
        transcribe_start_time = time.time()

        # Prepare pipeline arguments
        generate_kwargs = {"task": task}
        if language:
            generate_kwargs["language"] = language
        else:
             generate_kwargs["language"] = None # Let pipeline handle auto-detection

        # Set optimal batch size and chunk length from settings, if provided
        batch_size = settings.WHISPER_BATCH_SIZE if settings.WHISPER_BATCH_SIZE is not None else 16
        chunk_length_s = settings.WHISPER_CHUNK_LENGTH if settings.WHISPER_CHUNK_LENGTH is not None else 30

        # Read audio file using utility compatible with pipeline
        try:
            if pipe.feature_extractor:
                 sampling_rate = pipe.feature_extractor.sampling_rate
            else:
                 logger.warning(f"[MODEL:{self.name}] Pipeline feature_extractor not found, assuming 16kHz sampling rate.")
                 sampling_rate = 16000

            # Read audio file into bytes first to avoid potential issues with ffmpeg_read path handling
            with open(audio_path, "rb") as f:
                audio_bytes = f.read()

            inputs = ffmpeg_read(audio_bytes, sampling_rate=sampling_rate)
            duration = len(inputs) / sampling_rate if inputs is not None else 0
            logger.info(f"[MODEL:{self.name}] Audio loaded ({len(audio_bytes)} bytes): duration={duration:.2f}s, target_sr={sampling_rate}")

        except FileNotFoundError:
             logger.error(f"[MODEL:{self.name}] Audio file not found at path: {audio_path}")
             raise TranscriptionError(f"Audio file not found: {audio_path}")
        except Exception as e:
             logger.exception(f"[MODEL:{self.name}] Failed to read or process audio file {audio_path}: {e}")
             raise TranscriptionError(f"[MODEL:{self.name}] Failed to read or process audio file: {e}")


        if progress_callback: progress_callback(0.1)

        # Execute transcription
        try:
            with torch.inference_mode():
                result = pipe(
                    inputs,
                    chunk_length_s=chunk_length_s,
                    batch_size=batch_size,
                    return_timestamps=True,
                    generate_kwargs=generate_kwargs,
                    # return_language=True # Optional: Check if needed based on transformers version
                )
            if progress_callback: progress_callback(0.9)

        except Exception as e:
            logger.exception(f"[MODEL:{self.name}] Pipeline execution failed: {e}")
            # Explicit GC/cache clearing might be needed here on error if memory issues occur
            raise TranscriptionError(f"[MODEL:{self.name}] Transcription pipeline failed: {e}")

        # Process results
        full_text = result.get("text", "")
        # Language detection result might be nested or top-level depending on transformers version.
        # Fallback to requested language or 'unknown'.
        detected_language = result.get("language", generate_kwargs.get("language") or "unknown")
        segments = []
        if "chunks" in result and isinstance(result["chunks"], list):
            for chunk in result["chunks"]:
                timestamp = chunk.get("timestamp")
                if isinstance(timestamp, (tuple, list)) and len(timestamp) == 2:
                    start, end = timestamp
                    if isinstance(start, (int, float)) and isinstance(end, (int, float)):
                        segments.append({
                            "start": round(start, 3),
                            "end": round(end, 3),
                            "text": chunk.get("text", "").strip()
                        })
                    else:
                        logger.warning(f"[MODEL:{self.name}] Skipping chunk with invalid start/end timestamps: start={start}, end={end}")
                else:
                     logger.warning(f"[MODEL:{self.name}] Skipping chunk due to invalid timestamp format: {timestamp}")
        else:
             logger.warning(f"[MODEL:{self.name}] Pipeline result did not contain 'chunks' list. Timestamp accuracy might be limited.")
             if full_text and duration > 0 :
                  segments.append({"start": 0.0, "end": round(duration, 3), "text": full_text.strip()})


        processing_time = time.time() - transcribe_start_time

        logger.info(f"[MODEL:{self.name}] Transcription completed in {processing_time:.2f}s. Language: {detected_language}")
        if duration > 0 and processing_time > 0:
             logger.info(f"[MODEL:{self.name}] Realtime factor: {duration / processing_time:.2f}x")

        if progress_callback: progress_callback(1.0)

        # Explicit GC/cache clearing might be needed here if memory issues occur

        return {
            "text": full_text.strip(),
            "segments": segments,
            "language": detected_language,
            "duration": round(duration, 3),
            "processing_time": round(processing_time, 3),
            "model": self.name,
        }