"""
Speaker diarization service using pyannote.audio.

Handles loading the diarization pipeline, preprocessing audio, performing diarization
(including chunking for large files), and formatting the results.
Optimized for GPU usage and memory management.
"""

import os
import tempfile
import logging
import threading
import time
import gc
import asyncio
import subprocess
import shutil
import uuid  # Add missing import for UUID generation
import concurrent.futures  # Add missing import for executor
from typing import List, Dict, Any, Optional, Callable, Tuple, Union

import torch
import numpy as np
import pandas as pd
from huggingface_hub import login as hf_login # Add explicit login import

# Configure logging
logger = logging.getLogger(__name__)

# Import settings and exceptions safely
try:
    from app.config import settings
    from app.exceptions import DiarizationError, ConfigurationError, FileProcessingError
    from app.services.task_manager import TaskStatus # For progress reporting
except ImportError as e:
    logger.error(f"Failed to import required modules for DiarizationService: {e}")
    # Define dummy settings/exceptions if config failed, service will likely fail later
    class SettingsMock: DIARIZATION_ENABLED=False; USE_CUDA=False; CUDA_DEVICE=0; DIARIZATION_CHUNK_DURATION=300; HUGGINGFACE_TOKEN=None; RESULTS_DIR="/tmp"; PARALLEL_PROCESSING=False; USE_TF32=False; MODELS_CACHE_DIR="./models"
    settings = SettingsMock()
    DiarizationError = Exception
    ConfigurationError = Exception
    FileProcessingError = Exception
    TaskStatus = None # Progress reporting won't work

# --- Constants ---
SAMPLE_RATE = 16000  # pyannote pipelines typically expect 16kHz
DIARIZATION_AVAILABLE = False
pipeline_instance = None # Module-level pipeline instance
pipeline_lock = threading.Lock() # Lock for initializing the pipeline

# --- Conditional Import and Setup ---
# Check if enabled in settings first
if settings and settings.DIARIZATION_ENABLED:
    try:
        # Check NumPy version first (pyannote 3.1 requires < 2.0)
        if np.__version__ >= "2.0.0":
            np_warning = (
                f"NumPy version {np.__version__} detected. pyannote.audio 3.1 requires numpy<2.0. "
                "Diarization might fail. Downgrade NumPy (`uv pip install numpy~=1.24`) if issues occur."
            )
            logger.warning(np_warning)

        # Import pyannote and related libraries
        from pyannote.audio import Pipeline
        from pyannote.core import Annotation, Segment # IMPORT HERE

        DIARIZATION_AVAILABLE = True
        logger.info("pyannote.audio imported successfully. Diarization enabled.")

    except ImportError as e:
        logger.warning(f"Failed to import diarization dependencies: {e}. Diarization will be UNAVAILABLE.")
        logger.warning("Install dependencies: `uv pip install pyannote.audio==3.1.1` (and ensure numpy<2.0)")
        DIARIZATION_AVAILABLE = False
    except Exception as e:
        logger.error(f"Unexpected error importing diarization dependencies: {e}", exc_info=True)
        DIARIZATION_AVAILABLE = False
else:
     if settings: logger.info("Diarization is disabled in settings.")
     else: logger.warning("DiarizationService cannot initialize: settings not loaded.")


class DiarizationService:
    """Manages the diarization pipeline and processing."""

    def __init__(self):
        if not settings:
             raise ConfigurationError("Settings not loaded, cannot initialize DiarizationService.")
        if not settings.DIARIZATION_ENABLED:
            logger.info("[DIARIZATION] DiarizationService initialized but disabled by settings.")
            return
        if not DIARIZATION_AVAILABLE:
             logger.warning("[DIARIZATION] DiarizationService initialized but dependencies are missing.")
             return

        self._pipeline = None # Instance variable to hold loaded pipeline
        self._device = self._get_device()
        self._chunk_duration_sec = self._determine_optimal_chunk_size()
        self._apply_optimizations()
        # Initialize semaphore for limiting chunk concurrency
        # TODO: Make limit configurable via settings.DIARIZATION_MAX_CHUNK_CONCURRENCY
        self._chunk_concurrency_limit = getattr(settings, 'DIARIZATION_MAX_CHUNK_CONCURRENCY', 4)
        self._chunk_semaphore = asyncio.Semaphore(self._chunk_concurrency_limit)
        logger.info(f"[DIARIZATION] DiarizationService initialized. Device: {self._device}, Chunk Duration: {self._chunk_duration_sec}s, Chunk Concurrency: {self._chunk_concurrency_limit}")

    def _get_device(self) -> torch.device:
        """Determine the appropriate torch device."""
        if settings.USE_CUDA and torch and torch.cuda.is_available():
             try:
                  # Validate device index
                  num_devices = torch.cuda.device_count()
                  if settings.CUDA_DEVICE < num_devices:
                       return torch.device(f"cuda:{settings.CUDA_DEVICE}")
                  else:
                       logger.warning(f"[DIARIZATION] Configured CUDA device {settings.CUDA_DEVICE} invalid (found {num_devices}). Using cuda:0.")
                       return torch.device("cuda:0") if num_devices > 0 else torch.device("cpu")
             except Exception as e:
                  logger.error(f"[DIARIZATION] Error checking CUDA devices: {e}. Falling back to CPU.")
                  return torch.device("cpu")
        else:
            return torch.device("cpu")

    def _determine_optimal_chunk_size(self) -> int:
         """Sets optimal chunk size based on config and available GPU memory."""
         base_chunk_size = settings.DIARIZATION_CHUNK_DURATION

         if self._device.type == "cuda" and hasattr(torch.cuda, 'mem_get_info'):
             try:
                 # Ensure device index is valid before getting memory info
                 device_idx = self._device.index if self._device.index is not None else 0
                 if device_idx < torch.cuda.device_count():
                     free_mem_bytes, _ = torch.cuda.mem_get_info(device_idx)
                     free_gb = free_mem_bytes / (1024**3)

                     # Adjust based on free memory (example thresholds)
                     if free_gb > 20: base_chunk_size = 600 # 10 min
                     elif free_gb > 12: base_chunk_size = 480 # 8 min
                     elif free_gb > 8: base_chunk_size = 360  # 6 min
                     elif free_gb > 5: base_chunk_size = 240  # 4 min
                     else: base_chunk_size = 180              # 3 min (safer minimum)

                     logger.info(f"[DIARIZATION] Adjusted diarization chunk duration to {base_chunk_size}s based on {free_gb:.1f}GB free VRAM.")
                 else:
                     logger.warning(f"[DIARIZATION] Cannot get memory info for invalid device index {device_idx}.")

             except Exception as e:
                  logger.warning(f"[DIARIZATION] Could not get GPU memory info to optimize chunk size: {e}")

         # Ensure chunk size is reasonable (e.g., at least 30 seconds)
         return max(30, base_chunk_size)


    def _apply_optimizations(self):
        """Apply PyTorch performance settings."""
        if not DIARIZATION_AVAILABLE or not torch: return

        # PyTorch optimizations (only relevant if using GPU)
        if self._device.type == 'cuda':
            try:
                # Ensure device capability check is safe
                if hasattr(torch.cuda, 'get_device_capability'):
                     capability = torch.cuda.get_device_capability(self._device)
                     # Enable TF32 for compatible GPUs (Ampere+)
                     if settings.USE_TF32 and capability[0] >= 8:
                         torch.backends.cuda.matmul.allow_tf32 = True
                         if hasattr(torch.backends.cudnn, 'allow_tf32'):
                             torch.backends.cudnn.allow_tf32 = True
                         logger.info("[DIARIZATION] Enabled TF32 optimization for Diarization.")

                # Enable cuDNN benchmark mode (can speed up fixed-size inputs)
                if hasattr(torch.backends, 'cudnn') and hasattr(torch.backends.cudnn, 'benchmark'):
                     torch.backends.cudnn.benchmark = True
                     logger.info("[DIARIZATION] Enabled cuDNN benchmark mode for Diarization.")
            except Exception as e:
                 logger.warning(f"[DIARIZATION] Could not apply all PyTorch optimizations: {e}")


    def _load_pipeline_instance(self):
        """Loads the pyannote.audio pipeline instance if not already loaded. Thread-safe."""
        # Use the instance variable now, still protected by the module-level lock
        if self._pipeline is not None:
            # Double check pipeline is still valid (not garbage collected)
            try:
                # Simple attribute access to check if pipeline is still valid
                _ = self._pipeline.device
                return self._pipeline
            except (AttributeError, RuntimeError):
                logger.warning("[DIARIZATION] Diarization pipeline reference is invalid - will reload")
                self._pipeline = None
                # Continue to reload below
        
        # This check prevents loading if deps are missing or disabled
        if not DIARIZATION_AVAILABLE:
             raise ConfigurationError("Diarization dependencies not available, cannot load pipeline.")

        with pipeline_lock:
            # Double-check after acquiring lock
            if self._pipeline is not None:
                # Check again inside lock
                try:
                    _ = self._pipeline.device
                    return self._pipeline
                except (AttributeError, RuntimeError):
                    self._pipeline = None
                    # Continue to reload
            
            hf_token = settings.HUGGINGFACE_TOKEN or os.environ.get("HUGGINGFACE_TOKEN")
            if not hf_token:
                raise ConfigurationError("Hugging Face token (HUGGINGFACE_TOKEN) is required for diarization.")

            logger.info(f"[DIARIZATION] Loading diarization pipeline: {settings.DIARIZATION_MODEL} on device: {self._device}...")
            load_start_time = time.time()

            try:
                # Clean memory before loading (Removed explicit gc.collect/empty_cache)
                # gc.collect()
                if self._device.type == "cuda": torch.cuda.empty_cache()

                # Explicitly login using the token before loading pipeline
                try:
                    logger.info("[DIARIZATION] Authenticating with Hugging Face Hub...")
                    hf_login(token=hf_token)
                    logger.info("[DIARIZATION] Hugging Face Hub authentication successful.")
                except Exception as e:
                    logger.error(f"[DIARIZATION] Failed to authenticate with Hugging Face Hub: {e}", exc_info=True)
                    # Raise an error here as loading will likely fail without auth
                    raise DiarizationError(f"Failed to authenticate with Hugging Face: {e}")

                # Load the pipeline (Annotation is imported globally now)
                loaded_pipeline = Pipeline.from_pretrained(
                    settings.DIARIZATION_MODEL,
                    use_auth_token=hf_token, # Still pass token here as well
                )
                # Explicitly check if loading failed (returned None) before proceeding
                if loaded_pipeline is None:
                    # This usually happens due to auth/gating issues logged by pyannote/HF library
                    logger.error("[DIARIZATION] Pipeline.from_pretrained returned None. Check logs above for authentication/gating errors from Hugging Face.")
                    raise DiarizationError("Failed to instantiate diarization pipeline, likely due to authentication or model access issues.")

                loaded_pipeline.to(self._device)

                # Store on the instance
                self._pipeline = loaded_pipeline
                load_time = time.time() - load_start_time
                logger.info(f"[DIARIZATION] ✓ Diarization pipeline loaded in {load_time:.2f}s.")
                return self._pipeline

            except Exception as e:
                logger.exception(f"[DIARIZATION] Failed to load diarization pipeline: {e}")
                self._pipeline = None # Ensure it's None on failure
                raise DiarizationError(f"Failed to load diarization pipeline: {e}")

    async def _estimate_audio_duration(self, file_path: str) -> float:
        """Estimate audio duration using ffprobe."""
        try:
            # Attempt to import here to avoid circular dependency issues at module level
            from app.services.processor import estimate_audio_duration as estimate_duration_ext
            return await estimate_duration_ext(file_path)
        except ImportError as e:
            logger.error("[DIARIZATION] Failed to import 'estimate_audio_duration' from processor. This indicates a potential structure issue.", exc_info=True)
            raise ConfigurationError("Core utility function 'estimate_audio_duration' is unavailable.") from e
        except Exception as e:
            # Catch potential errors from the imported function itself
            logger.error(f"[DIARIZATION] Error calling imported 'estimate_audio_duration': {e}", exc_info=True)
            raise DiarizationError(f"Failed to estimate audio duration: {e}") from e


    async def _preprocess_audio(self, file_path: str, task_id: Optional[str] = None) -> Tuple[str, bool]:
        """
        Preprocess audio using FFmpeg for normalization, resampling, mono conversion.
        Returns the path to the processed file and a flag indicating if it's temporary.
        """
        prefix = f"task_{task_id}_" if task_id else ""
        # Use results dir for temporary processed files for better isolation/cleanup
        try:
            os.makedirs(settings.RESULTS_DIR, mode=0o700, exist_ok=True)
        except OSError as e:
            logger.error(f"[DIARIZATION][{task_id}] Cannot create temporary directory for preprocessing: {settings.RESULTS_DIR}. Error: {e}")
            raise FileProcessingError(f"Cannot access results directory: {settings.RESULTS_DIR}") from e

        temp_filename = f"{prefix}processed_{uuid.uuid4()}.wav"
        processed_path = os.path.join(settings.RESULTS_DIR, temp_filename)
        is_temporary = False

        preprocess_start_time = time.time()
        logger.info(f"[DIARIZATION][{task_id}] Preprocessing audio '{os.path.basename(file_path)}' -> '{os.path.basename(processed_path)}'...")

        try:
            # Build command for normalization, resampling, mono conversion, and basic filtering
            cmd = [
                "ffmpeg", "-y", "-i", file_path,
                "-af", f"loudnorm=I=-16:TP=-1.5:LRA=11,highpass=f=80,lowpass=f={SAMPLE_RATE // 2 - 500}",
                "-ar", str(SAMPLE_RATE), "-ac", "1", "-vn",
                "-hide_banner", "-loglevel", "warning",
                processed_path
            ]

            proc = await asyncio.create_subprocess_exec(
                *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await proc.communicate()

            if proc.returncode != 0:
                error_msg = stderr.decode(errors='ignore').strip()
                logger.warning(f"[DIARIZATION][{task_id}] Audio preprocessing with ffmpeg failed for '{os.path.basename(file_path)}': {error_msg}. Using original file.")
                if os.path.exists(processed_path):
                    try: os.remove(processed_path)
                    except OSError: pass
                return file_path, False # Return original path, not temporary
            else:
                if os.path.exists(processed_path) and os.path.getsize(processed_path) > 0:
                     preprocess_time = time.time() - preprocess_start_time
                     logger.info(f"[DIARIZATION][{task_id}] Audio preprocessing successful: '{os.path.basename(processed_path)}' in {preprocess_time:.2f}s")
                     is_temporary = True
                     return processed_path, is_temporary
                else:
                     logger.warning(f"[DIARIZATION][{task_id}] Audio preprocessing command succeeded but output file '{processed_path}' is missing or empty. Using original file.")
                     if os.path.exists(processed_path):
                          try: os.remove(processed_path)
                          except OSError: pass
                     return file_path, False

        except FileNotFoundError:
             logger.error("[DIARIZATION] ffmpeg command not found. Preprocessing requires ffmpeg in PATH.")
             return file_path, False
        except Exception as e:
            logger.error(f"[DIARIZATION][{task_id}] Error during audio preprocessing execution: {e}", exc_info=True)
            if os.path.exists(processed_path):
                try: os.remove(processed_path)
                except OSError: pass
            return file_path, False


    async def _split_audio_chunks(self, file_path: str, temp_dir: str, chunk_size: int, overlap: int = 5) -> List[str]:
        """Splits audio into overlapping chunks using FFmpeg."""
        chunk_files = []
        tasks = []
        base_name = "".join(c for c in os.path.splitext(os.path.basename(file_path))[0] if c.isalnum() or c in ['_', '-'])[:50]

        try:
            duration = await self._estimate_audio_duration(file_path)
            if duration <= 0:
                 logger.warning(f"[DIARIZATION] Could not get valid duration for {os.path.basename(file_path)}, cannot split accurately.")
                 return []

            num_chunks = int(np.ceil(duration / chunk_size))
            if num_chunks <= 1:
                 logger.info(f"[DIARIZATION] Audio duration ({duration:.1f}s) fits within one chunk ({chunk_size}s). No splitting needed for '{os.path.basename(file_path)}'.")
                 return []

            split_start_time = time.time()
            logger.info(f"[DIARIZATION] Splitting audio '{os.path.basename(file_path)}' ({duration:.1f}s) into {num_chunks} chunks of ~{chunk_size}s (overlap {overlap}s)...")

            for i in range(num_chunks):
                chunk_start = max(0.0, i * chunk_size - overlap if i > 0 else 0.0)
                base_chunk_dur = chunk_size
                current_overlap = overlap if i > 0 else 0
                next_overlap = overlap if i < num_chunks - 1 else 0
                chunk_duration = base_chunk_dur + current_overlap + next_overlap
                chunk_start = min(chunk_start, duration)
                chunk_duration = min(chunk_duration, duration - chunk_start)

                if chunk_duration <= 0.1: continue

                output_chunk_path = os.path.join(temp_dir, f"{base_name}_chunk_{i:03d}.wav")
                chunk_files.append(output_chunk_path)

                cmd = [
                    "ffmpeg", "-y", "-i", file_path,
                    "-ss", f"{chunk_start:.3f}", "-t", f"{chunk_duration:.3f}",
                    "-ar", str(SAMPLE_RATE), "-ac", "1", "-vn",
                    "-hide_banner", "-loglevel", "error",
                    output_chunk_path
                ]

                async def run_ffmpeg_split(command, chunk_path):
                    proc = await asyncio.create_subprocess_exec(*command, stderr=asyncio.subprocess.PIPE)
                    _, stderr = await proc.communicate()
                    if proc.returncode != 0:
                        logger.warning(f"[DIARIZATION] FFmpeg chunk split failed ({os.path.basename(chunk_path)}): {stderr.decode(errors='ignore')}")
                    return proc.returncode, chunk_path

                tasks.append(run_ffmpeg_split(cmd, output_chunk_path))

            results = await asyncio.gather(*tasks)
            successful_chunks = [path for code, path in results if code == 0]

            split_time = time.time() - split_start_time
            if len(successful_chunks) != len(chunk_files):
                 logger.warning(f"[DIARIZATION] Only {len(successful_chunks)}/{len(chunk_files)} audio chunks were split successfully in {split_time:.2f}s.")
                 failed_chunks = [os.path.basename(path) for code, path in results if code != 0]
                 logger.warning(f"[DIARIZATION] Failed chunk base names: {failed_chunks}")
                 for path in failed_chunks:
                      if os.path.exists(path):
                           try: os.remove(path)
                           except OSError: pass
                 if not successful_chunks:
                      raise DiarizationError("Failed to split audio into any processable chunks.")
                 return successful_chunks

            logger.info(f"[DIARIZATION] Audio splitting completed in {split_time:.2f}s, {len(successful_chunks)} chunks created.")
            return successful_chunks # Return only successful ones

        except Exception as e:
            logger.error(f"[DIARIZATION] Error splitting audio '{os.path.basename(file_path)}' into chunks: {e}", exc_info=True)
            for f in chunk_files:
                 if os.path.exists(f):
                      try: os.remove(f)
                      except OSError: pass
            raise DiarizationError(f"Failed to split audio: {e}")


    # Use string forward reference for Annotation type hint
    def _run_pipeline_on_chunk(self, pipeline, chunk_path: str, params: Dict[str, Any], pipeline_hyperparams: Dict[str, Any]) -> Optional['Annotation']:
        """Runs the pipeline on a single chunk (synchronous wrapper for executor)."""
        chunk_name = os.path.basename(chunk_path)
        chunk_start_time = time.time()
        # Task ID isn't directly passed here, but chunk_name implies context
        logger.debug(f"[DIARIZATION] Processing chunk: {chunk_name}...")
        try:
            if not DIARIZATION_AVAILABLE: # Guard against calling if deps are missing
                 logger.error(f"[DIARIZATION] Attempted to run diarization pipeline on {chunk_name} but dependencies are not available.")
                 return None

            audio_input = chunk_path
            # Combine speaker count params and pipeline hyperparams
            all_params = {**params, **pipeline_hyperparams}
            logger.debug(f"[DIARIZATION] Running pipeline on {chunk_name} with params: {all_params}")
            result: Annotation = pipeline(audio_input, **all_params) # Result should be Annotation type
            chunk_time = time.time() - chunk_start_time
            logger.debug(f"[DIARIZATION] Chunk processing successful: {chunk_name} -> {len(result.labels())} speakers in {chunk_time:.2f}s")
            return result
        except Exception as e:
             logger.error(f"[DIARIZATION] Diarization pipeline failed for chunk {chunk_name}: {e}", exc_info=True)
             return None


    async def diarize_file(
        self,
        file_path: str,
        num_speakers: Optional[int] = None,
        min_speakers: Optional[int] = None, # Speaker count constraints
        max_speakers: Optional[int] = None,
        # --- Pyannote Hyperparameters ---
        segmentation_onset: Optional[float] = None, # e.g., 0.5
        clustering_threshold: Optional[float] = None, # e.g., 0.5 (depends on model)
        segmentation_min_duration_off: Optional[float] = None, # e.g., 0.1
        # --- Other Params ---
        progress_callback: Optional[Callable[[str, float, Dict[str, Any]], None]] = None,
        language: Optional[str] = None, # Language hint (might be used by some pipelines)
        task_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Perform speaker diarization, handling preprocessing, chunking, and merging.
        """
        if not settings or not settings.DIARIZATION_ENABLED or not DIARIZATION_AVAILABLE:
            raise ConfigurationError("Diarization is disabled or unavailable.")
        if not os.path.exists(file_path):
             raise FileNotFoundError(f"Input audio file not found: {file_path}")

        start_time = time.time()
        processed_file_path = file_path
        is_temp_processed_file = False
        temp_chunk_dir = None
        final_diarization_result = None

        def _update_progress(progress: float, stage: str, extra: Optional[Dict] = None):
            if progress_callback and TaskStatus:
                 info = {"stage": stage}
                 if extra: info.update(extra)
                 clamped_progress = min(max(0.0, progress), 1.0)
                 # Pass the TaskStatus.PROCESSING enum member directly
                 progress_callback(TaskStatus.PROCESSING, clamped_progress, info)

        try:
            # 1. Load Pipeline
            _update_progress(0.01, "loading_pipeline")
            pipeline = self._load_pipeline_instance()
            if pipeline is None:
                raise DiarizationError("Failed to load diarization pipeline - got None")

            # 2. Preprocess Audio
            _update_progress(0.05, "preprocessing_audio")
            processed_file_path, is_temp_processed_file = await self._preprocess_audio(file_path, task_id)

            # 3. Get Duration
            _update_progress(0.08, "estimating_duration")
            duration = await self._estimate_audio_duration(processed_file_path)
            if duration <= 0.1:
                 logger.warning(f"[DIARIZATION][{task_id}] Audio file '{os.path.basename(processed_file_path)}' is too short or has invalid duration ({duration:.2f}s). Skipping diarization.")
                 return []

            # 4. Prepare Diarization Parameters (Speaker Count & Hyperparameters)
            _update_progress(0.10, "preparing_parameters")
            speaker_params = {}
            if num_speakers:
                speaker_params["num_speakers"] = num_speakers
                min_speakers, max_speakers = None, None # Override min/max if num_speakers is set
            else:
                speaker_params["min_speakers"] = min_speakers if min_speakers is not None else 1
                # Dynamic default max based on duration
                default_max = 10 if duration > 1800 else (8 if duration > 600 else 6)
                max_effective = max_speakers if max_speakers is not None else default_max
                speaker_params["max_speakers"] = max(speaker_params["min_speakers"], max_effective)

            # Collect pipeline hyperparameters if provided
            pipeline_hyperparams = {}
            if segmentation_onset is not None: pipeline_hyperparams["segmentation_onset"] = segmentation_onset
            if clustering_threshold is not None: pipeline_hyperparams["clustering_threshold"] = clustering_threshold
            if segmentation_min_duration_off is not None: pipeline_hyperparams["segmentation_min_duration_off"] = segmentation_min_duration_off

            logger.info(f"[DIARIZATION][{task_id}] Running diarization for '{os.path.basename(processed_file_path)}' ({duration:.1f}s)")
            logger.info(f"[DIARIZATION][{task_id}] Effective speaker params: {speaker_params}")
            if pipeline_hyperparams:
                logger.info(f"[DIARIZATION][{task_id}] Pipeline hyperparameters: {pipeline_hyperparams}")


            # 5. Execute Diarization (Chunked or Standard)
            chunk_size = self._chunk_duration_sec
            loop = asyncio.get_running_loop()
            chunk_overlap = 5
            needs_chunking = duration > max(chunk_size * 1.1, 900) # Use chunking logic

            if needs_chunking:
                 # --- Chunked Processing ---
                 _update_progress(0.12, "splitting_chunks")
                 logger.info(f"[DIARIZATION][{task_id}] Audio ({duration:.1f}s) requires chunked processing (chunk size {chunk_size}s).")
                 # Ensure results dir exists for temp chunk dir creation
                 os.makedirs(settings.RESULTS_DIR, mode=0o700, exist_ok=True)
                 temp_chunk_dir = tempfile.mkdtemp(prefix=f"diarize_chunks_{task_id}_", dir=settings.RESULTS_DIR)
                 chunk_paths = await self._split_audio_chunks(processed_file_path, temp_chunk_dir, chunk_size, chunk_overlap)

                 if not chunk_paths: # chunk_paths now only contains successful chunks
                      logger.warning("Audio splitting resulted in no chunks. Attempting standard processing on full file.")
                      needs_chunking = False # Fallback
                 else:
                      num_chunks = len(chunk_paths)
                      _update_progress(0.15, "processing_chunks", {"total": num_chunks})
                      logger.info(f"Processing {num_chunks} audio chunks...")

                      all_chunk_results = []
                      # Use asyncio.to_thread with semaphore to limit concurrency
                      tasks = []
                      num_chunks = len(chunk_paths)
                      logger.info(f"Processing {num_chunks} chunks using asyncio.to_thread with concurrency limit {self._chunk_semaphore._value}...")

                      async def process_chunk_wrapper(chunk_path, index):
                          # Acquire semaphore before running the thread
                          async with self._chunk_semaphore:
                              logger.debug(f"Acquired semaphore for chunk {index}")
                              try:
                                  # Run the synchronous function in a thread pool
                                  result = await asyncio.to_thread(
                                      self._run_pipeline_on_chunk,
                                      pipeline, # The loaded pipeline object
                                      chunk_path, # Path to the audio chunk
                                      speaker_params, # Speaker count constraints
                                      pipeline_hyperparams # Other pipeline params
                                  )
                                  return result
                              finally:
                                  # Semaphore is released automatically by 'async with'
                                  logger.debug(f"Released semaphore for chunk {index}")

                      for i, chunk_path in enumerate(chunk_paths):
                          task = asyncio.create_task(
                              process_chunk_wrapper(chunk_path, i),
                              name=f"diarize_chunk_{i}"
                          )
                          tasks.append(task)

                      # Wait for all chunk processing tasks to complete
                      raw_results = await asyncio.gather(*tasks)

                      # Process results
                      for i, result in enumerate(raw_results):
                          chunk_offset = max(0.0, i * chunk_size - chunk_overlap if i > 0 else 0.0)
                          all_chunk_results.append({"diarization": result, "chunk_offset": chunk_offset, "chunk_index": i})
                          progress = 0.15 + (0.8 * (i + 1) / num_chunks)
                          _update_progress(progress, "processing_chunks", {"completed": i + 1, "total": num_chunks})

                      # Merge results
                      _update_progress(0.95, "merging_results")
                      final_diarization_result = self._merge_diarization_chunks(all_chunk_results, chunk_size, chunk_overlap)

            # --- Standard Processing (if not chunking or fallback) ---
            if not needs_chunking:
                 _update_progress(0.15, "processing_audio")
                 logger.info("Audio duration within limit or chunking failed. Using standard diarization processing.")
                 final_diarization_result = await loop.run_in_executor(None, self._run_pipeline_on_chunk, pipeline, processed_file_path, speaker_params, pipeline_hyperparams)
                 _update_progress(0.95, "processing_complete")

            # 6. Format Results
            _update_progress(0.98, "formatting_results")
            formatted_segments = self._format_results(final_diarization_result, speaker_params.get("min_speakers"), speaker_params.get("max_speakers"), task_id=task_id)

            processing_time = time.time() - start_time
            num_speakers_found = len(set(s['speaker'] for s in formatted_segments))
            logger.info(f"Diarization finished in {processing_time:.2f}s. Found {num_speakers_found} unique speaker labels.")

            _update_progress(1.0, "completed") # Final progress update

            return formatted_segments

        except Exception as e:
             logger.error(f"Diarization pipeline failed for {file_path}: {e}", exc_info=True)
             _update_progress(0.99, "diarization_error", {"error": str(e)})
             raise DiarizationError(f"Diarization processing failed: {e}") from e

        finally:
            # 7. Cleanup Temporary Files and Memory
            # Cleanup: Processed file
            if is_temp_processed_file and processed_file_path != file_path and os.path.exists(processed_file_path):
                try: os.remove(processed_file_path); logger.debug(f"Removed temp processed file: {processed_file_path}")
                except OSError as e: logger.warning(f"Failed to remove temporary processed file {processed_file_path}: {e}")
            # Cleanup: Chunk directory
            if temp_chunk_dir and os.path.isdir(temp_chunk_dir): # Check if it's a directory
                try: shutil.rmtree(temp_chunk_dir); logger.debug(f"Removed temp chunk dir: {temp_chunk_dir}")
                except OSError as e: logger.warning(f"Failed to remove temporary chunk directory {temp_chunk_dir}: {e}")

            # gc.collect() # Removed explicit call
            # if self._device.type == 'cuda' and torch and torch.cuda.is_available():
            #      torch.cuda.empty_cache() # Removed explicit call

    # Use string forward reference for Annotation type hint
    def _merge_diarization_chunks(self, chunk_results: List[Dict], chunk_size: int, overlap: int, task_id: Optional[str] = None) -> Optional['Annotation']:
        """Merges diarization Annotation objects from overlapping chunks."""
        # Ensure Annotation is available before proceeding
        if not DIARIZATION_AVAILABLE:
            logger.error(f"[DIARIZATION][{task_id}] Cannot merge chunks: Diarization dependencies (pyannote.core) not available.")
            return None

        valid_chunks = [c for c in chunk_results if c.get('diarization') is not None and isinstance(c.get('diarization'), Annotation)]
        if not valid_chunks:
             logger.warning(f"[DIARIZATION][{task_id}] No valid diarization Annotation objects found in chunk results to merge.")
             return None

        merge_start_time = time.time()
        logger.info(f"[DIARIZATION][{task_id}] Merging results from {len(valid_chunks)} valid diarization chunks...")
        merged = Annotation()

        sorted_chunks = sorted(valid_chunks, key=lambda x: x['chunk_index'])

        for i, chunk in enumerate(sorted_chunks):
            diarization: Annotation = chunk['diarization']
            offset = chunk['chunk_offset']
            index = chunk['chunk_index']

            for segment, track, speaker in diarization.itertracks(yield_label=True):
                 start = segment.start + offset
                 end = segment.end + offset
                 effective_chunk_start = offset + overlap if i > 0 else 0.0
                 if start < effective_chunk_start: start = effective_chunk_start
                 if end <= start + 0.05: continue
                 merged[Segment(start, end), f"T{index}_{track}"] = speaker

        try:
            simplified = merged.support(collar=0.15)
            num_labels_before = len(merged.labels())
            num_labels_after = len(simplified.labels())
            merge_time = time.time() - merge_start_time
            logger.info(f"[DIARIZATION][{task_id}] Merged annotation simplified in {merge_time:.2f}s. Speaker labels before: {num_labels_before}, after: {num_labels_after}.")
            return simplified
        except Exception as e:
             logger.error(f"[DIARIZATION][{task_id}] Error simplifying merged annotation: {e}", exc_info=True)
             return merged


    # Use string forward reference for Annotation type hint
    def _format_results(self, diarization_result: Optional['Annotation'], min_speakers: Optional[int], max_speakers: Optional[int], task_id: Optional[str] = None, min_segment_duration: float = 0.75) -> List[Dict[str, Any]]:
        """Formats the final pyannote Annotation object into the API response list."""
        if diarization_result is None: return []
        if not DIARIZATION_AVAILABLE: # Check dependencies again before using Annotation
             logger.error(f"[DIARIZATION][{task_id}] Cannot format results: Diarization dependencies not available.")
             return []

        formatted = []
        format_start_time = time.time() # Start timer before formatting
        try:
            if not isinstance(diarization_result, Annotation):
                logger.warning(f"[DIARIZATION][{task_id}] Expected Annotation object for formatting, got {type(diarization_result)}. Cannot format.")
                return []

            # Initial formatting from Annotation
            for segment, _, speaker in diarization_result.itertracks(yield_label=True):
                formatted.append({
                    "start": round(segment.start, 3),
                    "end": round(segment.end, 3),
                    "speaker": str(speaker)
                })
            formatted.sort(key=lambda x: x['start'])

            unique_speakers = sorted(list(set(s['speaker'] for s in formatted)))
            num_detected = len(unique_speakers)
            logger.info(f"[DIARIZATION][{task_id}] Initial formatting produced {len(formatted)} segments and {num_detected} unique speaker labels.")

            # --- Post-processing: Merge short segments ---
            if formatted and min_segment_duration > 0:
                merged_short_segments = []
                i = 0
                while i < len(formatted):
                    current_seg = formatted[i]
                    duration = current_seg['end'] - current_seg['start']

                    if duration < min_segment_duration:
                        merged = False
                        # Try merging with previous segment if same speaker
                        if i > 0 and merged_short_segments:
                            prev_seg = merged_short_segments[-1]
                            if prev_seg['speaker'] == current_seg['speaker']:
                                # Merge current into previous
                                prev_seg['end'] = current_seg['end']
                                logger.debug(f"[DIARIZATION][{task_id}] Merged short segment (idx {i}, {duration:.2f}s) into previous.")
                                merged = True

                        # Try merging with next segment if same speaker (and not already merged with prev)
                        if not merged and i + 1 < len(formatted):
                            next_seg = formatted[i+1]
                            if next_seg['speaker'] == current_seg['speaker']:
                                # Merge current into next (adjust next's start)
                                next_seg['start'] = current_seg['start']
                                # Add the adjusted next segment and skip the original next one in the outer loop
                                merged_short_segments.append(next_seg)
                                logger.debug(f"[DIARIZATION][{task_id}] Merged short segment (idx {i}, {duration:.2f}s) into next.")
                                i += 1 # Skip the next segment as it's now incorporated
                                merged = True

                        # If it couldn't be merged, keep it (it might be a valid short utterance)
                        if not merged:
                            merged_short_segments.append(current_seg)
                            logger.debug(f"[DIARIZATION][{task_id}] Kept short segment (idx {i}, {duration:.2f}s) as it couldn't be merged.")
                    else:
                        # Segment is long enough, keep it
                        merged_short_segments.append(current_seg)

                    i += 1 # Move to the next segment

                if len(merged_short_segments) < len(formatted):
                    logger.info(f"[DIARIZATION][{task_id}] Merged {len(formatted) - len(merged_short_segments)} segments shorter than {min_segment_duration}s.")
                    formatted = merged_short_segments
                    # Recalculate unique speakers after merging short segments
                    unique_speakers = sorted(list(set(s['speaker'] for s in formatted)))
                    num_detected = len(unique_speakers)


            # --- Post-processing: Enforce Max Speakers (basic heuristic) ---
            if max_speakers is not None and num_detected > max_speakers:
                logger.warning(f"[DIARIZATION][{task_id}] Detected {num_detected} speakers after short segment merge, exceeding max limit {max_speakers}. Applying basic merge heuristic.")
                speaker_durations = {spk: 0.0 for spk in unique_speakers}
                for seg in formatted:
                    speaker = seg['speaker']
                    speaker_durations[speaker] += (seg['end'] - seg['start'])
                speakers_sorted_by_duration = sorted(speaker_durations.keys(), key=lambda s: speaker_durations[s], reverse=True)
                speakers_to_keep = set(speakers_sorted_by_duration[:max_speakers])
                speakers_to_remap = speakers_sorted_by_duration[max_speakers:]
                remap_dict = {}
                if speakers_to_keep:
                    most_dominant_speaker = speakers_sorted_by_duration[0]
                    for spk_remap in speakers_to_remap: remap_dict[spk_remap] = most_dominant_speaker
                if remap_dict:
                    new_formatted = []
                    for seg in formatted:
                        original_speaker = seg['speaker']
                        seg['speaker'] = remap_dict.get(original_speaker, original_speaker)
                        new_formatted.append(seg)
                    formatted = new_formatted
                    final_unique_speakers = sorted(list(set(s['speaker'] for s in formatted)))
                    logger.info(f"[DIARIZATION][{task_id}] After merging to max {max_speakers} speakers, {len(final_unique_speakers)} unique labels remain.")
                    # Update num_detected after max speaker enforcement
                    num_detected = len(final_unique_speakers)


            # --- Post-processing: Warn if Below Min Speakers ---
            if min_speakers is not None and num_detected < min_speakers:
                logger.warning(f"[DIARIZATION][{task_id}] Detected only {num_detected} speakers after all processing, below the minimum requested/default of {min_speakers}. Results may be less accurate.")

        except Exception as e:
            # Ensure format_time is calculated even on error
            format_time = time.time() - format_start_time
            logger.error(f"[DIARIZATION][{task_id}] Error during formatting/post-processing diarization results (after {format_time:.2f}s): {e}", exc_info=True)
            return [] # Return empty list on error

        # Calculate final format time if no error occurred
        format_time = time.time() - format_start_time
        logger.info(f"[DIARIZATION][{task_id}] Formatting and post-processing completed in {format_time:.2f}s.")
        return formatted


# --- Create Global Service Instance ---
diarization_service: Optional[DiarizationService] = None
if settings and settings.DIARIZATION_ENABLED and DIARIZATION_AVAILABLE:
    try:
        diarization_service = DiarizationService()
    except Exception as e:
        logger.error(f"Failed to initialize DiarizationService during global creation: {e}", exc_info=True)
elif settings and settings.DIARIZATION_ENABLED:
     logger.warning("Diarization enabled in settings, but service could not be initialized (likely missing dependencies).")