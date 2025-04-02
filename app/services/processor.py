"""
Processor for audio transcription and diarization tasks.

Coordinates model execution via the ModelRegistry and TaskManager.
Handles the overall workflow for processing audio files.
"""

import os
import time
import logging
import asyncio
import gc
import subprocess
import concurrent.futures
from typing import Dict, Any, Optional, List, Callable, Tuple

import torch
import pandas as pd
import numpy as np # Diarization utils might use numpy

from app.config import settings
# Correct import path for exceptions
from app.exceptions import TranscriptionError, DiarizationError, ConfigurationError, ModelNotFoundError, FileProcessingError
from app.services.model_registry import ModelRegistry, TranscriptionModel
from app.services.task_manager import TaskStatus

# Check for diarization availability and import service safely
diarization_service = None
DIARIZATION_AVAILABLE = False
try:
    # Only attempt import if enabled in settings
    if settings.DIARIZATION_ENABLED:
        from app.services.diarization import diarization_service as ds, DIARIZATION_AVAILABLE as da
        diarization_service = ds
        DIARIZATION_AVAILABLE = da
except ImportError:
    # Log warning if enabled but import fails
    if settings.DIARIZATION_ENABLED:
         logger = logging.getLogger(__name__)
         logger.warning("Diarization is enabled but failed to import dependencies. Diarization will be unavailable.")
except Exception as e:
     # Log other unexpected errors during import
     logger = logging.getLogger(__name__)
     logger.error(f"Unexpected error importing diarization service: {e}", exc_info=True)


# Configure logging
logger = logging.getLogger(__name__)

# --- Helper Functions ---

async def estimate_audio_duration(file_path: str) -> float:
    """Estimate audio duration using ffprobe for efficiency."""
    try:
        cmd = [
            "ffprobe", "-v", "error", "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1", file_path
        ]
        proc = await asyncio.create_subprocess_exec(
            *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await proc.communicate()

        if proc.returncode == 0 and stdout:
            try:
                 return float(stdout.decode('utf-8').strip())
            except ValueError:
                 logger.error(f"[PROCESSOR] ffprobe returned non-numeric duration for {os.path.basename(file_path)}: {stdout.decode()}")
                 return 0.0 # Treat as failure
        else:
            error_msg = stderr.decode('utf-8').strip() if stderr else "Unknown ffprobe error"
            logger.warning(f"[PROCESSOR] ffprobe failed for {os.path.basename(file_path)}: {error_msg}")
            # Fallback to basic file size estimate
            try:
                file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
                est_duration = file_size_mb * 60
                logger.info(f"[PROCESSOR] Falling back to duration estimate based on file size for {os.path.basename(file_path)}: {est_duration:.2f}s")
                return est_duration
            except Exception as file_e:
                logger.error(f"[PROCESSOR] Could not estimate duration from file size for {os.path.basename(file_path)}: {file_e}")
                return 0.0 # Cannot determine duration

    except FileNotFoundError:
        logger.error("[PROCESSOR] ffprobe command not found. Ensure ffmpeg is installed and in PATH.")
        return 0.0
    except Exception as e:
        logger.error(f"[PROCESSOR] Error estimating duration for {os.path.basename(file_path)}: {e}")
        return 0.0

def _create_scaled_callback(
    original_callback: Optional[Callable[[str, float, Dict[str, Any]], None]],
    step_weight: float,
    starting_progress: float
) -> Optional[Callable[[TaskStatus, float, Dict[str, Any]], None]]: # Use TaskStatus type hint
    """Creates a wrapped callback that scales progress within a defined range."""
    if not original_callback or not TaskStatus: # Check TaskStatus imported
        return None

    def scaled_callback(status: TaskStatus, step_progress: float, info: Dict[str, Any]):
        overall_progress = starting_progress + (step_progress * step_weight * 0.95)
        overall_progress = max(0.05, min(overall_progress, 0.98))
        # Pass the original status and info, but scaled progress
        original_callback(status, overall_progress, info)

    return scaled_callback

# --- Core Processing Functions ---

async def run_transcription(
    model: TranscriptionModel,
    file_path: str,
    language: Optional[str],
    progress_callback: Optional[Callable[[TaskStatus, float, Dict[str, Any]], None]] # Use TaskStatus type hint
) -> Dict[str, Any]:
    """Handles the transcription part of the process."""
    start_time = time.time()
    # Task ID is not directly passed here, but context implies it
    logger.info(f"[PROCESSOR] Starting transcription run for '{os.path.basename(file_path)}' with model {model.name}...")
    if progress_callback:
        progress_callback(TaskStatus.PROCESSING, 0.0, {"stage": "transcribing_start"})

    try:
        # Verify model is loaded BEFORE we start the async process
        if not model.is_loaded():
            logger.warning(f"[PROCESSOR] Model {model.name} was not loaded before transcription run. Attempting load now.")
            # If using CUDA, we should load on the appropriate device
            device = "cuda" if settings.USE_CUDA and torch and torch.cuda.is_available() else "cpu"
            model.load(device=device)
            
            # Double-check model is now properly loaded
            if not model.is_loaded():
                raise TranscriptionError(f"Failed to load model {model.name} before transcription")
            
            logger.info(f"[PROCESSOR] Model {model.name} loaded successfully after pre-check.")

        # Define an inner callback for the model's transcribe method if it supports it
        model_progress_callback = None
        if progress_callback:
            # Scale model progress (0-1) to the transcription phase (e.g., 0.1 to 0.8)
            def model_progress_inner(prog: float):
                 # Note: Actual progress updates from within model.transcribe might be limited
                 scaled_prog = 0.1 + prog * 0.7 # Example scaling
                 progress_callback(TaskStatus.PROCESSING, scaled_prog, {"stage": "transcribing_progress"})
            # Check if the specific model instance actually accepts a progress_callback
            # This requires inspecting the signature or adding an attribute/method to TranscriptionModel
            # For simplicity, we'll pass it assuming the WhisperModel implementation handles it (even if it just calls it at start/end)
            model_progress_callback = model_progress_inner

        # Instead of using asyncio.to_thread which might introduce reference issues,
        # we'll create a more controlled wrapper function that maintains strong references
        def transcribe_wrapper():
            # First, verify model is still loaded right before transcription
            if not model.is_loaded():
                raise TranscriptionError(f"Model {model.name} was invalidated between verification and transcription")
                
            # Now perform the actual transcription with all parameters
            return model.transcribe(
                audio_path=file_path,
                language=language,
                task="transcribe",
                progress_callback=model_progress_callback
            )
            
        # Use more verbose logging to track the process
        # Log moved up before the to_thread call
        
        # Run transcription in a separate thread using asyncio.to_thread
        # to avoid blocking the event loop.
        result = await asyncio.to_thread(transcribe_wrapper)

        processing_time = time.time() - start_time
        result["transcription_time"] = processing_time # Add specific timing
        logger.info(f"[PROCESSOR] Transcription run for '{os.path.basename(file_path)}' completed in {processing_time:.2f}s.")

        if progress_callback:
            # Signal 100% completion *of this step*
            progress_callback(TaskStatus.PROCESSING, 1.0, {"stage": "transcription_complete"})

        return result

    except Exception as e:
        logger.error(f"[PROCESSOR] Transcription run failed for '{os.path.basename(file_path)}': {e}", exc_info=True)
        if progress_callback:
             # Signal failure *of this step*
             progress_callback(TaskStatus.FAILED, 0.9, {"stage": "transcription_failed", "error": str(e)})
        # Re-raise as a TranscriptionError to be handled by the main processor
        raise TranscriptionError(f"Transcription execution failed: {e}") from e


async def run_diarization(
    file_path: str,
    task_params: Dict[str, Any],
    audio_duration: float,
    task_id: str, # Pass task_id for temp file naming
    progress_callback: Optional[Callable[[TaskStatus, float, Dict[str, Any]], None]] # Use TaskStatus type hint
) -> Optional[pd.DataFrame]:
    """Handles the diarization part of the process."""
    start_time = time.time()

    # Check availability again, as it might depend on runtime factors too
    diar_available_runtime = settings.DIARIZATION_ENABLED and DIARIZATION_AVAILABLE and diarization_service is not None

    if not diar_available_runtime:
        logger.warning(f"[PROCESSOR][{task_id}] Diarization requested but service is not available/enabled at runtime. Skipping.")
        if progress_callback: progress_callback(TaskStatus.PROCESSING, 1.0, {"stage": "diarization_skipped"})
        return None

    logger.info(f"[PROCESSOR][{task_id}] Starting diarization run for '{os.path.basename(file_path)}'...")
    if progress_callback:
        progress_callback(TaskStatus.PROCESSING, 0.0, {"stage": "diarizing"})

    try:
        # Extract relevant parameters
        language = task_params.get('language')
        user_num_speakers = task_params.get('num_speakers')
        user_min_speakers = task_params.get('min_speakers')
        user_max_speakers = task_params.get('max_speakers')
        # Extract new hyperparameters
        segmentation_onset = task_params.get('segmentation_onset')
        clustering_threshold = task_params.get('clustering_threshold')
        segmentation_min_duration_off = task_params.get('segmentation_min_duration_off')

        # Execute diarization using the service instance
        # Pass task_id for better temporary file management if needed
        diarization_result_list = await diarization_service.diarize_file(
            file_path=file_path,
            num_speakers=user_num_speakers,
            min_speakers=user_min_speakers,
            max_speakers=user_max_speakers,
            # Pass new hyperparameters
            segmentation_onset=segmentation_onset,
            clustering_threshold=clustering_threshold,
            segmentation_min_duration_off=segmentation_min_duration_off,
            # Other params
            progress_callback=progress_callback, # Pass the scaled callback directly
            language=language,
            task_id=task_id # Pass task_id
        )

        processing_time = time.time() - start_time
        logger.info(f"[PROCESSOR][{task_id}] Diarization run completed in {processing_time:.2f}s.")

        # Process results into DataFrame
        diarization_df = pd.DataFrame()
        if diarization_result_list:
             try:
                  temp_df = pd.DataFrame(diarization_result_list)
                  required_cols = ['start', 'end', 'speaker']
                  if all(col in temp_df.columns for col in required_cols):
                       diarization_df = temp_df[required_cols].copy() # Use copy to avoid SettingWithCopyWarning
                       # Ensure types are correct
                       diarization_df['start'] = pd.to_numeric(diarization_df['start'], errors='coerce')
                       diarization_df['end'] = pd.to_numeric(diarization_df['end'], errors='coerce')
                       diarization_df['speaker'] = diarization_df['speaker'].astype(str)
                       # Drop rows where conversion failed or start/end invalid
                       diarization_df = diarization_df.dropna(subset=['start', 'end'])
                       diarization_df = diarization_df[diarization_df['end'] > diarization_df['start']]

                       if not diarization_df.empty:
                            logger.info(f"[PROCESSOR][{task_id}] Diarization created DataFrame with {len(diarization_df)} segments, {len(diarization_df['speaker'].unique())} unique speaker labels.")
                       else:
                            logger.warning(f"[PROCESSOR][{task_id}] Diarization result list processed into an empty or invalid DataFrame.")
                  else:
                       logger.warning(f"[PROCESSOR][{task_id}] Diarization result list missing required columns (start, end, speaker). Discarding.")
             except Exception as df_e:
                  logger.error(f"[PROCESSOR][{task_id}] Error converting diarization result list to DataFrame: {df_e}", exc_info=True)
                  diarization_df = pd.DataFrame() # Ensure it's empty on error
        else:
             logger.warning(f"[PROCESSOR][{task_id}] Diarization service returned no results (empty list).")


        if progress_callback:
            # Signal 100% completion *of this step*
            progress_callback(TaskStatus.PROCESSING, 1.0, {"stage": "diarization_complete"})

        return diarization_df if not diarization_df.empty else None

    except Exception as e:
        logger.error(f"[PROCESSOR][{task_id}] Diarization run failed for '{os.path.basename(file_path)}': {e}", exc_info=True)
        if progress_callback:
             # Signal failure *of this step*
             progress_callback(TaskStatus.FAILED, 0.9, {"stage": "diarization_failed", "error": str(e)})
        # Return None to indicate failure, allowing main process to potentially continue
        return None

# --- Main Task Handler ---

async def process_audio(
    task_id: str, # Added task_id
    task_params: Dict[str, Any],
    progress_callback: Optional[Callable[[TaskStatus, float, Dict[str, Any]], None]] = None # Use TaskStatus
) -> Dict[str, Any]:
    """
    Main audio processing pipeline: transcription + optional diarization.
    Invoked by the TaskManager.
    """
    start_time = time.time()
    # Extract parameters
    file_path = task_params.get('file_path')
    model_size = task_params.get('model_size', settings.DEFAULT_MODEL)
    language = task_params.get('language', settings.DEFAULT_LANGUAGE) # Use configured default
    enable_diarization = task_params.get('enable_diarization', False)
    # task_id is passed as a direct argument, no need to get from params again

    if not file_path or not os.path.exists(file_path):
         # This should ideally be caught before queuing, but check again
         raise FileProcessingError(f"Input audio file path invalid or missing in task parameters: {file_path}")

    logger.info(f"[PROCESSOR][{task_id}] Starting audio processing. File: {os.path.basename(file_path)}, Model: {model_size}, Diarization: {enable_diarization}")

    # --- Preparation ---
    if progress_callback: progress_callback(TaskStatus.PROCESSING, 0.01, {"stage": "preparation"})

    # Get audio duration
    audio_duration = await estimate_audio_duration(file_path)
    logger.info(f"[PROCESSOR][{task_id}] Estimated audio duration: {audio_duration:.2f}s")

    # Select model
    model_key = f"whisper-{model_size}"
    try:
        model: TranscriptionModel = ModelRegistry.get_model(model_key)
    except ModelNotFoundError as e:
         logger.error(f"[PROCESSOR][{task_id}] Model '{model_key}' not found.")
         raise e # Re-raise to fail the task

    # Ensure model is loaded (can take time)
    if not model.is_loaded():
        logger.info(f"[PROCESSOR][{task_id}] Model {model_key} not loaded. Loading now...")
        if progress_callback: progress_callback(TaskStatus.LOADING_MODEL, 0.05, {"stage": "loading_model", "model": model_key})
        load_start = time.time()
        try:
             device = "cuda" if settings.USE_CUDA and torch and torch.cuda.is_available() else "cpu"
             # Use a dedicated thread for loading to avoid GC issues
             await asyncio.to_thread(model.load, device=device)
             logger.info(f"[PROCESSOR][{task_id}] Model {model_key} loaded in {time.time() - load_start:.2f}s.")
        except Exception as load_err:
             logger.error(f"[PROCESSOR][{task_id}] Failed to load model {model_key}: {load_err}", exc_info=True)
             raise TranscriptionError(f"Failed to load required model {model_key}") from load_err # Fail task if model won't load
    else:
         logger.info(f"[PROCESSOR][{task_id}] Model {model_key} is already loaded.")


    # Determine if diarization should run
    diar_available_runtime = settings.DIARIZATION_ENABLED and DIARIZATION_AVAILABLE and diarization_service is not None
    do_diarization = enable_diarization and diar_available_runtime

    if enable_diarization and not diar_available_runtime:
        logger.warning(f"[PROCESSOR][{task_id}] Diarization requested but is not available or not enabled. Proceeding without diarization.")


    # Define progress step weights
    diarization_weight = 0.4 if do_diarization else 0.0
    transcription_weight = 1.0 - diarization_weight

    # --- Transcription Step ---
    transcription_start_progress = 0.1 # Start after model load/prep
    transcription_scaled_cb = _create_scaled_callback(progress_callback, transcription_weight, transcription_start_progress)

    logger.info(f"[PROCESSOR][{task_id}] Starting transcription step...")
    try:
        transcription_result = await run_transcription(
            model, file_path, language, transcription_scaled_cb
        )
    except TranscriptionError as e:
        # Error already logged in run_transcription, re-raise to fail task
        raise e
    except Exception as e:
         # Catch unexpected errors during transcription call
         logger.exception(f"[PROCESSOR][{task_id}] Unexpected error during transcription step: {e}")
         raise TranscriptionError("Unexpected error during transcription") from e

    # Update overall progress after transcription finishes
    current_progress = transcription_start_progress + transcription_weight # Mark transcription step as done
    if progress_callback: progress_callback(TaskStatus.PROCESSING, current_progress, {"stage": "transcription_done"})

    # --- Diarization Step ---
    diarization_df = None
    if do_diarization:
        diarization_start_progress = current_progress
        diarization_scaled_cb = _create_scaled_callback(progress_callback, diarization_weight, diarization_start_progress)

        logger.info(f"[PROCESSOR][{task_id}] Starting diarization step...")
        # Ensure GPU memory is clear if possible before starting diarization (Removed explicit calls)
        # gc.collect()
        # if settings.USE_CUDA and torch and torch.cuda.is_available():
        #      torch.cuda.empty_cache()

        try:
            diarization_df = await run_diarization(
                # Use the *original* file path if preprocessing failed in diarization service?
                # Diarization service preprocessing handles this now. Pass original path.
                file_path,
                task_params,
                audio_duration,
                task_id, # Pass task_id
                diarization_scaled_cb
            )
        except DiarizationError as e:
             # Log the error but allow processing to potentially continue without speaker labels
             logger.error(f"[PROCESSOR][{task_id}] Diarization step failed: {e}. Proceeding without speaker labels.")
             diarization_df = None # Ensure df is None
        except Exception as e:
             # Catch unexpected errors during diarization call
             logger.exception(f"[PROCESSOR][{task_id}] Unexpected error during diarization step: {e}")
             diarization_df = None # Ensure df is None

        # Update overall progress after diarization attempt
        current_progress = diarization_start_progress + diarization_weight
        stage = "diarization_done" if diarization_df is not None else "diarization_failed_or_skipped"
        if progress_callback: progress_callback(TaskStatus.PROCESSING, current_progress, {"stage": stage})

    # --- Final Assembly ---
    if progress_callback: progress_callback(TaskStatus.COMPLETING, 0.98, {"stage": "finalizing"})

    final_result = {
        "transcription": transcription_result.get("text", ""),
        "segments": transcription_result.get("segments", []),
        "duration": transcription_result.get("duration", audio_duration),
        "language": transcription_result.get("language", "unknown"),
        "model": transcription_result.get("model", model_key),
        "speakers": [],
        "processing_time": time.time() - start_time, # Overall time
        "transcription_time": transcription_result.get("transcription_time"),
        # Add diarization time if available later
    }

    # Combine with diarization if successful
    if diarization_df is not None and not diarization_df.empty:
        logger.info(f"[PROCESSOR][{task_id}] Assigning speakers to transcription segments...")
        try:
            # Pass only the necessary parts to the assignment function
            assignment_input = {
                 "segments": final_result["segments"],
                 "duration": final_result["duration"] # Pass duration for context if needed
            }
            # assign_speakers_to_segments modifies the input dict directly
            assign_speakers_to_segments(diarization_df, assignment_input)
            # Update only the segments in final_result with the speaker-assigned segments
            final_result["segments"] = assignment_input["segments"]

            # Extract unique speakers from the final assigned segments
            assigned_speakers = sorted(list(set(
                 seg.get("speaker") for seg in final_result["segments"] if seg.get("speaker")
            )))
            final_result["speakers"] = assigned_speakers
            logger.info(f"[PROCESSOR][{task_id}] Final unique assigned speakers: {assigned_speakers}")
            _log_speaker_count_warnings(task_params, assigned_speakers, audio_duration)

        except Exception as e:
            logger.error(f"[PROCESSOR][{task_id}] Error assigning speakers to segments: {e}", exc_info=True)
            # Ensure segments don't have partial speaker assignments if failed
            for seg in final_result["segments"]: seg["speaker"] = None
            final_result["speakers"] = []
    else:
         # Ensure speaker field is present but None if no diarization
         for seg in final_result["segments"]: seg["speaker"] = None
         if enable_diarization: # Log only if user requested it but it failed/was unavailable
             logger.warning(f"[PROCESSOR][{task_id}] Speaker assignment skipped: No valid diarization data was available.")

    # --- Cleanup --- (Removed explicit calls)
    # gc.collect()
    # if settings.USE_CUDA and torch and torch.cuda.is_available():
    #     torch.cuda.empty_cache()

    # Final completion signal
    if progress_callback:
        progress_callback(TaskStatus.COMPLETED, 1.0, {"stage": "completed"})

    # Log performance
    total_time = final_result["processing_time"]
    audio_len = final_result.get("duration", 0)
    if audio_len > 0 and total_time > 0:
        realtime_factor = audio_len / total_time
        logger.info(f"[PROCESSOR][{task_id}] Processing completed in {total_time:.2f}s for {audio_len:.2f}s audio (RTF: {realtime_factor:.2f}x)")
    else:
        logger.info(f"[PROCESSOR][{task_id}] Processing completed in {total_time:.2f}s")

    return final_result


def assign_speakers_to_segments(
    diarize_df: pd.DataFrame,
    transcript_result: Dict[str, Any],
    min_overlap_ratio: float = 0.1, # Min overlap needed to consider assignment
    confidence_threshold: float = 0.5, # Confidence threshold for weighted overlap
    fill_nearest_threshold_sec: float = 1.5 # Max distance for nearest speaker fallback (Reduced from 3.0)
) -> Dict[str, Any]:
    """
    Assign speakers from diarization results to transcription segments.
    Modifies transcript_result['segments'] in place.

    Args:
        diarize_df: DataFrame with 'start', 'end', 'speaker' columns from diarization.
        transcript_result: Dictionary containing 'segments' list and 'duration'.
        min_overlap_ratio: Minimum ratio of segment duration covered by speaker turn.
        confidence_threshold: Minimum confidence for weighted overlap assignment.
        fill_nearest_threshold_sec: Max time distance to assign nearest speaker if overlap fails.

    Returns:
        The updated transcript_result dictionary with 'speaker' assigned in segments.
    """
    # --- Input Validation ---
    if diarize_df is None or diarize_df.empty:
        logger.warning("[PROCESSOR][ASSIGN] Diarization data is empty. Skipping assignment.")
        for seg in transcript_result.get("segments", []): seg["speaker"] = None
        return transcript_result

    required_cols = ['start', 'end', 'speaker']
    if not all(col in diarize_df.columns for col in required_cols):
        logger.warning(f"[PROCESSOR][ASSIGN] Diarization data missing columns ({required_cols}). Skipping.")
        for seg in transcript_result.get("segments", []): seg["speaker"] = None
        return transcript_result

    transcript_segments = transcript_result.get("segments", [])
    if not transcript_segments:
        logger.warning("[PROCESSOR][ASSIGN] Transcription result has no segments. Skipping.")
        return transcript_result

    # Ensure numeric types and sort diarization data
    try:
        # Work on a copy to avoid modifying the original DataFrame passed in
        diarize_df_processed = diarize_df.copy()
        diarize_df_processed['start'] = pd.to_numeric(diarize_df_processed['start'], errors='coerce')
        diarize_df_processed['end'] = pd.to_numeric(diarize_df_processed['end'], errors='coerce')
        diarize_df_processed['speaker'] = diarize_df_processed['speaker'].astype(str)
        # Drop invalid rows after conversion
        diarize_df_processed = diarize_df_processed.dropna(subset=['start', 'end'])
        diarize_df_processed = diarize_df_processed[diarize_df_processed['end'] > diarize_df_processed['start']]
        diarize_df_processed = diarize_df_processed.sort_values('start').reset_index(drop=True)
        if diarize_df_processed.empty:
             logger.warning("[PROCESSOR][ASSIGN] Diarization data became empty after cleaning.")
             for seg in transcript_segments: seg["speaker"] = None
             return transcript_result

    except Exception as e:
        logger.error(f"[PROCESSOR][ASSIGN] Failed to process diarization data types for speaker assignment: {e}")
        for seg in transcript_segments: seg["speaker"] = None
        return transcript_result

    # --- Preprocessing ---
    # Create IntervalIndex for efficient overlap calculation
    try:
        diarize_intervals = pd.IntervalIndex.from_arrays(
            diarize_df_processed['start'], diarize_df_processed['end'], closed='both'
        )
    except Exception as e:
         logger.error(f"[PROCESSOR][ASSIGN] Failed to create IntervalIndex from diarization data: {e}. Check for invalid times.")
         for seg in transcript_segments: seg["speaker"] = None
         return transcript_result

    # --- Main Assignment Loop ---
    num_assigned_overlap = 0
    num_assigned_nearest = 0
    num_unassigned = 0

    for seg_idx, seg in enumerate(transcript_segments):
        seg["speaker"] = None # Default to unassigned
        assigned = False

        try:
            seg_start = float(seg['start'])
            seg_end = float(seg['end'])
            seg_duration = seg_end - seg_start
            if seg_duration <= 0.01: continue # Skip tiny or invalid segments
        except (TypeError, ValueError):
            logger.warning(f"[PROCESSOR][ASSIGN] Skipping segment {seg_idx} due to invalid start/end times: start={seg.get('start')}, end={seg.get('end')}")
            continue

        # Find overlapping diarization intervals
        try:
             # Create segment interval safely
             seg_interval = pd.Interval(seg_start, seg_end, closed='both')
             overlapping_indices = diarize_intervals.overlaps(seg_interval)
        except Exception as e:
             logger.warning(f"[PROCESSOR][ASSIGN] Overlap calculation failed for segment {seg_idx}: {e}")
             overlapping_indices = np.array([False] * len(diarize_df_processed)) # Assume no overlap on error


        if overlapping_indices.any():
            overlapping_dia = diarize_df_processed[overlapping_indices]
            overlap_scores = {} # speaker -> weighted overlap score

            for _, dia_seg in overlapping_dia.iterrows():
                # Calculate actual intersection duration
                overlap_start = max(seg_start, dia_seg['start'])
                overlap_end = min(seg_end, dia_seg['end'])
                overlap_duration = max(0, overlap_end - overlap_start) # Ensure non-negative

                if overlap_duration / seg_duration >= min_overlap_ratio:
                    speaker = dia_seg['speaker']
                    # Simple weighted score (duration) - can be enhanced (e.g., center weighting)
                    score = overlap_duration
                    overlap_scores[speaker] = overlap_scores.get(speaker, 0) + score

            if overlap_scores:
                best_speaker, best_score = max(overlap_scores.items(), key=lambda item: item[1])
                total_score = sum(overlap_scores.values())
                confidence = best_score / total_score if total_score > 0 else 0

                if confidence >= confidence_threshold:
                    seg["speaker"] = best_speaker
                    num_assigned_overlap += 1
                    assigned = True
                # else: confidence too low for overlap assignment

        # If not assigned by overlap (or low confidence), try nearest speaker fallback
        if not assigned and fill_nearest_threshold_sec > 0:
            nearest_speaker, min_distance = _find_nearest_speaker(seg_start, seg_end, diarize_df_processed)
            # Assign if a nearest speaker was found within the time threshold
            if nearest_speaker is not None and min_distance <= fill_nearest_threshold_sec:
                seg["speaker"] = nearest_speaker
                num_assigned_nearest += 1
                assigned = True

        if not assigned:
            num_unassigned += 1

    logger.info(f"[PROCESSOR][ASSIGN] Speaker assignment stats: Overlap={num_assigned_overlap}, Nearest={num_assigned_nearest}, Unassigned={num_unassigned} / Total={len(transcript_segments)}")

    # --- Post-processing: Merge adjacent segments with same speaker ---
    merged_segments = _merge_adjacent_segments(transcript_segments)
    transcript_result["segments"] = merged_segments
    logger.info(f"[PROCESSOR][ASSIGN] Segment merging reduced segments from {len(transcript_segments)} to {len(merged_segments)}")

    return transcript_result # Return modified dict


def _find_nearest_speaker(seg_start, seg_end, diarize_df) -> Tuple[Optional[str], float]:
    """Helper to find the closest speaker turn (by midpoint distance) to a segment midpoint."""
    if diarize_df.empty:
        return None, float('inf')

    try:
        seg_midpoint = seg_start + (seg_end - seg_start) / 2

        # Calculate midpoints and distances for all diarization segments efficiently
        dia_midpoints = diarize_df['start'] + (diarize_df['end'] - diarize_df['start']) / 2
        distances = np.abs(dia_midpoints - seg_midpoint)

        # Find the index of the minimum distance using idxmin()
        min_idx = distances.idxmin()

        # Return the speaker and distance at that index
        return diarize_df.loc[min_idx, 'speaker'], distances.loc[min_idx]
    except Exception as e:
         logger.error(f"[PROCESSOR][ASSIGN_HELPER] Error finding nearest speaker: {e}")
         return None, float('inf')


def _merge_adjacent_segments(segments: List[Dict[str, Any]], max_gap_sec: float = 0.15) -> List[Dict[str, Any]]:
    """Merges adjacent segments if they have the same speaker and the gap is small."""
    if not segments:
        return []

    merged = []
    current_segment = segments[0].copy() # Start with the first segment

    for i in range(1, len(segments)):
        next_segment = segments[i]
        gap = next_segment['start'] - current_segment['end']

        # Check for same speaker (and speaker is not None) and small gap
        if (current_segment.get('speaker') is not None and
            current_segment.get('speaker') == next_segment.get('speaker') and
            0 <= gap <= max_gap_sec):
            # Merge: extend end time and combine text (handle potential word timings later if needed)
            current_segment['end'] = next_segment['end']
            # Simple text concatenation, might need refinement if word timestamps exist
            current_segment['text'] += " " + next_segment['text']
            # If merging word timestamps, logic would go here
        else:
            # No merge possible, add the completed current segment and start new
            merged.append(current_segment)
            current_segment = next_segment.copy()

    # Add the last segment (either merged or the original last one)
    merged.append(current_segment)

    return merged


def _log_speaker_count_warnings(task_params, assigned_speakers, audio_duration):
    """Logs warnings if detected speaker count mismatches requested parameters."""
    detected_count = len(assigned_speakers)
    user_num = task_params.get('num_speakers')
    user_min = task_params.get('min_speakers')
    user_max = task_params.get('max_speakers')

    if user_num is not None:
        if detected_count != user_num:
             logger.warning(f"[PROCESSOR][WARN] Detected {detected_count} speakers, but user requested exactly {user_num}.")
    else:
        # Determine effective min/max used by diarization if user didn't specify
        effective_min = user_min if user_min is not None else 1
        if user_max is None:
             default_max = 10 if audio_duration > 1800 else (8 if audio_duration > 600 else 6)
             effective_max = max(default_max, effective_min)
        else:
             effective_max = max(user_max, effective_min)

        if detected_count < effective_min:
             logger.warning(f"[PROCESSOR][WARN] Detected only {detected_count} speakers, below the minimum requested/default of {effective_min}.")
        # Check against effective_max, not user_max directly unless user_max was provided
        if detected_count > effective_max:
             logger.warning(f"[PROCESSOR][WARN] Detected {detected_count} speakers, exceeding the maximum requested/default of {effective_max}.")


async def handle_diarization_only(
    task_id: str, # Added task_id
    task_params: Dict[str, Any],
    progress_callback: Optional[Callable[[TaskStatus, float, Dict[str, Any]], None]] = None # Use TaskStatus
) -> Dict[str, Any]:
    """
    Handle diarization-only task execution.
    Invoked by the TaskManager for 'diarization_only' tasks.
    """
    start_time = time.time()
    file_path = task_params.get('file_path')
    # task_id is passed as a direct argument, no need to get from params again

    if not file_path or not os.path.exists(file_path):
         raise FileProcessingError(f"Invalid or missing file_path in task parameters: {file_path}")

    logger.info(f"[PROCESSOR][{task_id}] Starting diarization-only task. File: {os.path.basename(file_path)}")

    # --- Validation & Preparation ---
    # Re-check runtime availability
    diar_available_runtime = settings.DIARIZATION_ENABLED and DIARIZATION_AVAILABLE and diarization_service is not None
    if not diar_available_runtime:
        error_msg = "Diarization service not available or not properly initialized at runtime."
        logger.error(f"[PROCESSOR][{task_id}] {error_msg}")
        if progress_callback: progress_callback(TaskStatus.FAILED, 0.0, {"error": error_msg})
        raise ConfigurationError(error_msg)

    if not settings.HUGGINGFACE_TOKEN:
         error_msg = "Hugging Face token required for diarization is not configured."
         logger.error(f"[PROCESSOR][{task_id}] {error_msg}")
         if progress_callback: progress_callback(TaskStatus.FAILED, 0.0, {"error": error_msg})
         raise ConfigurationError(error_msg)


    if progress_callback: progress_callback(TaskStatus.PROCESSING, 0.05, {"stage": "preparation"})

    # Get audio duration
    audio_duration = await estimate_audio_duration(file_path)
    logger.info(f"[PROCESSOR][{task_id}] Estimated audio duration: {audio_duration:.2f}s")

    # --- Diarization Step ---
    # Full weight for this single step
    # Create a scaled callback starting from 10% progress
    diarization_scaled_cb = _create_scaled_callback(progress_callback, 1.0, 0.1)

    logger.info(f"[PROCESSOR][{task_id}] Starting diarization execution...")
    try:
        diarization_segments = await run_diarization( # Use the common run_diarization function
            file_path=file_path,
            task_params=task_params,
            audio_duration=audio_duration,
            task_id=task_id,
            progress_callback=diarization_scaled_cb
        )
        # Note: run_diarization now returns a DataFrame or None
        if diarization_segments is None:
             logger.warning(f"[PROCESSOR][{task_id}] Diarization execution returned no results.")
             diarization_df = pd.DataFrame() # Ensure empty df if None returned
        else:
             diarization_df = diarization_segments # Already a DataFrame

    except Exception as e:
         # Errors are logged within run_diarization, just re-raise
         raise e

    # --- Final Assembly ---
    if progress_callback: progress_callback(TaskStatus.COMPLETING, 0.98, {"stage": "finalizing"})

    # Format results from DataFrame
    final_segments = []
    final_speakers = []
    if diarization_df is not None and not diarization_df.empty:
         try:
              # Convert DataFrame records to list of dicts for the final result
              final_segments = diarization_df.to_dict('records')
              final_speakers = sorted(list(diarization_df['speaker'].unique()))
              logger.info(f"[PROCESSOR][{task_id}] Diarization-only successful, found {len(final_speakers)} speakers.")
              _log_speaker_count_warnings(task_params, final_speakers, audio_duration)
         except Exception as format_e:
              logger.error(f"[PROCESSOR][{task_id}] Failed to format diarization DataFrame results: {format_e}")
              final_segments = []
              final_speakers = []
    else:
         logger.warning(f"[PROCESSOR][{task_id}] Diarization-only resulted in no speaker segments.")


    final_result = {
        "segments": final_segments,
        "speakers": final_speakers,
        "duration": audio_duration,
        "processing_time": time.time() - start_time
    }

    # --- Cleanup --- (Removed explicit calls)
    # gc.collect()
    # if settings.USE_CUDA and torch and torch.cuda.is_available():
    #     logger.debug(f"[PROCESSOR][{task_id}] Clearing CUDA cache after diarization-only task.")
    #     torch.cuda.empty_cache()

    # Final completion signal
    if progress_callback:
        progress_callback(TaskStatus.COMPLETED, 1.0, {"stage": "completed"})

    logger.info(f"Diarization-only task {task_id} completed in {final_result['processing_time']:.2f}s")

    return final_result