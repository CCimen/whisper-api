"""
API Endpoints for submitting and managing speaker diarization tasks.
"""

import os
import shutil
import logging
import uuid
import asyncio
from typing import List, Optional, Dict, Any

from fastapi import (
    APIRouter, File, UploadFile, Query, HTTPException, status, Depends, Body
)
from pydantic import BaseModel, Field, validator

# Use the central router registry
from app.api.router_registry import router_diarization
from app.config import settings
from app.services.task_manager import task_manager, TaskStatus
from app.exceptions import ConfigurationError

# Import security dependency
from app.main import get_api_key # Import from main where it's defined

# Import helper from transcription route
from .transcription import _save_upload_file

# Configure logging
logger = logging.getLogger(__name__)

# Check for diarization support
try:
    from app.services.diarization import DIARIZATION_AVAILABLE
except ImportError:
    DIARIZATION_AVAILABLE = False
except Exception as e:
     logger.error(f"Failed to check DIARIZATION_AVAILABLE: {e}")
     DIARIZATION_AVAILABLE = False

# --- Pydantic Models ---

class DiarizationRequest(BaseModel):
    """Parameters for submitting a diarization-only job."""
    language: Optional[str] = Field(
        default=None,
        description="Language code (e.g., 'sv') - can help optimize diarization.",
        examples=["sv"]
    )
    num_speakers: Optional[int] = Field(
        default=None,
        ge=1,
        description="Exact number of speakers expected (overrides min/max_speakers)."
    )
    min_speakers: Optional[int] = Field(
        default=None,
        ge=1,
        description="Minimum number of speakers expected (used if num_speakers is not set)."
    )
    max_speakers: Optional[int] = Field(
        default=None,
        ge=1,
        description="Maximum number of speakers expected (used if num_speakers is not set)."
    )
    # --- Pyannote Hyperparameters ---
    segmentation_onset: Optional[float] = Field(
        default=None, # Pipeline default usually ~0.5
        description="Speech activity detection onset threshold (probability). Higher values require more confidence (e.g., 0.7), lower values are more sensitive (e.g., 0.3). Default ~0.5.",
        examples=[0.3, 0.5, 0.7]
    )
    clustering_threshold: Optional[float] = Field(
        default=None, # Pipeline default varies, e.g., ~0.7 for pyannote/speaker-diarization-3.1
        description="Speaker clustering threshold (distance/similarity). Lower values merge more (e.g., 0.5), higher values require more distinct speakers (e.g., 0.8). Model-dependent, requires experimentation.",
        examples=[0.5, 0.7, 0.8]
    )
    segmentation_min_duration_off: Optional[float] = Field(
        default=None, # Pipeline default usually ~0.0 or 0.1
        description="Minimum silence duration (seconds) to split speech segments. Increasing merges segments separated by shorter pauses (e.g., 0.5). Default ~0.0-0.1.",
        examples=[0.0, 0.1, 0.5]
    )

    @validator('max_speakers')
    def check_max_speakers(cls, v, values):
        min_val = values.get('min_speakers')
        if v is not None and min_val is not None and v < min_val:
            raise ValueError('max_speakers must be greater than or equal to min_speakers')
        return v

    @validator('num_speakers')
    def check_num_speakers_exclusive(cls, v, values):
        if v is not None and (values.get('min_speakers') is not None or values.get('max_speakers') is not None):
            raise ValueError('num_speakers cannot be used together with min_speakers or max_speakers')
        return v


class DiarizationSegment(BaseModel):
    """Represents a segment with an identified speaker and timing."""
    start: float = Field(..., description="Start time in seconds.")
    end: float = Field(..., description="End time in seconds.")
    speaker: str = Field(..., description="Identified speaker label (e.g., 'SPEAKER_00').")


class DiarizationResponse(BaseModel):
    """Response model for diarization status and results."""
    id: str = Field(..., description="Unique ID of the diarization task.")
    status: str = Field(..., description="Current status of the task (e.g., queued, processing, completed).")
    progress: Optional[float] = Field(None, description="Processing progress (0.0 to 1.0).")
    queue_position: Optional[int] = Field(None, description="Position in the queue (if status is queued).")
    error: Optional[str] = Field(None, description="Error message if the task failed.")
    # Result fields (populated when completed)
    segments: Optional[List[DiarizationSegment]] = Field(None, description="List of timed speaker segments.")
    speakers: Optional[List[str]] = Field(None, description="List of unique speaker labels identified.")
    duration: Optional[float] = Field(None, description="Duration of the processed audio in seconds.")
    processing_time: Optional[float] = Field(None, description="Total processing time in seconds.")


# --- Helper Function ---
def _validate_diarization_availability():
    """Raises HTTPException if diarization is not configured/available."""
    if not settings.DIARIZATION_ENABLED:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Diarization is disabled on this server.")
    if not DIARIZATION_AVAILABLE:
        raise HTTPException(status_code=status.HTTP_501_NOT_IMPLEMENTED, detail="Diarization dependencies are not installed on this server.")
    if not settings.HUGGINGFACE_TOKEN:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Hugging Face token required for diarization is not configured.")

# --- API Endpoints ---

@router_diarization.post("/", response_model=DiarizationResponse, status_code=status.HTTP_202_ACCEPTED)
async def submit_diarization_only_job(
    request_params: DiarizationRequest = Depends(), # Inject model using Depends for Query params
    audio_file: UploadFile = File(..., description="The audio file to diarize."),
    # API Key Dependency applied at the router level in main.py
    # _: bool = Depends(get_api_key)
):
    """
    Submit an audio file for speaker diarization ONLY (no transcription).

    Returns a response immediately with the task ID and initial status.
    Use the GET endpoints with the returned ID to check progress and retrieve results.
    """
    _validate_diarization_availability()

    # --- Input Validation ---
    if audio_file.size > settings.MAX_UPLOAD_SIZE_MB * 1024 * 1024:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File size ({audio_file.size / (1024*1024):.1f} MB) exceeds maximum limit of {settings.MAX_UPLOAD_SIZE_MB} MB."
        )

    # --- File Handling ---
    try:
        saved_file_path = await _save_upload_file(audio_file)
    finally:
        await audio_file.close()

    # --- Task Creation ---
    try:
        task_params = {
            "file_path": saved_file_path,
            "language": request_params.language,
            "num_speakers": request_params.num_speakers,
            "min_speakers": request_params.min_speakers,
            "max_speakers": request_params.max_speakers,
            # Add new hyperparameters
            "segmentation_onset": request_params.segmentation_onset,
            "clustering_threshold": request_params.clustering_threshold,
            "segmentation_min_duration_off": request_params.segmentation_min_duration_off,
        }
        task_params = {k: v for k, v in task_params.items() if v is not None} # Filter None

        logger.info(f"Creating diarization_only task with parameters: {task_params}")
        task_id = task_manager.create_task("diarization_only", task_params)
        await task_manager.queue_task(task_id)

        initial_task_info = task_manager.get_task(task_id, include_result=False)
        if not initial_task_info:
            raise HTTPException(status_code=500, detail="Failed to retrieve task status after creation.")

        response_data = DiarizationResponse(
             id=initial_task_info["id"],
             status=initial_task_info["status"],
             progress=initial_task_info["progress"],
             queue_position=initial_task_info.get("queue_position"),
        )
        return response_data

    except (ValueError, ConfigurationError) as e:
         logger.error(f"Configuration or Value error creating task: {e}")
         if os.path.exists(saved_file_path):
             try: os.remove(saved_file_path)
             except OSError: pass
         raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.exception(f"Unexpected error creating diarization task: {e}")
        if os.path.exists(saved_file_path):
            try: os.remove(saved_file_path)
            except OSError: pass
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to create diarization task.")


@router_diarization.get("/{task_id}/status", response_model=DiarizationResponse)
async def get_diarization_task_status(task_id: str):
     # API Key Dependency applied at the router level in main.py
    """Check the current status and progress of a diarization-only job."""
    task_info = task_manager.get_task(task_id, include_result=False)
    if not task_info:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Task ID '{task_id}' not found.")
    if task_info.get("type") != "diarization_only":
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Task ID '{task_id}' is not a diarization_only task.")

    return DiarizationResponse(
         id=task_info["id"],
         status=task_info["status"],
         progress=task_info["progress"],
         queue_position=task_info.get("queue_position"),
         error=task_info.get("error"),
    )


@router_diarization.get("/{task_id}", response_model=DiarizationResponse)
async def get_diarization_task_result(task_id: str):
     # API Key Dependency applied at the router level in main.py
    """
    Retrieve the results of a completed diarization-only job.
    """
    task_info = task_manager.get_task(task_id, include_result=True)
    if not task_info:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Task ID '{task_id}' not found.")
    if task_info.get("type") != "diarization_only":
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Task ID '{task_id}' is not a diarization_only task.")

    result_data = task_info.get("result") or {}
    return DiarizationResponse(
        id=task_info["id"],
        status=task_info["status"],
        progress=task_info["progress"],
        error=task_info.get("error"),
        segments=result_data.get("segments"),
        speakers=result_data.get("speakers"),
        duration=result_data.get("duration"),
        processing_time=result_data.get("processing_time"),
    )


@router_diarization.delete("/{task_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_diarization_task(task_id: str):
     # API Key Dependency applied at the router level in main.py
    """
    Delete a diarization-only task record and associated temporary files.
    If the task is running, it will be cancelled first.
    """
    # 1. Get task info and validate
    task_info = task_manager.get_task(task_id, include_result=False)
    if not task_info:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Task ID '{task_id}' not found.")
    if task_info.get("type") != "diarization_only":
        # This check might be redundant if get_task already filters, but good for clarity
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Task ID '{task_id}' is not a diarization_only task.")

    current_status = TaskStatus(task_info["status"]) # Convert string back to enum

    # 2. Check if already terminal
    if TaskStatus.is_terminal(current_status):
        logger.info(f"Task {task_id} is already in terminal state ({current_status}). Proceeding with deletion.")
    else:
        # 3. Cancel the task if active or queued
        logger.info(f"Task {task_id} is in state {current_status}. Attempting cancellation before deletion.")
        try:
            cancelled_successfully = await task_manager.cancel_task(task_id)
            if not cancelled_successfully:
                 # This might happen if the task finished between the get_task and cancel_task calls
                 logger.warning(f"Cancellation signal for task {task_id} returned False. Checking status again.")
                 task_info = task_manager.get_task(task_id, include_result=False)
                 if task_info and not TaskStatus.is_terminal(TaskStatus(task_info["status"])):
                      # If still not terminal after failed cancel signal, something is wrong
                      raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to effectively cancel task {task_id} before deletion.")

            # 4. Wait for cancellation to complete (polling)
            wait_start_time = asyncio.get_event_loop().time()
            timeout_seconds = 30.0 # Adjust as needed
            poll_interval = 0.5   # Adjust as needed

            while True:
                task_info = task_manager.get_task(task_id, include_result=False)
                if not task_info: # Task might have been deleted concurrently? Unlikely but possible
                    logger.warning(f"Task {task_id} disappeared during cancellation wait.")
                    return None # Treat as deleted

                current_status = TaskStatus(task_info["status"])
                if TaskStatus.is_terminal(current_status):
                    logger.info(f"Task {task_id} reached terminal state ({current_status}) after cancellation request.")
                    break # Exit loop, ready to delete

                if (asyncio.get_event_loop().time() - wait_start_time) > timeout_seconds:
                    logger.error(f"Timeout waiting for task {task_id} to reach terminal state after cancellation request. Proceeding with deletion anyway.")
                    # Optionally raise 500 here instead?
                    # raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Timeout waiting for task {task_id} to cancel.")
                    break # Exit loop, attempt deletion despite timeout

                await asyncio.sleep(poll_interval)

        except Exception as e:
             logger.error(f"Error during cancellation/wait for task {task_id}: {e}", exc_info=True)
             # Decide if deletion should still be attempted or raise 500
             raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Error cancelling task {task_id}: {e}")


    # 5. Delete the task record (now that it's terminal or timed out)
    deleted = task_manager.delete_task(task_id) # delete_task handles the actual file cleanup
    if not deleted:
         # This could happen if the task was deleted by another process between wait and delete
         logger.warning(f"Attempted to delete task {task_id}, but it was not found (possibly deleted concurrently).")
         # Return 204 anyway, as the desired state (deleted) is achieved
         # Alternatively, could raise 404 here, but 204 seems more idempotent
         # raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Task ID '{task_id}' not found during final deletion step.")

    logger.info(f"Successfully processed deletion request for diarization task {task_id}")
    return None # Return None for 204 No Content