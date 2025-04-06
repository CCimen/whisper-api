"""
API Endpoints for submitting and managing audio transcription tasks.
"""

import os
import shutil
import logging
import uuid
from typing import List, Optional, Dict, Any
from enum import Enum # Import Enum

from fastapi import (
    APIRouter, File, UploadFile, Query, HTTPException, status, Depends, Body
)
from pydantic import BaseModel, Field, validator

# Use the central router registry
from app.api.router_registry import router_transcription
from app.config import settings, WHISPER_MODEL_MAPPING
from app.services.task_manager import TaskManager, TaskStatus # Import TaskManager type hint
from app.exceptions import ConfigurationError, FileProcessingError # For validation/file errors

# Import security dependency
from app.main import get_api_key # Keep API key import if needed here
from app.dependencies import get_task_manager # Import from new location

# Configure logging
logger = logging.getLogger(__name__)

# Check for diarization support during import to avoid runtime checks everywhere
try:
    from app.services.diarization import DIARIZATION_AVAILABLE
except ImportError:
    DIARIZATION_AVAILABLE = False
except Exception as e:
     logger.error(f"Failed to check DIARIZATION_AVAILABLE: {e}")
     DIARIZATION_AVAILABLE = False


# --- Pydantic Models ---

# Use Enums for allowed model sizes based on config
AvailableModels = list(WHISPER_MODEL_MAPPING.keys())
# Ensure AvailableModels is not empty before creating Enum
if not AvailableModels:
    logger.error("WHISPER_MODEL_MAPPING is empty in config. Cannot create ModelSizeEnum.")
    # Provide a fallback or raise a ConfigurationError? Fallback for now.
    ModelSizeEnum = Enum("ModelSizeEnum", {"medium": "medium"}) # Default fallback
else:
    ModelSizeEnum = Enum("ModelSizeEnum", {size: size for size in AvailableModels})


class TranscriptionRequest(BaseModel):
    """Parameters for submitting a transcription job."""
    language: Optional[str] = Field(
        default=None,
        description="Target language code (e.g., 'en', 'sv'). If None, attempts auto-detection.",
        examples=["sv", "en"]
    )
    model_size: ModelSizeEnum = Field(
        # Ensure the default model from settings is actually in the Enum
        default=ModelSizeEnum(settings.DEFAULT_MODEL) if settings.DEFAULT_MODEL in AvailableModels else ModelSizeEnum(AvailableModels[0]),
        description="Whisper model size/type to use."
    )
    diarization: bool = Field(
        default=False,
        description="Enable speaker diarization (identifies different speakers)."
    )
    num_speakers: Optional[int] = Field(
        default=None,
        ge=1, # Must be at least 1 speaker
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


class TimestampedSegment(BaseModel):
    """Represents a segment of the transcript with timing and speaker."""
    start: float = Field(..., description="Start time in seconds.")
    end: float = Field(..., description="End time in seconds.")
    text: str = Field(..., description="Transcribed text of the segment.")
    speaker: Optional[str] = Field(None, description="Identified speaker label (if diarization is enabled).")


class TranscriptionResponse(BaseModel):
    """Response model for transcription status and results."""
    id: str = Field(..., description="Unique ID of the transcription task.")
    status: str = Field(..., description="Current status of the task (e.g., queued, processing, completed).")
    progress: Optional[float] = Field(None, ge=0.0, le=1.0, description="Processing progress (0.0 to 1.0).")
    queue_position: Optional[int] = Field(None, ge=1, description="Position in the queue (if status is queued).")
    error: Optional[str] = Field(None, description="Error message if the task failed.")
    # Result fields (populated when completed)
    transcription: Optional[str] = Field(None, description="The full transcribed text.")
    segments: Optional[List[TimestampedSegment]] = Field(None, description="List of timed text segments.")
    speakers: Optional[List[str]] = Field(None, description="List of unique speaker labels identified.")
    duration: Optional[float] = Field(None, ge=0.0, description="Duration of the processed audio in seconds.")
    processing_time: Optional[float] = Field(None, ge=0.0, description="Total processing time in seconds.")
    model: Optional[str] = Field(None, description="The specific model used for transcription.")
    language: Optional[str] = Field(None, description="Detected or specified language code.")


# --- Helper Functions ---

async def _save_upload_file(upload_file: UploadFile) -> str:
    """Saves uploaded file securely to the configured upload directory."""
    base_filename = os.path.basename(upload_file.filename or "audio_file")
    safe_filename = "".join(c for c in base_filename if c.isalnum() or c in ['.', '_', '-'])[:100]
    file_extension = os.path.splitext(safe_filename)[1].lower()

    if file_extension not in settings.ALLOWED_FILE_EXTENSIONS:
         logger.warning(f"File upload with potentially disallowed extension: {file_extension} (from: {upload_file.filename})")

    try:
        os.makedirs(settings.UPLOAD_DIR, mode=0o700, exist_ok=True)
    except OSError as e:
        logger.error(f"Cannot create upload directory: {settings.UPLOAD_DIR}. Error: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Server storage configuration error.")

    file_id = str(uuid.uuid4())
    save_path = os.path.join(settings.UPLOAD_DIR, f"{file_id}{file_extension or '.tmp'}")

    try:
        logger.debug(f"Attempting to save uploaded file '{upload_file.filename}' to '{save_path}'")
        with open(save_path, "wb") as buffer:
            shutil.copyfileobj(upload_file.file, buffer)
        os.chmod(save_path, 0o600)
        logger.info(f"Saved uploaded file '{upload_file.filename}' to '{save_path}'")
        return save_path
    except Exception as e:
        logger.error(f"Failed to save uploaded file '{upload_file.filename}': {e}", exc_info=True)
        if os.path.exists(save_path):
            try: os.remove(save_path)
            except OSError: pass
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to save uploaded file.")


# --- API Endpoints ---

@router_transcription.post(
    "/",
    response_model=TranscriptionResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Submit Transcription Job",
    description="Upload an audio file to start an asynchronous transcription task with optional speaker diarization."
)
async def submit_transcription_job(
    request_params: TranscriptionRequest = Depends(),
    audio_file: UploadFile = File(..., description="The audio file to transcribe (e.g., mp3, wav, m4a)."),
    tm: TaskManager = Depends(get_task_manager)
    # _: bool = Depends(get_api_key) # API key dep is handled by router inclusion in main.py
):
    """
    Submit an audio file for asynchronous transcription and optional diarization.
    """
    # --- Input Validation ---
    if audio_file.size is None or audio_file.size <= 0:
         raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid or empty file uploaded.")
    if audio_file.size > settings.MAX_UPLOAD_SIZE_MB * 1024 * 1024:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File size ({audio_file.size / (1024*1024):.1f} MB) exceeds maximum limit of {settings.MAX_UPLOAD_SIZE_MB} MB."
        )
    if request_params.diarization:
        if not settings.DIARIZATION_ENABLED:
             raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Diarization feature is disabled on this server.")
        if not DIARIZATION_AVAILABLE:
             raise HTTPException(status_code=status.HTTP_501_NOT_IMPLEMENTED, detail="Diarization dependencies are not installed or available on this server.")
        # Consider warning vs error for missing token as before
        # if not settings.HUGGINGFACE_TOKEN: logger.warning(...)

    # --- File Handling ---
    saved_file_path = None
    try:
        saved_file_path = await _save_upload_file(audio_file)
    finally:
        await audio_file.close()

    # TaskManager is now injected, check should not be needed here, but keep for safety?
    # Or rely on the dependency raising HTTPException if tm is None. Let's rely on dependency.
    # if not tm:
         # logger.critical("TaskManager not initialized. Cannot create task.") # Redundant - handled by dependency
         # if saved_file_path and os.path.exists(saved_file_path):
         #     try: os.remove(saved_file_path)
             # except OSError: pass
         # raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Task processing service is unavailable.")

    # --- Task Creation ---
    try:
        task_params = request_params.model_dump(exclude_unset=True)
        task_params["file_path"] = saved_file_path
        task_params["model_size"] = request_params.model_size.value
        task_params["enable_diarization"] = task_params.pop("diarization", False)

        logger.info(f"Creating transcription task with parameters: {task_params}")

        # *** FIX: Call create_task correctly, it returns the generated ID ***
        task_id = await tm.create_task(task_type="transcription", params=task_params) # Use injected tm and await
        # Now task_id holds the generated UUID

        # Add the generated task_id back into params *IF* the processor needs it
        # (optional, depending on processor implementation)
        # task_params["task_id"] = task_id # Add if needed by processor

        await tm.queue_task(task_id) # Use injected tm

        initial_task_info = await tm.get_task(task_id, include_result=False) # Use injected tm and await
        if not initial_task_info:
             raise HTTPException(status_code=500, detail="Failed to retrieve task status after creation.")

        response_data = TranscriptionResponse(
             id=initial_task_info["id"], # Use the ID from the task info
             status=initial_task_info["status"],
             progress=initial_task_info["progress"],
             queue_position=initial_task_info.get("queue_position"),
             model=f"whisper-{request_params.model_size.value}",
             # Initialize other fields
             error=None, transcription=None, segments=None, speakers=None,
             duration=None, processing_time=None, language=None,
        )
        return response_data

    except (ValueError, ConfigurationError, FileProcessingError) as e:
         logger.error(f"Known error type creating/queuing task: {type(e).__name__} - {e}")
         if saved_file_path and os.path.exists(saved_file_path):
             try: os.remove(saved_file_path)
             except OSError: pass
         status_code = status.HTTP_400_BAD_REQUEST if isinstance(e, (ValueError, FileProcessingError)) else status.HTTP_500_INTERNAL_SERVER_ERROR
         raise HTTPException(status_code=status_code, detail=str(e))
    except Exception as e:
        logger.exception(f"Unexpected error creating/queuing transcription task: {e}")
        if saved_file_path and os.path.exists(saved_file_path):
            try: os.remove(saved_file_path)
            except OSError: pass
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to create transcription task.")


@router_transcription.get(
    "/{task_id}/status",
    response_model=TranscriptionResponse,
    summary="Get Transcription Job Status",
    description="Check the current status and progress of a specific transcription job by its ID."
)
async def get_transcription_task_status(
    task_id: str,
    tm: TaskManager = Depends(get_task_manager)
    # _: bool = Depends(get_api_key) # API key dep handled by router
):
     """Check the current status and progress of a transcription job."""
     # No need to check tm availability, dependency handles it.
     task_info = await tm.get_task(task_id, include_result=False) # Use injected tm and await
     if not task_info:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Task ID '{task_id}' not found.")
     if task_info.get("type") != "transcription":
         raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Task ID '{task_id}' is not a transcription task.")

     result_data = task_info.get("result") or {}
     additional_info = task_info.get("additional_info") or {}

     return TranscriptionResponse(
         id=task_info["id"],
         status=task_info["status"],
         progress=task_info.get("progress"),
         queue_position=task_info.get("queue_position"),
         error=task_info.get("error"),
         model=result_data.get("model", additional_info.get("model")),
         language=result_data.get("language", additional_info.get("language")),
     )


@router_transcription.get(
    "/{task_id}",
    response_model=TranscriptionResponse,
    summary="Get Transcription Job Result",
    description="Retrieve the full results of a transcription job (transcription text, segments, speakers, etc.). Should be called when job status is 'completed'."
)
async def get_transcription_task_result(
    task_id: str,
    include_segments: bool = Query(True, description="Set to false to exclude the detailed (potentially large) segments list from the response."),
    tm: TaskManager = Depends(get_task_manager)
    # _: bool = Depends(get_api_key) # API key dep handled by router
):
    """
    Retrieve the results of a transcription job.
    """
    # No need to check tm availability, dependency handles it.
    task_info = await tm.get_task(task_id, include_result=True) # Use injected tm and await
    if not task_info:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Task ID '{task_id}' not found.")
    if task_info.get("type") != "transcription":
         raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Task ID '{task_id}' is not a transcription task.")

    result_data = task_info.get("result") or {}
    response_data = TranscriptionResponse(
        id=task_info["id"],
        status=task_info["status"],
        progress=task_info.get("progress"),
        error=task_info.get("error"),
        transcription=result_data.get("transcription"),
        speakers=result_data.get("speakers"),
        duration=result_data.get("duration"),
        processing_time=result_data.get("processing_time"),
        model=result_data.get("model"),
        language=result_data.get("language"),
        segments=result_data.get("segments") if include_segments else None,
        queue_position=task_info.get("queue_position")
    )
    return response_data


@router_transcription.delete(
    "/{task_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete Transcription Job",
    description="Delete a transcription task record and trigger cleanup of associated temporary files (if not already deleted by auto-cleanup)."
)
async def delete_transcription_task(
    task_id: str,
    tm: TaskManager = Depends(get_task_manager)
    # _: bool = Depends(get_api_key) # API key dep handled by router
):
    """
    Delete a transcription task record and associated temporary files.
    """
    # No need to check tm availability, dependency handles it.
    deleted = await tm.delete_task(task_id) # Use injected tm and await
    if not deleted:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Task ID '{task_id}' not found or already deleted.")

    logger.info(f"Deleted transcription task {task_id}")
    return None