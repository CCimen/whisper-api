"""
Main FastAPI application for the Whisper Transcription API.

This module provides the API endpoints for transcription and diarization
with optimized performance, resource management, and production-ready features.
"""

import os
import uuid
import shutil
import tempfile
import time
import logging
import asyncio
from typing import List, Dict, Optional, Any
from enum import Enum

import torch
from fastapi import FastAPI, UploadFile, File, HTTPException, Query, BackgroundTasks, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.openapi.utils import get_openapi
from fastapi.openapi.docs import get_swagger_ui_html
from pydantic import BaseModel, Field

from app.config import settings
from app.transcriber import check_gpu, model_manager
from app.processor import process_audio
from app.diarization import DIARIZATION_AVAILABLE
from app.exceptions import TranscriptionError, DiarizationError

# Configure logging
logging.basicConfig(
    level=logging.INFO if not settings.DEBUG else logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Status tracking with more detailed states
class ProcessingStatus(str, Enum):
    """Detailed processing status for better tracking."""
    PENDING = "pending"
    TRANSCRIBING = "transcribing"
    DIARIZING = "diarizing"
    COMPLETING = "completing"
    COMPLETED = "completed"
    ERROR = "error"

# API Models
class ModelSize(str, Enum):
    """Available model sizes for transcription."""
    tiny = "tiny"
    small = "small"
    medium = "medium"
    large = "large"

class TimestampedSegment(BaseModel):
    """Segment of transcription with timing and optional speaker."""
    start: float = Field(..., description="Start time of segment in seconds")
    end: float = Field(..., description="End time of segment in seconds")
    text: str = Field(..., description="Transcribed text for this segment")
    speaker: Optional[str] = Field(None, description="Speaker identifier (if diarization enabled)")

class TranscriptionResult(BaseModel):
    """Complete transcription result model."""
    id: str = Field(..., description="Unique identifier for this transcription job")
    status: str = Field(..., description="Current processing status")
    progress: float = Field(0.0, description="Processing progress from 0.0 to 1.0")
    transcription: Optional[str] = Field(None, description="Full transcription text")
    segments: Optional[List[TimestampedSegment]] = Field(None, description="Timestamped segments")
    speakers: Optional[List[str]] = Field(None, description="List of identified speakers (if diarization enabled)")
    duration: Optional[float] = Field(None, description="Audio duration in seconds")
    processing_time: Optional[float] = Field(None, description="Time taken to process in seconds")
    error: Optional[str] = Field(None, description="Error message if processing failed")

class APIStatus(BaseModel):
    """API status and capability information."""
    status: str = Field(..., description="API operational status")
    version: str = Field(..., description="API version")
    gpu: Dict[str, Any] = Field(..., description="GPU information and status")
    diarization: Dict[str, Any] = Field(..., description="Diarization capability status")
    default_model: str = Field(..., description="Default transcription model")
    models_in_memory: List[str] = Field(..., description="Currently loaded models")

# In-memory storage for transcription jobs
transcription_jobs = {}

# Create FastAPI app
app = FastAPI(
    title="Whisper Transcription API",
    description="API for transcribing audio files using KB-Whisper models with optional speaker diarization",
    version="1.0.0",
    docs_url=None,  # Custom docs URL below
    redoc_url=None  # Disable ReDoc
)

# Add performance middleware
app.add_middleware(GZipMiddleware, minimum_size=1000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For production, specify exact domains
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
    max_age=86400,  # Cache preflight requests for 24 hours
)

# Custom OpenAPI documentation
def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    
    openapi_schema = get_openapi(
        title="Whisper Transcription API",
        version="1.0.0",
        description="""
        # Audio Transcription API with Speaker Identification
        
        This API provides high-performance audio transcription with optional speaker identification
        (diarization). It uses Whisper models with GPU acceleration for processing.
        
        ## Key Features:
        - Transcribe audio files with timestamps
        - Identify speakers in audio (optional diarization)
        - Multiple model sizes for different needs
        - Asynchronous processing for large files
        - Real-time status updates
        
        ## Integration Guide
        For frontend integration, use the following endpoint flow:
        1. Submit audio with `POST /transcriptions`
        2. Poll status with `GET /transcriptions/{job_id}/status`
        3. Retrieve results with `GET /transcriptions/{job_id}`
        
        ## Performance Considerations
        - GPU processing gives ~5-10x realtime performance
        - Larger models are more accurate but slower
        - Diarization adds processing time but enables speaker tracking
        """,
        routes=app.routes,
    )
    
    # Add custom components and schemas
    app.openapi_schema = openapi_schema
    return app.openapi_schema

@app.get("/docs", include_in_schema=False)
async def custom_swagger_ui_html():
    """Custom Swagger UI with additional metadata."""
    return get_swagger_ui_html(
        openapi_url="/openapi.json",
        title="Whisper Transcription API",
        swagger_js_url="https://cdn.jsdelivr.net/npm/swagger-ui-dist@4/swagger-ui-bundle.js",
        swagger_css_url="https://cdn.jsdelivr.net/npm/swagger-ui-dist@4/swagger-ui.css",
    )

app.openapi = custom_openapi

@app.get("/", tags=["Health"])
async def root():
    """Basic health check endpoint."""
    return {"message": "Whisper Transcription API is running"}

@app.get("/status", response_model=APIStatus, tags=["System"])
async def get_status():
    """
    Get detailed system status including GPU and diarization availability.
    
    Returns information about the API's capabilities, loaded models,
    and hardware status for monitoring and diagnostics.
    """
    gpu_info = check_gpu()
    
    # Add diarization status
    diarization_status = {
        "enabled": settings.DIARIZATION_ENABLED,
        "available": DIARIZATION_AVAILABLE,
        "huggingface_token_set": bool(settings.HUGGINGFACE_TOKEN)
    }
    
    return {
        "status": "ok",
        "version": "1.0.0",
        "gpu": gpu_info,
        "diarization": diarization_status,
        "default_model": settings.DEFAULT_MODEL,
        "models_in_memory": list(model_manager._model_cache.keys()) if hasattr(model_manager, "_model_cache") else [],
    }

@app.get("/gpu-status", tags=["System"])
async def gpu_status():
    """
    Get detailed information about available GPUs.
    
    Returns comprehensive GPU information including memory usage,
    capabilities, and current status.
    """
    return check_gpu()

@app.post("/transcriptions", response_model=Dict[str, Any], tags=["Transcription"])
async def create_transcription(
    audio_file: UploadFile = File(..., description="Audio file to transcribe (MP3, WAV, etc.)"),
    language: str = Query(settings.DEFAULT_LANGUAGE, description="Language code (e.g., 'sv' for Swedish)"),
    model_size: ModelSize = Query(ModelSize(settings.DEFAULT_MODEL), description="Model size to use for transcription"),
    diarization: bool = Query(False, description="Enable speaker diarization (requires Hugging Face token)"),
    num_speakers: Optional[int] = Query(None, description="Fixed number of speakers (overrides min/max)"),
    min_speakers: Optional[int] = Query(None, description="Minimum number of speakers"),
    max_speakers: Optional[int] = Query(None, description="Maximum number of speakers")
):
    """
    Start a new transcription job with optional speaker diarization.
    
    This endpoint accepts an audio file and starts an asynchronous transcription process.
    It returns a job ID that can be used to check status and retrieve results.
    
    If diarization is enabled, speakers will be identified in the output segments.
    """
    # Check GPU availability
    gpu_info = check_gpu()
    if not gpu_info["available"]:
        raise HTTPException(status_code=503, detail="GPU not available. This service requires CUDA.")
    
    # Check file size
    max_file_size_mb = 1000  # 1GB file size limit
    if audio_file.size > max_file_size_mb * 1024 * 1024:
        raise HTTPException(
            status_code=413, 
            detail=f"File too large. Maximum allowed size is {max_file_size_mb}MB."
        )
    
    # Check content type
    valid_audio_types = [
        'audio/mpeg', 'audio/mp3', 'audio/wav', 'audio/wave', 'audio/x-wav',
        'audio/ogg', 'audio/flac', 'audio/x-flac'
    ]
    if audio_file.content_type not in valid_audio_types and not audio_file.filename.endswith(
        ('.mp3', '.wav', '.ogg', '.flac', '.m4a', '.aac')
    ):
        logger.warning(f"Suspicious content type: {audio_file.content_type}, filename: {audio_file.filename}")
        # Allow but warn - content type is sometimes incorrect in uploads
    
    # Check if diarization is enabled but not configured
    if diarization and not settings.DIARIZATION_ENABLED:
        raise HTTPException(
            status_code=400, 
            detail="Diarization is not enabled in server configuration. Set DIARIZATION_ENABLED=True in .env"
        )
        
    # Check if diarization is requested but dependencies not available
    if diarization and settings.DIARIZATION_ENABLED and not DIARIZATION_AVAILABLE:
        logger.warning("Diarization requested but dependencies not available")
        # We'll still process the request, but without diarization
        diarization = False
    
    # Generate unique ID for this transcription
    job_id = str(uuid.uuid4())
    
    # Save uploaded file to a temporary location
    temp_dir = tempfile.mkdtemp()
    file_path = os.path.join(temp_dir, f"{job_id}_{audio_file.filename}")
    
    try:
        # Log job creation
        logger.info(f"Creating job {job_id} for file {audio_file.filename}")
        
        # Save the uploaded file
        with open(file_path, "wb") as f:
            shutil.copyfileobj(audio_file.file, f)
        
        # Initialize job in our storage
        transcription_jobs[job_id] = {
            "id": job_id,
            "status": ProcessingStatus.PENDING,
            "progress": 0.0,
            "file_path": file_path,
            "language": language,
            "model_size": model_size.value,
            "diarization": diarization,
            "num_speakers": num_speakers,
            "min_speakers": min_speakers,
            "max_speakers": max_speakers,
            "start_time": time.time()
        }
        
        # Start processing in background
        asyncio.create_task(process_audio_job(job_id))
        
        # Return job ID to client with initial status
        return {
            "id": job_id, 
            "status": ProcessingStatus.PENDING,
            "progress": 0.0
        }
    
    except Exception as e:
        # Clean up on error
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        logger.exception(f"Error creating transcription job: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/transcriptions/{job_id}/status", response_model=Dict[str, Any], tags=["Transcription"])
async def get_transcription_status(job_id: str):
    """
    Check the status of a transcription job.
    
    Returns the current status and progress of the transcription job.
    This endpoint can be polled regularly to track job progress.
    """
    if job_id not in transcription_jobs:
        raise HTTPException(status_code=404, detail=f"Transcription job {job_id} not found")
    
    # Get a copy of the current job status to avoid blocking
    job = transcription_jobs[job_id].copy()
    
    return {
        "id": job_id, 
        "status": job["status"],
        "progress": job.get("progress", 0.0),
        "error": job.get("error") if job["status"] == ProcessingStatus.ERROR else None
    }

@app.get("/transcriptions/{job_id}", response_model=TranscriptionResult, tags=["Transcription"])
async def get_transcription_result(
    job_id: str,
    include_segments: bool = Query(True, description="Include timestamped segments in the response")
):
    """
    Get the result of a completed transcription job.
    
    Returns the full transcription result including text, segments, and
    speaker information if diarization was enabled. This endpoint should be
    called after the job status is 'completed'.
    
    For bandwidth efficiency with large transcriptions, you can set
    include_segments=false to get only the full transcription text.
    """
    if job_id not in transcription_jobs:
        raise HTTPException(status_code=404, detail=f"Transcription job {job_id} not found")
    
    job = transcription_jobs[job_id]
    
    # Include segments based on parameter
    segments = job.get("segments", []) if include_segments else []
    
    return {
        "id": job_id,
        "status": job["status"],
        "progress": job.get("progress", 1.0) if job["status"] == ProcessingStatus.COMPLETED else job.get("progress", 0.0),
        "transcription": job.get("transcription"),
        "segments": segments,
        "speakers": job.get("speakers", []),
        "duration": job.get("duration"),
        "processing_time": job.get("processing_time"),
        "error": job.get("error") if job["status"] == ProcessingStatus.ERROR else None
    }

@app.delete("/transcriptions/{job_id}", status_code=204, tags=["Transcription"])
async def delete_transcription(job_id: str):
    """
    Delete a transcription job and its resources.
    
    This endpoint allows clients to clean up a job after they're done with it,
    freeing up server resources. It works regardless of job status.
    """
    if job_id not in transcription_jobs:
        raise HTTPException(status_code=404, detail=f"Transcription job {job_id} not found")
    
    job = transcription_jobs[job_id]
    
    # Clean up any files if they still exist
    file_path = job.get("file_path")
    if file_path and os.path.exists(file_path):
        try:
            os.remove(file_path)
            dir_path = os.path.dirname(file_path)
            if os.path.exists(dir_path) and not os.listdir(dir_path):
                os.rmdir(dir_path)
        except Exception as e:
            logger.warning(f"Error cleaning up files for job {job_id}: {e}")
    
    # Remove the job from storage
    del transcription_jobs[job_id]
    return None  # 204 No Content

async def process_audio_job(job_id: str):
    """
    Process a transcription/diarization job in the background with progress tracking.
    """
    job = transcription_jobs[job_id]
    file_path = job["file_path"]
    
    # Update status function
    def update_job_status(status, progress):
        if job_id in transcription_jobs:
            job["status"] = status
            job["progress"] = progress
    
    try:
        logger.info(f"Processing job {job_id}")
        update_job_status(ProcessingStatus.PENDING, 0.0)
        
        # Make sure the file exists
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Input file {file_path} not found")
        
        # Process audio with transcription and optional diarization
        result = await process_audio(
            file_path=file_path,
            language=job["language"],
            model_size=job["model_size"],
            enable_diarization=job.get("diarization", False),
            num_speakers=job.get("num_speakers"),
            min_speakers=job.get("min_speakers"),
            max_speakers=job.get("max_speakers"),
            progress_callback=update_job_status
        )
        
        # Update job with results
        update_job_status(ProcessingStatus.COMPLETED, 1.0)
        job.update({
            "transcription": result["transcription"],
            "segments": result["segments"],
            "speakers": result.get("speakers", []),
            "duration": result["duration"],
            "processing_time": result["processing_time"]
        })
        
        logger.info(f"Job {job_id} completed successfully")
        
    except (TranscriptionError, DiarizationError) as e:
        # Handle specific errors
        logger.error(f"Job {job_id} failed with {type(e).__name__}: {e}")
        update_job_status(ProcessingStatus.ERROR, 0.0)
        job["error"] = str(e)
    
    except Exception as e:
        # Handle unexpected errors
        logger.exception(f"Job {job_id} failed with unexpected error: {e}")
        update_job_status(ProcessingStatus.ERROR, 0.0)
        job["error"] = f"Unexpected error: {str(e)}"
    
    finally:
        # Clean up temporary file ONLY after processing is complete
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                # Only try to remove directory if it exists
                dir_path = os.path.dirname(file_path)
                if os.path.exists(dir_path) and not os.listdir(dir_path):
                    os.rmdir(dir_path)
                logger.debug(f"Cleaned up temporary files for job {job_id}")
        except Exception as e:
            logger.error(f"Error cleaning up files for job {job_id}: {e}")

# Cleanup old jobs periodically (runs every hour)
@app.on_event("startup")
async def setup_periodic_tasks():
    # Start the cleanup task
    asyncio.create_task(periodic_cleanup())
    
    # Pre-load the default model if configured
    if settings.PRELOAD_DEFAULT_MODEL:
        asyncio.create_task(preload_default_model())

async def preload_default_model():
    """Pre-load the default model at startup."""
    try:
        logger.info(f"Pre-loading default model: {settings.DEFAULT_MODEL}")
        # This will trigger model loading through the manager
        _ = model_manager.get_pipeline(settings.DEFAULT_MODEL)
        logger.info(f"Default model pre-loaded successfully")
    except Exception as e:
        logger.error(f"Failed to pre-load default model: {e}")

async def periodic_cleanup():
    """Run periodic cleanup tasks."""
    while True:
        await asyncio.sleep(3600)  # Wait for 1 hour
        try:
            cleanup_old_jobs()
            # Also run memory cleanup
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as e:
            logger.error(f"Error in periodic cleanup: {e}")

def cleanup_old_jobs(max_age_hours=settings.JOB_CLEANUP_HOURS):
    """Clean up transcription jobs older than the specified age."""
    current_time = time.time()
    jobs_to_remove = []
    
    for job_id, job in transcription_jobs.items():
        # If job is older than max_age_hours, mark for removal
        if current_time - job["start_time"] > max_age_hours * 3600:
            # Make sure the job is not still processing
            if job["status"] not in [ProcessingStatus.PENDING, ProcessingStatus.TRANSCRIBING, ProcessingStatus.DIARIZING]:
                jobs_to_remove.append(job_id)
    
    # Remove marked jobs
    for job_id in jobs_to_remove:
        # Clean up any remaining files
        job = transcription_jobs[job_id]
        file_path = job.get("file_path")
        if file_path and os.path.exists(file_path):
            try:
                os.remove(file_path)
                dir_path = os.path.dirname(file_path)
                if os.path.exists(dir_path) and not os.listdir(dir_path):
                    os.rmdir(dir_path)
            except Exception:
                pass
                
        # Remove from cache
        del transcription_jobs[job_id]
    
    if jobs_to_remove:
        logger.info(f"Cleaned up {len(jobs_to_remove)} old transcription jobs")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app", 
        host="0.0.0.0", 
        port=8000,
        reload=settings.DEBUG,
        workers=1  # Multiple workers can cause issues with GPU memory
    )