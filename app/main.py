import os
import uuid
import shutil
import tempfile
import time
import logging
import asyncio
from typing import List, Dict, Optional
from enum import Enum

import torch
from fastapi import FastAPI, UploadFile, File, HTTPException, Query, BackgroundTasks
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

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

# Models for API
class ModelSize(str, Enum):
    tiny = "tiny"
    small = "small"
    medium = "medium"
    large = "large"

class TimestampedSegment(BaseModel):
    start: float
    end: float
    text: str
    speaker: Optional[str] = None

class TranscriptionResult(BaseModel):
    id: str
    status: str
    transcription: Optional[str] = None
    segments: Optional[List[TimestampedSegment]] = None
    speakers: Optional[List[str]] = None
    duration: Optional[float] = None
    processing_time: Optional[float] = None
    error: Optional[str] = None

# In-memory storage for transcription jobs
transcription_jobs = {}

# Create FastAPI app
app = FastAPI(
    title="Whisper Transcription API",
    description="API for transcribing audio files using KB-Whisper models with optional speaker diarization",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins in development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    """Check if the API is running"""
    return {"message": "Whisper Transcription API is running"}

@app.get("/status")
async def get_status():
    """Get system status including GPU and diarization availability"""
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

@app.get("/gpu-status")
async def gpu_status():
    """Get information about available GPUs"""
    return check_gpu()

@app.post("/transcriptions", response_model=dict)
async def create_transcription(
    audio_file: UploadFile = File(...),
    language: str = Query(settings.DEFAULT_LANGUAGE, description="Language code (e.g., 'sv' for Swedish)"),
    model_size: ModelSize = Query(settings.DEFAULT_MODEL, description="Model size to use for transcription"),
    diarization: bool = Query(False, description="Enable speaker diarization (requires Hugging Face token)"),
    num_speakers: Optional[int] = Query(None, description="Fixed number of speakers (overrides min/max)"),
    min_speakers: Optional[int] = Query(None, description="Minimum number of speakers"),
    max_speakers: Optional[int] = Query(None, description="Maximum number of speakers")
):
    """Start a new transcription job with optional speaker diarization"""
    # Check GPU availability
    gpu_info = check_gpu()
    if not gpu_info["available"]:
        raise HTTPException(status_code=503, detail="GPU not available. This service requires CUDA.")
    
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
            "status": "processing",
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
        
        # Return job ID to client
        return {"id": job_id, "status": "processing"}
    
    except Exception as e:
        # Clean up on error
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        logger.exception(f"Error creating transcription job: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/transcriptions/{job_id}/status")
async def get_transcription_status(job_id: str):
    """Check the status of a transcription job"""
    if job_id not in transcription_jobs:
        raise HTTPException(status_code=404, detail=f"Transcription job {job_id} not found")
    
    job = transcription_jobs[job_id]
    return {"id": job_id, "status": job["status"]}

@app.get("/transcriptions/{job_id}", response_model=TranscriptionResult)
async def get_transcription_result(job_id: str):
    """Get the result of a completed transcription job"""
    if job_id not in transcription_jobs:
        raise HTTPException(status_code=404, detail=f"Transcription job {job_id} not found")
    
    job = transcription_jobs[job_id]
    
    if job["status"] == "processing":
        return {"id": job_id, "status": "processing"}
    
    if job["status"] == "error":
        return {"id": job_id, "status": "error", "error": job.get("error", "Unknown error")}
    
    return {
        "id": job_id,
        "status": job["status"],
        "transcription": job.get("transcription"),
        "segments": job.get("segments"),
        "speakers": job.get("speakers", []),
        "duration": job.get("duration"),
        "processing_time": job.get("processing_time")
    }

async def process_audio_job(job_id: str):
    """Process a transcription/diarization job in the background"""
    job = transcription_jobs[job_id]
    file_path = job["file_path"]
    
    try:
        logger.info(f"Processing job {job_id}")
        
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
            max_speakers=job.get("max_speakers")
        )
        
        # Update job with results
        job.update({
            "status": "completed",
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
        job["status"] = "error"
        job["error"] = str(e)
    
    except Exception as e:
        # Handle unexpected errors
        logger.exception(f"Job {job_id} failed with unexpected error: {e}")
        job["status"] = "error"
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
async def setup_periodic_cleanup():
    asyncio.create_task(periodic_cleanup())

async def periodic_cleanup():
    while True:
        await asyncio.sleep(3600)  # Wait for 1 hour
        cleanup_old_jobs()

def cleanup_old_jobs(max_age_hours=settings.JOB_CLEANUP_HOURS):
    """Clean up transcription jobs older than the specified age"""
    current_time = time.time()
    jobs_to_remove = []
    
    for job_id, job in transcription_jobs.items():
        # If job is older than max_age_hours, mark for removal
        if current_time - job["start_time"] > max_age_hours * 3600:
            # Make sure the job is not still processing
            if job["status"] != "processing":
                jobs_to_remove.append(job_id)
    
    # Remove marked jobs
    for job_id in jobs_to_remove:
        del transcription_jobs[job_id]
    
    if jobs_to_remove:
        logger.info(f"Cleaned up {len(jobs_to_remove)} old transcription jobs")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)