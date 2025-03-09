"""
Combined processor for audio transcription and diarization.
"""

import logging
import asyncio
import time
from typing import Dict, Any, Optional, List

from app.transcriber import transcribe_audio
from app.diarization import diarization_service, DIARIZATION_AVAILABLE
from app.config import settings

logger = logging.getLogger(__name__)

async def process_audio(
    file_path: str,
    language: str = "sv",
    model_size: str = "medium",
    enable_diarization: bool = False,
    num_speakers: Optional[int] = None,
    min_speakers: Optional[int] = None,
    max_speakers: Optional[int] = None
) -> Dict[str, Any]:
    """
    Process audio file with transcription and optional diarization.
    Can run in parallel if configured.
    """
    start_time = time.time()
    
    # Track tasks and results
    tasks = {}
    results = {}
    
    # Check if diarization is actually available when requested
    do_diarization = enable_diarization and settings.DIARIZATION_ENABLED and DIARIZATION_AVAILABLE
    if enable_diarization and not do_diarization:
        if not settings.DIARIZATION_ENABLED:
            logger.warning("Diarization requested but disabled in settings")
        elif not DIARIZATION_AVAILABLE:
            logger.warning("Diarization requested but dependencies not available")
    
    # Flag to determine if we should run in parallel
    run_parallel = settings.PARALLEL_PROCESSING and do_diarization
    
    # Start transcription
    logger.info("Starting transcription")
    if run_parallel:
        # Run as asyncio task
        tasks["transcription"] = asyncio.create_task(
            transcribe_audio(file_path, language, model_size)
        )
    else:
        # Run synchronously
        results["transcription"] = await transcribe_audio(file_path, language, model_size)
    
    # Start diarization if enabled and available
    if do_diarization:
        logger.info("Starting diarization")
        try:
            if run_parallel:
                # Run as asyncio task
                tasks["diarization"] = asyncio.create_task(
                    diarization_service.diarize_file(
                        file_path,
                        num_speakers=num_speakers,
                        min_speakers=min_speakers,
                        max_speakers=max_speakers
                    )
                )
            else:
                # Run synchronously
                results["diarization"] = await diarization_service.diarize_file(
                    file_path,
                    num_speakers=num_speakers,
                    min_speakers=min_speakers,
                    max_speakers=max_speakers
                )
        except Exception as e:
            logger.error(f"Diarization failed: {e}")
            results["diarization"] = []
    
    # Wait for parallel tasks to complete
    if run_parallel:
        for name, task in tasks.items():
            try:
                results[name] = await task
                logger.info(f"{name.capitalize()} completed")
            except Exception as e:
                logger.error(f"{name.capitalize()} task failed: {e}")
                results[name] = {"error": str(e)} if name == "transcription" else []
    
    # Get transcription results
    transcription = results.get("transcription", {})
    
    # Base result structure
    result = {
        "transcription": transcription.get("text", ""),
        "segments": transcription.get("segments", []),
        "duration": transcription.get("duration", 0),
        "processing_time": time.time() - start_time
    }
    
    # If diarization was performed and successful, combine results
    if do_diarization:
        diarization = results.get("diarization", [])
        if diarization and isinstance(diarization, list):  # Make sure it's a valid list
            # Add speakers to segments
            segments_with_speakers = combine_transcription_and_diarization(
                result["segments"], 
                diarization
            )
            result["segments"] = segments_with_speakers
            result["speakers"] = list(set(seg["speaker"] for seg in diarization))
    
    logger.info(f"Audio processing completed in {result['processing_time']:.2f}s")
    return result

def combine_transcription_and_diarization(
    transcription_segments: List[Dict[str, Any]],
    diarization_segments: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Combine transcription segments with speaker information from diarization.
    
    This is a simple algorithm that assigns a speaker to each transcription segment
    based on the maximum overlap with speaker segments.
    """
    if not diarization_segments:
        return transcription_segments
    
    # Create a copy of transcription segments
    combined_segments = []
    
    for trans_seg in transcription_segments:
        trans_start = trans_seg["start"]
        trans_end = trans_seg["end"]
        
        # Find overlapping speaker segments
        overlaps = []
        
        for spk_seg in diarization_segments:
            spk_start = spk_seg["start"]
            spk_end = spk_seg["end"]
            
            # Calculate overlap
            overlap_start = max(trans_start, spk_start)
            overlap_end = min(trans_end, spk_end)
            
            # If there is overlap
            if overlap_end > overlap_start:
                overlap_duration = overlap_end - overlap_start
                overlaps.append({
                    "speaker": spk_seg["speaker"],
                    "duration": overlap_duration
                })
        
        # Sort overlaps by duration
        overlaps.sort(key=lambda x: x["duration"], reverse=True)
        
        # Create new segment with assigned speaker
        new_segment = trans_seg.copy()
        
        # If we found overlaps, assign the speaker with most overlap
        if overlaps:
            new_segment["speaker"] = overlaps[0]["speaker"]
        else:
            # No overlap found, assign unknown speaker
            new_segment["speaker"] = "UNKNOWN"
        
        combined_segments.append(new_segment)
    
    return combined_segments