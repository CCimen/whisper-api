"""
Combined processor for audio transcription and diarization with performance optimizations.

This module provides functionality to process audio files with transcription and optional 
speaker diarization. It includes performance enhancements for GPU utilization, 
parallel processing, and progress tracking.
"""

import logging
import asyncio
import time
from typing import Dict, Any, Optional, List, Callable

from app.transcriber import transcribe_audio, estimate_audio_duration
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
    max_speakers: Optional[int] = None,
    progress_callback: Optional[Callable[[str, float], None]] = None
) -> Dict[str, Any]:
    """
    Process audio file with transcription and optional diarization.
    
    Args:
        file_path: Path to the audio file
        language: Language code for transcription
        model_size: Size of the whisper model to use
        enable_diarization: Whether to perform speaker diarization
        num_speakers: Exact number of speakers (if known)
        min_speakers: Minimum number of speakers
        max_speakers: Maximum number of speakers
        progress_callback: Function to call with status updates
        
    Returns:
        Dictionary containing transcription and diarization results
    """
    start_time = time.time()
    
    # Track tasks and results
    tasks = {}
    results = {}
    
    # Estimate full duration for progress reporting
    try:
        audio_duration = await estimate_audio_duration(file_path)
    except Exception as e:
        logger.warning(f"Could not estimate audio duration: {e}")
        audio_duration = None
    
    # Check if diarization is actually available when requested
    do_diarization = enable_diarization and settings.DIARIZATION_ENABLED and DIARIZATION_AVAILABLE
    if enable_diarization and not do_diarization:
        if not settings.DIARIZATION_ENABLED:
            logger.warning("Diarization requested but disabled in settings")
        elif not DIARIZATION_AVAILABLE:
            logger.warning("Diarization requested but dependencies not available")
    
    # Update initial status
    if progress_callback:
        progress_callback("transcribing", 0.0)
    
    # Flag to determine if we should run in parallel
    run_parallel = settings.PARALLEL_PROCESSING and do_diarization
    
    # Start transcription
    logger.info("Starting transcription")
    transcription_progress_callback = None
    if progress_callback:
        # Create a transcription progress callback that reports overall progress
        diarization_weight = 0.4 if do_diarization else 0.0
        transcription_weight = 1.0 - diarization_weight
        
        def transcription_progress_callback(progress: float):
            progress_callback("transcribing", progress * transcription_weight)
    
    if run_parallel:
        # Run as asyncio task
        tasks["transcription"] = asyncio.create_task(
            transcribe_audio(
                file_path, 
                language, 
                model_size,
                progress_callback=transcription_progress_callback
            )
        )
    else:
        # Run synchronously
        results["transcription"] = await transcribe_audio(
            file_path, 
            language, 
            model_size,
            progress_callback=transcription_progress_callback
        )
        
        # If we're not doing diarization, we're done
        if progress_callback and not do_diarization:
            progress_callback("completing", 0.9)
    
    # Start diarization if enabled and available
    if do_diarization:
        logger.info("Starting diarization")
        if progress_callback:
            progress_callback("diarizing", transcription_weight)
            
        diarization_progress_callback = None
        if progress_callback:
            def diarization_progress_callback(progress: float):
                overall_progress = transcription_weight + (progress * diarization_weight)
                progress_callback("diarizing", overall_progress)
                
        try:
            if run_parallel:
                # Run as asyncio task
                tasks["diarization"] = asyncio.create_task(
                    diarization_service.diarize_file(
                        file_path,
                        num_speakers=num_speakers,
                        min_speakers=min_speakers,
                        max_speakers=max_speakers,
                        progress_callback=diarization_progress_callback
                    )
                )
            else:
                # Run synchronously
                results["diarization"] = await diarization_service.diarize_file(
                    file_path,
                    num_speakers=num_speakers,
                    min_speakers=min_speakers,
                    max_speakers=max_speakers,
                    progress_callback=diarization_progress_callback
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
    
    # Final progress update before completing
    if progress_callback:
        progress_callback("completing", 0.95)
    
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
    
    # Final progress update
    if progress_callback:
        progress_callback("completed", 1.0)
    
    logger.info(f"Audio processing completed in {result['processing_time']:.2f}s")
    return result

def combine_transcription_and_diarization(
    transcription_segments: List[Dict[str, Any]],
    diarization_segments: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Combine transcription segments with speaker information from diarization.
    
    This algorithm assigns a speaker to each transcription segment
    based on the maximum overlap with speaker segments.
    
    Args:
        transcription_segments: List of segments from transcription with start/end times
        diarization_segments: List of segments from diarization with speaker labels
        
    Returns:
        List of combined segments with speaker information
    """
    if not diarization_segments:
        return transcription_segments
    
    # Create a copy of transcription segments
    combined_segments = []
    
    # For faster lookups, create an index of diarization segments
    # Based on start times, sorted by time
    diarization_index = sorted(diarization_segments, key=lambda x: x["start"])
    
    for trans_seg in transcription_segments:
        trans_start = trans_seg["start"]
        trans_end = trans_seg["end"]
        
        # Find overlapping speaker segments 
        # Use binary search to find closest starting segment
        overlaps = []
        
        # Find segments that might overlap with the transcription segment
        relevant_segments = [
            seg for seg in diarization_index 
            if (seg["start"] <= trans_end and seg["end"] >= trans_start)
        ]
        
        for spk_seg in relevant_segments:
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