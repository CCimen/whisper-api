"""
Speaker diarization service with forced CPU processing for FFT operations to avoid cuFFT errors.
"""

import os
import tempfile
import logging
import threading
import time
import gc
import torch
import numpy as np
from typing import List, Dict, Any, Optional

from app.config import settings
from app.exceptions import DiarizationError, ConfigurationError

# Configure logging
logger = logging.getLogger(__name__)

# Sample rate expected by pyannote
SAMPLE_RATE = 16000

# Flag to track diarization availability
DIARIZATION_AVAILABLE = False

# Conditionally import pyannote only if diarization is enabled
if settings.DIARIZATION_ENABLED:
    try:
        # Check numpy version first
        numpy_version = np.__version__
        if numpy_version.startswith("2."):
            logger.warning(
                f"NumPy {numpy_version} detected. pyannote.audio requires NumPy < 2.0. "
                "Please downgrade NumPy with: pip install numpy<2.0.0"
            )
        
        from pyannote.audio import Pipeline
        from pyannote.audio.core.io import AudioFile
        from pydub import AudioSegment
        import subprocess
        
        DIARIZATION_AVAILABLE = True
        logger.info("Successfully imported pyannote.audio for diarization")
    except ImportError as e:
        DIARIZATION_AVAILABLE = False
        logger.warning(f"Failed to import diarization dependencies: {e}. "
                     "Please install with 'pip install pyannote.audio==3.1.1 pydub==0.25.1 numpy<2.0.0'")
    except Exception as e:
        DIARIZATION_AVAILABLE = False
        logger.warning(f"Unexpected error importing diarization dependencies: {e}")


class DiarizationService:
    """
    Speaker diarization service using Pyannote with forced CPU audio processing.
    """
    # Class-level lock for thread safety
    _init_lock = threading.RLock()
    
    # Singleton pipeline instance
    _pipeline = None
    _initialized = False
    
    # If audio is longer than this, we chunk it
    CHUNK_DURATION_SEC = settings.DIARIZATION_CHUNK_DURATION

    def __init__(self):
        """Initialize the diarization service."""
        with DiarizationService._init_lock:
            if DiarizationService._initialized:
                logger.debug("DiarizationService already initialized, reusing instance")
                return
            
            logger.info("Initializing DiarizationService with CPU-forced FFT processing")
            
            if not settings.DIARIZATION_ENABLED:
                logger.info("Diarization is disabled in settings, skipping initialization")
                DiarizationService._initialized = True
                return
                
            if not DIARIZATION_AVAILABLE:
                logger.warning("Diarization dependencies not available, service will be disabled")
                DiarizationService._initialized = True
                return
            
            # Mark as initialized
            DiarizationService._initialized = True
            
            # Apply monkey patch for FFT operations
            self._monkey_patch_torch_fft()

    def _monkey_patch_torch_fft(self):
        """Apply monkey patches to ensure FFT operations run on CPU to avoid cuFFT errors."""
        if not settings.DIARIZATION_ENABLED:
            return
            
        logger.info("Applying FFT CPU forcing patches")
        
        # Save original FFT functions
        original_rfft = torch.fft.rfft
        original_irfft = torch.fft.irfft
        
        # Create CPU-forcing versions
        def cpu_rfft(input, *args, **kwargs):
            original_device = None
            if torch.is_tensor(input) and input.device.type == "cuda":
                original_device = input.device
                input = input.cpu()
            
            result = original_rfft(input, *args, **kwargs)
            
            if original_device is not None:
                # Return to original device
                result = result.to(original_device)
            
            return result
        
        def cpu_irfft(input, *args, **kwargs):
            original_device = None
            if torch.is_tensor(input) and input.device.type == "cuda":
                original_device = input.device
                input = input.cpu()
            
            result = original_irfft(input, *args, **kwargs)
            
            if original_device is not None:
                # Return to original device
                result = result.to(original_device)
            
            return result
        
        # Replace the FFT functions
        torch.fft.rfft = cpu_rfft
        torch.fft.irfft = cpu_irfft
        
        logger.info("✓ FFT operations will now run on CPU to avoid cuFFT errors")

    def load_pipeline(self):
        """Load the Pyannote pipeline once, move to GPU if available."""
        if not settings.DIARIZATION_ENABLED:
            raise ConfigurationError("Diarization is disabled in configuration")
            
        if not DIARIZATION_AVAILABLE:
            raise ConfigurationError("Diarization dependencies not available. Please install required packages.")
        
        # Use a lock to prevent multiple threads from loading simultaneously
        with DiarizationService._init_lock:
            if DiarizationService._pipeline is not None:
                return DiarizationService._pipeline
            
            hf_token = settings.HUGGINGFACE_TOKEN
            if not hf_token:
                raise ConfigurationError("No Hugging Face token found. Set HUGGINGFACE_TOKEN in .env")
            
            logger.info("Loading pyannote/speaker-diarization-3.1 pipeline")
            
            try:
                # Check CUDA setup
                if torch.cuda.is_available():
                    logger.info(f"CUDA available - version: {torch.version.cuda}")
                    device = torch.device("cuda:0")
                else:
                    logger.info("No CUDA available, using CPU only")
                    device = torch.device("cpu")
                
                # Clean memory before loading
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                # Load pipeline from Hugging Face
                pipeline = Pipeline.from_pretrained(
                    "pyannote/speaker-diarization-3.1",
                    use_auth_token=hf_token,
                )
                
                if pipeline is None:
                    raise DiarizationError("Pipeline.from_pretrained returned None")
                
                # Move to GPU if available
                if device.type == "cuda":
                    logger.info(f"Moving diarization pipeline to {device}")
                    pipeline = pipeline.to(device)
                
                # Validate pipeline is on correct device
                self._validate_pipeline_devices(pipeline)
                
                # Store pipeline
                DiarizationService._pipeline = pipeline
                logger.info("✓ Diarization pipeline loaded successfully")
                
                return pipeline
            except Exception as e:
                logger.exception(f"Failed to load diarization pipeline: {e}")
                raise DiarizationError(f"Loading diarization pipeline failed: {e}")

    def _validate_pipeline_devices(self, pipeline):
        """Ensure pipeline components are on the correct devices."""
        if not hasattr(pipeline, "_models"):
            logger.info("Pipeline doesn't have _models attribute - can't validate devices")
            return
        
        # Target device should be CUDA if available
        target_device_type = "cuda" if torch.cuda.is_available() else "cpu"
        
        for name, model in pipeline._models.items():
            device_found = False
            for param in model.parameters():
                device_found = True
                current_device = param.device
                
                if current_device.type != target_device_type:
                    logger.warning(f"Model {name} is on {current_device}, not {target_device_type}")
                    # Try to fix by moving to correct device
                    try:
                        if torch.cuda.is_available():
                            model.to(torch.device("cuda:0"))
                            logger.info(f"Moved model {name} to cuda:0")
                        else:
                            model.to(torch.device("cpu"))
                            logger.info(f"Moved model {name} to cpu")
                    except Exception as e:
                        logger.error(f"Failed to move model {name} to correct device: {e}")
                else:
                    logger.info(f"Model {name} correctly on {current_device}")
                break
            
            if not device_found:
                logger.warning(f"Could not determine device for model {name}")

    def load_audio(self, file_path: str, sr: int = SAMPLE_RATE) -> np.ndarray:
        """Load audio file using ffmpeg for better reliability."""
        try:
            # Use ffmpeg to decode audio
            cmd = [
                "ffmpeg",
                "-nostdin",
                "-threads", "0",
                "-i", file_path,
                "-f", "s16le",
                "-ac", "1",
                "-acodec", "pcm_s16le",
                "-ar", str(sr),
                "-"
            ]
            out = subprocess.run(cmd, capture_output=True, check=True).stdout
            return np.frombuffer(out, np.int16).flatten().astype(np.float32) / 32768.0
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Failed to load audio: {e.stderr.decode()}")

    def get_audio_info(self, file_path):
        """Get basic audio information without loading entire file."""
        try:
            # Use ffprobe to get duration
            cmd = [
                "ffprobe", 
                "-v", "error", 
                "-show_entries", "format=duration", 
                "-of", "default=noprint_wrappers=1:nokey=1", 
                file_path
            ]
            duration = float(subprocess.check_output(cmd).decode('utf-8').strip())
            
            # Get sample rate
            cmd = [
                "ffprobe", 
                "-v", "error", 
                "-select_streams", "a:0", 
                "-show_entries", "stream=sample_rate", 
                "-of", "default=noprint_wrappers=1:nokey=1", 
                file_path
            ]
            sample_rate = int(subprocess.check_output(cmd).decode('utf-8').strip())
            
            return {
                'duration': duration,
                'sample_rate': sample_rate
            }
        except Exception as e:
            logger.warning(f"Could not get audio info: {e}, using estimates")
            # Default estimates
            return {
                'duration': 0,  # Unknown
                'sample_rate': 16000
            }

    async def diarize_file(
        self,
        file_path: str,
        num_speakers: Optional[int] = None,
        min_speakers: Optional[int] = None,
        max_speakers: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Perform diarization on an audio file, handling chunking for large files.
        Returns a list of speaker segments.
        
        This is an async function to be compatible with asyncio.
        """
        if not settings.DIARIZATION_ENABLED:
            logger.warning("Diarization requested but disabled in config")
            return []
        
        if not DIARIZATION_AVAILABLE:
            logger.warning("Diarization dependencies not available")
            return []
        
        start_time = time.time()
        
        try:
            # Convert zero to None
            if num_speakers == 0:
                num_speakers = None
            if min_speakers == 0:
                min_speakers = None
            if max_speakers == 0:
                max_speakers = None
            
            # If no speaker constraints provided, default to 2
            if not any([num_speakers, min_speakers, max_speakers]):
                num_speakers = 2
                logger.info("No speaker constraints provided, defaulting to 2 speakers")
            
            # Load the pipeline
            pipeline = self.load_pipeline()
            
            # Prepare kwargs for diarization
            diar_kwargs = {}
            if num_speakers is not None:
                diar_kwargs["num_speakers"] = num_speakers
            else:
                if min_speakers is not None:
                    diar_kwargs["min_speakers"] = min_speakers
                if max_speakers is not None:
                    diar_kwargs["max_speakers"] = max_speakers
            
            # Get audio info
            audio_info = self.get_audio_info(file_path)
            logger.info(f"Audio duration: {audio_info['duration']:.1f}s, sample rate: {audio_info['sample_rate']}Hz")
            
            # Process based on audio length
            if audio_info['duration'] <= self.CHUNK_DURATION_SEC:
                logger.info("Processing audio in single pass")
                with torch.no_grad():
                    diarization = pipeline(file_path, **diar_kwargs)
                segments = self._extract_segments(diarization)
            else:
                logger.info(f"Processing audio in chunks of {self.CHUNK_DURATION_SEC}s")
                segments = self._process_in_chunks(pipeline, file_path, audio_info, diar_kwargs)
            
            # Normalize segment labels for consistency
            normalized_segments = self._normalize_speaker_labels(segments)
            
            processing_time = time.time() - start_time
            logger.info(f"Diarization completed in {processing_time:.2f}s for {audio_info['duration']:.1f}s audio")
            logger.info(f"Found {len(normalized_segments)} speaker segments")
            
            return normalized_segments
            
        except Exception as e:
            logger.exception(f"Diarization failed: {e}")
            raise DiarizationError(f"Diarization failed: {e}")

    async def diarize_bytes(
        self,
        audio_bytes: bytes,
        num_speakers: Optional[int] = None,
        min_speakers: Optional[int] = None,
        max_speakers: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Perform diarization on audio bytes by saving to a temporary file first.
        Returns a list of speaker segments.
        """
        if not settings.DIARIZATION_ENABLED:
            logger.warning("Diarization requested but disabled in config")
            return []
            
        if not DIARIZATION_AVAILABLE:
            logger.warning("Diarization dependencies not available")
            return []
        
        try:
            # Save bytes to a temporary file
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                temp_file.write(audio_bytes)
                temp_file.flush()
                temp_path = temp_file.name
            
            try:
                # Process the temporary file
                return await self.diarize_file(
                    temp_path,
                    num_speakers=num_speakers,
                    min_speakers=min_speakers,
                    max_speakers=max_speakers
                )
            finally:
                # Clean up the temporary file
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
        
        except Exception as e:
            logger.exception(f"Diarization failed: {e}")
            raise DiarizationError(f"Diarization failed: {e}")

    def _process_in_chunks(self, pipeline, file_path, audio_info, diar_kwargs):
        """Process audio in chunks to avoid memory issues with long files."""
        duration = audio_info['duration']
        chunk_duration = self.CHUNK_DURATION_SEC
        
        # Use overlap between chunks for better continuity
        overlap_duration = min(5.0, chunk_duration * 0.1)  # 10% overlap or 5 seconds max
        
        all_segments = []
        
        # Calculate total chunks for logging
        total_chunks = int(duration / (chunk_duration - overlap_duration)) + 1
        logger.info(f"Processing audio in {total_chunks} chunks with {overlap_duration}s overlap")
        
        # Create temporary directory for chunks
        with tempfile.TemporaryDirectory() as temp_dir:
            # Split audio into chunks
            chunk_files = self._split_audio(file_path, temp_dir, chunk_duration, overlap_duration)
            
            # Process each chunk
            for i, (chunk_file, chunk_start) in enumerate(chunk_files):
                logger.info(f"Processing chunk {i+1}/{len(chunk_files)} (starts at {chunk_start:.2f}s)")
                
                try:
                    # Process with retry logic (try max 2 times)
                    for attempt in range(2):
                        try:
                            with torch.no_grad():
                                diarization = pipeline(chunk_file, **diar_kwargs)
                            
                            # Extract and adjust segments
                            chunk_segments = self._extract_segments(diarization)
                            
                            # Adjust timestamps to account for chunk position
                            for segment in chunk_segments:
                                segment['start'] += chunk_start
                                segment['end'] += chunk_start
                            
                            all_segments.extend(chunk_segments)
                            logger.info(f"Chunk {i+1}: Found {len(chunk_segments)} segments")
                            break
                        
                        except Exception as e:
                            if attempt < 1:  # First attempt failed
                                logger.warning(f"Error in chunk {i+1}, retrying: {e}")
                                # Clean memory
                                gc.collect()
                                if torch.cuda.is_available():
                                    torch.cuda.empty_cache()
                                time.sleep(1)
                            else:
                                logger.error(f"Failed to process chunk {i+1}: {e}")
                                # Continue to next chunk instead of failing completely
                except Exception as e:
                    logger.error(f"Error processing chunk {i+1}: {e}")
                    # Continue with next chunk
        
        # If we got segments, process them
        if all_segments:
            # Sort by start time
            all_segments.sort(key=lambda x: x['start'])
            
            # Merge overlapping segments from the same speaker
            merged_segments = self._merge_adjacent_segments(all_segments)
            
            return merged_segments
        else:
            logger.warning("No segments found in any chunk")
            return []

    def _split_audio(self, file_path, output_dir, chunk_duration, overlap_duration):
        """
        Split audio file into chunks using ffmpeg.
        Returns a list of (chunk_file_path, start_time) tuples.
        """
        # Get audio duration
        cmd = [
            "ffprobe",
            "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            file_path
        ]
        duration = float(subprocess.check_output(cmd).decode('utf-8').strip())
        
        chunk_files = []
        chunk_start = 0.0
        
        while chunk_start < duration:
            # Generate output filename
            output_file = os.path.join(output_dir, f"chunk_{len(chunk_files)}.wav")
            
            # Calculate chunk end
            chunk_end = min(chunk_start + chunk_duration, duration)
            chunk_length = chunk_end - chunk_start
            
            # Extract chunk
            cmd = [
                "ffmpeg",
                "-y",  # Overwrite existing files
                "-ss", str(chunk_start),
                "-t", str(chunk_length),
                "-i", file_path,
                "-ac", "1",  # Mono
                "-ar", str(SAMPLE_RATE),  # 16kHz
                "-acodec", "pcm_s16le",  # PCM 16-bit
                output_file
            ]
            
            subprocess.run(cmd, check=True, capture_output=True)
            
            # Save chunk info
            chunk_files.append((output_file, chunk_start))
            
            # Update for next chunk (with overlap)
            chunk_start += (chunk_duration - overlap_duration)
        
        return chunk_files

    @staticmethod
    def _extract_segments(diarization):
        """Extract speaker segments from diarization result"""
        segments = []
        
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            segments.append({
                'start': round(turn.start, 2),
                'end': round(turn.end, 2),
                'speaker': speaker
            })
        
        return segments
    
    @staticmethod
    def _normalize_speaker_labels(segments):
        """Rename speaker IDs to SPEAKER_0, SPEAKER_1, etc."""
        if not segments:
            return []
            
        speaker_map = {}
        normalized = []
        
        for seg in segments:
            spk = seg["speaker"]
            if spk not in speaker_map:
                speaker_map[spk] = f"SPEAKER_{len(speaker_map)}"
            
            normalized.append({
                "start": seg["start"],
                "end": seg["end"],
                "speaker": speaker_map[spk]
            })
        
        return normalized
    
    @staticmethod
    def _merge_adjacent_segments(segments, max_gap=0.5):
        """Merge adjacent segments from the same speaker if close enough."""
        if not segments:
            return []
            
        sorted_segs = sorted(segments, key=lambda x: x["start"])
        merged = [sorted_segs[0].copy()]
        
        for seg in sorted_segs[1:]:
            prev = merged[-1]
            
            if seg["speaker"] == prev["speaker"]:
                # Merge if segments are close enough
                if seg["start"] <= prev["end"] + max_gap:
                    prev["end"] = max(prev["end"], seg["end"])
                else:
                    merged.append(seg.copy())
            else:
                merged.append(seg.copy())
        
        return merged


# Create a singleton instance
diarization_service = DiarizationService()