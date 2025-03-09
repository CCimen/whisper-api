"""
Configuration settings for the Whisper Transcription API.
"""
import os
from pydantic import BaseSettings

# Default model cache configuration
MODEL_CACHE_CONFIG = {
    "tiny": {"max_memory_gb": 1.0, "preload": True},
    "small": {"max_memory_gb": 2.0, "preload": False},
    "medium": {"max_memory_gb": 4.0, "preload": False},
    "large": {"max_memory_gb": 8.0, "preload": False}
}

# Default whisper models with mapping to HF model IDs
WHISPER_MODELS = {
    "tiny": "KBLab/kb-whisper-tiny", 
    "small": "KBLab/kb-whisper-small",
    "medium": "KBLab/kb-whisper-medium",
    "large": "KBLab/kb-whisper-large"
}

class Settings(BaseSettings):
    """Application settings."""
    
    # API settings
    APP_NAME: str = "Whisper Transcription API"
    DEBUG: bool = bool(os.getenv("DEBUG", "False").lower() in ("true", "1", "t"))
    
    # Whisper model settings
    DEFAULT_MODEL: str = os.getenv("DEFAULT_MODEL", "medium")
    DEFAULT_LANGUAGE: str = os.getenv("DEFAULT_LANGUAGE", "sv")
    MODELS_CACHE_DIR: str = os.getenv("MODELS_CACHE_DIR", "./models")
    KEEP_MULTIPLE_MODELS_IN_MEMORY: bool = bool(os.getenv("KEEP_MULTIPLE_MODELS_IN_MEMORY", "False").lower() in ("true", "1", "t"))
    MAX_MODELS_IN_MEMORY: int = int(os.getenv("MAX_MODELS_IN_MEMORY", "1"))
    MIN_FREE_MEMORY_GB: float = float(os.getenv("MIN_FREE_MEMORY_GB", "3.0"))
    PRELOAD_DEFAULT_MODEL: bool = bool(os.getenv("PRELOAD_DEFAULT_MODEL", "True").lower() in ("true", "1", "t"))
    
    # Diarization settings
    DIARIZATION_ENABLED: bool = bool(os.getenv("DIARIZATION_ENABLED", "False").lower() in ("true", "1", "t"))
    HUGGINGFACE_TOKEN: str = os.getenv("HUGGINGFACE_TOKEN", "")
    PARALLEL_PROCESSING: bool = bool(os.getenv("PARALLEL_PROCESSING", "False").lower() in ("true", "1", "t"))
    DIARIZATION_CHUNK_DURATION: int = int(os.getenv("DIARIZATION_CHUNK_DURATION", "300"))
    
    # Job settings
    JOB_CLEANUP_HOURS: int = int(os.getenv("JOB_CLEANUP_HOURS", "24"))
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

# Create the settings instance
settings = Settings()