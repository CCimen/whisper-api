"""
Configuration settings for the Whisper Transcription API with privacy enhancements.

This module provides a centralized configuration system with environment variable
support, sensible defaults, and privacy-focused options. Uses pydantic-settings.
"""
import os
import sys
import logging
import secrets
import base64
from typing import Dict, Any, Optional, List, Union

import torch
from pydantic import validator, Field, AnyHttpUrl
from pydantic_settings import BaseSettings, SettingsConfigDict

# Configure logging
logger = logging.getLogger(__name__)

# --- Constants ---
KB_WHISPER_TINY_ID = "KBLab/kb-whisper-tiny"
KB_WHISPER_SMALL_ID = "KBLab/kb-whisper-small"
KB_WHISPER_MEDIUM_ID = "KBLab/kb-whisper-medium"
KB_WHISPER_LARGE_ID = "KBLab/kb-whisper-large" # Use kb-whisper-large as requested

# Define Whisper model mappings (size -> Hugging Face ID)
# Include the KB-Whisper models you want to support
WHISPER_MODEL_MAPPING = {
    # General Purpose / English Optimized
    "openai-large-v3-turbo": "openai/whisper-large-v3-turbo",

    # KBLab Swedish Optimized Models
    "kblab-tiny": KB_WHISPER_TINY_ID,
    "kblab-small": KB_WHISPER_SMALL_ID,
    "kblab-medium": KB_WHISPER_MEDIUM_ID,
    "kblab-large": KB_WHISPER_LARGE_ID,
}


# --- Helper Functions ---

def detect_system_capabilities() -> Dict[str, Any]:
    """Detect system capabilities for optimal default settings."""
    capabilities = {
        "cuda_available": False,
        "device_count": 0,
        "total_memory_gb": 0,
        "free_memory_gb": 0,
        "compute_capabilities": [],
        "cpu_count": os.cpu_count() or 4,
    }
    try:
        if torch and torch.cuda.is_available(): # Check torch exists
            capabilities["cuda_available"] = True
            capabilities["device_count"] = torch.cuda.device_count()

            total_mem = 0
            for i in range(capabilities["device_count"]):
                device_info = {"index": i}
                props = torch.cuda.get_device_properties(i)
                device_info["name"] = props.name
                device_info["compute_capability"] = f"{props.major}.{props.minor}"
                device_mem_gb = props.total_memory / (1024**3)
                device_info["memory_gb"] = round(device_mem_gb, 2)
                total_mem += device_mem_gb
                capabilities["compute_capabilities"].append(device_info)

            capabilities["total_memory_gb"] = round(total_mem, 2)

            if hasattr(torch.cuda, 'mem_get_info'):
                # Ensure device index is valid
                device_idx = 0 # Default to 0
                if settings and settings.CUDA_DEVICE < capabilities["device_count"]:
                     device_idx = settings.CUDA_DEVICE
                elif capabilities["device_count"] > 0:
                     logger.warning(f"Defaulting memory check to device 0, configured device {getattr(settings,'CUDA_DEVICE', 'N/A')} might be invalid.")
                else: # No devices, skip mem check
                     return capabilities

                free_mem_bytes, _ = torch.cuda.mem_get_info(device_idx)
                capabilities["free_memory_gb"] = round(free_mem_bytes / (1024**3), 2)

    except Exception as e:
        logger.warning(f"Could not fully detect system capabilities: {e}")

    return capabilities

def get_optimal_defaults(capabilities: Dict[str, Any]) -> Dict[str, Any]:
    """Get optimal default settings based on detected system capabilities."""
    defaults = {}
    total_mem_gb = capabilities.get("total_memory_gb", 0)
    cuda_available = capabilities.get("cuda_available", False)

    defaults["use_cuda"] = cuda_available

    # Set default model based on memory, preferring KB-Whisper if enough memory
    # Check if kb models exist in mapping before setting as default
    has_kblab_large = "kblab-large" in WHISPER_MODEL_MAPPING
    has_kblab_tiny = "kblab-tiny" in WHISPER_MODEL_MAPPING

    if cuda_available:
        if total_mem_gb >= 12 and has_kblab_large:
            defaults["default_model"] = "kblab-large"
        elif total_mem_gb >= 8:
            defaults["default_model"] = "medium"
        elif total_mem_gb >= 4:
            defaults["default_model"] = "small"
        elif has_kblab_tiny: # Prefer kb-tiny if available on smaller GPUs
             defaults["default_model"] = "kblab-tiny"
        else:
            defaults["default_model"] = "tiny" # Fallback to openai tiny

        # Adjust concurrency based on memory
        if total_mem_gb >= 24: defaults["max_concurrent_tasks"] = 3
        elif total_mem_gb >= 12: defaults["max_concurrent_tasks"] = 2
        else: defaults["max_concurrent_tasks"] = 1

        defaults["keep_multiple_models"] = total_mem_gb >= 16
        defaults["max_models_in_memory"] = 2 if defaults["keep_multiple_models"] else 1
        defaults["parallel_processing"] = total_mem_gb >= 16 # Diarization parallel processing
    else: # CPU Defaults
         defaults["default_model"] = "kblab-tiny" if has_kblab_tiny else "tiny" # Smallest available
         defaults["max_concurrent_tasks"] = max(1, capabilities.get("cpu_count", 4) // 4)
         defaults["keep_multiple_models"] = False
         defaults["max_models_in_memory"] = 1
         defaults["parallel_processing"] = False

    # Log detected capabilities and chosen defaults
    if cuda_available:
        logger.info(f"Detected {capabilities['device_count']} CUDA device(s) with "
                    f"{capabilities['total_memory_gb']:.1f}GB total memory.")
    else:
        logger.warning("No CUDA devices detected. Using CPU-compatible defaults. Performance will be significantly lower.")

    logger.info(f"Setting optimal defaults: model={defaults['default_model']}, "
                f"concurrent_tasks={defaults['max_concurrent_tasks']}, "
                f"parallel_diarization={defaults['parallel_processing']}")

    return defaults

def generate_secure_key(byte_length: int = 32) -> str:
    """Generate a secure URL-safe key."""
    key = secrets.token_bytes(byte_length)
    return base64.urlsafe_b64encode(key).decode()

# --- Detect Capabilities and Set Defaults ---
# Note: This runs at import time. Defaults might not reflect runtime env vars yet.
# Settings class will read env vars later.
SYSTEM_CAPABILITIES = detect_system_capabilities()
OPTIMAL_DEFAULTS = get_optimal_defaults(SYSTEM_CAPABILITIES)

# --- Settings Class ---
class Settings(BaseSettings):
    """Application settings with enhanced privacy features."""

    APP_NAME: str = "Whisper Transcription API"
    APP_VERSION: str = "1.1.0" # Incremented version
    DEBUG: bool = False
    LOG_LEVEL: str = "INFO"

    # Server settings
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    WORKERS: int = 1 # Uvicorn workers. For production, use Gunicorn or similar process manager.
    TIMEOUT: int = 300 # Request timeout in seconds

    # TLS/HTTPS settings (Recommended for production behind a reverse proxy)
    ENABLE_HTTPS: bool = False
    SSL_CERT_FILE: Optional[str] = None
    SSL_KEY_FILE: Optional[str] = None

    # Whisper model settings
    # Default model determined by available resources, but overridden by .env
    DEFAULT_MODEL: str = OPTIMAL_DEFAULTS.get("default_model", "medium")
    DEFAULT_LANGUAGE: Optional[str] = "sv" # Optional, model can detect
    MODELS_CACHE_DIR: str = "./models" # Recommend mounting a volume here in Docker

    # GPU / Compute settings
    USE_CUDA: bool = OPTIMAL_DEFAULTS.get("use_cuda", False)
    CUDA_DEVICE: int = 0 # Index of the GPU to use
    COMPUTE_TYPE: str = "auto" # Options: auto, float16, float32, bfloat16
    USE_TF32: bool = True # Enable TensorFloat32 on Ampere+ GPUs

    # Memory management settings
    KEEP_MULTIPLE_MODELS_IN_MEMORY: bool = OPTIMAL_DEFAULTS.get("keep_multiple_models", False)
    MAX_MODELS_IN_MEMORY: int = OPTIMAL_DEFAULTS.get("max_models_in_memory", 1)
    MIN_FREE_MEMORY_GB: float = 1.5 # Min free VRAM before trying to unload models
    PRELOAD_DEFAULT_MODEL: bool = True # Preload the default model on startup

    # Task management settings
    MAX_CONCURRENT_TASKS: int = OPTIMAL_DEFAULTS.get("max_concurrent_tasks", 1)
    JOB_CLEANUP_HOURS: int = 24 # How long to keep task results (0 for immediate cleanup after first request)
    TASK_TIMEOUT_MINUTES: int = 60 # Max time a single task can run
    RETRY_FAILED_TASKS: bool = False # Whether to retry tasks that fail
    MAX_RETRIES: int = 1 # Number of retries if enabled

    # Diarization settings
    DIARIZATION_ENABLED: bool = True # Enable/disable diarization feature globally
    HUGGINGFACE_TOKEN: Optional[str] = None # Required for pyannote models
    PARALLEL_PROCESSING: bool = OPTIMAL_DEFAULTS.get("parallel_processing", False) # Parallel chunk processing for diarization
    DIARIZATION_CHUNK_DURATION: int = 300 # Default chunk size in seconds (can be auto-adjusted)
    DIARIZATION_MODEL: str = "pyannote/speaker-diarization-3.1"

    # Storage settings (Crucial for privacy)
    UPLOAD_DIR: str = "./uploads" # Default relative path, use absolute for clarity
    RESULTS_DIR: str = "./results" # Default relative path
    MAX_UPLOAD_SIZE_MB: int = 1024 # Max audio file size in MB

    # Optimization settings (Passed to Whisper model)
    WHISPER_BATCH_SIZE: Optional[int] = None # Override default batch size if needed
    WHISPER_CHUNK_LENGTH: Optional[int] = None # Override default chunk length (in seconds)

    # Privacy settings
    AUTO_DELETE_AFTER_COMPLETION: bool = True # Delete audio files immediately after task finishes (success/fail)
    AUTO_DELETE_INTERVAL_MINUTES: int = 15 # Interval for periodic cleanup task (also cleans old task records)
    SECURE_MEMORY_WIPING: bool = False # Attempt to wipe memory (expensive, limited effectiveness)
    CLEAN_ALL_ON_SHUTDOWN: bool = True # Clean all temp files and tasks on graceful shutdown

    # Security settings
    API_AUTH_REQUIRED: bool = False # Set to True for production
    API_KEY: Optional[str] = None # Generate a strong key if auth is required
    ALLOWED_HOSTS: List[str] = ["*"] # Configure for specific hosts in production

    # Audit logging
    AUDIT_LOGGING_ENABLED: bool = False
    AUDIT_LOG_PATH: str = "./logs/audit.log" # Default relative path

    # Input validation
    ALLOWED_FILE_EXTENSIONS: List[str] = Field(
        default_factory=lambda: [".mp3", ".wav", ".ogg", ".flac", ".m4a", ".aac", ".opus", ".webm", ".mpga"]
    )
    ALLOWED_MIME_TYPES: List[str] = Field(
        default_factory=lambda: ["audio/*", "video/*", "application/octet-stream"] # Allow common audio/video types
    )

    # Rate limiting (Requires external setup or additional library like slowapi)
    RATE_LIMITING_ENABLED: bool = False
    RATE_LIMIT_REQUESTS: int = 100
    RATE_LIMIT_WINDOW_SECONDS: int = 3600

    # Pydantic V2 configuration using model_config
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False, # Environment variables are typically uppercase
        extra='ignore' # Ignore extra fields from environment
    )

    # --- Validators ---
    @validator('LOG_LEVEL', pre=True)
    def uppercase_log_level(cls, v):
        if isinstance(v, str):
            return v.upper()
        return "INFO" # Default if not string

    @validator('SSL_CERT_FILE', 'SSL_KEY_FILE')
    def validate_ssl_files(cls, v, values):
        # Check if 'ENABLE_HTTPS' exists and is True in the values dict
        if values.get('ENABLE_HTTPS') and v and not os.path.exists(v):
            logger.warning(f"SSL file does not exist: {v}")
        return v

    @validator('API_KEY', always=True)
    def validate_api_key(cls, v, values):
        if values.get('API_AUTH_REQUIRED') and not v:
            logger.warning("API_AUTH_REQUIRED is true, but no API_KEY is set. Generating a temporary key.")
            return generate_secure_key(32)
        return v

    @validator('HUGGINGFACE_TOKEN', always=True)
    def validate_hf_token(cls, v, values):
        if values.get('DIARIZATION_ENABLED') and not v:
            env_token = os.environ.get("HUGGINGFACE_TOKEN")
            if env_token:
                logger.info("Using HUGGINGFACE_TOKEN from environment variable.")
                return env_token
            else:
                logger.warning("Diarization is enabled, but HUGGINGFACE_TOKEN is not set.")
        return v

    @validator('DEFAULT_MODEL')
    def validate_default_model(cls, v):
        # Use the globally defined mapping
        if v not in WHISPER_MODEL_MAPPING:
            raise ValueError(f"Invalid DEFAULT_MODEL '{v}'. Must be one of {list(WHISPER_MODEL_MAPPING.keys())}")
        return v

    # Validator to ensure paths are absolute after loading
    # This helps avoid issues with relative paths when running from different locations
    @validator('MODELS_CACHE_DIR', 'UPLOAD_DIR', 'RESULTS_DIR', 'AUDIT_LOG_PATH', always=True)
    def make_paths_absolute(cls, v):
        # Only make absolute if it's not already (e.g., /dev/shm) and is set
        if v and not os.path.isabs(v):
            # Assume relative to project root (where .env likely is)
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # ../ from config.py
            abs_path = os.path.abspath(os.path.join(project_root, v))
            logger.debug(f"Converting relative path '{v}' to absolute '{abs_path}'")
            return abs_path
        return v


# --- Global Settings Instance ---
try:
    settings = Settings()
except Exception as e:
     # Catch validation errors during Settings instantiation
     logger.critical(f"CRITICAL: Failed to initialize application settings: {e}", exc_info=True)
     logger.critical("Application cannot start without valid settings. Exiting.")
     sys.exit(1) # Exit immediately if settings fail


# --- Apply Global Settings ---

def apply_torch_settings():
    """Apply settings to PyTorch."""
    if not settings or not settings.USE_CUDA or not torch or not torch.cuda.is_available():
        return # Skip if settings not loaded, CUDA disabled, or torch/cuda unavailable

    # Set active device
    try:
        num_devices = torch.cuda.device_count()
        if settings.CUDA_DEVICE < num_devices:
            torch.cuda.set_device(settings.CUDA_DEVICE)
            logger.info(f"Set active CUDA device to: cuda:{settings.CUDA_DEVICE}")
        elif num_devices > 0:
             logger.warning(f"CUDA_DEVICE index {settings.CUDA_DEVICE} is invalid (found {num_devices}). Using default device 0.")
             torch.cuda.set_device(0)
        # No else needed if num_devices is 0, already handled by is_available() check
    except Exception as e:
        logger.error(f"Could not set CUDA device: {e}. Check nvidia-smi.")

    # Enable TF32 if specified and supported
    try:
         if settings.USE_TF32 and torch.cuda.get_device_capability(settings.CUDA_DEVICE)[0] >= 8:
             torch.backends.cuda.matmul.allow_tf32 = True
             if hasattr(torch.backends.cudnn, 'allow_tf32'):
                  torch.backends.cudnn.allow_tf32 = True
             logger.info("Enabled TensorFloat32 (TF32) for faster matmul on compatible GPUs.")
         else:
             if settings.USE_TF32: logger.info("GPU does not support TF32 or USE_TF32 is false.")
             torch.backends.cuda.matmul.allow_tf32 = False
             if hasattr(torch.backends.cudnn, 'allow_tf32'):
                  torch.backends.cudnn.allow_tf32 = False
    except Exception as e:
         logger.warning(f"Could not configure TF32 settings: {e}")

    # Enable cuDNN benchmark mode (can improve performance for fixed input sizes)
    if hasattr(torch.backends, 'cudnn'):
         torch.backends.cudnn.benchmark = True
         logger.info("Enabled cuDNN benchmark mode.")

def create_secure_directories():
    """Create required directories with secure permissions (owner only)."""
    if not settings: # Skip if settings failed to load
         logger.error("Settings not loaded, cannot create directories.")
         return

    directories = [
        settings.MODELS_CACHE_DIR,
        settings.UPLOAD_DIR,
        settings.RESULTS_DIR,
    ]
    if settings.AUDIT_LOGGING_ENABLED and settings.AUDIT_LOG_PATH:
        audit_log_dir = os.path.dirname(settings.AUDIT_LOG_PATH)
        if audit_log_dir: # Ensure audit path isn't just a filename in root
            directories.append(audit_log_dir)

    for directory in directories:
        if not directory: continue # Skip empty paths
        try:
            # Use absolute paths ensured by validator
            abs_directory = directory # Already absolute from validator
            os.makedirs(abs_directory, mode=0o700, exist_ok=True)
            # Verify and set permissions if directory already existed but had wrong permissions
            current_mode = os.stat(abs_directory).st_mode & 0o777
            if current_mode != 0o700:
                os.chmod(abs_directory, 0o700)
                logger.info(f"Corrected permissions for directory: {abs_directory} to 700")
            else:
                 logger.debug(f"Directory exists with correct permissions: {abs_directory}")
        except PermissionError as e:
             logger.error(f"Permission denied creating/securing directory {directory}: {e}. Check user permissions for path.")
        except Exception as e:
            logger.error(f"Could not create/secure directory {directory}: {e}. Check path and permissions.", exc_info=True)


def check_storage_paths():
    """Warn if using potentially insecure default storage paths."""
    if not settings: return

    # Use absolute paths for comparison
    insecure_indicators = ["./", "../", "/tmp/"] # Indicate relative or standard tmp
    upload_abs = settings.UPLOAD_DIR
    results_abs = settings.RESULTS_DIR

    if any(upload_abs.startswith(indicator) for indicator in insecure_indicators):
         logger.warning(f"UPLOAD_DIR ('{upload_abs}') seems to be relative or in standard /tmp. Consider using a dedicated, secure path or memory-based storage (like /dev/shm) for production.")
    if any(results_abs.startswith(indicator) for indicator in insecure_indicators):
         logger.warning(f"RESULTS_DIR ('{results_abs}') seems to be relative or in standard /tmp. Consider using a dedicated, secure path or memory-based storage (like /dev/shm) for production.")

    if "/dev/shm" in upload_abs or "/dev/shm" in results_abs:
        if not os.path.exists("/dev/shm"):
            logger.warning("/dev/shm specified in paths but may not be available or sufficiently sized on this system.")


# --- Run Initial Setup only if settings loaded successfully ---
if settings:
    apply_torch_settings()
    create_secure_directories()
    check_storage_paths()

    # --- Log Final Configuration Summary ---
    logger.info("--- Configuration Summary ---")
    logger.info(f"Default Model: {settings.DEFAULT_MODEL}")
    logger.info(f"Use CUDA: {settings.USE_CUDA} (Device: {settings.CUDA_DEVICE if settings.USE_CUDA else 'N/A'})")
    logger.info(f"Max Concurrent Tasks: {settings.MAX_CONCURRENT_TASKS}")
    logger.info(f"Diarization Enabled: {settings.DIARIZATION_ENABLED}")
    logger.info(f"Auto Delete Files: {settings.AUTO_DELETE_AFTER_COMPLETION}")
    logger.info(f"API Auth Required: {settings.API_AUTH_REQUIRED}")
    logger.info(f"Upload Dir: {settings.UPLOAD_DIR}")
    logger.info(f"Results Dir: {settings.RESULTS_DIR}")
    logger.info(f"Models Cache Dir: {settings.MODELS_CACHE_DIR}")
    logger.info(f"Audit Logging Enabled: {settings.AUDIT_LOGGING_ENABLED}")
    logger.info("--- End Configuration Summary ---")

    # --- Security Warnings ---
    if not settings.API_AUTH_REQUIRED:
        logger.warning("Security Warning: API authentication is DISABLED. The API is open.")
    if settings.HOST == "0.0.0.0" and not settings.API_AUTH_REQUIRED:
        logger.critical("CRITICAL Security Warning: API is binding to all interfaces (0.0.0.0) WITHOUT authentication!")
    if not settings.ENABLE_HTTPS and settings.API_AUTH_REQUIRED:
         logger.warning("Security Warning: API authentication is enabled, but HTTPS is not. API key may be sent in plaintext if not behind a TLS-terminating proxy.")

else:
     logger.critical("Application settings failed to load. Cannot proceed with setup.")
     # Application will likely fail later during import in main.py