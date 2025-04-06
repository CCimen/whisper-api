"""
Main FastAPI application for the Whisper Transcription API.

Initializes the application, sets up middleware, lifespan management,
exception handling, and includes API routers.
"""

import os
import asyncio
import logging
import redis.asyncio as redis # For Redis connection pool
# import json # No longer needed for JSON logging here
# from logging.handlers import RotatingFileHandler # Already imported for audit, but good practice
from rich.logging import RichHandler
from rich.console import Console
from rich.theme import Theme
from rich.style import Style
from rich.panel import Panel
from rich.text import Text
from rich.highlighter import RegexHighlighter
from rich.table import Table
from rich.box import ROUNDED
from rich.traceback import install as install_rich_traceback
import time
import gc
import uuid
import secrets
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, Request, status, Depends, HTTPException
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.security import APIKeyHeader
# Import for Prometheus metrics
from starlette_exporter import PrometheusMiddleware, handle_metrics
# Import for rate limiting
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

# Install rich traceback handler
install_rich_traceback(show_locals=True)

# Custom highlighter for log messages
class WhisperAPIHighlighter(RegexHighlighter):
    """Custom highlighter for WhisperAPI log messages."""
    
    base_style = "log."
    highlights = [
        # Task statuses with capture groups for different styling
        r"Status changed (?P<status_from>\w+) -> (?P<status_to>\w+)",
        r"\[(?P<component>MODEL|PROCESSOR|TASKMGR|TASKR)\]",
        
        # Model references
        r"(?P<model_tiny>whisper-kblab-tiny)",
        r"(?P<model_small>whisper-kblab-small)",
        r"(?P<model_medium>whisper-kblab-medium)",
        r"(?P<model_large>whisper-kblab-large|whisper-openai-large-v3)",
        r"(?P<model_base>whisper-base)",
        
        # Request information
        r"(?P<http_method>GET|POST|DELETE|PUT) (?P<url>/[^\s]+)",
        r"Duration=(?P<duration>\d+\.?\d*m?s)",
        r"Status=(?P<status>\d{3})",
        
        # File information
        r"(?P<file_path>/tmp/[^\s]+)",
        r"(?P<file_size>\d+) bytes\)",
        r"R_ID=(?P<request_id>[a-f0-9\-]+)",
        
        # Task IDs
        r"task (?P<task_id>[a-f0-9\-]+)"
    ]

api_highlighter = WhisperAPIHighlighter()

# Define custom theme for the console - using color-blind friendly palette
CUSTOM_THEME = Theme({
    # General message types
    "info": "bold cyan",
    "warning": "bold yellow",
    "danger": "bold bright_red",
    "success": "bold bright_green",
    
    # Application components
    "app.title": "bold bright_magenta",
    "app.subtitle": "italic cyan",
    
    # Models - using blue which is generally visible to most color-blind people
    "model.name": "bold bright_blue",
    "model.tiny": "bold blue",
    "model.small": "bold cyan",
    "model.medium": "bold blue on black",
    "model.large": "bold bright_blue on black",
    "model.base": "bold cyan on black",
    "model.status": "bright_green",
    "model.loading": "yellow",
    "model.ready": "bold bright_green",
    
    # API related
    "api.request": "bold cyan",
    "api.response": "cyan",
    "api.method": "bold bright_green",
    "api.status.success": "bright_green",
    "api.status.error": "bright_red",
    "api.duration": "bold bright_white",
    "api.url": "underline cyan",
    
    # Configuration
    "config.key": "bold bright_blue",
    "config.value": "bold bright_white",
    "config.enabled": "bold bright_green",
    "config.disabled": "italic yellow",
    
    # Task management - using high contrast
    "task.id": "bold bright_yellow",
    "task.status": "bold bright_white",
    "task.status.change": "bold bright_yellow",
    "taskmgr": "bold bright_blue",
    "taskr": "bold blue",
    "processor": "bold bright_magenta",
    
    # Log components
    "log.timestamp": "dim white",
    "log.scope": "bold bright_blue",
    
    # File operations
    "file.path": "underline bright_white",
    "file.size": "bold bright_white",
    "file.duration": "italic bright_white",
})

# Logging will be configured via --log-config passed to Uvicorn
# Get logger instance for this module
logger = logging.getLogger(__name__)

# Import exceptions FIRST so they are available for the try/except block
try:
    from app.exceptions import (
        ModelNotFoundError, TranscriptionError, DiarizationError,
        ConfigurationError, BaseApiException, FileProcessingError
    )
except ImportError as e:
    print(f"[bold red]CRITICAL:[/bold red] Failed to import base exception classes: {e}. Cannot start.")
    exit(1)

# Import settings and dependencies safely within a try/except
# Global instances will be stored in app.state instead
# task_manager = None
# model_registry = None
try:
    from app.config import settings, WHISPER_MODEL_MAPPING
    # Exit if settings failed to load in config.py
    if settings is None:
        raise ConfigurationError("Settings object failed to initialize in config.py")

    # Core services will be initialized within the lifespan function now
    from app.services.model_registry import ModelRegistry
    from app.services.task_manager import TaskManager
    from app.services.diarization import DiarizationService, DIARIZATION_AVAILABLE # Import DiarizationService

except ImportError as e:
    print(f"[bold red]CRITICAL:[/bold red] Failed to import core modules: {e}. Please ensure all dependencies are installed and configuration is present.")
    exit(1)
except ConfigurationError as e:
    print(f"[bold red]CRITICAL:[/bold red] Configuration Error on startup: {e}")
    exit(1)
except Exception as e:
    print(f"[bold red]CRITICAL:[/bold red] Unexpected error during initial imports: {e}")
    exit(1)

# --- Audit Logging Setup (Remains here as it needs settings) ---
# (Audit logging code remains unchanged)

# --- Audit Logging Setup ---
# --- Audit Logging Setup (Moved inside lifespan for app context if needed, or keep global if simple) ---
# Simplified global setup for audit logger if enabled
audit_logger = None
if settings.AUDIT_LOGGING_ENABLED:
    try:
        audit_logger = logging.getLogger("audit")
        # Basic configuration check - detailed setup might need app context if complex
        if not settings.AUDIT_LOG_PATH:
             raise ConfigurationError("AUDIT_LOGGING_ENABLED is true, but AUDIT_LOG_PATH is not set.")
        # Ensure directory exists (permissions handled in config.py)
        log_dir = os.path.dirname(settings.AUDIT_LOG_PATH)
        if log_dir: os.makedirs(log_dir, exist_ok=True)
        # Handler setup will happen in lifespan or via logging config file
        # logger.info(f":lock: Audit logging configured in settings. Path: [italic]{settings.AUDIT_LOG_PATH}[/italic]") # Reduced verbosity
    except Exception as e:
        logger.error(f":warning: [danger]Initial check for audit logging failed:[/danger] {e}")
        audit_logger = None # Ensure it's None if setup fails

# --- PyTorch & GPU Availability Check ---
pytorch_available = False
gpu_available = False
try:
    import torch
    pytorch_available = True
    # logger.info(f":package: PyTorch version: [config.value]{torch.__version__}[/config.value]") # Reduced verbosity
    if settings.USE_CUDA:
        if torch.cuda.is_available():
            gpu_available = True
            num_devices = torch.cuda.device_count()
            # logger.info(f":zap: [success]CUDA available[/success]. Found [config.value]{num_devices}[/config.value] device(s).") # Reduced verbosity
            if settings.CUDA_DEVICE >= num_devices:
                 logger.error(f":warning: [danger]Invalid CUDA_DEVICE index {settings.CUDA_DEVICE}[/danger] (found {num_devices}). Defaulting to 0.")
                 settings.CUDA_DEVICE = 0 # Correct setting
            # logger.info(f":computer: Using CUDA device [config.value]{settings.CUDA_DEVICE}[/config.value]") # Reduced verbosity
        else:
            # logger.warning(":warning: CUDA enabled but not available. Check drivers/installation. Using CPU.") # Reduced verbosity
            settings.USE_CUDA = False # Correct setting
    # else: # Reduced verbosity
        # logger.info(":information_source: CUDA is disabled in settings. Using CPU.")

except ImportError:
    logger.warning(":warning: [warning]PyTorch not found[/warning]. Transcription/Diarization will likely fail.")
    torch = None # Keep torch=None check available


# --- API Key Security ---
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False, description="API key for authenticating requests.")

async def get_api_key(x_api_key: Optional[str] = Depends(api_key_header)):
    """Dependency function to validate the provided API key."""
    if not settings.API_AUTH_REQUIRED:
        return True

    if not settings.API_KEY:
        logger.critical(":rotating_light: [danger]CRITICAL: API Authentication is required, but no API_KEY is configured![/danger]")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="API authentication misconfigured on server."
        )

    if not x_api_key:
        logger.debug(":key: API key missing in request header.")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API key required.",
            headers={"WWW-Authenticate": 'ApiKey realm="Restricted Area"'},
        )

    if not secrets.compare_digest(x_api_key, settings.API_KEY):
        logger.warning(f":key: [warning]Invalid API key received.[/warning]")
        await asyncio.sleep(0.1)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key.",
            headers={"WWW-Authenticate": 'ApiKey realm="Restricted Area"'},
        )

    logger.debug(":white_check_mark: API key validated successfully.")
    return True


# --- Background Tasks (defined outside lifespan for clarity) ---

async def preload_model_background(model_key: str):
    """Loads a model in a background task."""
    if not task_manager:
        logger.error(f":x: [danger]TaskManager not available.[/danger] Cannot load model {model_key}.")
        return

    # Determine model size for styling
    model_style = "model.name"
    if "-tiny" in model_key:
        model_style = "model.tiny"
    elif "-small" in model_key:
        model_style = "model.small"
    elif "-medium" in model_key:
        model_style = "model.medium"
    elif "-large" in model_key:
        model_style = "model.large"
    elif "-base" in model_key:
        model_style = "model.base"

    logger.info(f"[model.loading]⏳ [Preload Task][/model.loading] Starting preload for [{model_style}]{model_key}[/{model_style}]")
    try:
        model = ModelRegistry.get_model(model_key)
        if not model.is_loaded():
            device_str = "cuda" if gpu_available and settings.USE_CUDA else "cpu"
            await asyncio.to_thread(model.load, device=device_str)
            logger.info(f"[model.ready]✓ [Preload Task][/model.ready] Successfully preloaded model: [{model_style}]{model_key}[/{model_style}] on [config.value]{device_str}[/config.value]")
        else:
            logger.info(f"[model.status]ℹ [Preload Task][/model.status] Model [{model_style}]{model_key}[/{model_style}] was already loaded.")
    except ModelNotFoundError:
        logger.error(f"[model.loading]❌ [Preload Task][/model.loading] Model [{model_style}]{model_key}[/{model_style}] [danger]not found[/danger] in registry.")
    except Exception as e:
        logger.error(f"[model.loading]❌ [Preload Task][/model.loading] [danger]Failed to preload model[/danger] [{model_style}]{model_key}[/{model_style}]: {e}")

async def run_periodic_cleanup():
    """Runs background task for cleaning up old jobs and potentially files."""
    if not task_manager:
        logger.error("🧹 [danger]TaskManager not available.[/danger] Periodic cleanup task cannot run.")
        return

    logger.info("🧹 [taskmgr]Periodic cleanup task started.[/taskmgr]")
    interval_seconds = max(300, settings.AUTO_DELETE_INTERVAL_MINUTES * 60)
    logger.info(f"⏰ Cleanup interval set to [config.value]{interval_seconds}[/config.value] seconds.")

    while True:
        try:
            await asyncio.sleep(interval_seconds)
            logger.debug("🧹 Running periodic cleanup cycle...")

            cleaned_tasks = await task_manager.cleanup_old_tasks()
            if cleaned_tasks > 0:
                # Create a small table for cleanup results
                table = Table(
                    title="Cleanup Results", 
                    box=ROUNDED, 
                    title_style="bold bright_blue",
                    border_style="bright_blue"
                )
                table.add_column("Action", style="bright_white")
                table.add_column("Count", style="bright_yellow")
                table.add_column("Status", style="bright_green")
                
                table.add_row(
                    "Task Records Removed", 
                    str(cleaned_tasks), 
                    "✓ Complete"
                )
                
                logger.info(table)

            gc.collect()
            if gpu_available and torch:
                torch.cuda.empty_cache()
                logger.debug("♻️ [taskmgr]Periodic GPU cache clear performed.[/taskmgr]")

        except asyncio.CancelledError:
            logger.info("🛑 [taskmgr]Periodic cleanup task stopping due to cancellation.[/taskmgr]")
            break
        except Exception as e:
            logger.error(f"⚠️ [danger]Error in periodic cleanup loop:[/danger] {e}")
            await asyncio.sleep(60)

# --- Application Lifespan ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handles application startup and shutdown events using async context manager."""
    # === Startup ===
    start_time = time.time()
    # Logging is now handled by Uvicorn's --log-config, setup happens before lifespan

    # --- Initialize Redis Pool ---
    redis_pool = None
    if settings.REDIS_HOST:
        try:
            logger.info(f":globe_with_meridians: Attempting to connect to Redis at [config.value]{settings.REDIS_HOST}:{settings.REDIS_PORT}[/config.value] (DB: {settings.REDIS_DB})")
            redis_pool = redis.ConnectionPool(
                host=settings.REDIS_HOST,
                port=settings.REDIS_PORT,
                db=settings.REDIS_DB,
                password=settings.REDIS_PASSWORD,
                socket_timeout=settings.REDIS_TIMEOUT,
                decode_responses=True # Decode responses to strings
            )
            # Test connection
            r = redis.Redis(connection_pool=redis_pool)
            await r.ping()
            app.state.redis_pool = redis_pool # Store pool in app state
            logger.info(":white_check_mark: [success]Successfully connected to Redis.[/success]")
        except Exception as e:
            logger.error(f":x: [danger]Failed to connect to Redis:[/danger] {e}. Task management and session storage may fail.")
            app.state.redis_pool = None # Ensure it's None if connection failed
            redis_pool = None # Ensure local variable is also None
    else:
        logger.warning(":warning: [warning]REDIS_HOST not configured.[/warning] Task management features requiring Redis will not work.")
        app.state.redis_pool = None

    # --- Instantiate DiarizationService (if enabled and available) ---
    # Moved BEFORE TaskManager initialization
    app.state.diarization_service = None # Initialize in state first
    if settings.DIARIZATION_ENABLED and DIARIZATION_AVAILABLE:
        try:
            app.state.diarization_service = DiarizationService()
            logger.info(":loud_sound: DiarizationService initialized and stored in app.state.")
        except Exception as e:
            logger.error(f":warning: [danger]Failed to initialize DiarizationService during startup:[/danger] {e}", exc_info=True)
            # App can continue, but diarization routes/features might fail later
    elif settings.DIARIZATION_ENABLED:
         logger.warning(":warning: Diarization enabled in settings, but dependencies are missing. DiarizationService not initialized.")

    # --- Instantiate TaskManager (Now depends on Redis pool AND DiarizationService) ---
    # global task_manager # No longer using module global
    try:
        # Import moved inside lifespan to ensure config is loaded
        from app.services.task_manager import TaskManager
        # Pass the potentially None redis_pool and diarization_service to TaskManager
        # Instantiate and store in app.state
        app.state.task_manager = TaskManager(
            redis_pool=app.state.redis_pool,
            diarization_service=app.state.diarization_service # Pass the potentially None diarization service instance
        )
        # logger.info(":gear: TaskManager initialized and stored in app.state.") # Reduced verbosity
    except Exception as e:
        logger.critical(f":rotating_light: [danger]CRITICAL: Failed to initialize TaskManager:[/danger] {e}", exc_info=True)
        # Decide if the app should exit if TaskManager is critical
        # sys.exit(1) # Uncomment to make TaskManager essential

    # --- Instantiate ModelRegistry ---
    # global model_registry # No longer using module global
    try:
        # Instantiate and store in app.state
        app.state.model_registry = ModelRegistry()
        # logger.info(":books: ModelRegistry initialized and stored in app.state.") # Reduced verbosity
    except Exception as e:
        logger.critical(f":rotating_light: [danger]CRITICAL: Failed to initialize ModelRegistry:[/danger] {e}", exc_info=True)
        # sys.exit(1) # Uncomment if ModelRegistry is essential

    # --- Continue with Startup ---
    # --- Continue with Startup ---
    # app_title = Text(f"--- Starting {settings.APP_NAME} v{settings.APP_VERSION} ---") # Reduced verbosity
    # app_title.stylize("bold magenta")
    # logger.info(app_title)
    
    # Log critical settings resolved after loading .env
    # logger.info(Panel.fit( # Commented out to reduce duplicate logs from multiple workers
    #     f"[config.key]API Authentication Required:[/config.key] [config.value]{settings.API_AUTH_REQUIRED}[/config.value]\n"
    #     f"[config.key]Diarization Enabled:[/config.key] [config.value]{settings.DIARIZATION_ENABLED}[/config.value]\n"
    #     f"[config.key]Max Concurrent Tasks:[/config.key] [config.value]{settings.MAX_CONCURRENT_TASKS}[/config.value]\n"
    #     f"[config.key]Default Model:[/config.key] [config.value]{settings.DEFAULT_MODEL}[/config.value]\n"
    #     f"[config.key]Using CUDA:[/config.key] [config.value]{settings.USE_CUDA}[/config.value] | [config.key]GPU Available:[/config.key] [config.value]{gpu_available}[/config.value]",
    #     title="[app.title]Configuration[/app.title]",
    #     border_style="cyan"
    # ))

    # Ensure ModelRegistry discovers models early
    # Ensure ModelRegistry discovers models early (using the instance)
    try:
        # Use instance from app.state
        app.state.model_registry.discover_models()
        available_models = app.state.model_registry.available_models()
        # logger.info(f":mag: Available models discovered: [model.name]{', '.join(available_models)}[/model.name]") # Reduced verbosity
    except Exception as e:
        logger.error(f":x: [danger]Failed during model discovery:[/danger] {e}", exc_info=True) # Add exc_info

    # Preload default model if configured
    preload_task = None
    if settings.PRELOAD_DEFAULT_MODEL and settings.DEFAULT_MODEL:
        model_key = f"whisper-{settings.DEFAULT_MODEL}"
        # Use instance from app.state
        if model_key in app.state.model_registry.available_models():
            # logger.info(f":hourglass: Initiating preload for default model: [model.name]{model_key}[/model.name]...") # Reduced verbosity
            logger.info(f":hourglass: Initiating preload for default model: [model.name]{model_key}[/model.name]...") # Keep preload initiation log
            preload_task = asyncio.create_task(preload_model_background(model_key))
        else:
            # Use instance from app.state
            logger.warning(f":warning: Default model '[model.name]{settings.DEFAULT_MODEL}[/model.name]' (key: [model.name]{model_key}[/model.name]) specified for preload but [warning]not found in registry[/warning]. Available: [model.name]{', '.join(app.state.model_registry.available_models())}[/model.name]")

    # Start periodic cleanup task
    # logger.info("🧹 Starting periodic cleanup task...") # Reduced verbosity (run_periodic_cleanup logs its own start)
    cleanup_task = asyncio.create_task(run_periodic_cleanup())

    # --- Log Final Configuration Summary (after settings and services are loaded) ---
    logger.info("--- Application Startup Configuration ---")
    # Use f-strings and access settings directly
    cuda_status = f"[success]Enabled[/success] (Device: {settings.CUDA_DEVICE})" if settings.USE_CUDA and gpu_available else "[warning]Disabled or Unavailable[/warning]"
    diar_status = f"[success]Enabled[/success]" if settings.DIARIZATION_ENABLED and DIARIZATION_AVAILABLE else "[warning]Disabled or Unavailable[/warning]"
    auth_status = f"[success]Enabled[/success]" if settings.API_AUTH_REQUIRED else "[danger]DISABLED[/danger]"
    redis_status = "[success]Connected[/success]" if app.state.redis_pool else "[danger]Not Connected[/danger]"

    # Create a simple summary string or use Rich Panel if preferred
    summary_text = (
        f"  Default Model: [config.value]{settings.DEFAULT_MODEL}[/config.value]\n"
        f"  Use CUDA: {cuda_status}\n"
        f"  Max Concurrent Tasks: [config.value]{settings.MAX_CONCURRENT_TASKS}[/config.value]\n"
        f"  Diarization: {diar_status}\n"
        f"  API Auth: {auth_status}\n"
        f"  Redis Status: {redis_status}\n"
        f"  Upload Dir: [file.path]{settings.UPLOAD_DIR}[/file.path]\n"
        f"  Results Dir: [file.path]{settings.RESULTS_DIR}[/file.path]"
    )
    logger.info(Panel(summary_text, title="[app.title]Key Settings[/app.title]", border_style="cyan", expand=False))
    logger.info("--- Application Startup Complete ---")

    yield  # Application runs here

    # === Shutdown ===
    shutdown_title = Text(f"--- Shutting down {settings.APP_NAME} ---")
    shutdown_title.stylize("bold yellow")
    logger.info(shutdown_title)
    shutdown_start_time = time.time()

    # 1. Signal TaskManager to shut down
    # Use instance from app.state
    if hasattr(app.state, 'task_manager') and app.state.task_manager:
        logger.info(":gear: Initiating TaskManager shutdown...")
        await app.state.task_manager.shutdown()
        logger.info(":white_check_mark: TaskManager shutdown complete.")
    else:
        logger.warning(":warning: [warning]TaskManager not available during shutdown sequence.[/warning]")

    # 2. Cancel background tasks
    if preload_task and not preload_task.done():
        preload_task.cancel()
        logger.info(":stop_sign: Cancelled ongoing preload task.")
    if cleanup_task and not cleanup_task.done():
        cleanup_task.cancel()
        try:
            await asyncio.wait_for(cleanup_task, timeout=5.0)
        except asyncio.CancelledError:
            logger.info(":white_check_mark: Periodic cleanup task cancelled successfully.")
        except asyncio.TimeoutError:
            logger.warning(":hourglass: [warning]Periodic cleanup task did not finish within timeout during shutdown.[/warning]")
        except Exception as e:
            logger.error(f":x: [danger]Error waiting for cleanup task during shutdown:[/danger] {e}")

    # 3. Close Redis Pool
    if hasattr(app.state, 'redis_pool') and app.state.redis_pool:
        logger.info(":electric_plug: Closing Redis connection pool...")
        try:
            # Use disconnect() which closes all connections in the pool
            await app.state.redis_pool.disconnect(inuse_connections=True)
            logger.info(":white_check_mark: Redis connection pool closed.")
        except Exception as e:
            logger.error(f":x: [danger]Error closing Redis pool:[/danger] {e}")

    # 4. Unload all models
    # Use instance from app.state
    try:
        if hasattr(app.state, 'model_registry') and app.state.model_registry:
            app.state.model_registry.unload_all()
    except Exception as e:
        logger.error(f":x: [danger]Error unloading models during shutdown:[/danger] {e}")

    # 5. Final memory cleanup
    gc.collect()
    if gpu_available and torch:
        try:
            torch.cuda.empty_cache()
            logger.info(":recycle: Final GPU cache clear performed.")
        except Exception as e:
            logger.warning(f":warning: [warning]Error during final GPU cache clear:[/warning] {e}")

    shutdown_duration = time.time() - shutdown_start_time
    logger.info(f":white_check_mark: [success]Application shutdown complete[/success] ([config.value]{shutdown_duration:.2f}s[/config.value])")


# --- FastAPI App Initialization ---
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="High-performance API for audio transcription with optional speaker diarization using Whisper models. Optimized for privacy and GPU acceleration.",
    lifespan=lifespan,
)

# --- Rate Limiter Setup ---
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
# Construct rate limit string from settings if enabled
rate_limit_value = None
if settings.RATE_LIMITING_ENABLED:
    rate_limit_value = f"{settings.RATE_LIMIT_REQUESTS}/{settings.RATE_LIMIT_WINDOW_SECONDS}s"
    logger.info(f"Rate limiting enabled: {rate_limit_value}")

# --- Middleware Configuration ---
# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_HOSTS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
    allow_headers=["*", "X-API-Key", "Authorization", "Content-Type"],
)
# Prometheus Middleware (Add BEFORE other custom middleware if possible)
app.add_middleware(PrometheusMiddleware)
# GZip Middleware
app.add_middleware(GZipMiddleware, minimum_size=1000)
# Security Headers Middleware
@app.middleware("http")
async def add_security_headers_middleware(request: Request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    return response
# Request ID and Logging Context Middleware
@app.middleware("http")
async def request_context_middleware(request: Request, call_next):
    request_id = str(uuid.uuid4())
    request.state.request_id = request_id
    request.state.client_ip = request.client.host if request.client else "unknown"
    start_time = time.time()
    
    # Format the short request ID for display
    short_id = request_id[:8]
    
    logger.info(
        f"→ [api.request]Request started:[/api.request] "
        f"[task.id]R_ID={short_id}...[/task.id] "
        f"IP={request.state.client_ip} "
        f"[api.method]{request.method}[/api.method] "
        f"URL=[api.url]{request.url.path}[/api.url]"
    )
    
    try:
        response = await call_next(request)
        process_time = (time.time() - start_time) * 1000
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Process-Time-MS"] = f"{process_time:.2f}"
        
        status_style = "api.status.success" if response.status_code < 400 else "api.status.error"
        logger.info(
            f"✓ [api.response]Request finished:[/api.response] "
            f"[task.id]R_ID={short_id}...[/task.id] "
            f"Status=[{status_style}]{response.status_code}[/{status_style}] "
            f"Duration=[api.duration]{process_time:.2f}ms[/api.duration]"
        )
        
        return response
    except Exception as e:
        process_time = (time.time() - start_time) * 1000
        logger.error(
            f"❌ [danger]Request failed:[/danger] "
            f"[task.id]R_ID={short_id}...[/task.id] "
            f"Error=[danger]{type(e).__name__}[/danger] "
            f"Duration=[api.duration]{process_time:.2f}ms[/api.duration]"
        )
        raise e

# Audit Logging Middleware
@app.middleware("http")
async def audit_logging_middleware(request: Request, call_next):
    if not settings.AUDIT_LOGGING_ENABLED or not audit_logger:
        return await call_next(request)
    
    start_time = time.time()
    request_id = getattr(request.state, "request_id", "N/A")
    client_ip = getattr(request.state, "client_ip", "unknown")
    log_extra = {'request_id': request_id, 'client_ip': client_ip}
    
    audit_logger.info(f"REQ Start: {request.method} {request.url.path} QS='{request.url.query}'", extra=log_extra)
    
    try:
        response = await call_next(request)
        duration_ms = (time.time() - start_time) * 1000
        audit_logger.info(f"RES End: Status={response.status_code} Duration={duration_ms:.2f}ms", extra=log_extra)
        return response
    except Exception as e:
        duration_ms = (time.time() - start_time) * 1000
        status_code = 500
        if isinstance(e, HTTPException): 
            status_code = e.status_code
        audit_logger.error(f"REQ Error: Status={status_code} Type={type(e).__name__} Msg='{e}' Duration={duration_ms:.2f}ms", extra=log_extra)
        raise e

# --- Custom Exception Handlers ---
@app.exception_handler(BaseApiException)
async def handle_custom_api_exceptions(request: Request, exc: BaseApiException):
    status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
    
    # Determine the error category and style
    error_category = "SERVER ERROR"
    error_style = "danger"
    
    if isinstance(exc, ModelNotFoundError): 
        status_code = status.HTTP_404_NOT_FOUND
        error_category = "NOT FOUND"
        error_style = "warning"
    elif isinstance(exc, ConfigurationError): 
        status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        error_category = "SERVICE UNAVAILABLE"
        error_style = "danger"
    elif isinstance(exc, FileProcessingError): 
        status_code = status.HTTP_400_BAD_REQUEST
        error_category = "BAD REQUEST"
        error_style = "warning"
    # Transcription/DiarizationError remain 500
    
    request_id = getattr(request.state, "request_id", "N/A")
    short_id = request_id[:8] if len(request_id) > 8 else request_id
    log_level = logging.ERROR if status_code >= 500 else logging.WARNING
    
    # Create a styled panel for exception
    error_message = Panel(
        f"[{error_style}]{type(exc).__name__}[/{error_style}]: {exc.message}",
        title=f"[{error_style}]{error_category}[/{error_style}]",
        subtitle=f"Request ID: {short_id}...",
        border_style=error_style
    )
    
    logger.log(log_level, error_message, exc_info=status_code >= 500)
    
    return JSONResponse(
        status_code=status_code,
        content={"detail": exc.message, "request_id": request_id, "error_type": type(exc).__name__},
    )

@app.exception_handler(HTTPException)
async def http_exception_handler_override(request: Request, exc: HTTPException):
    request_id = getattr(request.state, "request_id", "N/A")
    
    logger.warning(
        f":warning: [warning]HTTPException[/warning] [task.id][{request_id}][/task.id]: "
        f"Status=[api.status.error]{exc.status_code}[/api.status.error], "
        f"Detail={exc.detail}"
    )
    
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail, "request_id": request_id, "error_type": "HTTPException"},
        headers=exc.headers,
    )

@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    request_id = getattr(request.state, "request_id", "N/A")
    
    logger.error(
        f":rotating_light: [danger]Unhandled exception[/danger] [task.id][{request_id}][/task.id]: "
        f"[danger]{type(exc).__name__}[/danger]: {exc}", 
        exc_info=True
    )
    
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "An unexpected internal server error occurred.", "request_id": request_id, "error_type": type(exc).__name__},
    )

# --- API Routers ---
from app.api.router_registry import (
    router_health, router_system, router_transcription, router_diarization
)
# Import route modules AFTER app/handlers defined
from app.api.routes import health, system, transcription, diarization
# Define base dependencies (API key)
base_api_dependencies = [Depends(get_api_key)] if settings.API_AUTH_REQUIRED else []

# Define rate limited dependencies
rate_limited_api_dependencies = base_api_dependencies[:] # Copy base dependencies
if rate_limit_value:
    rate_limited_api_dependencies.append(Depends(limiter.limit(rate_limit_value)))

# Include routers
app.include_router(router_health, prefix="/health") # Health check should not be rate limited
app.include_router(router_system, dependencies=rate_limited_api_dependencies)
app.include_router(router_transcription, dependencies=rate_limited_api_dependencies)
app.include_router(router_diarization, dependencies=rate_limited_api_dependencies)

# --- Metrics Endpoint ---
# Add route for Prometheus metrics
app.add_route("/metrics", handle_metrics)

# --- Root Endpoint ---
@app.get("/", include_in_schema=False)
async def read_root():
    """Redirects root path to API documentation."""
    return RedirectResponse(url="/docs")
# --- Dependency functions moved to app/dependencies.py to avoid circular imports ---