# logging_config.py
import os
import logging.config
from rich.logging import RichHandler
from rich.console import Console
from rich.theme import Theme

# --- Constants copied from app/main.py ---
# Define custom theme for the console - using color-blind friendly palette
CUSTOM_THEME = Theme({
    # General message types
    "info": "bold cyan", "warning": "bold yellow", "danger": "bold bright_red", "success": "bold bright_green",
    # Application components
    "app.title": "bold bright_magenta", "app.subtitle": "italic cyan",
    # Models
    "model.name": "bold bright_blue", "model.tiny": "bold blue", "model.small": "bold cyan",
    "model.medium": "bold blue on black", "model.large": "bold bright_blue on black", "model.base": "bold cyan on black",
    "model.status": "bright_green", "model.loading": "yellow", "model.ready": "bold bright_green",
    # API related
    "api.request": "bold cyan", "api.response": "cyan", "api.method": "bold bright_green",
    "api.status.success": "bright_green", "api.status.error": "bright_red", "api.duration": "bold bright_white",
    "api.url": "underline cyan",
    # Configuration
    "config.key": "bold bright_blue", "config.value": "bold bright_white", "config.enabled": "bold bright_green",
    "config.disabled": "italic yellow",
    # Task management
    "task.id": "bold bright_yellow", "task.status": "bold bright_white", "task.status.change": "bold bright_yellow",
    "taskmgr": "bold bright_blue", "taskr": "bold blue", "processor": "bold bright_magenta",
    # Log components
    "log.timestamp": "dim white", "log.scope": "bold bright_blue",
    # File operations
    "file.path": "underline bright_white", "file.size": "bold bright_white", "file.duration": "italic bright_white",
})

# Define keywords to highlight in logs
LOG_KEYWORDS = [
    "Status changed", "queued", "preparing", "processing", "completed", "failed",
    "whisper-kblab-tiny", "whisper-kblab-small", "whisper-kblab-medium", "whisper-kblab-large",
    "whisper-base", "whisper-openai-large-v3",
    "[MODEL]", "[PROCESSOR]", "[TASKMGR]", "[TASKR]", "[Preload Task]",
    "Duration=", "bytes)", "factor:", "Segments:", "Speakers:",
    "GET /health", "POST /transcriptions", "Request started", "Request finished"
]
# --- End Constants ---

# Read log level from environment variable, default to INFO
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()
# Read debug mode for tracebacks (assuming settings.DEBUG maps to this)
DEBUG_MODE = os.environ.get("DEBUG", "False").lower() == "true"

# Create console for RichHandler
# Ensure force_terminal=True for Docker environments
force_console = Console(
    force_terminal=True,
    width=120,
    theme=CUSTOM_THEME,
    highlight=True # Enable Rich's highlighting
)

# Define the logging configuration dictionary
LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False, # Don't disable loggers like uvicorn
    "formatters": {
        "default": { # Uvicorn default formatter
            "()": "uvicorn.logging.DefaultFormatter",
            "fmt": "%(levelprefix)s %(message)s",
            "use_colors": None,
        },
        "access": { # Uvicorn access formatter
            "()": "uvicorn.logging.AccessFormatter",
            "fmt": '%(levelprefix)s %(client_addr)s - "%(request_line)s" %(status_code)s',
        },
    },
    "handlers": {
        "rich_default": { # Use Rich handler for general console output
            "class": "rich.logging.RichHandler",
            "level": LOG_LEVEL,
            "console": force_console, # Use pre-configured console
            "rich_tracebacks": True,
            "tracebacks_show_locals": DEBUG_MODE,
            "markup": True,
            "show_path": False, # Keep logs cleaner
            "log_time_format": "[%X]",
            "omit_repeated_times": False, # Show time for each log entry
            "keywords": LOG_KEYWORDS, # Use defined keywords
            # RichHandler uses its own formatting based on level/markup
        },
        "uvicorn_access": { # Standard handler for Uvicorn access logs
            "formatter": "access",
            "class": "logging.StreamHandler",
            "stream": "ext://sys.stdout", # Use stdout for access logs
        },
    },
    "loggers": {
        # Root logger: Configure to use the Rich handler
        "": {
            "handlers": ["rich_default"],
            "level": LOG_LEVEL,
            "propagate": False, # Don't propagate to higher level (none)
        },
        # Uvicorn error logger: Use Rich handler
        "uvicorn.error": {
            "level": LOG_LEVEL, # Match root level
            "handlers": ["rich_default"],
            "propagate": False,
        },
        # Uvicorn access logger: Use specific access handler
        "uvicorn.access": {
            "handlers": ["uvicorn_access"],
            "level": "INFO", # Access logs are typically INFO
            "propagate": False,
        },
        # Optional: Configure specific app loggers if needed
        # "app": {
        #     "handlers": ["rich_default"],
        #     "level": LOG_LEVEL,
        #     "propagate": False,
        # },
    },
}

# Note: Audit logging setup remains in app/main.py lifespan
# because it needs access to settings and might need to create directories.