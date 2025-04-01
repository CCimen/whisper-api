#!/usr/bin/env python3
"""
Command-line interface for launching the Whisper Transcription API server.
Uses uvicorn to run the FastAPI application defined in app.main.
"""

import argparse
import logging
import os
import sys
import platform
import uvicorn

# Set up basic logging BEFORE loading the full app/config
# to catch early errors. The main app config will refine this later.
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("whisper-api-cli")

# --- Argument Parsing ---
def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run the Whisper Transcription API Server",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter # Show defaults in help
    )
    # Get defaults from environment or hardcoded values if ENV not set
    default_host = os.environ.get("HOST", "0.0.0.0")
    default_port = int(os.environ.get("PORT", 8000))
    default_workers = int(os.environ.get("WORKERS", 1)) # Default to 1, use Gunicorn for multi-worker prod
    default_log_level = os.environ.get("LOG_LEVEL", "info").lower()
    default_reload = os.environ.get("RELOAD", "False").lower() == "true"

    parser.add_argument(
        "--host", type=str, default=default_host,
        help="Host address to bind the server to. Overrides HOST env var."
    )
    parser.add_argument(
        "--port", type=int, default=default_port,
        help="Port to bind the server to. Overrides PORT env var."
    )
    parser.add_argument(
        "--workers", type=int, default=default_workers,
        help="Number of uvicorn worker processes. Overrides WORKERS env var. (For production, consider Gunicorn/Hypercorn)."
    )
    parser.add_argument(
        "--log-level", type=str, default=default_log_level,
        choices=["debug", "info", "warning", "error", "critical"],
        help="Set the logging level. Overrides LOG_LEVEL env var."
    )
    parser.add_argument(
        "--reload", action='store_true', default=default_reload,
        help="Enable auto-reload for development. Overrides RELOAD env var."
    )
    return parser.parse_args()

# --- Platform Checks ---
def is_running_as_root():
    """Check if the process is running as root/administrator."""
    if platform.system() == "Windows":
        try:
            import ctypes
            # Checks if the process token is elevated
            return ctypes.windll.shell32.IsUserAnAdmin() != 0
        except ImportError:
            logger.warning("Could not import ctypes on Windows to check for admin privileges.")
            return False
        except Exception as e:
            logger.warning(f"Error checking admin status on Windows: {e}")
            return False
    else: # Unix-like systems (Linux, macOS)
        try:
            return os.geteuid() == 0
        except AttributeError:
             logger.warning("Could not check user ID (geteuid not available). Assuming not root.")
             return False # Cannot determine, assume not root


# --- Main Execution ---
def main():
    """Parse arguments and run the API server with uvicorn."""
    args = parse_args()

    # Apply log level from args to the basic logger setup
    log_level_cli = getattr(logging, args.log_level.upper(), logging.INFO)
    logging.getLogger().setLevel(log_level_cli) # Update root logger level

    # Print startup information
    logger.info("=" * 60)
    logger.info("🚀 Launching Whisper Transcription API Server...")
    logger.info(f"   Host:         {args.host}")
    logger.info(f"   Port:         {args.port}")
    logger.info(f"   Workers:      {args.workers} (Note: Use Gunicorn/Hypercorn for multi-worker production)")
    logger.info(f"   Log Level:    {args.log_level.upper()}")
    logger.info(f"   Reload Mode:  {'Enabled (Development)' if args.reload else 'Disabled (Production)'}")
    logger.info("-" * 60)
    logger.info(f"   Platform:     {platform.system()} {platform.release()}")
    logger.info(f"   Python Ver:   {platform.python_version()}")

    # Check if running as root/admin
    if is_running_as_root():
        logger.critical("🚨 Security Warning: Running as root/administrator is strongly discouraged!")

    # Check critical environment variables (optional, config handles defaults/errors)
    # critical_vars = ["HUGGINGFACE_TOKEN"] # Example if diarization is always required
    # for var in critical_vars:
    #     if not os.environ.get(var):
    #         logger.warning(f"Environment variable '{var}' is not set. Required for some features.")

    # --- Run Uvicorn ---
    uvicorn_config = uvicorn.Config(
        "app.main:app", # Path to the FastAPI app instance
        host=args.host,
        port=args.port,
        workers=args.workers if args.workers > 1 else None, # Uvicorn handles single worker differently
        log_level=args.log_level.lower(),
        reload=args.reload,
        # Consider adding interface='asgi3' or 'wsgi3' if needed
        # Use loop='uvloop' for potential performance gains (install uvloop first)
        # loop='uvloop',
        # http='httptools', # Install httptools for performance
    )
    server = uvicorn.Server(config=uvicorn_config)

    try:
        logger.info("=" * 60)
        # Uvicorn will take over logging based on its config now
        server.run()
    except KeyboardInterrupt:
         logger.info("Server stopped by user (KeyboardInterrupt).")
    except Exception as e:
        logger.critical(f"💥 Failed to start or run server: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    # This allows running the script directly (python app/cli.py)
    # The run.py script also calls this main function for compatibility.
    main()