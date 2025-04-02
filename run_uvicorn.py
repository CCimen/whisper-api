import os
import logging.config
import uvicorn

# Import the configuration dictionary
try:
    from app.logging_config import LOGGING_CONFIG
except ImportError as e:
    print(f"Error: Could not import LOGGING_CONFIG from app/logging_config.py: {e}")
    print("Ensure logging_config.py exists and is valid.")
    exit(1)
except Exception as e:
    print(f"Error loading logging_config.py: {e}")
    exit(1)

# Apply the logging configuration from the dictionary
try:
    logging.config.dictConfig(LOGGING_CONFIG)
    # Get a logger instance AFTER config is applied
    logger = logging.getLogger(__name__)
    logger.info("Logging configured successfully from app/logging_config.py using dictConfig.")
except Exception as e:
    # Fallback basic config if dictConfig fails
    logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO").upper())
    logger = logging.getLogger(__name__)
    logger.error(f"Failed to apply logging config from dictionary: {e}. Falling back to basic config.", exc_info=True)


if __name__ == "__main__":
    # Get Uvicorn settings from environment or defaults
    # These should match the previous CMD line arguments
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "8000"))
    workers = int(os.environ.get("WORKERS", "4")) # Keep 4 workers
    use_uvloop = os.environ.get("LOOP", "uvloop").lower() == "uvloop"
    use_proxy_headers = os.environ.get("PROXY_HEADERS", "true").lower() == "true"

    logger.info(f"Starting Uvicorn: host={host}, port={port}, workers={workers}, loop={'uvloop' if use_uvloop else 'asyncio'}, proxy_headers={use_proxy_headers}")

    uvicorn.run(
        "app.main:app", # App import string
        host=host,
        port=port,
        workers=workers,
        loop="uvloop" if use_uvloop else "asyncio",
        proxy_headers=use_proxy_headers,
        # No log_config here, as we applied it above
        # Uvicorn will use the already configured logging setup
    )