"""
API Endpoint for basic health checks.
"""
from fastapi import APIRouter

# Import the specific router instance for health checks
from app.api.router_registry import router_health

@router_health.get(
    "/", # Typically mounted at /health in main.py
    summary="API Health Check",
    description="A simple endpoint to verify that the API service is running and responsive.",
    tags=["Health"] # Match tag defined in registry
)
async def health_check():
    """
    Returns a simple 'ok' status if the API is running.
    """
    return {"status": "ok", "message": "Whisper Transcription API is healthy"}

# Potential extensions:
# - Check database connectivity (if used)
# - Check dependency service status (if any)
# - Check basic model loading status (e.g., can ModelRegistry list models?)