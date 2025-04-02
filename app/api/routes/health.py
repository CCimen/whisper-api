"""
API Endpoint for basic health checks.
"""
import logging
import torch
from fastapi import APIRouter, Response, status

# Import the specific router instance for health checks
from app.api.router_registry import router_health
# Import necessary components for checks
from app.config import settings
from app.services.model_registry import ModelRegistry
from app.services.diarization import diarization_service, DIARIZATION_AVAILABLE

logger = logging.getLogger(__name__)

@router_health.get(
    "/", # Typically mounted at /health in main.py
    summary="API Health Check",
    description="A simple endpoint to verify that the API service is running and responsive.",
    tags=["Health"] # Match tag defined in registry
)
async def health_check(response: Response):
    """
    Performs health checks on critical components and returns the status.
    Returns 503 Service Unavailable if a critical check fails.
    """
    healthy = True
    checks = {
        "api_responsive": True,
        "model_registry_accessible": False,
        "cuda_status": "not_checked", # not_checked, ok, error, disabled
        "diarization_status": "not_checked" # not_checked, ok, error, disabled
    }

    # 1. Check Model Registry
    try:
        available_models = ModelRegistry.available_models()
        if available_models:
            checks["model_registry_accessible"] = True
            checks["available_models"] = available_models
        else:
            checks["model_registry_accessible"] = False
            healthy = False # Critical if no models can be listed
            logger.warning("[Health Check] ModelRegistry accessible but no models found.")
    except Exception as e:
        logger.error(f"[Health Check] Failed to access ModelRegistry: {e}", exc_info=True)
        checks["model_registry_accessible"] = False
        healthy = False # Critical failure

    # 2. Check CUDA Status (if enabled)
    if settings.USE_CUDA:
        try:
            if torch and torch.cuda.is_available():
                checks["cuda_status"] = "ok"
                checks["cuda_devices"] = torch.cuda.device_count()
            else:
                checks["cuda_status"] = "error"
                healthy = False # Critical if CUDA is expected but not available
                logger.warning("[Health Check] CUDA enabled in settings but torch.cuda.is_available() is False.")
        except Exception as e:
            logger.error(f"[Health Check] Error checking CUDA status: {e}", exc_info=True)
            checks["cuda_status"] = "error"
            healthy = False # Treat error as critical failure
    else:
        checks["cuda_status"] = "disabled"

    # 3. Check Diarization Status (if enabled)
    if settings.DIARIZATION_ENABLED:
        if DIARIZATION_AVAILABLE and diarization_service is not None:
            # Could add a lightweight check on the service instance if needed
            checks["diarization_status"] = "ok"
        else:
            checks["diarization_status"] = "error"
            # Decide if this is critical. Maybe not if transcription still works?
            # For now, let's consider it non-critical for overall health, but log it.
            logger.warning("[Health Check] Diarization enabled but service/dependencies are not available.")
            # healthy = False # Optional: Make this critical if needed
    else:
        checks["diarization_status"] = "disabled"

    # Set final status code
    if not healthy:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return {"status": "unhealthy", "checks": checks}
    else:
        response.status_code = status.HTTP_200_OK
        return {"status": "ok", "checks": checks}

# Potential extensions:
# - Check database connectivity (if used)
# - Check dependency service status (if any)
# - Check basic model loading status (e.g., can ModelRegistry list models?)