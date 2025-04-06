import logging
from fastapi import Request, HTTPException, status
from typing import Optional # Import Optional
from app.config import settings # Import settings
# Import the actual classes for type hinting and checking
from app.services.task_manager import TaskManager
from app.services.model_registry import ModelRegistry
from app.services.diarization import DiarizationService, DIARIZATION_AVAILABLE # Import DiarizationService

logger = logging.getLogger(__name__)

# --- Dependency for accessing TaskManager ---
async def get_task_manager(request: Request) -> TaskManager:
    """Dependency function to get the TaskManager instance from app state."""
    if not hasattr(request.app.state, 'task_manager') or not request.app.state.task_manager:
        # This should ideally not happen if lifespan completes successfully
        logger.error("TaskManager not found in application state during request.")
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Task processing service is not available.")

    # Ensure the object retrieved is actually a TaskManager instance
    tm = request.app.state.task_manager
    if not isinstance(tm, TaskManager):
         logger.error(f"Object in app.state.task_manager is not a TaskManager instance (Type: {type(tm)}).")
         raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Task processing service is misconfigured.")
    return tm

# --- Dependency for accessing ModelRegistry ---
async def get_model_registry(request: Request) -> ModelRegistry:
    """Dependency function to get the ModelRegistry instance from app state."""
    if not hasattr(request.app.state, 'model_registry') or not request.app.state.model_registry:
        logger.error("ModelRegistry not found in application state during request.")
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Model registry service is not available.")

    # Ensure the object retrieved is actually a ModelRegistry instance
    mr = request.app.state.model_registry
    if not isinstance(mr, ModelRegistry):
         logger.error(f"Object in app.state.model_registry is not a ModelRegistry instance (Type: {type(mr)}).")
         raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Model registry service is misconfigured.")
    return mr

# --- Dependency for accessing DiarizationService ---
async def get_diarization_service(request: Request) -> Optional[DiarizationService]:
    """
    Dependency function to get the DiarizationService instance from app state.
    Returns None if diarization is disabled or unavailable.
    """
    if not settings.DIARIZATION_ENABLED or not DIARIZATION_AVAILABLE:
        # Return None or raise specific exception if caller expects it?
        # Returning None allows callers to gracefully handle disabled feature.
        return None

    if not hasattr(request.app.state, 'diarization_service') or not request.app.state.diarization_service:
        logger.error("DiarizationService enabled but not found in application state during request.")
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Diarization service is not available or misconfigured.")

    ds = request.app.state.diarization_service
    if not isinstance(ds, DiarizationService):
         logger.error(f"Object in app.state.diarization_service is not a DiarizationService instance (Type: {type(ds)}).")
         raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Diarization service is misconfigured.")
    return ds