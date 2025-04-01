"""
API Endpoints for system status, monitoring, and configuration information.
"""

import logging
import asyncio
import torch
from fastapi import Depends, HTTPException, status, APIRouter
from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional

from app.api.router_registry import router_system # Use central registry
from app.config import settings, SYSTEM_CAPABILITIES # Use settings and detected caps
# Correctly import ModelNotFoundError from app.exceptions
from app.exceptions import ModelNotFoundError
from app.services.model_registry import ModelRegistry # Keep ModelRegistry import
from app.services.task_manager import task_manager

# Import security dependency
from app.main import get_api_key # Import from main where it's defined

# Configure logging
logger = logging.getLogger(__name__)

# Import diarization availability safely
try:
    from app.services.diarization import DIARIZATION_AVAILABLE
except ImportError:
    DIARIZATION_AVAILABLE = False
except Exception as e:
     logger.error(f"Failed to check DIARIZATION_AVAILABLE in system route: {e}")
     DIARIZATION_AVAILABLE = False


# --- Pydantic Models ---

class GPUDeviceInfo(BaseModel):
    id: int
    name: str
    memory_total_gb: float
    memory_free_gb: float
    memory_used_gb: float
    utilization_pct: float
    compute_capability: Optional[str] = None
    multi_processor_count: Optional[int] = None
    error: Optional[str] = None # Field to indicate issues getting full details


class GPUStatus(BaseModel):
    """Detailed GPU status information."""
    available: bool
    details: Optional[str] = None # Add details field for unavailable status
    device_count: int = 0
    devices: List[GPUDeviceInfo] = []
    active_device_id: Optional[int] = None


class DiarizationStatus(BaseModel):
    enabled_in_config: bool
    dependencies_available: bool
    huggingface_token_set: bool


class QueueStatus(BaseModel):
    queued_tasks: int
    active_tasks: int
    max_concurrent_tasks: int
    is_processing: bool

class ModelInfo(BaseModel):
     name: str
     type: Optional[str] = None
     loaded: bool
     size: Optional[str] = None
     estimated_memory_gb: Optional[float] = None
     loaded_device: Optional[str] = None
     error: Optional[str] = None


class SystemStatusResponse(BaseModel):
    """Comprehensive API status and capability information."""
    status: str = "ok"
    app_name: str = settings.APP_NAME
    app_version: str = settings.APP_VERSION
    gpu_status: GPUStatus
    diarization_status: DiarizationStatus
    model_status: Dict[str, ModelInfo] # Dictionary of models
    queue_status: QueueStatus
    default_model: str = settings.DEFAULT_MODEL
    max_upload_size_mb: int = settings.MAX_UPLOAD_SIZE_MB


# --- Helper Functions ---

def get_gpu_status() -> GPUStatus:
    """Checks GPU availability and retrieves detailed information."""
    # Check if CUDA is enabled in settings and PyTorch is available
    if not settings.USE_CUDA or not torch:
        return GPUStatus(available=False, details="CUDA disabled in settings or PyTorch not found.")

    if not torch.cuda.is_available():
        return GPUStatus(available=False, details="torch.cuda.is_available() returned False.")

    devices_info = []
    device_count = 0
    active_device_id = None
    try:
        device_count = torch.cuda.device_count()
        if device_count == 0:
             return GPUStatus(available=False, device_count=0, details="torch.cuda.device_count() returned 0.")

        active_device_id = torch.cuda.current_device()

        for i in range(device_count):
            try:
                props = torch.cuda.get_device_properties(i)
                # Prefer mem_get_info for more accurate free memory if available
                if hasattr(torch.cuda, 'mem_get_info'):
                     free_mem, total_mem_actual = torch.cuda.mem_get_info(i)
                     total_mem_prop = props.total_memory # Can sometimes differ slightly
                     total_gb = total_mem_prop / (1024**3) # Use property total for consistency
                     free_gb = free_mem / (1024**3)
                     used_gb = total_gb - free_gb
                else: # Fallback if mem_get_info not available
                     total_gb = props.total_memory / (1024**3)
                     # Cannot accurately get free memory without mem_get_info
                     free_gb = -1.0 # Indicate unavailable
                     used_gb = -1.0

                util_pct = round((used_gb / total_gb) * 100, 1) if total_gb > 0 and used_gb >= 0 else 0.0

                devices_info.append(GPUDeviceInfo(
                    id=i,
                    name=props.name,
                    memory_total_gb=round(total_gb, 2),
                    memory_free_gb=round(free_gb, 2),
                    memory_used_gb=round(used_gb, 2),
                    utilization_pct=util_pct,
                    compute_capability=f"{props.major}.{props.minor}",
                    multi_processor_count=props.multi_processor_count
                ))
            except Exception as e:
                logger.error(f"Could not get full properties for GPU device {i}: {e}")
                try:
                     # Attempt to add basic info even if properties fail
                     devices_info.append(GPUDeviceInfo(
                          id=i, name=torch.cuda.get_device_name(i),
                          memory_total_gb=0.0, memory_free_gb=0.0, memory_used_gb=0.0, utilization_pct=0.0, # Indicate failure
                          error=f"Failed to get full details: {e}"
                     ))
                except Exception as basic_e:
                     logger.error(f"Could not even get basic name for GPU device {i}: {basic_e}")
                     # Append placeholder if even name fails
                     devices_info.append(GPUDeviceInfo(
                          id=i, name=f"GPU {i} (Error)",
                          memory_total_gb=0.0, memory_free_gb=0.0, memory_used_gb=0.0, utilization_pct=0.0,
                          error=f"Failed to get any details: {basic_e}"
                     ))

    except Exception as e:
        logger.error(f"Failed to query CUDA devices: {e}", exc_info=True)
        return GPUStatus(available=False, details=f"Error querying CUDA devices: {e}")


    return GPUStatus(
        available=True,
        device_count=device_count,
        devices=devices_info,
        active_device_id=active_device_id
    )

# --- Background Task for Loading Models ---
# This function is used by the load endpoint and potentially by main.py for preload
async def _load_model_background(model_key: str):
    """Loads a model in the background without blocking the request."""
    # Check if task manager is available before proceeding
    if not task_manager:
        logger.error(f"[Background Load Task] TaskManager not available. Cannot load model {model_key}.")
        return

    logger.info(f"[Background Load Task] Starting load for {model_key}")
    try:
        model = ModelRegistry.get_model(model_key) # Instantiates if needed
        if not model.is_loaded():
            device = "cuda" if settings.USE_CUDA and torch and torch.cuda.is_available() else "cpu"
            # Run load within the task manager's executor if available?
            # Or just use asyncio.to_thread for simplicity here.
            await asyncio.to_thread(model.load, device=device)
            logger.info(f"[Background Load Task] Successfully loaded model: {model_key}")
        else:
            logger.info(f"[Background Load Task] Model {model_key} was already loaded.")
    except ModelNotFoundError:
         logger.error(f"[Background Load Task] Model {model_key} not found in registry.")
    except Exception as e:
        logger.error(f"[Background Load Task] Failed to load model {model_key}: {e}", exc_info=True)

# --- API Endpoints ---

@router_system.get(
    "/status",
    response_model=SystemStatusResponse,
    summary="Get Overall System Status",
    description="Provides a comprehensive overview of the API's status, including GPU, diarization, models, and task queue information."
)
async def get_system_status(
    # API Key Dependency applied at the router level in main.py
    # _: bool = Depends(get_api_key) # Keep if applying per-route
):
    """
    Get the overall status of the API, including hardware and configuration.
    """
    gpu_stat = get_gpu_status()
    diar_stat = DiarizationStatus(
        enabled_in_config=settings.DIARIZATION_ENABLED,
        dependencies_available=DIARIZATION_AVAILABLE,
        huggingface_token_set=bool(settings.HUGGINGFACE_TOKEN)
    )
    # Check if task_manager is initialized before calling
    q_stat_dict = task_manager.get_queue_status() if task_manager else {
        "queued_tasks": 0, "active_tasks": 0, "max_concurrent_tasks": settings.MAX_CONCURRENT_TASKS, "is_processing": False
    }
    q_stat = QueueStatus(**q_stat_dict) # Unpack dict into model
    model_stat = ModelRegistry.get_model_info() # Returns dict directly

    return SystemStatusResponse(
        gpu_status=gpu_stat,
        diarization_status=diar_stat,
        model_status=model_stat,
        queue_status=q_stat,
        # Other fields use defaults from model definition or settings
    )


@router_system.get(
    "/gpu",
    response_model=GPUStatus,
    summary="Get Detailed GPU Status",
    description="Retrieves detailed information about available GPU(s), including memory usage and capabilities."
)
async def get_gpu_details(
    # _: bool = Depends(get_api_key)
):
    """
    Get detailed information specifically about the available GPU(s).
    """
    return get_gpu_status()


@router_system.get(
    "/models",
    response_model=Dict[str, ModelInfo],
    summary="List Available Models",
    description="Lists all transcription models configured in the system and their current load status."
)
async def list_available_models(
    # _: bool = Depends(get_api_key)
):
    """
    List all available transcription models and their current status.
    """
    try:
        return ModelRegistry.get_model_info()
    except Exception as e:
        logger.error(f"Failed to get model info: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to retrieve model information.")


@router_system.post(
    "/models/{model_name}/load",
    status_code=status.HTTP_202_ACCEPTED,
    summary="Load a Model",
    description="Requests the server to load a specified model into memory asynchronously. Check status later."
)
async def load_model_on_demand(
    model_name: str
    # _: bool = Depends(get_api_key)
):
    """
    Request the server to load a specific model into memory. (Asynchronous)
    """
    available_models = ModelRegistry.available_models()
    if model_name not in available_models:
         raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Model '{model_name}' not available. Available: {available_models}")

    try:
         # Get model instance (this might instantiate it but not load it)
         model = ModelRegistry.get_model(model_name)
         if model.is_loaded():
              logger.info(f"Model '{model_name}' is already loaded.")
              return {"message": f"Model '{model_name}' is already loaded."}

         logger.info(f"Received request to load model '{model_name}'. Initiating background load...")
         # Trigger background load
         asyncio.create_task(_load_model_background(model_name))
         return {"message": f"Background loading initiated for model '{model_name}'. Check /system/models endpoint for status."}

    except ModelNotFoundError as e: # Catch specific error from get_model
         raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except Exception as e:
         logger.error(f"Failed to initiate load for model '{model_name}': {e}", exc_info=True)
         raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Could not initiate load for model '{model_name}': {e}")


@router_system.post(
    "/models/{model_name}/unload",
    status_code=status.HTTP_200_OK,
    summary="Unload a Model",
    description="Requests the server to unload a specific model from memory to free resources."
)
async def unload_model_on_demand(
    model_name: str
    # _: bool = Depends(get_api_key)
):
    """
    Request the server to unload a specific model from memory.
    """
    # Check if the model *could* exist, even if not instantiated/loaded
    available_models = ModelRegistry.available_models()
    if model_name not in available_models:
         raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Model '{model_name}' is not recognized by the registry.")

    try:
        # Check internal instance dict directly to see if it's even instantiated
        # Use ModelRegistry method if available, otherwise access _instances (less ideal)
        # Ensure ModelRegistry._instances is accessible or provide a method
        instance = ModelRegistry._instances.get(model_name) # Assuming direct access for check
        if not instance or not instance.is_loaded():
             logger.info(f"Model '{model_name}' is not currently loaded.")
             return {"message": f"Model '{model_name}' is not currently loaded."}

        logger.info(f"Received request to unload model '{model_name}'. Initiating unload...")
        # Run unload in a thread to avoid blocking the API request if unload is slow
        await asyncio.to_thread(instance.unload)

        # Verify unload and remove from instances dict (still need lock if multithreaded access to _instances)
        # For simplicity here, we assume unload worked and remove. Registry should handle internally if possible.
        if model_name in ModelRegistry._instances:
            # Ideally, ModelRegistry would have an unload_instance method
             del ModelRegistry._instances[model_name]

        logger.info(f"Model '{model_name}' successfully unloaded.")
        return {"message": f"Model '{model_name}' successfully unloaded."}

    except Exception as e:
        logger.error(f"Failed to unload model '{model_name}': {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Could not unload model '{model_name}': {e}")


@router_system.get(
    "/queue",
    response_model=QueueStatus,
    summary="Get Task Queue Status",
    description="Retrieves the current status of the task processing queue, including active and pending tasks."
)
async def get_task_queue_status(
    # _: bool = Depends(get_api_key)
):
    """
    Get the current status of the task processing queue.
    """
    if not task_manager:
         raise HTTPException(status_code=503, detail="TaskManager is not available.")
    q_status_dict = task_manager.get_queue_status()
    return QueueStatus(**q_status_dict)