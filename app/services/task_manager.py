"""
Task Manager with enhanced privacy and security features.

Manages asynchronous execution of transcription and diarization tasks,
handling queuing, resource limits, retries, status tracking, and cleanup.
Focuses on secure file handling and memory management.
"""

import asyncio
import time
import uuid
import logging
import threading
import os
import shutil
import secrets
import gc
from typing import Dict, Any, Optional, List, Callable, Awaitable, Deque
from enum import Enum
from collections import deque
import weakref # Use weakref for tasks if memory becomes an issue

import torch # For memory cleanup

# Configure logging
logger = logging.getLogger(__name__)

# Import settings and exceptions safely
try:
    from app.config import settings
    from app.exceptions import ConfigurationError
except ImportError:
    # This allows the module to be imported even if config isn't fully set up yet
    # Handle missing settings gracefully later.
    logger.error("Could not import app.config or app.exceptions. Task Manager may not function correctly.")
    settings = None # Indicate missing settings


class TaskStatus(str, Enum):
    """Detailed task status for better tracking."""
    PENDING = "pending"      # Task created but not yet queued (or ready for queue)
    QUEUED = "queued"        # Task in queue waiting for worker slot
    PREPARING = "preparing"  # Worker starting up, initial checks
    LOADING_MODEL = "loading_model" # Downloading/loading the required model
    PROCESSING = "processing"# Actively running the core logic (transcription/diarization)
    COMPLETING = "completing"# Final steps after core logic (e.g., speaker assignment)
    COMPLETED = "completed"  # Task finished successfully
    FAILED = "failed"        # Task failed with an error
    CANCELLED = "cancelled"  # Task was cancelled by user or system

    @classmethod
    def is_terminal(cls, status: "TaskStatus") -> bool:
        """Check if a status is final (no further updates expected)."""
        return status in (cls.COMPLETED, cls.FAILED, cls.CANCELLED)

    @classmethod
    def is_active(cls, status: "TaskStatus") -> bool:
        """Check if a status means the task is currently being worked on."""
        return status in (cls.PREPARING, cls.LOADING_MODEL, cls.PROCESSING, cls.COMPLETING)


class Task:
    """
    Represents an asynchronous task with state, progress, results, and cleanup management.
    """
    def __init__(
        self,
        task_id: str,
        task_type: str,
        params: Dict[str, Any],
        max_retries: int = 0
    ):
        if not settings:
            raise ConfigurationError("Settings module not loaded. Task cannot be initialized.")

        self.id: str = task_id
        self.type: str = task_type
        self.params: Dict[str, Any] = params
        self.status: TaskStatus = TaskStatus.PENDING
        self.progress: float = 0.0
        self.result: Optional[Any] = None
        self.error: Optional[str] = None
        self.created_at: float = time.time()
        self.started_at: Optional[float] = None
        self.completed_at: Optional[float] = None
        self.max_retries: int = max_retries
        self.retry_count: int = 0
        self.queue_position: Optional[int] = None
        self.additional_info: Dict[str, Any] = {}
        self._files_to_clean: List[str] = [] # Track files for secure deletion

        # Register initial input file for cleanup if present
        input_file = params.get('file_path')
        if input_file:
             self.register_file_for_cleanup(input_file)

    def to_dict(self, include_result: bool = True) -> Dict[str, Any]:
        """Convert task state to a dictionary for API responses."""
        data = {
            "id": self.id,
            "type": self.type,
            "status": self.status.value, # Use enum value
            "progress": round(self.progress, 3),
            "error": self.error,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "queue_position": self.queue_position,
            "additional_info": self.additional_info,
        }
        if include_result:
            data["result"] = self.result # Include result by default
        return data

    def update_progress(self, status: Optional[TaskStatus] = None, progress: Optional[float] = None,
                       info: Optional[Dict[str, Any]] = None, error: Optional[str] = None,
                       result: Optional[Any] = None) -> None:
        """Update task state atomically (within TaskManager's lock)."""
        now = time.time()
        if status is not None and self.status != status:
            # Prevent moving *back* from a terminal state unless retrying
            if TaskStatus.is_terminal(self.status) and status != TaskStatus.QUEUED:
                logger.warning(f"[TASK][{self.id}] Ignoring status update from {self.status} to {status}")
                return

            logger.info(f"[TASK][{self.id}] Status changed {self.status.value} -> {status.value}")
            self.status = status

            if TaskStatus.is_active(status) and self.started_at is None:
                self.started_at = now
            elif TaskStatus.is_terminal(status) and self.completed_at is None:
                self.completed_at = now
                self.queue_position = None # No longer in queue

        if progress is not None:
            # Ensure progress doesn't go backward unless status resets
            if status == self.status or status is None: # Only update progress if status is same or not changing
                 self.progress = max(self.progress, progress) # Avoid progress going down
            else: # Status changed, allow progress reset
                 self.progress = progress
            self.progress = min(max(0.0, self.progress), 1.0) # Clamp between 0 and 1

        if info is not None:
            self.additional_info.update(info)

        if error is not None:
            self.error = error

        if result is not None:
            self.result = result
            # Register output files if result contains paths
            if isinstance(result, dict):
                 output_file = result.get('output_file_path') # Example key
                 if output_file: self.register_file_for_cleanup(output_file)


    def register_file_for_cleanup(self, file_path: str) -> None:
        """Register a file path for secure deletion upon task completion/deletion."""
        if not settings: return # Cannot function without settings

        try:
            if not file_path or not isinstance(file_path, str):
                 logger.warning(f"[TASK][{self.id}] Invalid file path provided for cleanup: {file_path}")
                 return

            abs_file_path = os.path.abspath(file_path)

            # Security check: Ensure file is within designated temp/upload areas
            allowed_dirs = [
                os.path.abspath(settings.UPLOAD_DIR),
                os.path.abspath(settings.RESULTS_DIR)
            ]
            # Add system temp dir as a fallback? Be cautious.
            # import tempfile
            # allowed_dirs.append(os.path.abspath(tempfile.gettempdir()))

            if not any(abs_file_path.startswith(allowed_dir) for allowed_dir in allowed_dirs):
                logger.error(f"[TASK][{self.id}] Attempted to register file outside allowed directories for cleanup: {abs_file_path}")
                return # Do not register potentially unsafe paths

            if os.path.exists(abs_file_path) and abs_file_path not in self._files_to_clean:
                self._files_to_clean.append(abs_file_path)
                logger.debug(f"[TASK][{self.id}] Registered file for cleanup: {abs_file_path}")

        except Exception as e:
            logger.error(f"[TASK][{self.id}] Error registering file {file_path} for cleanup: {e}")

    def perform_cleanup(self) -> None:
        """Securely delete all registered files associated with this task."""
        if not settings or not settings.AUTO_DELETE_AFTER_COMPLETION:
            self._files_to_clean.clear() # Clear list even if not deleting
            return

        if not self._files_to_clean:
            return

        # Process a copy of the list FIRST, before logging
        files_to_process = list(self._files_to_clean)
        logger.info(f"[TASK][{self.id}] Performing secure cleanup of {len(files_to_process)} registered file(s)...")
        self._files_to_clean.clear() # Clear original list immediately

        for file_path in files_to_process:
            self._secure_delete_file(file_path)

    def _secure_delete_file(self, file_path: str) -> None:
        """Securely delete a single file."""
        try:
            if not os.path.exists(file_path) or not os.path.isfile(file_path):
                logger.debug(f"[TASK][{self.id}] File not found or not a file, skipping delete: {file_path}")
                return

            file_size = os.path.getsize(file_path)
            if file_size == 0:
                os.remove(file_path)
                logger.info(f"[TASK][{self.id}] Removed empty file: {file_path}")
                return

            # Secure wipe if enabled in settings (can be slow)
            perform_wipe = getattr(settings, 'SECURE_FILE_WIPING', False) # Add this setting if needed

            if perform_wipe:
                 logger.debug(f"[TASK][{self.id}] Securely wiping file: {file_path} ({file_size} bytes)")
                 # Overwrite with random data, then zeros
                 with open(file_path, "wb") as f:
                     # Overwrite 1: Random data
                     random_data = secrets.token_bytes(file_size)
                     f.seek(0)
                     f.write(random_data)
                     f.flush()
                     os.fsync(f.fileno())

                     # Overwrite 2: Zeros
                     zeros = bytes(file_size)
                     f.seek(0)
                     f.write(zeros)
                     f.flush()
                     os.fsync(f.fileno())

                     # Truncate (optional, but good practice)
                     f.truncate(0)

            # Final removal
            os.remove(file_path)
            logger.info(f"[TASK][{self.id}] Successfully deleted file: {file_path}{' (wiped)' if perform_wipe else ''}")

        except Exception as e:
            logger.error(f"[TASK][{self.id}] Error during secure file deletion for {file_path}: {e}", exc_info=True)
            # Attempt standard removal as fallback if secure wipe failed but file might still exist
            try:
                 if os.path.exists(file_path):
                      os.remove(file_path)
                      logger.warning(f"[TASK][{self.id}] Fell back to standard deletion for: {file_path}")
            except Exception as fallback_e:
                 logger.error(f"[TASK][{self.id}] Failed standard deletion fallback for {file_path}: {fallback_e}")


class TaskManager:
    """
    Manages asynchronous task execution with queuing, resource limits, and cleanup.
    """
    def __init__(self):
        if not settings:
            raise ConfigurationError("Settings module not loaded. TaskManager cannot be initialized.")

        self.tasks: Dict[str, Task] = {} # Store actual Task objects
        self.max_concurrent_tasks: int = settings.MAX_CONCURRENT_TASKS
        self._active_workers: Dict[str, asyncio.Task] = {} # task_id -> asyncio.Task handle
        self._task_queue: Deque[str] = deque() # Queue of task IDs
        self._queue_lock = asyncio.Lock() # Lock for queue modifications
        self._task_lock = threading.RLock() # Lock for self.tasks dictionary access
        self._queue_processor_task: Optional[asyncio.Task] = None
        self._shutdown_event = asyncio.Event()
        self._handlers: Dict[str, Callable[..., Awaitable[Any]]] = {} # Registered task handlers

        self._register_handlers()
        self._ensure_directories_secure()
        logger.info(f"[TASKMGR] TaskManager initialized. Max concurrent tasks: {self.max_concurrent_tasks}")


    def _ensure_directories_secure(self) -> None:
        """Ensure required directories exist with secure permissions (called by config now)."""
        # This logic is moved to config.py's create_secure_directories
        # to ensure it runs early. Keep this stub in case it's needed later.
        pass


    def _register_handlers(self):
        """Dynamically register task handlers."""
        try:
            from app.services.processor import process_audio, handle_diarization_only
            self._handlers['transcription'] = process_audio
            self._handlers['diarization_only'] = handle_diarization_only
            logger.info(f"[TASKMGR] Registered task handlers: {list(self._handlers.keys())}")
        except ImportError as e:
            logger.error(f"[TASKMGR] Failed to import task handlers: {e}. Tasks cannot be processed.")
        except Exception as e:
             logger.error(f"[TASKMGR] Unexpected error registering handlers: {e}", exc_info=True)


    def create_task(self, task_type: str, params: Dict[str, Any]) -> str:
        """Create a new task and return its ID."""
        task_id = str(uuid.uuid4())
        effective_max_retries = settings.MAX_RETRIES if settings.RETRY_FAILED_TASKS else 0

        with self._task_lock:
            if task_type not in self._handlers:
                raise ValueError(f"No handler registered for task type: '{task_type}'")

            task = Task(task_id, task_type, params, effective_max_retries)
            self.tasks[task_id] = task # Store the Task object

        logger.info(f"[TASKMGR] Created task {task_id} (Type: {task_type}). Initial status: {task.status.value}")
        return task_id

    async def queue_task(self, task_id: str) -> None:
        """Add a task to the processing queue."""
        async with self._queue_lock:
             with self._task_lock:
                 task = self.tasks.get(task_id)
                 if not task:
                     logger.error(f"[TASKMGR] Task {task_id} not found for queuing.")
                     raise ValueError(f"Task {task_id} not found for queuing.")
                 if TaskStatus.is_terminal(task.status):
                     logger.warning(f"[TASKMGR] Task {task_id} is already in terminal state {task.status}. Cannot queue.")
                     return
                 if task.status == TaskStatus.QUEUED or task_id in self._task_queue:
                      logger.warning(f"[TASKMGR] Task {task_id} is already queued. Ignoring.")
                      return
                 if task_id in self._active_workers:
                      logger.warning(f"[TASKMGR] Task {task_id} is already active. Cannot queue.")
                      return

                 # Update status and add to queue
                 task.update_progress(status=TaskStatus.QUEUED)
                 self._task_queue.append(task_id)
                 self._update_queue_positions_nolock() # Update positions within lock

                 logger.info(f"[TASKMGR] Task {task_id} queued at position {task.queue_position}. Total queue size: {len(self._task_queue)}")

        # Ensure the queue processor is running
        self._start_queue_processor_if_needed()


    def _start_queue_processor_if_needed(self):
         """Starts the background queue processor task if it's not running."""
         # Check if processor is running or if shutdown is initiated
         if (self._queue_processor_task is None or self._queue_processor_task.done()) and \
            not self._shutdown_event.is_set():
              logger.info("[TASKMGR] Starting queue processor background task.")
              self._queue_processor_task = asyncio.create_task(self._process_queue())
              self._queue_processor_task.add_done_callback(self._queue_processor_done_callback)


    def _queue_processor_done_callback(self, future: asyncio.Future):
        """Callback when the queue processor task finishes (e.g., due to error)."""
        try:
            future.result() # Raise exception if one occurred
            logger.info("[TASKMGR] Queue processor task finished normally.")
        except asyncio.CancelledError:
            logger.info("[TASKMGR] Queue processor task was cancelled.")
        except Exception as e:
            logger.error(f"[TASKMGR] Queue processor task failed unexpectedly: {e}", exc_info=True)
            # Optionally restart the processor after a delay if not shutting down
            if not self._shutdown_event.is_set():
                 logger.info("[TASKMGR] Attempting to restart queue processor after failure...")
                 # Consider adding a delay here before restarting
                 # asyncio.create_task(asyncio.sleep(5)) # Example delay
                 self._queue_processor_task = None # Reset task handle
                 self._start_queue_processor_if_needed()

        # Ensure handle is cleared if task is done
        if self._queue_processor_task and self._queue_processor_task.done():
            self._queue_processor_task = None


    async def _process_queue(self) -> None:
        """Continuously process tasks from the queue when workers are available."""
        logger.info("[TASKMGR] Queue processor loop started.")
        while not self._shutdown_event.is_set():
            task_id_to_run = None
            async with self._queue_lock:
                if self._task_queue and len(self._active_workers) < self.max_concurrent_tasks:
                    task_id_to_run = self._task_queue.popleft()
                    self._update_queue_positions_nolock()

            if task_id_to_run:
                logger.info(f"[TASKMGR] Dequeued task {task_id_to_run}. Active workers: {len(self._active_workers) + 1}/{self.max_concurrent_tasks}")
                # Start the task execution in the background
                worker_task = asyncio.create_task(self._execute_task_wrapper(task_id_to_run))
                self._active_workers[task_id_to_run] = worker_task
                # Add callback for when the worker finishes
                worker_task.add_done_callback(
                    lambda fut, tid=task_id_to_run: self._worker_done_callback(tid, fut)
                )
            else:
                # No task to run, wait briefly before checking again
                try:
                    # Wait for shutdown signal or timeout
                    await asyncio.wait_for(self._shutdown_event.wait(), timeout=0.5)
                except asyncio.TimeoutError:
                    continue # Continue loop normally
                except asyncio.CancelledError:
                     logger.info("Queue processor wait cancelled.")
                     break # Exit loop if cancelled

        logger.info("[TASKMGR] Queue processor loop stopped.")


    def _worker_done_callback(self, task_id: str, future: asyncio.Future):
         """Callback executed when a worker asyncio.Task finishes."""
         logger.debug(f"[TASKMGR] Worker finished for task {task_id}.")

         # Remove worker from active list
         if task_id in self._active_workers:
             del self._active_workers[task_id]
         else:
              logger.warning(f"[TASKMGR] Task {task_id} not found in active workers during completion callback.")

         # Check for exceptions in the worker task itself (separate from task logic errors)
         try:
             future.result()
         except asyncio.CancelledError:
              logger.info(f"[TASKMGR] Worker task {task_id} was cancelled.")
         except Exception as e:
              logger.error(f"[TASKMGR] Worker task {task_id} failed with unhandled exception: {e}", exc_info=True)
              # Mark the application task as failed if it wasn't already terminal
              with self._task_lock:
                   task = self.tasks.get(task_id) # Fetch task again inside lock
                   if task and not TaskStatus.is_terminal(task.status):
                        task.update_progress(status=TaskStatus.FAILED, error=f"Worker execution error: {e}")
                        # Cleanup will happen when task is eventually deleted or by periodic cleanup
                        # task.perform_cleanup() # Removed immediate cleanup call


         # Clean up memory after task finishes
         self._cleanup_memory()

         # Ensure queue processor keeps running if needed
         self._start_queue_processor_if_needed()


    def _update_queue_positions_nolock(self) -> None:
        """Internal: Update queue positions assuming queue_lock is held."""
        with self._task_lock:
            for i, task_id in enumerate(self._task_queue):
                if task_id in self.tasks:
                    self.tasks[task_id].queue_position = i + 1


    async def _execute_task_wrapper(self, task_id: str) -> None:
        """Wrapper to execute a task handler, manage state, and handle retries."""
        task = None
        handler = None
        with self._task_lock:
            task = self.tasks.get(task_id)
            if not task:
                logger.error(f"[TASKMGR] Task {task_id} not found for execution wrapper.")
                return
            if task.status != TaskStatus.QUEUED:
                 logger.warning(f"[TASKMGR] Task {task_id} status is {task.status}, not QUEUED. Aborting execution wrapper.")
                 return

            handler = self._handlers.get(task.type)
            if not handler:
                # This should be caught earlier, but double-check
                logger.error(f"[TASKMGR] No handler found for task type {task.type}. Marking task {task_id} as failed.")
                task.update_progress(status=TaskStatus.FAILED, error=f"No handler for task type {task.type}")
                task.perform_cleanup()
                return

            # Mark as preparing
            task.update_progress(status=TaskStatus.PREPARING, progress=0.01, info={"stage": "worker_starting"})
            task.queue_position = None # No longer queued

        logger.info(f"[TASKMGR] Executing task {task_id} (Type: {task.type})")

        try:
            # Define the progress callback wrapper for the handler
            def progress_callback_internal(status: TaskStatus, progress: float, info: Dict[str, Any]):
                 with self._task_lock:
                     # Check if task still exists and is not cancelled
                     current_task = self.tasks.get(task_id)
                     if current_task and current_task.status != TaskStatus.CANCELLED:
                          # Use the provided status from the handler if it's active
                          update_status = status if TaskStatus.is_active(status) else None
                          current_task.update_progress(status=update_status, progress=progress, info=info)

            # Execute the actual task handler
            result = await handler(task_id=task_id, task_params=task.params, progress_callback=progress_callback_internal)

            # Mark as completed successfully
            with self._task_lock:
                task = self.tasks.get(task_id) # Re-get task in case it was modified
                if task and task.status != TaskStatus.CANCELLED:
                     task.update_progress(status=TaskStatus.COMPLETED, progress=1.0, result=result, info={"stage": "task_successful"})
                     # Add summary log
                     duration = task.completed_at - task.started_at if task.completed_at and task.started_at else 0
                     # Enhanced Success Log
                     result_summary = f"Result type: {type(result).__name__}" if result is not None else "No result object"
                     if isinstance(result, dict) and 'segments' in result:
                         result_summary += f", Segments: {len(result['segments'])}"
                     if isinstance(result, dict) and 'speakers' in result:
                         result_summary += f", Speakers: {len(result['speakers'])}"
                     logger.info(f"[TASKMGR] Task {task_id} (Type: {task.type}) completed successfully in {duration:.2f}s. {result_summary}")
                     task.perform_cleanup() # Perform cleanup on success


        except asyncio.CancelledError:
             # Handle cancellation initiated by cancel_task() or shutdown()
             with self._task_lock:
                 task = self.tasks.get(task_id) # Fetch task again inside lock
                 task_type = task.type if task else "unknown" # Get type before potential deletion
                 if task and task.status != TaskStatus.CANCELLED: # Ensure status is updated if not already
                      task.update_progress(status=TaskStatus.CANCELLED, info={"stage": "task_cancelled"})
                 # Log safely, checking if task exists
                 logger.info(f"[TASKMGR] Task {task_id} (Type: {task_type}) execution was cancelled.")
                 # Cleanup will happen when task is eventually deleted or by periodic cleanup
                 # if task: task.perform_cleanup() # Removed immediate cleanup call


        except Exception as e:
            # Error log already enhanced in previous diff, ensure exc_info=True is kept
            logger.error(f"[TASKMGR] Error executing task {task_id} (Type: {task.type}): {e}", exc_info=True)
            should_retry = False
            with self._task_lock:
                 task = self.tasks.get(task_id)
                 if not task: return # Task deleted concurrently

                 # Check if cancelled during execution
                 if task.status == TaskStatus.CANCELLED:
                      logger.info(f"[TASKMGR] Task {task_id} (Type: {task.type if task else 'unknown'}) failed but was already cancelled.")
                      # Cleanup will happen when task is eventually deleted or by periodic cleanup
                      # task.perform_cleanup() # Removed immediate cleanup call
                      return # Don't retry cancelled tasks

                 # Handle retries
                 if task.retry_count < task.max_retries:
                     task.retry_count += 1
                     should_retry = True
                     task.update_progress(status=TaskStatus.QUEUED, error=str(e), info={"stage": f"retrying_attempt_{task.retry_count}"})
                     logger.info(f"[TASKMGR] Task {task_id} (Type: {task.type}) failed. Scheduling retry {task.retry_count}/{task.max_retries}.")
                 else:
                     # Max retries reached, mark as terminally failed
                     task.update_progress(status=TaskStatus.FAILED, error=str(e), info={"stage": "task_failed_max_retries"})
                     # Enhanced Failure Log
                     duration = task.completed_at - task.started_at if task.completed_at and task.started_at else 0
                     logger.error(f"[TASKMGR] Task {task_id} (Type: {task.type if task else 'unknown'}) failed permanently after {task.retry_count} retries in {duration:.2f}s. Final Error: {task.error}")
                     # Cleanup will happen when task is eventually deleted or by periodic cleanup
                     # task.perform_cleanup() # Removed immediate cleanup call

            # If retry is needed, re-queue the task
            if should_retry:
                try:
                    # Use create_task to avoid blocking the worker completion
                    asyncio.create_task(self.queue_task(task_id))
                except Exception as queue_err:
                    logger.error(f"[TASKMGR] Failed to re-queue task {task_id} for retry: {queue_err}")
                    # Mark as failed if re-queue fails
                    with self._task_lock:
                         task = self.tasks.get(task_id)
                         if task and task.status == TaskStatus.QUEUED: # If still in retry state
                              task.update_progress(status=TaskStatus.FAILED, error=f"Failed to re-queue after error: {e}")
                              task.perform_cleanup()

    def _cleanup_memory(self) -> None:
        """Perform garbage collection and clear GPU cache."""
        logger.debug("[TASKMGR] Running post-task memory cleanup...") # Already has prefix
        gc.collect()
        if settings.USE_CUDA and torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
                logger.debug("[TASKMGR] CUDA cache cleared.") # Already has prefix
            except Exception as e:
                logger.warning(f"[TASKMGR] Error clearing CUDA cache: {e}")
        # Secure memory wiping is generally not recommended unless strictly required
        # due to performance impact and limited guarantees.
        if settings.SECURE_MEMORY_WIPING:
             logger.warning("[TASKMGR] Secure memory wiping enabled - this may impact performance.") # Already has prefix
             # Implement wiping logic if truly necessary, otherwise remove setting


    def get_task(self, task_id: str, include_result: bool = True) -> Optional[Dict[str, Any]]:
        """Get task status and optionally results."""
        with self._task_lock:
            task = self.tasks.get(task_id)
            return task.to_dict(include_result=include_result) if task else None

    def get_all_tasks(self, status_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get a list of all tasks, optionally filtered by status."""
        with self._task_lock:
            # Return a copy to avoid issues with concurrent modifications
            tasks_snapshot = list(self.tasks.values())

        filtered_tasks = []
        for task in tasks_snapshot:
            if status_filter is None or task.status.value == status_filter:
                filtered_tasks.append(task.to_dict(include_result=False)) # Don't include results in list view
        return filtered_tasks

    def get_queue_status(self) -> Dict[str, Any]:
        """Get the current status of the task queue and workers."""
        queued_tasks = len(self._task_queue)
        active_tasks = len(self._active_workers)
        return {
            "queued_tasks": queued_tasks,
            "active_tasks": active_tasks,
            "max_concurrent_tasks": self.max_concurrent_tasks,
            "is_processing": self._queue_processor_task is not None and not self._queue_processor_task.done()
        }

    async def cancel_task(self, task_id: str) -> bool:
        """Attempt to cancel a pending or active task."""
        async_worker_task_to_cancel: Optional[asyncio.Task] = None
        task_status_updated = False

        with self._task_lock:
            task = self.tasks.get(task_id)
            if not task:
                logger.warning(f"[TASKMGR] Cannot cancel task {task_id}: Not found.")
                return False
            if TaskStatus.is_terminal(task.status):
                logger.info(f"[TASKMGR] Cannot cancel task {task_id}: Already in terminal state {task.status}.")
                return False

            logger.info(f"[TASKMGR] Attempting to cancel task {task_id} (Type: {task.type}, Current status: {task.status.value})")

            # Update status immediately to prevent race conditions
            original_status = task.status
            task.update_progress(status=TaskStatus.CANCELLED, info={"stage": "cancellation_requested"})
            task_status_updated = True

            # If queued, try removing from queue
            if original_status == TaskStatus.QUEUED:
                 async with self._queue_lock:
                      try:
                           self._task_queue.remove(task_id)
                           self._update_queue_positions_nolock()
                           logger.info(f"[TASKMGR] Removed task {task_id} (Type: {task.type}) from queue during cancellation.")
                      except ValueError:
                           # Might have been picked up by worker between locks
                           logger.warning(f"[TASKMGR] Task {task_id} not found in queue during cancellation (might have started).")
                           # Check if it's now active
                           if task_id in self._active_workers:
                                async_worker_task_to_cancel = self._active_workers.get(task_id)
            # If active, get the asyncio task handle
            elif task_id in self._active_workers:
                 async_worker_task_to_cancel = self._active_workers.get(task_id)

            # If cancelled, the worker wrapper or periodic cleanup will handle file cleanup
            # task.perform_cleanup() # Removed immediate cleanup call

        # Cancel the asyncio task outside the lock
        if async_worker_task_to_cancel and not async_worker_task_to_cancel.done():
            logger.info(f"[TASKMGR] Sending cancellation signal to worker for task {task_id} (Type: {task.type if task else 'unknown'}).")
            cancelled = async_worker_task_to_cancel.cancel()
            if cancelled:
                 logger.info(f"[TASKMGR] Cancellation signal sent successfully to worker for task {task_id}.")
                 return True
            else:
                 logger.warning(f"[TASKMGR] Failed to send cancel signal to worker {task_id} (already done?).")
                 # Status was already set to CANCELLED, so return True conceptually
                 return True
        elif task_status_updated:
            # Task was queued or preparing and status was updated
            return True
        else:
            # Should not happen if status updated, but as fallback
             logger.warning(f"[TASKMGR] Cancellation for task {task_id} did not result in queue removal or worker signal.")
             return False

    def delete_task(self, task_id: str) -> bool:
        """Delete a task record. Performs cleanup."""
        with self._task_lock:
            task = self.tasks.pop(task_id, None) # Remove from dict atomically
            if task:
                 logger.info(f"[TASKMGR] Deleting task record {task_id} (Type: {task.type})...")
                 # Cleanup is deferred to worker completion or periodic cleanup
                 # task.perform_cleanup() # Removed immediate cleanup call
                 # Also ensure it's removed from active workers if somehow still there
                 if task_id in self._active_workers:
                      worker = self._active_workers.pop(task_id)
                      if not worker.done(): worker.cancel() # Cancel worker if deleting active task
                 logger.info(f"[TASKMGR] Deleted task record {task_id} (Type: {task.type}) and initiated cleanup.")
                 return True
            else:
                 logger.warning(f"[TASKMGR] Cannot delete task {task_id}: Not found.") # Already has prefix
                 return False

    def cleanup_old_tasks(self) -> int:
        """Clean up terminal tasks older than configured JOB_CLEANUP_HOURS."""
        if not settings: return 0

        max_age_hours = settings.JOB_CLEANUP_HOURS
        if max_age_hours <= 0: # Interpret 0 as very short retention (e.g., few minutes)
            max_age_seconds = 60 * 5 # 5 minutes
            logger.info(f"[TASKMGR] JOB_CLEANUP_HOURS is {max_age_hours}, using short retention of {max_age_seconds}s.")
        else:
            max_age_seconds = max_age_hours * 3600

        current_time = time.time()
        tasks_to_remove: List[str] = []

        with self._task_lock:
            # Iterate safely over a copy of keys
            for task_id in list(self.tasks.keys()):
                task = self.tasks.get(task_id)
                if task and TaskStatus.is_terminal(task.status) and task.completed_at:
                    age = current_time - task.completed_at
                    if age > max_age_seconds:
                        tasks_to_remove.append(task_id)

        removed_count = 0
        for task_id in tasks_to_remove:
            if self.delete_task(task_id): # Use the delete method which includes cleanup
                removed_count += 1

        if removed_count > 0:
            logger.info(f"[TASKMGR] Cleaned up {removed_count} old tasks (older than {max_age_hours:.2f} hours).") # Already has prefix
        return removed_count

    async def shutdown(self) -> None:
        """Gracefully shutdown the TaskManager."""
        if self._shutdown_event.is_set():
            logger.info("[TASKMGR] Shutdown already in progress.")
            return

        logger.info("[TASKMGR] Initiating TaskManager shutdown...")
        self._shutdown_event.set() # Signal loops to stop

        # Stop the queue processor task
        if self._queue_processor_task and not self._queue_processor_task.done():
            self._queue_processor_task.cancel()
            try:
                await self._queue_processor_task
            except asyncio.CancelledError:
                logger.info("[TASKMGR] Queue processor cancelled successfully.")
            except Exception as e:
                 logger.error(f"[TASKMGR] Error during queue processor shutdown: {e}")


        # Cancel all active worker tasks
        active_worker_ids = list(self._active_workers.keys())
        if active_worker_ids:
            logger.info(f"[TASKMGR] Cancelling {len(active_worker_ids)} active worker tasks...") # Already has prefix
            tasks_to_await = []
            for task_id in active_worker_ids:
                 worker = self._active_workers.get(task_id)
                 if worker and not worker.done():
                      worker.cancel()
                      tasks_to_await.append(worker)

            # Wait briefly for workers to finish cancellation
            if tasks_to_await:
                 _, pending = await asyncio.wait(tasks_to_await, timeout=10.0)
                 if pending:
                     logger.warning(f"[TASKMGR] {len(pending)} worker tasks did not terminate gracefully after cancellation signal.")

        # Clear active workers dict
        # Clear active workers dictionary
        self._active_workers.clear()

        # Clean up all remaining tasks if configured
        if settings.CLEAN_ALL_ON_SHUTDOWN:
            logger.info("[TASKMGR] Performing final cleanup of all remaining tasks and files...") # Already has prefix
            with self._task_lock:
                all_task_ids = list(self.tasks.keys())
            for task_id in all_task_ids:
                self.delete_task(task_id) # Use delete which handles cleanup
            logger.info("[TASKMGR] Final cleanup complete.")
        else:
             logger.info("[TASKMGR] Skipping final task cleanup as CLEAN_ALL_ON_SHUTDOWN is false.")

        # Clear the queue
        async with self._queue_lock:
             # Clear task queue
             async with self._queue_lock:
                 self._task_queue.clear()

        logger.info("[TASKMGR] TaskManager shutdown complete.")


# --- Global Instance ---
# Ensure settings are loaded before creating the instance
task_manager = TaskManager() if settings else None
if not task_manager:
     logger.critical("TaskManager could not be initialized due to missing settings.")