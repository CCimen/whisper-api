"""
Task Manager using Redis for robust state management across multiple workers.

Manages asynchronous execution of transcription and diarization tasks,
handling queuing, resource limits, status tracking, cancellation, and cleanup.
"""

import asyncio
import time
import uuid
import logging
import gc
import os
import json
from typing import Dict, Any, Optional, List, Callable, Awaitable
from enum import Enum

import redis.asyncio as redis
import torch # For memory cleanup

# Configure logging
logger = logging.getLogger(__name__)

# Import settings and exceptions safely
try:
    from app.config import settings
    from app.exceptions import ConfigurationError, FileProcessingError
except ImportError:
    logger.critical("Could not import app.config or app.exceptions. Task Manager cannot function.")
    # Define a minimal placeholder if imports fail to allow basic structure loading
    class SettingsMock:
        REDIS_HOST=None; REDIS_PORT=6379; REDIS_DB=0; REDIS_PASSWORD=None; REDIS_TIMEOUT=10;
        REDIS_KEY_PREFIX="whisper_api:"; MAX_CONCURRENT_TASKS=1; JOB_CLEANUP_HOURS=24;
        MAX_RETRIES=0; RETRY_FAILED_TASKS=False; UPLOAD_DIR="/tmp/uploads"; RESULTS_DIR="/tmp/results";
        USE_CUDA=False; CLEAN_ALL_ON_SHUTDOWN=True; SECURE_MEMORY_WIPING=False
    settings = SettingsMock()
    ConfigurationError = RuntimeError
    FileProcessingError = RuntimeError


class TaskStatus(str, Enum):
    """Detailed task status for better tracking."""
    PENDING = "pending"      # Task created but not yet queued
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
        """Check if a status is final."""
        return status in (cls.COMPLETED, cls.FAILED, cls.CANCELLED)

    @classmethod
    def is_active(cls, status: "TaskStatus") -> bool:
        """Check if a status means the task is currently being worked on by a worker."""
        return status in (cls.PREPARING, cls.LOADING_MODEL, cls.PROCESSING, cls.COMPLETING)


# Forward reference DiarizationService for type hinting
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from app.services.diarization import DiarizationService

class TaskManager:
    """
    Manages asynchronous task execution using Redis for state and queuing.
    """
    def __init__(
        self,
        redis_pool: Optional[redis.ConnectionPool] = None,
        diarization_service: Optional['DiarizationService'] = None # Add diarization service dependency
    ):
        """
        Initializes the TaskManager.

        Args:
            redis_pool: An asyncio-redis ConnectionPool instance.
            diarization_service: An optional DiarizationService instance.
        """
        if not settings:
            raise ConfigurationError("Settings module not loaded. TaskManager cannot be initialized.")

        self.redis_pool = redis_pool
        if not self.redis_pool:
            logger.error("[TASKMGR] Redis connection pool is NOT provided. TaskManager cannot operate.")
            self.redis: Optional[redis.Redis] = None
        else:
            self.redis: Optional[redis.Redis] = redis.Redis(connection_pool=self.redis_pool)
            logger.info("[TASKMGR] Redis client initialized from pool.")

        # Define Redis keys using prefix from settings
        self.redis_prefix = settings.REDIS_KEY_PREFIX
        self.task_key_prefix = f"{self.redis_prefix}task:"
        self.result_key_prefix = f"{self.redis_prefix}result:"
        self.queue_key = f"{self.redis_prefix}task_queue"
        self.active_set_key = f"{self.redis_prefix}active_tasks"

        self.max_concurrent_tasks: int = settings.MAX_CONCURRENT_TASKS
        # These track state *local* to the current worker process
        self._active_workers: Dict[str, asyncio.Task] = {}
        self._worker_management_lock = asyncio.Lock()
        self._queue_processor_task: Optional[asyncio.Task] = None
        self._shutdown_event = asyncio.Event()
        self._handlers: Dict[str, Callable[..., Awaitable[Any]]] = {}
        self._diarization_service = diarization_service

        self._register_handlers()

        log_redis = "[success]Redis pool provided.[/success]" if self.redis else "[danger]Redis pool NOT provided.[/danger]"
        logger.info(
            f"[TASKMGR] TaskManager initialized. "
            f"Max concurrent: [config.value]{self.max_concurrent_tasks}[/config.value]. "
            f"{log_redis} Key Prefix: [config.value]'{self.redis_prefix}'[/config.value]"
        )
        # Start the queue processor automatically if Redis is available
        if self.redis:
            self._start_queue_processor_if_needed()


    def _register_handlers(self):
        """Dynamically register task handlers from the processor module."""
        try:
            from app.services.processor import process_audio, handle_diarization_only
            self._handlers['transcription'] = process_audio
            self._handlers['diarization_only'] = handle_diarization_only
            logger.info(f"[TASKMGR] Registered task handlers: [config.value]{list(self._handlers.keys())}[/config.value]")
        except ImportError as e:
            logger.error(f"[TASKMGR] [danger]Failed to import task handlers:[/danger] {e}. Tasks cannot be processed.")
        except Exception as e:
             logger.error(f"[TASKMGR] [danger]Unexpected error registering handlers:[/danger] {e}", exc_info=True)


    async def create_task(self, task_type: str, params: Dict[str, Any]) -> str:
        """
        Creates a new task, stores its initial state in Redis, and returns its ID.
        Handles JSON serialization for complex fields.
        """
        if not self.redis:
            raise ConfigurationError("Redis is not available. Cannot create task.")

        task_id = str(uuid.uuid4())
        effective_max_retries = settings.MAX_RETRIES if settings.RETRY_FAILED_TASKS else 0

        if task_type not in self._handlers:
            raise ValueError(f"No handler registered for task type: '{task_type}'")

        # Prepare initial files_to_clean list from input file path
        files_to_clean = []
        input_file = params.get('file_path')
        if input_file and isinstance(input_file, str):
            try:
                # Use absolute paths for reliable checking
                abs_file_path = os.path.abspath(input_file)
                allowed_dirs = [os.path.abspath(d) for d in [settings.UPLOAD_DIR, settings.RESULTS_DIR]]
                if any(abs_file_path.startswith(allowed_dir) for allowed_dir in allowed_dirs):
                    files_to_clean.append(abs_file_path)
                else:
                    logger.warning(f"[TASKMGR][{task_id}] Initial file path '{input_file}' is outside allowed directories ([file.path]{allowed_dirs}[/file.path]). Not registering for cleanup.")
            except Exception as path_err:
                 logger.error(f"[TASKMGR][{task_id}] Error processing initial file path '{input_file}' for cleanup: {path_err}")

        # Prepare initial task data for Redis Hash
        task_data = {
            "id": task_id,
            "type": task_type,
            "status": TaskStatus.PENDING.value,
            "progress": "0.0",
            "params": json.dumps(params),
            "result_key": "",
            "error": "",
            "created_at": str(time.time()),
            "started_at": "0.0",
            "completed_at": "0.0",
            "max_retries": str(effective_max_retries),
            "retry_count": "0",
            "cancelled": "false",
            "additional_info": json.dumps({}),
            "files_to_clean": json.dumps(files_to_clean)
        }

        try:
            task_key = f"{self.task_key_prefix}{task_id}"
            # Use HSET with mapping for efficiency
            await self.redis.hset(task_key, mapping=task_data)

            # Set an initial expiry for the task hash itself
            # Use a longer expiry initially to prevent premature deletion if queuing/processing is delayed
            cleanup_seconds = settings.JOB_CLEANUP_HOURS * 3600
            buffer_seconds = 24 * 3600 # Add a buffer (e.g., 1 day)
            min_expiry_seconds = 3600 # Ensure at least 1 hour expiry
            initial_expiry_seconds = max(min_expiry_seconds, cleanup_seconds + buffer_seconds)

            await self.redis.expire(task_key, initial_expiry_seconds)

            logger.info(f"[TASKMGR] Created task [task.id]{task_id}[/task.id] (Type: [config.value]{task_type}[/config.value]) in Redis. Key: [config.value]{task_key}[/config.value]")
            return task_id
        except redis.RedisError as e:
            logger.error(f"[TASKMGR] [danger]Redis error creating task {task_id}:[/danger] {e}", exc_info=True)
            raise ConnectionError(f"Failed to communicate with Redis: {e}") from e
        except Exception as e:
            logger.error(f"[TASKMGR] [danger]Unexpected error creating task {task_id}:[/danger] {e}", exc_info=True)
            raise


    async def queue_task(self, task_id: str) -> None:
        """
        Adds a task ID to the Redis processing queue and updates its status.
        Uses WATCH/MULTI/EXEC for safer concurrent updates.
        """
        if not self.redis:
            raise ConfigurationError("Redis is not available. Cannot queue task.")

        task_key = f"{self.task_key_prefix}{task_id}"
        try:
            async with self.redis.pipeline(transaction=True) as pipe:
                # Watch task key and active set for changes during the transaction
                await pipe.watch(task_key, self.active_set_key)

                # Check task existence and status within the transaction
                task_exists = await pipe.exists(task_key)
                if not task_exists:
                    logger.error(f"[TASKMGR] Task [task.id]{task_id}[/task.id] not found in Redis for queuing.")
                    await pipe.unwatch()
                    raise ValueError(f"Task {task_id} not found for queuing.")

                # Fetch multiple fields efficiently
                task_fields = await pipe.hmget(task_key, ["status", "cancelled"])
                is_active_in_set = await pipe.sismember(self.active_set_key, task_id)

                current_status_val = task_fields[0]
                is_cancelled_val = task_fields[1]

                if not current_status_val:
                    logger.error(f"[TASKMGR] Task [task.id]{task_id}[/task.id] exists but status is missing. Cannot queue.")
                    await pipe.unwatch()
                    raise ValueError(f"Task {task_id} status missing.")

                current_status = TaskStatus(current_status_val)
                is_cancelled = is_cancelled_val == "true"

                # Check if task can be queued
                if TaskStatus.is_terminal(current_status):
                    logger.warning(f"[TASKMGR] Task [task.id]{task_id}[/task.id] is already in terminal state [task.status]{current_status}[/task.status]. Cannot queue.")
                    await pipe.unwatch()
                    return
                if current_status == TaskStatus.QUEUED:
                    logger.warning(f"[TASKMGR] Task [task.id]{task_id}[/task.id] status is already [task.status]QUEUED[/task.status].")
                    await pipe.unwatch()
                    return
                if is_active_in_set:
                     logger.warning(f"[TASKMGR] Task [task.id]{task_id}[/task.id] is already in active set. Cannot queue.")
                     await pipe.unwatch()
                     return
                if is_cancelled:
                     logger.warning(f"[TASKMGR] Task [task.id]{task_id}[/task.id] is marked as cancelled. Cannot queue.")
                     await pipe.unwatch()
                     return

                pipe.multi()
                pipe.hset(task_key, "status", TaskStatus.QUEUED.value)
                pipe.lpush(self.queue_key, task_id)
                results = await pipe.execute()

                # Check results (execute raises WatchError if watched keys changed)
                # Check HSET (index 0: should be 0 or 1) and LPUSH (index 1: should be > 0) results
                if not results or len(results) != 2 or results[0] not in [0, 1] or not isinstance(results[1], int) or results[1] <= 0:
                     logger.error(f"[TASKMGR] Redis transaction failed for queuing task [task.id]{task_id}[/task.id]. Results: {results}")
                     raise redis.RedisError("Failed to queue task atomically.")

            # Log queue length after successful queuing
            queue_len = await self.redis.llen(self.queue_key)
            logger.info(f"[TASKMGR] Task [task.id]{task_id}[/task.id] queued. Total queue size: [config.value]{queue_len}[/config.value]")

            # Ensure the queue processor is running in this worker process
            self._start_queue_processor_if_needed()

        except redis.WatchError:
             logger.warning(f"[TASKMGR] Task [task.id]{task_id}[/task.id] or active set was modified during queuing transaction. Another process may be acting on it.")
             # Let the other process handle it, or potentially retry after a delay if needed.
        except redis.RedisError as e:
            logger.error(f"[TASKMGR] [danger]Redis error queuing task {task_id}:[/danger] {e}", exc_info=True)
            # Attempt to mark as failed (best effort)
            try:
                await self.redis.hset(task_key, mapping={
                    "status": TaskStatus.FAILED.value,
                    "error": f"Failed during queuing: {e}"
                })
            except Exception as revert_e:
                 logger.error(f"[TASKMGR] [danger]Failed to mark task {task_id} as failed after queuing error:[/danger] {revert_e}")
            raise ConnectionError(f"Failed to communicate with Redis during queuing: {e}") from e
        except Exception as e:
             logger.error(f"[TASKMGR] [danger]Unexpected error queuing task {task_id}:[/danger] {e}", exc_info=True)
             raise


    def _start_queue_processor_if_needed(self):
        """Starts the background queue processor task if Redis is available and it's not running."""
        if not self.redis:
            logger.debug("[TASKMGR] Cannot start queue processor: Redis is unavailable.")
            return

        # Use lock to prevent race conditions when multiple workers start concurrently
        async def start_processor_task():
            async with self._worker_management_lock:
                if (self._queue_processor_task is None or self._queue_processor_task.done()) and \
                   not self._shutdown_event.is_set():
                    logger.info("[TASKMGR] Starting queue processor background task for this worker.")
                    self._queue_processor_task = asyncio.create_task(self._process_queue())
                    self._queue_processor_task.add_done_callback(self._queue_processor_done_callback)

        asyncio.create_task(start_processor_task())


    def _queue_processor_done_callback(self, future: asyncio.Future):
        """Callback when the local queue processor task finishes."""
        try:
            future.result()
            logger.info("[TASKMGR] Queue processor task finished normally.")
        except asyncio.CancelledError:
            logger.info("[TASKMGR] Queue processor task was cancelled.")
        except Exception as e:
            logger.error(f"[TASKMGR] [danger]Queue processor task failed unexpectedly:[/danger] {e}", exc_info=True)
            # Optionally restart the processor after a delay if not shutting down
            if not self._shutdown_event.is_set() and self.redis:
                logger.info("[TASKMGR] Attempting to restart queue processor after failure...")
                # Consider adding a delay: await asyncio.sleep(5)
                self._queue_processor_task = None
                self._start_queue_processor_if_needed()
        finally:
            # Ensure handle is cleared if task is done, using lock for safety
            async def clear_handle():
                async with self._worker_management_lock:
                    if self._queue_processor_task and self._queue_processor_task.done():
                        self._queue_processor_task = None
            asyncio.create_task(clear_handle())


    async def _process_queue(self) -> None:
        """Continuously process tasks from the Redis queue when local workers are available."""
        if not self.redis:
            logger.error("[TASKMGR] [danger]Redis unavailable, queue processor cannot run.[/danger]")
            return

        logger.info("[TASKMGR] Queue processor loop started.")
        while not self._shutdown_event.is_set():
            task_id_to_run = None
            worker_slot_available = False

            # Check local concurrency limit before blocking pop
            async with self._worker_management_lock:
                if len(self._active_workers) < self.max_concurrent_tasks:
                    worker_slot_available = True
                else:
                    # If max tasks reached locally, wait briefly before checking Redis again
                    await asyncio.sleep(0.2)
                    continue

            # If slot available, wait for a task from Redis queue
            if worker_slot_available:
                try:
                    result = await self.redis.brpop(self.queue_key, timeout=1) # Short timeout allows checking shutdown flag
                    if result:
                        _, task_id_to_run = result
                        logger.debug(f"[TASKMGR] Popped task [task.id]{task_id_to_run}[/task.id] from Redis queue '{self.queue_key}'.")
                    else:
                        continue
                except asyncio.CancelledError:
                     logger.info("[TASKMGR] Queue processor BRPOP cancelled.")
                     break
                except redis.RedisError as e:
                     logger.error(f"[TASKMGR] [danger]Redis error during BRPOP on {self.queue_key}:[/danger] {e}. Retrying...")
                     await asyncio.sleep(1)
                     continue
                except Exception as e:
                     logger.error(f"[TASKMGR] [danger]Unexpected error during BRPOP:[/danger] {e}. Retrying...", exc_info=True)
                     await asyncio.sleep(1)
                     continue

            # If a task was popped, try to claim and execute it
            if task_id_to_run:
                async with self._worker_management_lock:
                    # Double-check concurrency limit after popping
                    if len(self._active_workers) < self.max_concurrent_tasks:
                        # Atomically add to Redis active set to claim the task globally
                        try:
                            was_added = await self.redis.sadd(self.active_set_key, task_id_to_run)
                            if was_added == 1: # Successfully added (claimed)
                                logger.info(f"[TASKMGR] Claimed task [task.id]{task_id_to_run}[/task.id]. Active workers (local): {len(self._active_workers) + 1}/{self.max_concurrent_tasks}")
                                # Start the task execution in the background locally
                                worker_task = asyncio.create_task(self._execute_task_wrapper(task_id_to_run))
                                self._active_workers[task_id_to_run] = worker_task
                                worker_task.add_done_callback(
                                    lambda fut, tid=task_id_to_run: self._worker_done_callback(tid, fut)
                                )
                            else:
                                logger.warning(f"[TASKMGR] Task [task.id]{task_id_to_run}[/task.id] was already claimed (in Redis active set). Skipping execution in this worker.")
                        except redis.RedisError as e:
                             logger.error(f"[TASKMGR] [danger]Redis error adding task {task_id_to_run} to active set:[/danger] {e}. Re-queueing.")
                             await self.redis.lpush(self.queue_key, task_id_to_run)
                        except Exception as e:
                             logger.error(f"[TASKMGR] [danger]Unexpected error adding task {task_id_to_run} to active set:[/danger] {e}. Re-queueing.", exc_info=True)
                             await self.redis.lpush(self.queue_key, task_id_to_run)
                    else:
                        logger.warning(f"[TASKMGR] Max concurrent tasks reached locally before claiming {task_id_to_run}. Re-queueing.")
                        await self.redis.lpush(self.queue_key, task_id_to_run)

        logger.info("[TASKMGR] Queue processor loop stopped.")

    def _worker_done_callback(self, task_id: str, future: asyncio.Future):
        """Callback executed when a local worker asyncio.Task finishes."""
        logger.debug(f"[TASKMGR] Worker finished processing task [task.id]{task_id}[/task.id].")

        # Remove worker from local active list (use lock for safety)
        async def remove_worker_from_local():
            async with self._worker_management_lock:
                if task_id in self._active_workers:
                    del self._active_workers[task_id]
                    logger.debug(f"[TASKMGR] Removed task {task_id} from local active worker list.")
                else:
                    logger.warning(f"[TASKMGR] Task {task_id} not found in local active workers during completion callback.")

        asyncio.create_task(remove_worker_from_local())


        try:
            future.result()
        except asyncio.CancelledError:
            logger.info(f"[TASKMGR] Worker task [task.id]{task_id}[/task.id] was cancelled.")
        except Exception as e:
            logger.error(f"[TASKMGR] [danger]Worker task {task_id} failed with unhandled exception:[/danger] {e}", exc_info=True)
            # Mark the application task as failed in Redis if it wasn't already terminal
            async def mark_failed_in_redis():
                if not self.redis: return
                task_key = f"{self.task_key_prefix}{task_id}"
                try:
                    current_status_val = await self.redis.hget(task_key, "status")
                    if current_status_val and not TaskStatus.is_terminal(TaskStatus(current_status_val)):
                        await self.redis.hset(task_key, mapping={
                            "status": TaskStatus.FAILED.value,
                            "error": f"Worker execution error: {e}",
                            "completed_at": str(time.time())
                        })
                except Exception as mark_err:
                    logger.error(f"[TASKMGR] [danger]Failed to mark task {task_id} as failed in Redis after worker error:[/danger] {mark_err}")

            asyncio.create_task(mark_failed_in_redis())

        self._cleanup_memory()

        self._start_queue_processor_if_needed()

    async def _execute_task_wrapper(self, task_id: str) -> None:
        """Wrapper to execute a task handler, manage state in Redis, and handle retries."""
        if not self.redis:
            logger.error(f"[TASKMGR] [danger]Redis unavailable, cannot execute task {task_id}.[/danger]")
            try: await self.redis.srem(self.active_set_key, task_id)
            except: pass
            return

        task_key = f"{self.task_key_prefix}{task_id}"
        task_data = None
        handler = None
        task_type = "unknown"

        try:
            # 1. Fetch initial task data & Validate State
            task_data = await self.redis.hgetall(task_key)
            if not task_data:
                logger.error(f"[TASKMGR] Task [task.id]{task_id}[/task.id] data not found in Redis for execution.")
                return

            task_type = task_data.get("type", "unknown")
            current_status_val = task_data.get("status")
            is_cancelled_val = task_data.get("cancelled")

            if not current_status_val:
                 logger.error(f"[TASKMGR] Task {task_id} status missing in Redis data. Cannot execute.")
                 return

            current_status = TaskStatus(current_status_val)
            is_cancelled = is_cancelled_val == "true"

            if is_cancelled or TaskStatus.is_terminal(current_status):
                 logger.info(f"[TASKMGR] Task [task.id]{task_id}[/task.id] execution skipped: Status is [task.status]{current_status.value}[/task.status] or cancelled flag is set.")
                 return

            # 2. Get Handler
            handler = self._handlers.get(task_type)
            if not handler:
                raise ValueError(f"No handler for task type {task_type}")

            # 3. Mark as Preparing/Started in Redis
            await self.redis.hset(task_key, mapping={
                "status": TaskStatus.PREPARING.value,
                "started_at": str(time.time()),
                "progress": "0.01",
                "additional_info": json.dumps({"stage": "worker_starting"})
            })

            logger.info(f"[TASKMGR] Executing task [task.id]{task_id}[/task.id] (Type: [config.value]{task_type}[/config.value])")

            # 4. Define Progress Callback Wrapper for the handler
            async def progress_callback_internal(status: TaskStatus, progress: float, info: Dict[str, Any]):
                if not self.redis: return
                task_key_cb = f"{self.task_key_prefix}{task_id}"
                try:
                    exists_cancelled = await self.redis.hmget(task_key_cb, ["cancelled"]) # Check cancelled flag efficiently
                    if exists_cancelled and exists_cancelled[0] == "false":
                        update_data: Dict[str, Any] = {"progress": f"{min(max(0.0, progress), 1.0):.4f}"}
                        if TaskStatus.is_active(status):
                            update_data["status"] = status.value
                        if info:
                            existing_info_raw = await self.redis.hget(task_key_cb, "additional_info")
                            existing_info = json.loads(existing_info_raw) if existing_info_raw else {}
                            existing_info.update(info)
                            update_data["additional_info"] = json.dumps(existing_info)

                        await self.redis.hset(task_key_cb, mapping=update_data)
                except Exception as cb_err:
                    logger.warning(f"[TASKMGR] Error updating progress for task {task_id} in Redis: {cb_err}")

            # 5. Execute Task Handler
            task_params = json.loads(task_data.get("params", "{}"))
            # Pass dependencies to the handler
            handler_kwargs = {
                "task_id": task_id,
                "task_params": task_params,
                "progress_callback": progress_callback_internal,
                "diarization_service": self._diarization_service
            }
            result = await handler(**handler_kwargs)

            # --- Post-Execution Steps (Success Path) ---

            # 6. Store Result (if successful execution)
            result_key = f"{self.result_key_prefix}{task_id}"
            result_stored = False
            final_task_status = TaskStatus.COMPLETED
            final_error = None
            cleanup_seconds = settings.JOB_CLEANUP_HOURS * 3600
            result_expiry_seconds = cleanup_seconds if cleanup_seconds > 0 else (24 * 3600)

            try:
                await self.redis.set(result_key, json.dumps(result), ex=result_expiry_seconds)
                result_stored = True
                logger.info(f"[TASKMGR] Stored result for task {task_id} in Redis key [config.value]{result_key}[/config.value] with TTL {result_expiry_seconds}s.")
            except redis.RedisError as redis_err:
                logger.error(f"[TASKMGR] [danger]Failed to store result for task {task_id} in Redis:[/danger] {redis_err}", exc_info=True)
                final_task_status = TaskStatus.FAILED
                final_error = f"Failed to store result in Redis: {redis_err}"
                result_key = ""
            except TypeError as json_err:
                 logger.error(f"[TASKMGR] [danger]Failed to serialize result for task {task_id}:[/danger] {json_err}", exc_info=True)
                 final_task_status = TaskStatus.FAILED
                 final_error = f"Result serialization failed: {json_err}"
                 result_key = ""


            # 7. Update Final Task Status in Redis
            completed_at = time.time()
            final_task_update = {
                "status": final_task_status.value,
                "progress": "1.0",
                "result_key": result_key,
                "error": final_error or "",
                "completed_at": str(completed_at),
                "additional_info": json.dumps({"stage": "task_finished"})
            }
            await self.redis.hset(task_key, mapping=final_task_update)
            await self.redis.expire(task_key, result_expiry_seconds)


            # 8. Log Summary
            started_at = float(task_data.get("started_at", 0.0))
            duration = completed_at - started_at if started_at > 0 else 0
            if final_task_status == TaskStatus.COMPLETED:
                 logger.info(f"[TASKMGR] Task [task.id]{task_id}[/task.id] (Type: {task_type}) [success]completed successfully[/success] in {duration:.2f}s. Result stored: {result_stored}")
            else:
                 logger.error(f"[TASKMGR] Task [task.id]{task_id}[/task.id] (Type: {task_type}) [danger]failed[/danger] during result storage after {duration:.2f}s. Error: {final_error}")


            # 9. Perform File Cleanup (based on latest list in Redis)
            if settings.AUTO_DELETE_AFTER_COMPLETION:
                 try:
                     files_raw = await self.redis.hget(task_key, "files_to_clean")
                     files_to_clean = json.loads(files_raw) if files_raw else []
                     if files_to_clean:
                          self._perform_file_cleanup(task_id, files_to_clean)
                          # await self.redis.hset(task_key, "files_to_clean", json.dumps([]))
                 except Exception as clean_err:
                      logger.error(f"[TASKMGR] [danger]Error during post-completion file cleanup for task {task_id}:[/danger] {clean_err}")

        except asyncio.CancelledError:
             logger.info(f"[TASKMGR] Task [task.id]{task_id}[/task.id] (Type: {task_type}) execution was cancelled.")

        except Exception as task_exec_err:
            # --- Handle Task Execution Errors & Retries ---
            logger.error(f"[TASKMGR] [danger]Error executing task {task_id} (Type: {task_type}):[/danger] {task_exec_err}", exc_info=True)
            should_retry = False
            error_message = str(task_exec_err)

            if self.redis:
                try:
                    # Fetch latest data again to check retries/cancellation status
                    task_data_on_fail = await self.redis.hgetall(task_key)
                    if not task_data_on_fail: return

                    current_status_on_fail = TaskStatus(task_data_on_fail.get("status", TaskStatus.FAILED.value))
                    is_cancelled_on_fail = task_data_on_fail.get("cancelled") == "true"

                    if is_cancelled_on_fail or TaskStatus.is_terminal(current_status_on_fail):
                        logger.info(f"[TASKMGR] Task {task_id} failed but was already cancelled or terminal. No retry.")
                        return

                    retry_count = int(task_data_on_fail.get("retry_count", 0))
                    max_retries = int(task_data_on_fail.get("max_retries", 0))

                    if retry_count < max_retries:
                        retry_count += 1
                        should_retry = True
                        await self.redis.hset(task_key, mapping={
                            "status": TaskStatus.QUEUED.value,
                            "error": error_message,
                            "retry_count": str(retry_count),
                            "additional_info": json.dumps({"stage": f"retrying_attempt_{retry_count}"})
                        })
                        logger.info(f"[TASKMGR] Task [task.id]{task_id}[/task.id] failed. Scheduling retry {retry_count}/{max_retries}.")
                        await self.redis.lpush(self.queue_key, task_id)
                    else:
                        # Max retries reached, mark as terminally failed
                        completed_at_fail = time.time()
                        started_at_fail = float(task_data_on_fail.get("started_at", 0.0))
                        duration_fail = completed_at_fail - started_at_fail if started_at_fail > 0 else 0
                        await self.redis.hset(task_key, mapping={
                            "status": TaskStatus.FAILED.value,
                            "error": error_message,
                            "completed_at": str(completed_at_fail),
                            "additional_info": json.dumps({"stage": "task_failed_max_retries"})
                        })
                        logger.error(f"[TASKMGR] Task [task.id]{task_id}[/task.id] [danger]failed permanently[/danger] after {retry_count} retries in {duration_fail:.2f}s. Final Error: {error_message}")
                        # No cleanup on failure path to avoid deleting input files if processing failed early
                        pass

                except redis.RedisError as redis_err:
                    logger.error(f"[TASKMGR] [danger]Redis error during failure/retry handling for task {task_id}:[/danger] {redis_err}")
                except Exception as retry_err:
                    logger.error(f"[TASKMGR] [danger]Unexpected error during failure/retry handling for task {task_id}:[/danger] {retry_err}")

        finally:
            if self.redis:
                try:
                    await self.redis.srem(self.active_set_key, task_id)
                except Exception as e:
                    logger.error(f"[TASKMGR] [danger]Failed to remove task {task_id} from Redis active set in finally block:[/danger] {e}")


    async def get_task(self, task_id: str, include_result: bool = False) -> Optional[Dict[str, Any]]:
        """
        Gets task status and optionally the result from Redis.

        Args:
            task_id: The ID of the task.
            include_result: Whether to fetch the result data from its separate Redis key.

        Returns:
            A dictionary representing the task, or None if not found.
        """
        if not self.redis:
            logger.error("[TASKMGR] Redis unavailable, cannot get task.")
            return None

        task_key = f"{self.task_key_prefix}{task_id}"
        try:
            task_data = await self.redis.hgetall(task_key)
            if not task_data:
                return None

            # Basic processing of retrieved hash data
            task_info = {
                "id": task_data.get("id", task_id),
                "type": task_data.get("type"),
                "status": task_data.get("status"),
                "progress": float(task_data.get("progress", 0.0)),
                "params": json.loads(task_data.get("params", "{}")),
                "result_key": task_data.get("result_key"),
                "error": task_data.get("error"),
                "created_at": float(task_data.get("created_at", 0.0)),
                "started_at": float(task_data.get("started_at", 0.0)),
                "completed_at": float(task_data.get("completed_at", 0.0)),
                "retry_count": int(task_data.get("retry_count", 0)),
                "cancelled": task_data.get("cancelled") == "true",
                "additional_info": json.loads(task_data.get("additional_info", "{}")),
                "files_to_clean": json.loads(task_data.get("files_to_clean", "[]")),
                "result": None
            }

            # Fetch result data if requested and task is completed
            if include_result and \
               task_info["status"] == TaskStatus.COMPLETED.value and \
               task_info["result_key"]:
                try:
                    result_data_raw = await self.redis.get(task_info["result_key"])
                    if result_data_raw:
                        task_info["result"] = json.loads(result_data_raw)
                    else:
                         logger.warning(f"[TASKMGR] Task {task_id} completed but result key '{task_info['result_key']}' not found or empty in Redis.")
                         task_info["error"] = task_info.get("error", "") + " (Result data missing)"
                         task_info["status"] = TaskStatus.FAILED.value
                except Exception as e:
                    logger.error(f"[TASKMGR] [danger]Failed to fetch/parse result for task {task_id} from key {task_info['result_key']}:[/danger] {e}")
                    task_info["error"] = task_info.get("error", "") + f" (Result fetch/parse error: {e})"
                    task_info["status"] = TaskStatus.FAILED.value

            return task_info

        except redis.RedisError as e:
            logger.error(f"[TASKMGR] [danger]Redis error getting task {task_id}:[/danger] {e}", exc_info=True)
            return None
        except Exception as e:
            logger.error(f"[TASKMGR] [danger]Unexpected error getting task {task_id}:[/danger] {e}", exc_info=True)
            return None


    async def get_all_tasks(self, status_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get a list of all task IDs (requires scanning Redis keys).
        NOTE: This can be inefficient on large Redis instances. Use with caution.
        Consider alternative tracking methods if frequent listing is needed.
        """
        if not self.redis:
            logger.error("[TASKMGR] Redis unavailable, cannot list tasks.")
            return []

        task_ids = []
        try:
            cursor = '0'
            while cursor != 0:
                cursor, keys = await self.redis.scan(cursor=cursor, match=f"{self.task_key_prefix}*", count=100)
                task_ids.extend([key.decode('utf-8').split(':')[-1] for key in keys])

            tasks_details = []
            for task_id in task_ids:
                task_info = await self.get_task(task_id, include_result=False)
                if task_info:
                    if status_filter is None or task_info.get("status") == status_filter:
                        tasks_details.append(task_info)
            return tasks_details

        except redis.RedisError as e:
            logger.error(f"[TASKMGR] [danger]Redis error scanning for tasks:[/danger] {e}", exc_info=True)
            return []
        except Exception as e:
            logger.error(f"[TASKMGR] [danger]Unexpected error listing tasks:[/danger] {e}", exc_info=True)
            return []

    def get_queue_status(self) -> Dict[str, Any]:
        """
        Get the current status of the task queue (length from Redis) and local workers.
        Note: Queue length is fetched asynchronously.
        """
        # This should ideally be async to fetch queue length
        # For now, return local worker count and placeholder for queue length
        active_local_workers = len(self._active_workers)
        is_local_processor_running = self._queue_processor_task is not None and not self._queue_processor_task.done()

        # TODO: Make this method async to fetch actual queue length from Redis
        queue_len = "N/A (Sync method)"

        return {
            "queued_tasks": queue_len,
            "active_tasks_local": active_local_workers,
            "max_concurrent_tasks": self.max_concurrent_tasks,
            "is_processor_running_local": is_local_processor_running
        }
        # TODO: Make this method async to fetch queue_len from Redis


    async def cancel_task(self, task_id: str) -> bool:
        """
        Attempt to cancel a task by setting its 'cancelled' flag in Redis
        and removing it from the queue if present.
        """
        if not self.redis:
            raise ConfigurationError("Redis is not available. Cannot cancel task.")

        task_key = f"{self.task_key_prefix}{task_id}"
        logger.info(f"[TASKMGR] Attempting to cancel task [task.id]{task_id}[/task.id]...")

        try:
            # Use a transaction to ensure atomicity
            async with self.redis.pipeline(transaction=True) as pipe:
                await pipe.watch(task_key) # Watch the task key

                # Check if task exists and is not already terminal/cancelled
                task_fields = await pipe.hmget(task_key, ["status", "cancelled"])
                if not task_fields or task_fields[0] is None: # Task doesn't exist
                    logger.warning(f"[TASKMGR] Cannot cancel task {task_id}: Not found.")
                    await pipe.unwatch()
                    return False

                status_val = task_fields[0]
                cancelled_val = task_fields[1]
                current_status = TaskStatus(status_val)

                if TaskStatus.is_terminal(current_status):
                    logger.info(f"[TASKMGR] Cannot cancel task {task_id}: Already terminal ([task.status]{current_status.value}[/task.status]).")
                    await pipe.unwatch()
                    return False
                if cancelled_val == "true":
                    logger.info(f"[TASKMGR] Task {task_id} is already marked as cancelled.")
                    await pipe.unwatch()
                    return True # Already cancelled

                # Start transaction
                pipe.multi()
                # Set cancelled flag and potentially status
                pipe.hset(task_key, mapping={
                    "cancelled": "true",
                    "status": TaskStatus.CANCELLED.value, # Set status directly
                    "error": "Task cancelled by user/system.",
                    "completed_at": str(time.time()) # Mark completion time
                })
                # Attempt to remove from queue (LREM returns number of removed elements)
                pipe.lrem(self.queue_key, 0, task_id) # Remove all occurrences
                # Optionally remove from active set (though worker should handle this)
                # pipe.srem(self.active_set_key, task_id)

                results = await pipe.execute()

                # Check results: results[0] is HSET result (usually True/1), results[1] is LREM result (count)
                if not results or not results[0]:
                     logger.error(f"[TASKMGR] [danger]Redis transaction failed for cancelling task {task_id}.[/danger]")
                     raise redis.RedisError("Cancellation transaction failed")

                removed_count = results[1]
                logger.info(f"[TASKMGR] Cancellation processed for task [task.id]{task_id}[/task.id]. Marked as cancelled. Removed from queue: {removed_count > 0}.")
                return True

        except redis.WatchError:
            logger.warning(f"[TASKMGR] Task {task_id} was modified during cancellation attempt. Retrying or assuming concurrent modification.")
            # Could implement retry logic here if needed
            return False
        except redis.RedisError as e:
            logger.error(f"[TASKMGR] [danger]Redis error cancelling task {task_id}:[/danger] {e}", exc_info=True)
            return False
        except Exception as e:
            logger.error(f"[TASKMGR] [danger]Unexpected error cancelling task {task_id}:[/danger] {e}", exc_info=True)
            return False


    async def delete_task(self, task_id: str) -> bool:
        """
        Deletes a task record and its associated result from Redis.
        Also attempts removal from queue and active set for thorough cleanup.
        """
        if not self.redis:
            raise ConfigurationError("Redis is not available. Cannot delete task.")

        task_key = f"{self.task_key_prefix}{task_id}"
        result_key = f"{self.result_key_prefix}{task_id}" # Construct potential result key

        logger.info(f"[TASKMGR] Attempting to delete task [task.id]{task_id}[/task.id] and associated data...")

        try:
            # Use a pipeline for multiple deletions/removals
            async with self.redis.pipeline(transaction=False) as pipe: # No transaction needed for deletes
                pipe.delete(task_key)
                pipe.delete(result_key)
                pipe.lrem(self.queue_key, 0, task_id) # Attempt queue removal
                pipe.srem(self.active_set_key, task_id) # Attempt active set removal

                results = await pipe.execute()

            # results contains the number of keys deleted/elements removed for each command
            task_deleted_count = results[0]
            result_deleted_count = results[1]
            queue_removed_count = results[2]
            active_removed_count = results[3]

            if task_deleted_count > 0:
                 logger.info(f"[TASKMGR] Deleted task record {task_id} (Key: {task_key}).")
                 if result_deleted_count > 0: logger.info(f"--- Deleted associated result (Key: {result_key}).")
                 if queue_removed_count > 0: logger.info(f"--- Removed from queue '{self.queue_key}'.")
                 if active_removed_count > 0: logger.info(f"--- Removed from active set '{self.active_set_key}'.")
                 # Perform associated file cleanup after successful Redis deletion
                 # Fetch files_to_clean *before* deleting the task hash if needed,
                 # or rely on periodic cleanup if hash is gone. Assuming delete is for terminal tasks.
                 # For simplicity, we won't fetch here; rely on caller/periodic cleanup.
                 return True
            else:
                 logger.warning(f"[TASKMGR] Task record {task_id} not found for deletion (Key: {task_key}).")
                 return False

        except redis.RedisError as e:
            logger.error(f"[TASKMGR] [danger]Redis error deleting task {task_id}:[/danger] {e}", exc_info=True)
            return False
        except Exception as e:
            logger.error(f"[TASKMGR] [danger]Unexpected error deleting task {task_id}:[/danger] {e}", exc_info=True)
            return False


    def _perform_file_cleanup(self, task_id: str, files_to_clean: List[str]):
        """Securely delete files associated with a task."""
        if not files_to_clean: return

        logger.info(f"[TASKMGR] Performing file cleanup for task [task.id]{task_id}[/task.id]. Files: {files_to_clean}")
        allowed_dirs_abs = [os.path.abspath(d) for d in [settings.UPLOAD_DIR, settings.RESULTS_DIR]]

        for file_path in files_to_clean:
            try:
                abs_file_path = os.path.abspath(file_path)
                # Security Check: Ensure the file is within allowed directories
                if not any(abs_file_path.startswith(allowed_dir) for allowed_dir in allowed_dirs_abs):
                    logger.error(f"[TASKMGR] [danger]SECURITY RISK:[/danger] Attempted cleanup of file '{file_path}' outside allowed directories. Skipping.")
                    continue

                if os.path.exists(abs_file_path):
                    # Simple secure delete (overwrite slightly then remove)
                    # For more robust wiping, specific OS tools or libraries are needed.
                    file_size = os.path.getsize(abs_file_path)
                    if file_size > 0 and settings.SECURE_MEMORY_WIPING: # Check setting
                         try:
                             with open(abs_file_path, "wb") as f:
                                 f.write(os.urandom(min(file_size, 1024 * 1024))) # Overwrite up to 1MB
                         except Exception as write_err:
                              logger.warning(f"[TASKMGR] Failed to overwrite file {abs_file_path} before deletion: {write_err}")

                    os.remove(abs_file_path)
                    logger.info(f"[TASKMGR] Deleted file: [file.path]{abs_file_path}[/file.path]")
                else:
                    logger.debug(f"[TASKMGR] File not found for cleanup (already deleted?): {abs_file_path}")
            except Exception as e:
                logger.error(f"[TASKMGR] [danger]Error deleting file {file_path}:[/danger] {e}")


    async def cleanup_old_tasks(self) -> int:
        """
        Clean up terminal tasks older than configured JOB_CLEANUP_HOURS from Redis.
        NOTE: This requires SCAN and can be inefficient on large datasets.
        Returns the number of tasks deleted.
        """
        if not self.redis: return 0

        max_age_hours = settings.JOB_CLEANUP_HOURS
        if max_age_hours <= 0: # Treat 0 or negative as disabled periodic cleanup
            return 0

        max_age_seconds = max_age_hours * 3600
        current_time = time.time()
        tasks_to_remove_ids: List[str] = []
        removed_count = 0

        logger.info("[TASKMGR] Starting periodic cleanup of old tasks...")
        scan_count = 0
        iteration = 0

        try:
            async for key_bytes in self.redis.scan_iter(match=f"{self.task_key_prefix}*", count=100):
                 iteration += 1
                 task_key = key_bytes.decode('utf-8')
                 task_id = task_key.split(':')[-1]
                 scan_count += 1

                 try:
                      # Check status and completed_at timestamp efficiently
                      fields = await self.redis.hmget(task_key, ["status", "completed_at"])
                      status_val = fields[0]
                      completed_at_val = fields[1]

                      if status_val and completed_at_val:
                           status = TaskStatus(status_val)
                           completed_at = float(completed_at_val)
                           if TaskStatus.is_terminal(status) and (current_time - completed_at > max_age_seconds):
                                tasks_to_remove_ids.append(task_id)
                 except Exception as fetch_err:
                      logger.warning(f"[TASKMGR] Error fetching details for key {task_key} during cleanup scan: {fetch_err}")

                 # Log progress periodically during long scans
                 if iteration % 500 == 0:
                      logger.debug(f"[TASKMGR] Cleanup scan progress: Scanned {scan_count} keys...")


            logger.info(f"[TASKMGR] Cleanup scan complete. Found {len(tasks_to_remove_ids)} tasks older than {max_age_hours} hours.")

            # Delete the identified tasks
            for task_id in tasks_to_remove_ids:
                # Use delete_task for consistent deletion logic (includes result key)
                if await self.delete_task(task_id):
                    removed_count += 1

            logger.info(f"[TASKMGR] Periodic cleanup finished. Deleted {removed_count} old task records.")

        except redis.RedisError as e:
            logger.error(f"[TASKMGR] [danger]Redis error during cleanup scan:[/danger] {e}", exc_info=True)
        except Exception as e:
            logger.error(f"[TASKMGR] [danger]Unexpected error during periodic cleanup:[/danger] {e}", exc_info=True)

        return removed_count


    async def shutdown(self) -> None:
        """Gracefully shutdown the TaskManager and cancel local tasks."""
        if self._shutdown_event.is_set():
            logger.info("[TASKMGR] Shutdown already in progress.")
            return

        logger.info("[TASKMGR] Initiating TaskManager shutdown...")
        self._shutdown_event.set() # Signal loops to stop

        # Stop the local queue processor task
        if self._queue_processor_task and not self._queue_processor_task.done():
            self._queue_processor_task.cancel()
            try:
                await asyncio.wait_for(self._queue_processor_task, timeout=5.0)
            except asyncio.CancelledError:
                logger.info("[TASKMGR] Local queue processor cancelled successfully.")
            except asyncio.TimeoutError:
                logger.warning("[TASKMGR] Local queue processor did not stop within timeout.")
            except Exception as e:
                 logger.error(f"[TASKMGR] [danger]Error stopping local queue processor:[/danger] {e}")

        # Cancel all active local worker tasks
        active_worker_ids = list(self._active_workers.keys())
        if active_worker_ids:
            logger.info(f"[TASKMGR] Cancelling {len(active_worker_ids)} active local worker tasks...")
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
                     logger.warning(f"[TASKMGR] {len(pending)} local worker tasks did not finish gracefully after cancellation signal.")

        self._active_workers.clear()

        # Note: Redis connection pool is closed in main.py lifespan

        # Optional: Final cleanup if configured (might be better handled externally)
        if settings.CLEAN_ALL_ON_SHUTDOWN:
             logger.warning("[TASKMGR] CLEAN_ALL_ON_SHUTDOWN is enabled, but cleanup logic on shutdown is complex and potentially unsafe in multi-worker setup. Consider manual cleanup or robust periodic cleanup.")
             # Add cleanup logic here if absolutely necessary, but be cautious.

        logger.info("[TASKMGR] TaskManager shutdown complete for this worker.")


    def _cleanup_memory(self) -> None:
        """Perform garbage collection and potentially clear GPU cache."""
        logger.debug("[TASKMGR] Running post-task memory cleanup...")
        gc.collect()
        if settings.USE_CUDA and torch and torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
                logger.debug("[TASKMGR] CUDA cache cleared.")
            except Exception as e:
                logger.warning(f"[TASKMGR] Error clearing CUDA cache: {e}")
        if settings.SECURE_MEMORY_WIPING:
             logger.warning("[TASKMGR] Secure memory wiping is enabled - this may impact performance and effectiveness is limited.")
             # Implement actual wiping if needed, else remove this check


# --- Global Instance ---
# Initialization moved to main.py's lifespan function.
# The TaskManager instance is now accessed via app.state or dependency injection.
# task_manager: Optional[TaskManager] = None # Removed unused global variable