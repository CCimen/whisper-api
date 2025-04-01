# Privacy Review: app/services/task_manager.py

This module is central to managing the lifecycle of transcription tasks and plays a critical role in the application's privacy posture.

## Analysis

1.  **Task Representation (`Task` Class):**
    *   Each task is represented by a `Task` object holding its ID, type, parameters, status, progress, result, error messages, timestamps, and retry counts.
    *   Crucially, it includes a `_files_to_clean` list to track associated files needing deletion.

2.  **Task Storage:**
    *   A global `TaskManager` instance holds all active and recently completed `Task` objects in an **in-memory dictionary** (`self.tasks`).
    *   There is no indication of persistent storage (database, file system) for task data or results within this module. Data is lost on application restart unless saved externally.

3.  **Input File Handling & Cleanup:**
    *   **Registration:** When a `Task` is initialized, the input `file_path` from its parameters is automatically added to the `_files_to_clean` list via `register_file_for_cleanup`.
    *   **Security Check:** `register_file_for_cleanup` verifies that the file path is within allowed directories (`UPLOAD_DIR`, `RESULTS_DIR`) before adding it to the list, preventing accidental deletion of arbitrary files.
    *   **Cleanup Trigger:** The `perform_cleanup` method is called reliably when a task reaches a terminal state:
        *   Successful completion (`_execute_task_wrapper`).
        *   Permanent failure (`_execute_task_wrapper`).
        *   Cancellation (`cancel_task`, `_execute_task_wrapper`).
        *   Explicit deletion (`delete_task`).
        *   Automatic cleanup of old tasks (`cleanup_old_tasks`).
    *   **Secure Deletion:** `_secure_delete_file` handles the actual deletion.
        *   If `settings.SECURE_FILE_WIPING` is `True`, it overwrites the file with random data and then zeros before deleting (robust secure wipe).
        *   Otherwise, it performs a standard `os.remove()`.

4.  **Result Storage:**
    *   Upon successful task completion in `_execute_task_wrapper`, the result dictionary returned by the task handler (`process_audio`) is stored in the `task.result` attribute of the in-memory `Task` object.
    *   This result remains in memory until the `Task` object is deleted.

5.  **Memory Management (`_cleanup_memory`):**
    *   This method is called after a worker task finishes (`_worker_done_callback`).
    *   It explicitly triggers Python's garbage collector (`gc.collect()`) and clears the PyTorch CUDA cache (`torch.cuda.empty_cache()`) if CUDA is used. This helps release memory promptly.

6.  **Task Expiry (`cleanup_old_tasks`):**
    *   Provides a mechanism to automatically delete terminal tasks (Completed, Failed, Cancelled) older than a configured duration (`settings.JOB_CLEANUP_HOURS`).
    *   This deletion process also triggers `perform_cleanup`, ensuring associated files are deleted.
    *   **Note:** This method needs to be scheduled externally (e.g., via APScheduler) to run periodically.

## Privacy Considerations

*   **Excellent Input File Cleanup:** The `TaskManager` provides a robust and central mechanism for ensuring the original input audio files are deleted once they are no longer needed. The automatic registration and multiple trigger points for cleanup significantly reduce the risk of orphaned sensitive files.
*   **Secure Wipe Option:** The configurable `SECURE_FILE_WIPING` offers enhanced protection against data recovery for highly sensitive environments.
*   **In-Memory Results:** Storing results in memory limits their persistence footprint. However, they remain in memory until the task record is deleted (manually or via `cleanup_old_tasks`).
*   **Task Record Retention:** The `JOB_CLEANUP_HOURS` setting controls how long task records (and their in-memory results) are kept after completion/failure. Setting this appropriately balances the need for status checking with privacy requirements (shorter retention is generally better for privacy).
*   **Centralized Control:** Managing the task lifecycle and cleanup centrally within the `TaskManager` simplifies auditing and ensures consistency.

## Conclusion

The `TaskManager` is well-designed from a privacy perspective, particularly concerning the lifecycle management and cleanup of input audio files. Its main characteristic regarding results is the current in-memory storage tied to the task record. Adapting for session-based storage would involve intercepting the result before it's stored in `task.result` and redirecting it to the user's session, while still leveraging the `TaskManager`'s robust file cleanup mechanisms.