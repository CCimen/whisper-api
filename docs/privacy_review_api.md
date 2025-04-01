# Privacy Review: app/api/routes/transcription.py

This file defines the FastAPI endpoints for interacting with the transcription service.

## Analysis

1.  **Submission Endpoint (`POST /`)**
    *   **Input:** Receives transcription parameters (`TranscriptionRequest`) and the audio file (`UploadFile`). Performs validation (file size, diarization availability).
    *   **File Handling:**
        *   Calls `_save_upload_file` to save the uploaded audio to the configured `UPLOAD_DIR`.
        *   Uses a UUID for the filename (e.g., `uploads/some-uuid.mp3`).
        *   Sets file permissions to `0o600` (read/write for owner only).
        *   The path to this saved file (`saved_file_path`) is captured.
        *   **Error Handling:** If an error occurs *during submission* (before task creation/queuing), it attempts to delete the `saved_file_path`.
        *   **Success Handling:** If the task is successfully created and queued via `TaskManager`, this endpoint **does not delete** the `saved_file_path`. The file remains in `UPLOAD_DIR`.
    *   **Task Creation:** Creates a task using `task_manager.create_task`, passing the `saved_file_path` and other parameters. Queues the task using `task_manager.queue_task`.
    *   **Response:** Returns the initial task status (ID, status="queued", etc.), *not* the transcription result.

2.  **Status Endpoint (`GET /{task_id}/status`)**
    *   Retrieves task status information (excluding the result) from the `TaskManager`.
    *   Does not handle files or sensitive data directly.

3.  **Result Endpoint (`GET /{task_id}`)**
    *   Retrieves the full task information, including the result (`task_info.get("result")`), from the `TaskManager`.
    *   Returns the result dictionary (transcription, segments, etc.) directly in the API response body.
    *   Does not store the result itself or handle files.

4.  **Deletion Endpoint (`DELETE /{task_id}`)**
    *   Calls `task_manager.delete_task(task_id)`.
    *   Relies entirely on the `TaskManager` to handle the deletion logic, including triggering the cleanup of the associated input audio file.

## Privacy Considerations

*   **Input File Persistence:** The most significant point is that the API route saves the input file but **does not delete it upon successful task submission**. It correctly delegates this responsibility, but it means the file persists until the `TaskManager` performs cleanup based on task completion/failure/deletion.
*   **Temporary Storage Security:** Saving files with UUID names and restricted permissions (`0o600`) in a designated `UPLOAD_DIR` is a good practice for temporary storage.
*   **Result Transmission:** Results are fetched from the `TaskManager` and sent directly over HTTPS (assuming deployment behind a TLS-terminating proxy like Nginx/Traefik). No intermediate storage of results occurs in the API layer itself.
*   **Error Handling Cleanup:** The attempt to delete the saved file upon submission errors is good, preventing orphaned files in case of early failures.
*   **No Session Logic:** The API follows a standard asynchronous task pattern, not a session-based one. User context isn't explicitly managed here beyond potential API key authentication.

## Conclusion

The API layer correctly handles the initial saving of the uploaded file securely. Its primary privacy implication is the reliance on the `TaskManager` for the eventual deletion of this input file after successful submission. It acts as a pass-through for results stored by the `TaskManager`. Adapting this for session-based results would require significant changes, primarily around associating tasks with sessions and adding endpoints to interact with session data instead of fetching results directly via `task_id` from the `TaskManager`.