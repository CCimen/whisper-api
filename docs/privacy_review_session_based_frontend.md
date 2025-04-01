# Proposed Changes for Session-Based Frontend Integration

This document outlines the necessary modifications to the Whisper API backend to support a session-based frontend, where transcription results are stored in the user's server-side session data instead of the global `TaskManager`.

## Goal

*   Store transcription results associated with a specific user session.
*   Prevent results from being stored globally in the `TaskManager`'s memory.
*   Allow the frontend to retrieve results from its own session context.
*   Maintain the existing robust cleanup of input audio files.

## Core Idea

Modify the task execution flow so that upon successful completion, the result dictionary is placed into the relevant user's session data rather than the `Task` object's `result` attribute. Add API endpoints to manage this session-based interaction.

## Required Modifications

**1. Session Management Implementation (Prerequisite)**

*   A robust server-side session management system must be implemented for FastAPI. Options include:
    *   Using libraries like `fastapi-sessions` with a backend (e.g., Redis, database) or secure cookies.
    *   Implementing custom middleware using libraries like `itsdangerous` for signed cookies.
*   This system must provide:
    *   A way to get the current user's session ID or session data within API route handlers (e.g., via FastAPI dependencies).
    *   A mechanism to store and retrieve data associated with a session ID server-side.
    *   Secure session handling (e.g., HTTPS, HttpOnly cookies, secure secret keys).
    *   Session expiration and cleanup mechanisms.

**2. `app/services/task_manager.py` Changes**

*   **`create_task` Method:**
    *   Modify to expect a `session_id` (or similar identifier) within the `params` dictionary passed from the API layer.
    *   Store this `session_id` within the `task.params` or potentially a dedicated `task.session_id` attribute if preferred.
*   **`_execute_task_wrapper` Method:**
    *   Inside the `try` block, after successfully getting the `result` from the handler (line 487):
        ```python
        # Original line (to be modified/removed):
        # task.update_progress(status=TaskStatus.COMPLETED, progress=1.0, result=result, ...)

        # --- New Logic ---
        with self._task_lock:
            task = self.tasks.get(task_id) # Re-get task
            if task and task.status != TaskStatus.CANCELLED:
                session_id = task.params.get("session_id") # Retrieve session identifier
                if session_id:
                    try:
                        # !!! CRITICAL DEPENDENCY !!!
                        # Requires a function/method to access the session management system
                        # This function needs to retrieve the session store based on session_id
                        # and save the 'result' dictionary into it.
                        # Example placeholder:
                        session_store = get_session_store() # Hypothetical function
                        session_store.set(session_id, "transcription_result", result)
                        logger.info(f"[TASKMGR] Stored result for task {task_id} in session {session_id}")

                        # Update task status WITHOUT the full result
                        task.update_progress(status=TaskStatus.COMPLETED, progress=1.0, result={"status": "stored_in_session"}, info={"stage": "task_successful"})

                    except Exception as session_err:
                        logger.error(f"[TASKMGR] Failed to store result for task {task_id} in session {session_id}: {session_err}", exc_info=True)
                        # Decide how to handle: fail the task? Store locally as fallback?
                        # Failing is safer from a consistency perspective:
                        task.update_progress(status=TaskStatus.FAILED, error=f"Failed to store result in session: {session_err}", info={"stage": "session_storage_failed"})

                else:
                    logger.error(f"[TASKMGR] Cannot store result for task {task_id}: Missing session_id in task parameters.")
                    task.update_progress(status=TaskStatus.FAILED, error="Missing session_id for result storage", info={"stage": "session_storage_failed"})

                # Cleanup is still performed regardless of session storage success/failure
                task.perform_cleanup()
        # --- End New Logic ---
        ```
    *   Ensure `task.perform_cleanup()` (line 503 or in the failure path) is still called to delete the input audio file.

**3. `app/api/routes/transcription.py` Changes**

*   **Add Session Dependency:** Inject the session management dependency into relevant routes.
*   **`submit_transcription_job` (`POST /`) Endpoint:**
    *   Get the current user's `session_id` from the injected session dependency.
    *   Add this `session_id` to the `task_params` dictionary *before* calling `task_manager.create_task`.
        ```python
        # Example (assuming 'session_data' is the dependency)
        session_id = session_data.get("session_id") # Or however the session ID is obtained
        if not session_id:
             raise HTTPException(status_code=400, detail="Session not found")

        task_params = request_params.model_dump(exclude_unset=True)
        task_params["file_path"] = saved_file_path
        task_params["model_size"] = request_params.model_size.value
        task_params["enable_diarization"] = task_params.pop("diarization", False)
        task_params["session_id"] = session_id # Inject session ID here

        task_id = task_manager.create_task(task_type="transcription", params=task_params)
        # ... rest of the function
        ```
*   **`get_transcription_task_result` (`GET /{task_id}`) Endpoint:**
    *   This endpoint becomes less relevant as the result is no longer stored in `task.result`.
    *   Consider modifying it to only return status/metadata or removing it entirely in favor of the session-based endpoint.
*   **Create New Endpoint (`GET /session/transcription` or similar):**
    *   Define a new route, e.g., `@router_transcription.get("/session/result", response_model=YourResultModel)`.
    *   Inject the session dependency.
    *   Retrieve the transcription result directly from the user's session data using the session dependency.
    *   Handle cases where the result is not yet available (task still processing) or not found.
    *   Return the result.
    *   Add an endpoint to clear the result from the session if needed (`DELETE /session/result`?).

## Benefits

*   **Data Isolation:** Transcription results are tied to specific user sessions, preventing accidental exposure between users.
*   **Reduced Global State:** Less data stored globally in the `TaskManager`, potentially improving scalability and reducing memory footprint over time (as results are cleared with sessions).
*   **Frontend Simplicity:** Frontend interacts with its own session context rather than polling task IDs globally.

## Dependencies

*   Requires a fully implemented and secure server-side session management system integrated with FastAPI.
*   Requires a mechanism within the `TaskManager` (or accessible to it) to interact with the session store based on a session ID.

This approach provides enhanced privacy and aligns better with typical web application patterns where user data is scoped to their session.