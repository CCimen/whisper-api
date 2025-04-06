# 🏗️ Application Architecture

This document outlines the architecture and workflow of the Whisper Transcription API.

## 📋 Table of Contents
- [Project Structure](#-project-structure)
- [Core Workflow](#-core-workflow-transcription-task)
- [Key Components](#-key-components)
- [API Endpoints](#-api-endpoints)
- [Adding a New Transcription Model](#-adding-a-new-transcription-model)

## 📂 Project Structure

The project follows a standard structure for FastAPI applications:

```
app/
├── api/                      # API layer (FastAPI routers and request handling)
│   ├── router_registry.py    # Router definitions
│   └── routes/               # Route handlers (health, system, transcription, diarization)
├── models/                   # Model definitions (e.g., Whisper model wrapper)
│   └── whisper_model.py      # Whisper model implementation using Transformers
├── services/                 # Core business logic and services
│   ├── diarization.py        # Speaker diarization service (using pyannote)
│   ├── model_registry.py     # Manages loading/unloading of transcription models
│   ├── processor.py          # Orchestrates the audio processing pipeline
│   └── task_manager.py       # Handles asynchronous task queuing and execution
├── cli.py                    # Command-line interface script
├── config.py                 # Application configuration (settings from .env)
├── exceptions.py             # Custom exception classes
├── logging_config.py         # Logging setup
└── main.py                   # FastAPI application entry point
```

## 🌊 Core Workflow (Transcription Task)

### High-Level Overview

The following diagram shows the simplified flow of a transcription request through the system, highlighting the role of Redis:

```mermaid
sequenceDiagram
    participant User
    participant API as FastAPI Worker
    participant Redis
    participant WorkerN as FastAPI Worker N

    User->>API: POST /transcriptions/ (audio)
    API->>Redis: HSET task:<id> status=PENDING, params=..., cancelled=false
    API->>Redis: LPUSH task_queue <id>
    API-->>User: 202 Accepted (task_id)

    Note over WorkerN, Redis: Worker N waits via BRPOP task_queue
    Redis-->>WorkerN: <id> (Task ID popped)
    WorkerN->>Redis: SADD active_tasks <id>
    WorkerN->>Redis: HSET task:<id> status=PROCESSING, started_at=...
    Note over WorkerN: Worker checks HGET task:<id> cancelled periodically
    alt Task Cancelled During Processing
        WorkerN->>Redis: HGET task:<id> cancelled -> "true"
        Note over WorkerN: Abort processing, cleanup local files
        WorkerN->>Redis: HSET task:<id> status=CANCELLED, completed_at=...
        WorkerN->>Redis: SREM active_tasks <id>
    else Task Completes Normally
        Note over WorkerN: Processes audio (calls Processor -> Model/Diarization)
        WorkerN->>Redis: HSET task:<id> progress=... (Periodically via callback)
        Note over WorkerN: Processing complete
        WorkerN->>Redis: SET result:<id> {result_json} EX <ttl>
        WorkerN->>Redis: HSET task:<id> status=COMPLETED, result_key=result:<id>, completed_at=...
        WorkerN->>Redis: SREM active_tasks <id>
    else Task Fails During Processing
        Note over WorkerN: Catches exception
        WorkerN->>Redis: HSET task:<id> status=FAILED, error=..., completed_at=...
        WorkerN->>Redis: SREM active_tasks <id>
    end

    User->>API: GET /transcriptions/<id>
    API->>Redis: HGETALL task:<id>
    alt Task Completed
        Redis-->>API: Task details (status=COMPLETED, result_key=...)
        API->>Redis: GET result:<id>
        Redis-->>API: {result_json}
        API-->>User: 200 OK (Full result)
    else Task Pending/Processing/Cancelled
        Redis-->>API: Task details (status=...)
        API-->>User: 200 OK (Status only)
    else Task Failed
         Redis-->>API: Task details (status=FAILED, error=...)
         API-->>User: 200 OK (Status with error)
    end
```

<!-- Removed outdated detailed diagram -->

### Workflow Explained:

1. **Request Handling**:
   - User sends a POST request with audio file and parameters.
   - API (FastAPI Worker) validates the request.
   - `TaskManager` creates task metadata in a Redis Hash (`task:<id>`).
   - `TaskManager` pushes the `task_id` onto the Redis List (`task_queue`).
   - API returns the `task_id` and `queued` status to the user.

2. **Task Processing (by any available worker)**:
   - A worker process uses `BRPOP` to wait for and retrieve a `task_id` from `task_queue`.
   - Worker adds `task_id` to the `active_tasks` Redis Set.
   - Worker updates the task status to `PROCESSING` in the Redis Hash.
   - Worker calls the `Audio Processor`.
   - `Processor` gets the required model via `ModelRegistry` (loading if necessary).
   - `Processor` calls the model's `transcribe` method.
   - `Processor` periodically updates task progress in the Redis Hash via callbacks.

3. **Optional Diarization**:
   - If requested, the `Processor` calls the `DiarizationService`.
   - The `Processor` assigns identified speakers to transcription segments.

4. **Task Completion**:
   - On success: `Processor` returns the result dictionary. The worker stores the result JSON in a Redis String (`result:<id>`) with a TTL, updates the task hash status to `COMPLETED` (including the `result_key`), and removes the `task_id` from `active_tasks`.
   - On failure: Worker catches the exception, updates the task hash status to `FAILED` with the error message, and removes the `task_id` from `active_tasks`.
   - On cancellation: If the `cancelled` flag in the task hash is set to `true`, the worker aborts processing, updates status to `CANCELLED`, and removes from `active_tasks`.

5. **Result Retrieval**:
   - User polls `GET /transcriptions/{task_id}`.
   - API worker fetches task details from the Redis Hash (`task:<id>`).
   - If status is `COMPLETED`, the API worker fetches the result JSON from the `result:<id>` key in Redis.
   - API returns the status and potentially the full result to the user.

6. **Cleanup**:
   - Uploaded files are deleted after processing based on configuration.
   - Task results in Redis expire automatically based on `JOB_CLEANUP_HOURS`.
   - Task metadata hashes remain until explicitly deleted or cleaned up by a potential future cleanup job.

## 🔑 Key Components

* **FastAPI (`app/main.py`, `app/api/`)**: 
  Handles HTTP requests, routing, validation, and responses

* **Redis**: Acts as the central message broker and database for task queue, task state/metadata, results, and active task tracking.
* **TaskManager (`app/services/task_manager.py`)**: Interacts with Redis to manage the lifecycle of asynchronous tasks (creation, queuing, status updates, result storage, cancellation). Runs a background loop in each worker to process tasks from the Redis queue.

* **Audio Processor (`app/services/processor.py`)**: 
  Orchestrates audio processing steps - model acquisition, transcription, diarization

* **ModelRegistry (`app/services/model_registry.py`)**: 
  Handles model instances, loading/unloading, and resource management

* **WhisperModel (`app/models/whisper_model.py`)**: 
  Wraps the Hugging Face implementation of Whisper

* **DiarizationService (`app/services/diarization.py`)**:
  Handles speaker diarization using `pyannote.audio` (if dependencies are installed).

* **Configuration (`app/config.py`)**: 
  Loads settings from environment variables using pydantic-settings

## 📊 API Endpoints

Access interactive documentation at `/docs`

### Health Checks

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health/` | API Health Check |

### System & Monitoring

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/system/status` | Get Overall System Status |
| `GET` | `/system/gpu` | Get Detailed GPU Status |
| `GET` | `/system/models` | List Available Models |
| `POST` | `/system/models/{model_name}/load` | Load a Model |
| `POST` | `/system/models/{model_name}/unload` | Unload a Model |
| `GET` | `/system/queue` | Get Task Queue Status |

### Transcription

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/transcriptions/` | Submit Transcription Job |
| `GET` | `/transcriptions/{task_id}/status` | Get Transcription Job Status |
| `GET` | `/transcriptions/{task_id}` | Get Transcription Job Result |
| `DELETE` | `/transcriptions/{task_id}` | Delete Transcription Job |

### Diarization

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/diarize/` | Submit Diarization Only Job |
| `GET` | `/diarize/{task_id}/status` | Get Diarization Task Status |
| `GET` | `/diarize/{task_id}` | Get Diarization Task Result |
| `DELETE` | `/diarize/{task_id}` | Delete Diarization Task |

## ✨ Adding a New Transcription Model

To add support for a new transcription model:

1. **Implement Model Wrapper**:
   - Create a new Python file in `app/models/` (e.g., `my_new_model.py`).
   - Define a class inheriting from `app.services.model_registry.TranscriptionModel`.
   - Implement the required methods: `load`, `unload`, `is_loaded`, and `transcribe`.
   - **Crucially**, decorate the class with `@ModelRegistry.register` and ensure the class has a unique `name` attribute (e.g., `name = "my-new-model"`).
     ```python
     # In app/models/my_new_model.py
     from app.services.model_registry import TranscriptionModel, ModelRegistry

     @ModelRegistry.register
     class MyNewModel(TranscriptionModel):
         name = "my-new-model" # Unique name for registration

         def __init__(self, ...): # Add necessary init args
             super().__init__(...)
             self._loaded = False
             # ... other initialization ...
             
         def load(self, device: Optional[str] = None):
             # Logic to load model weights
             # ...
             self._loaded = True
             
         def unload(self):
             # Release resources
             # ...
             self._loaded = False
             
         def is_loaded(self) -> bool:
             return self._loaded
             
         def transcribe(self, audio_path: str, **kwargs) -> Dict[str, Any]:
             # Perform transcription using your model
             # ...
             return {"text": "...", "segments": [...]}
     ```

2. **Ensure Discovery**:
   - The `ModelRegistry` automatically discovers models in the `app.models` package when the application starts (specifically, when `ModelRegistry.available_models()` or `get_model()` is first called). Ensure your new file is imported correctly (e.g., via `app/models/__init__.py` if needed, though the decorator often handles this).

3. **Update Documentation**:
   - Add your new model key (e.g., `"my-new-model"`) to the relevant sections of the README.md.

4. **Restart API**:
   - The application needs to restart to pick up the new Python file and register the model class during discovery.
