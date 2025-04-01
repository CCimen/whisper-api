# 🏗️ Application Architecture

This document outlines the architecture and workflow of the Whisper Transcription API.

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
│   ├── processor.py          # Orchestrates the audio processing pipeline (transcription + diarization)
│   ├── task_manager.py       # Handles asynchronous task queuing and execution
│   └── transcriber.py        # (Potentially legacy or helper for transcription - Processor is main)
├── config.py                 # Application configuration (settings loading from .env)
├── exceptions.py             # Custom exception classes
└── main.py                   # FastAPI application entry point (middleware, lifespan, router inclusion)
```

## 🌊 Core Workflow (Transcription Task)

The following sequence diagram illustrates the typical flow when a user submits an audio file for transcription:

```mermaid
sequenceDiagram
    participant User
    participant FastAPI_API as FastAPI API<br>(main.py, routes/transcription.py)
    participant TaskManager as Task Manager<br>(services/task_manager.py)
    participant Processor as Audio Processor<br>(services/processor.py)
    participant ModelRegistry as Model Registry<br>(services/model_registry.py)
    participant WhisperModel as Whisper Model<br>(models/whisper_model.py)

    User->>+FastAPI_API: POST /transcriptions/ (audio file, params)
    FastAPI_API->>+TaskManager: create_task(type="transcription", params)
    TaskManager-->>-FastAPI_API: task_id
    FastAPI_API->>+TaskManager: queue_task(task_id)
    TaskManager-->>-FastAPI_API: Queued status (task_id, queue_position)
    FastAPI_API-->>-User: 202 Accepted (task_id, status="queued")

    Note over TaskManager: Worker becomes available
    TaskManager->>+Processor: process_audio(task_id, params, callback)
    Processor->>+ModelRegistry: get_model(model_key)
    ModelRegistry->>+WhisperModel: Check if loaded
    alt Model Not Loaded
        WhisperModel-->>-ModelRegistry: Not loaded
        ModelRegistry-->>-Processor: Model instance (not loaded)
        Processor->>TaskManager: Update Status: LOADING_MODEL (via callback)
        Processor->>+WhisperModel: load(device)
        Note over WhisperModel: Downloads/loads model weights & pipeline
        WhisperModel-->>-Processor: Model loaded
    else Model Already Loaded
        WhisperModel-->>-ModelRegistry: Loaded
        ModelRegistry-->>-Processor: Model instance (loaded)
    end
    Processor->>TaskManager: Update Status: PROCESSING (via callback)
    Processor->>+WhisperModel: transcribe(audio_path, ...)
    WhisperModel-->>-Processor: Transcription result (text, segments)
    Processor->>TaskManager: Update Status: COMPLETING (via callback)
    Note over Processor: Optional: Run Diarization & Assign Speakers
    Processor->>TaskManager: Update Status: COMPLETED, Result (via callback)
    Processor-->>-TaskManager: Return final result dict

    Note over User, TaskManager: User polls for status/result later
    User->>+FastAPI_API: GET /transcriptions/{task_id}
    FastAPI_API->>+TaskManager: get_task(task_id)
    TaskManager-->>-FastAPI_API: Task details (status="completed", result)
    FastAPI_API-->>-User: 200 OK (Transcription result)

```

**Explanation:**

1.  **Request:** The user sends a POST request with the audio file and parameters (language, model, diarization flag) to the FastAPI endpoint (`/transcriptions/`).
2.  **Task Creation:** The API route handler validates the request and calls the `TaskManager` to create a new task record, getting back a unique `task_id`.
3.  **Queuing:** The task is added to the `TaskManager`'s queue. The API immediately responds to the user with a `202 Accepted` status, including the `task_id` and the current status (e.g., `queued`).
4.  **Processing:** When a worker slot is free, the `TaskManager` dequeues the task and calls the `Audio Processor` (`process_audio` function).
5.  **Model Acquisition:** The `Processor` asks the `ModelRegistry` for the required `WhisperModel` instance.
6.  **Model Loading (if needed):** The `ModelRegistry` checks if the model is loaded. If not, the `Processor` updates the task status to `LOADING_MODEL` (via a callback to the `TaskManager`) and instructs the `WhisperModel` instance to load itself (which might involve downloading from Hugging Face).
7.  **Transcription:** Once the model is loaded, the `Processor` updates the status to `PROCESSING` and calls the `WhisperModel`'s `transcribe` method.
8.  **Diarization (Optional):** If requested, the `Processor` calls the `DiarizationService` and then assigns speaker labels to the transcription segments.
9.  **Completion:** The `Processor` updates the task status to `COMPLETED` via the `TaskManager` callback, storing the final result.
10. **Result Retrieval:** The user polls the `/transcriptions/{task_id}` endpoint using the `task_id` received earlier. Once the status is `completed`, the API retrieves the result from the `TaskManager` and returns it to the user.
11. **Cleanup:** The `TaskManager` handles automatic cleanup of associated files based on configuration.

## 🔑 Key Components

*   **FastAPI (`app/main.py`, `app/api/`)**: Handles HTTP requests, routing, request validation, and response formatting.
*   **TaskManager (`app/services/task_manager.py`)**: Manages the lifecycle of asynchronous tasks (queuing, execution, status tracking, results, cleanup). Decouples request handling from long-running processing.
*   **Audio Processor (`app/services/processor.py`)**: Orchestrates the steps involved in processing an audio file: getting the model, running transcription, running diarization (optional), and combining results.
*   **ModelRegistry (`app/services/model_registry.py`)**: Manages transcription model instances. Handles loading/unloading to manage resources (especially GPU VRAM). Allows different models to be used.
*   **WhisperModel (`app/models/whisper_model.py`)**: A wrapper around the Hugging Face `transformers` implementation of Whisper. Handles the specifics of loading a model and running the transcription pipeline.
*   **DiarizationService (`app/services/diarization.py`)**: Handles speaker diarization using `pyannote.audio`.
*   **Configuration (`app/config.py`)**: Loads settings from environment variables and `.env` files using `pydantic-settings`.

## ✨ Adding a New Transcription Model

To add support for a new transcription model (e.g., a different Whisper variant or a completely different ASR model):

1.  **Implement Model Wrapper:**
    *   Create a new class in `app/models/` (or modify `whisper_model.py` if it's a Whisper variant).
    *   This class **must** inherit from `app.services.model_registry.TranscriptionModel`.
    *   Implement the required methods:
        *   `__init__(self, model_size: str, ...)`: Initialize with necessary parameters. Call `super().__init__(...)`.
        *   `load(self, device: Optional[str] = None)`: Logic to load the model weights, processor, pipeline, etc., onto the specified device. Set `self._loaded = True` on success.
        *   `unload(self)`: Logic to release model resources (delete references, clear GPU cache). Set `self._loaded = False`.
        *   `is_loaded(self) -> bool`: Return `True` if the model is fully loaded and ready, `False` otherwise.
        *   `transcribe(self, audio_path: str, language: Optional[str] = None, task: str = "transcribe", ...) -> Dict[str, Any]`: Perform the actual transcription. Must return a dictionary containing at least `"text"` and `"segments"` (list of dicts with `"start"`, `"end"`, `"text"`).
    *   Add the `@ModelRegistry.register` class decorator above your new model class definition. This makes it discoverable.

2.  **Update Configuration (`app/config.py`):**
    *   Add a new key-value pair to the `WHISPER_MODEL_MAPPING` dictionary. The key is the short name you'll use in API requests (e.g., `"my-custom-model"`), and the value is the identifier your model's `__init__` method expects (e.g., its Hugging Face path or a specific size identifier).

    ```python
    # In app/config.py
    WHISPER_MODEL_MAPPING = {
        # ... existing models ...
        "my-custom-model": "huggingface/path-to-your-model",
        "whisper-tiny-en": "openai/whisper-tiny.en", # Example
    }
    ```

3.  **Update `README.md` (Optional but Recommended):**
    *   Add your new model key to the list of available models in the documentation.

4.  **Restart API:** The application needs to restart to pick up the new model registration and configuration.

Now, users can select your new model using the key you defined in `WHISPER_MODEL_MAPPING` (e.g., `model_size=my-custom-model`) when submitting transcription requests. The `ModelRegistry` and `Processor` will handle loading and using it automatically.