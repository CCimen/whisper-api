# 🎙️ Whisper Transcription API

[![Python](https://img.shields.io/badge/Python-3.10+-blue)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-green)](https://fastapi.tiangolo.com/)
[![PyTorch](https://img.shields.io/badge/PyTorch-CUDA-red)](https://pytorch.org/)
[![uv](https://img.shields.io/badge/uv-Managed-green)](https://github.com/astral-sh/uv)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Docker](https://img.shields.io/badge/Docker-Supported-blue)](https://www.docker.com/)

A high-performance API for audio transcription with optional speaker diarization, built with FastAPI and optimized for GPU acceleration and privacy. Designed for ease of use, maintainability, and integration, particularly for privacy-conscious users.

## ✨ Features

- **🚀 High-Performance Transcription**: Leverages GPU-accelerated Whisper models via `transformers`.
- **🇸🇪 Optimized for Swedish**: Easily configure KB-Whisper models (thanks to [KBLab](https://huggingface.co/KBLab)) via `.env` for superior Swedish transcription accuracy.
- **🔊 Speaker Identification**: Optional speaker diarization using `pyannote.audio` to identify _who_ spoke _when_.
- **⚙️ Tunable Diarization**: Control segmentation sensitivity and speaker clustering via API parameters (`segmentation_onset`, `clustering_threshold`, `segmentation_min_duration_off`).
- **🔒 Privacy Focused**:
  - **Automatic File Deletion**: Audio files automatically and securely deleted after processing (configurable).
  - **Secure Storage**: Options for memory-based storage (`tmpfs`) or persistent volumes with secure directory permissions.
  - **Anonymized Task IDs**: Uses UUIDs for task tracking.
  - **Optional Audit Logging**: Track request metadata without logging sensitive data.
- **⚙️ Asynchronous & Scalable**: Built with FastAPI and `asyncio`, uses a task queue (`TaskManager`) to handle concurrent requests efficiently based on configured limits (e.g., GPU VRAM).
- **🐳 Dockerized**: Optimized multi-stage Dockerfile for production using `uv`. Includes options for GPU (NVIDIA) and CPU builds. `docker-compose.yml` provided.
- **🔧 Configurable**: Settings managed via `.env` file and environment variables using `pydantic-settings`. Smart defaults based on detected system capabilities (e.g., GPU memory).
- **📊 API Monitoring**: Endpoints for system status, GPU details, model status, and task queue monitoring.
- **🎨 Enhanced Logging**: Uses `rich` for colorful, readable console logs, even in Docker.

## 🙏 Special Thanks

We extend our sincere gratitude to **Kungliga biblioteket (The National Library of Sweden)** for their outstanding work on the **KB-Whisper** models. Their models, trained on extensive Swedish speech data, significantly outperform standard Whisper models on Swedish tasks. This API makes it easy to utilize these models by simply changing the `DEFAULT_MODEL` setting in `.env` to `kblab-large`.

Visit [KBLab's Whisper models on Hugging Face](https://huggingface.co/KBLab) to learn more.

## 🚀 Quick Start

### Using Docker (Recommended)

This provides a consistent environment for deployment.

1.  **Clone Repository:**

    ```bash
    git clone https://github.com/CCimen/whisper-api.git
    cd whisper-api
    ```

2.  **Configure Environment (`.env`):**

    ```bash
    cp .env.example .env
    nano .env # Edit the file
    ```

    **Key settings to review:**

    - `API_AUTH_REQUIRED=true` (Recommended) & `API_KEY` (Generate: `python -c 'import secrets; print(secrets.token_urlsafe(32))'`)
    - `HUGGINGFACE_TOKEN` (Required if `DIARIZATION_ENABLED=true`. Get from [HF Settings](https://huggingface.co/settings/tokens)).
    - `DEFAULT_MODEL`: e.g., `kblab-large`, `large`, `medium`. See `app/config.py` for keys.
    - `AUTO_DELETE_AFTER_COMPLETION=true` (Recommended for privacy).
    - `UPLOAD_DIR`, `RESULTS_DIR`: Match `docker-compose.yml` volumes or `tmpfs` paths.
    - `MAX_CONCURRENT_TASKS`: Adjust based on your GPU VRAM (see `.env.example` comments).

3.  **Build & Run (Docker Compose):**

    - **GPU Version:** (Requires NVIDIA Drivers & Container Toolkit)
      ```bash
      docker compose build whisper-api
      docker compose up -d whisper-api
      ```
    - **CPU Version:**
      ```bash
      docker compose build --build-arg BASE_IMAGE=python:3.10-slim whisper-api
      docker compose up -d whisper-api
      ```

4.  **Access API:**
    - Docs: `http://localhost:8000/docs` (or your `${PORT}`)
    - Health: `http://localhost:8000/health/`

### Local Development (using `uv`)

1.  **Clone Repository & Install `uv`:** (See [astral.sh/uv](https://astral.sh/uv))
2.  **Create & Activate Virtual Environment:**
    ```bash
    uv venv
    source .venv/bin/activate # Or relevant activation command
    ```
3.  **Install Dependencies:**
    - **CPU-only (+ Diarization):** `uv pip install -e ".[diarization]"`
    - **GPU (+ Diarization):** Install PyTorch+CUDA first (see [pytorch.org](https://pytorch.org/)), then `uv pip install -e ".[diarization]"`. Example (CUDA 11.8):
      ```bash
      uv pip install torch==2.1.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu118
      uv pip install -e ".[diarization]"
      ```
4.  **Configure Environment (`.env`):** (As in Docker section)
5.  **Install FFmpeg:** (System dependency: `sudo apt install ffmpeg`, `brew install ffmpeg`, etc.)
6.  **Run API:**
    ```bash
    python app/cli.py --port 8000 # Or use --reload for dev
    ```
7.  **Access API:** `http://localhost:8000/docs`

## 🔧 Requirements

- Python 3.10+
- `uv` (for local setup)
- Docker & Docker Compose (for container deployment)
- NVIDIA GPU + Drivers + Container Toolkit (for GPU in Docker)
- FFmpeg (system dependency)
- Hugging Face Token (if using diarization)

## ⚙️ Configuration (`.env`)

Key settings explained (see `.env.example` for all):

| Variable                       | Description                                                                      | Example            |
| :----------------------------- | :------------------------------------------------------------------------------- | :----------------- |
| `DEFAULT_MODEL`                | Model to use by default (`tiny`, `small`, `medium`, `large`, `kblab-large`)      | `kblab-large`      |
| `USE_CUDA`                     | Use GPU if available (`true`/`false`)                                            | `true`             |
| `MAX_CONCURRENT_TASKS`         | Max simultaneous processing tasks (adjust based on VRAM)                         | `1`                |
| `DIARIZATION_ENABLED`          | Globally enable/disable speaker diarization                                      | `true`             |
| `HUGGINGFACE_TOKEN`            | **Required** for diarization models from Hugging Face                            | `hf_YourTokenHere` |
| `API_AUTH_REQUIRED`            | Enable API Key authentication (`true`/`false`)                                   | `true`             |
| `API_KEY`                      | Secret key if auth is enabled                                                    | `YourGeneratedKey` |
| `UPLOAD_DIR`                   | Path inside container for temporary uploads                                      | `/app/uploads`     |
| `RESULTS_DIR`                  | Path inside container for temporary results/processing files                     | `/app/results`     |
| `AUTO_DELETE_AFTER_COMPLETION` | **Privacy**: Delete audio files immediately after task finishes (`true`/`false`) | `true`             |
| `JOB_CLEANUP_HOURS`            | How long to keep task _metadata_ (results) in memory (0=very short)              | `24`               |
| `LOG_LEVEL`                    | App logging level (`DEBUG`, `INFO`, `WARNING`, `ERROR`)                          | `INFO`             |

## 🛡️ API Authentication

If `API_AUTH_REQUIRED=true` in `.env`:

1.  Set a strong, unique `API_KEY` in `.env`.
2.  Clients **must** send this key in the `X-API-Key` HTTP header.

```bash
curl -H "X-API-Key: your_secret_api_key" http://localhost:8000/system/status
```

## 📊 API Endpoints

Access interactive documentation (Swagger UI) at `/docs`.

- `/transcriptions/` (POST): Submit audio for transcription (+ optional diarization).
- `/diarize/` (POST): Submit audio for diarization only.
- `/transcriptions/{id}` / `/diarize/{id}` (GET): Get task results.
- `/transcriptions/{id}/status` / `/diarize/{id}/status` (GET): Get task status/progress.
- `/transcriptions/{id}` / `/diarize/{id}` (DELETE): Delete task record and files.
- `/health/` (GET): Basic API health check.
- `/system/status` (GET): Detailed system status (GPU, models, queue).
- `/system/models` (GET): List available models and load status.
- `/system/models/{name}/load` (POST): Request async model loading.
- `/system/models/{name}/unload` (POST): Request model unloading.

## 📖 Basic Usage

**Submit Transcription Job:**

```bash
curl -X POST "http://localhost:8000/transcriptions/" \
  -H "accept: application/json" \
  # -H "X-API-Key: your_secret_api_key" # Add if auth enabled
  -F "audio_file=@/path/to/audio.mp3" \
  -F "language=sv" \
  -F "model_size=kblab-large" \
  -F "diarization=true"
```

**Response (Example):**

```json
{
  "id": "generated-uuid-task-id",
  "status": "queued", // or pending, preparing, processing
  "progress": 0.0,
  "queue_position": 1,
  "error": null,
  "model": "whisper-kblab-large"
  // Other fields null initially
}
```

**Check Status:**

```bash
curl # -H "X-API-Key: your_secret_api_key" \
  "http://localhost:8000/transcriptions/generated-uuid-task-id/status"
```

**Get Results (when status is 'completed'):**

```bash
curl # -H "X-API-Key: your_secret_api_key" \
  "http://localhost:8000/transcriptions/generated-uuid-task-id"
```

## 🔒 Privacy Considerations

This API is designed with privacy in mind:

- **File Deletion**: Enable `AUTO_DELETE_AFTER_COMPLETION=true` (default is true in updated code). The TaskManager ensures the original uploaded audio and any temporary processed files (like preprocessed audio for diarization) associated with a task are securely deleted immediately after the task finishes (success, failure, or cancellation).

- **Storage**:

  - **Memory Storage** (Recommended for Privacy): Use tmpfs mounts in `docker-compose.yml` for `UPLOAD_DIR` and `RESULTS_DIR` (e.g., pointing them to `/dev/shm/...`). This ensures audio data resides only in RAM and is gone when the container stops. Requires sufficient host RAM.
  - **Persistent Storage**: If using standard Docker volumes, the host OS's filesystem security and encryption become important. The application creates directories with restrictive permissions (0o700), but data persists on disk until deleted by the app or volume removal.

- **Task IDs**: Anonymous UUIDs are used.

- **Logging**: Set `LOG_LEVEL=INFO` or `WARNING` in production. Filenames might appear in logs; sensitive payload data is avoided. Audit logs (`AUDIT_LOGGING_ENABLED=true`) track request metadata only. Ensure log file permissions are secure.

- **Authentication**: Strongly recommended. Enable `API_AUTH_REQUIRED=true` and use a strong `API_KEY`.

- **HTTPS**: Essential for production. Terminate TLS at a reverse proxy (Nginx, Traefik, Cloud Load Balancer) placed in front of the API container.

## 📄 License

This project is licensed under the MIT License. (You should add a LICENSE file with the MIT license text to the repository root).


## Additional info
Decision: We will keep the attn_implementation="flash_attention_2" line commented out in app/models/whisper_model.py for now.
Recommendation: For users with compatible hardware (Ampere+), they can achieve the performance boost by:
Installing the extra dependency: uv pip install flash-attn --no-build-isolation
Uncommenting the line # attn_implementation="flash_attention_2" in app/models/whisper_model.py or potentially making this configurable via an environment variable in the future. This provides the optimization path without breaking compatibility by default.
