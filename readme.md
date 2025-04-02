# 🎙️ Whisper Transcription API

<div align="center">

[![Python Version](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)](https://www.python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-05998b?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1+-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/Transformers-4.35+-ff69b4?logo=huggingface&logoColor=white)](https://huggingface.co/transformers)
[![Docker Support](https://img.shields.io/badge/Docker-Supported-blue?logo=docker&logoColor=white)](https://www.docker.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

</div>

<p align="center">
  A high-performance API for audio transcription with optional speaker diarization, built with FastAPI and optimized for GPU acceleration and privacy.
</p>

<div align="center">
  <a href="#-features">Features</a> •
  <a href="#-quick-start">Quick Start</a> •
  <a href="#-configuration">Configuration</a> •
  <a href="#-api-endpoints">API Endpoints</a> •
  <a href="#-basic-usage">Usage</a> •
  <a href="#-privacy-considerations">Privacy</a> •
  <a href="#-performance-tuning">Performance</a> •
  <a href="architecture.md">Architecture</a>
</div>

## ✨ Features

### 🚀 Performance
* GPU-accelerated Whisper models via `transformers`.
* Flash Attention 2 support for compatible GPUs.
* Asynchronous & scalable with FastAPI and `asyncio`.

### 🔊 Audio Processing
* Speaker identification (diarization) with `pyannote.audio`.
* Tunable diarization sensitivity.
* Swedish-optimized models via KBLab.

### 🔒 Privacy Focused
* Automatic file deletion after processing.
* Secure storage options (memory/persistent `tmpfs`).
* Anonymized task IDs using UUIDs.
* Optional audit logging.

### 🔧 Developer Friendly
* Dockerized with GPU/CPU support.
* Comprehensive API monitoring endpoints.
* Environment-based configuration (`.env`).
* Enhanced logging with `rich`.

For a detailed overview of the system's design and workflow, see the [Application Architecture](architecture.md) document.

## 🙏 Special Thanks

<p align="center">
  <a href="https://www.kb.se">
    <img src="https://www.kb.se/images/18.22ec630916021ebba936d5/1512642529193/logo.svg" alt="KB Logo" width="180">
  </a>
</p>

We extend our sincere gratitude to **Kungliga biblioteket (The National Library of Sweden)** for their outstanding work on the KB-Whisper models. Their models, trained on extensive Swedish speech data, significantly outperform standard Whisper models on Swedish tasks.

Visit [KBLab's Whisper models on Hugging Face](https://huggingface.co/KBLab/kb-whisper-large) to learn more.

## 🚀 Quick Start

### Using Docker (Recommended)

<details>
<summary><b>View Docker setup instructions</b></summary>

1.  **Clone Repository:**
    ```bash
    git clone https://github.com/CCimen/whisper-api.git
    cd whisper-api
    ```

2.  **Configure Environment:**
    * Copy the example environment file:
        ```bash
        cp .env.example .env
        ```
    * Edit the `.env` file with your preferred editor (e.g., `nano .env`).
    * **Key settings to review:**
        * `API_AUTH_REQUIRED=true` (Recommended). Generate a key:
            ```bash
            python -c 'import secrets; print(secrets.token_urlsafe(32))'
            ```
            Set this key to `API_KEY`.
        * `HUGGINGFACE_TOKEN`: Required if `DIARIZATION_ENABLED=true`. Get one from [Hugging Face](https://huggingface.co/settings/tokens).
        * `DEFAULT_MODEL`: Choose a model (e.g., `kblab-large`, `openai-large-v3-turbo`).
        * `AUTO_DELETE_AFTER_COMPLETION=true`: Enhances privacy by deleting files post-processing.

3.  **Build & Run:**
    * **GPU Version:**
        ```bash
        docker compose build whisper-api
        docker compose up -d whisper-api
        ```
    * **CPU Version:**
        ```bash
        # Build using a CPU base image
        docker compose build --build-arg BASE_IMAGE=python:3.10-slim whisper-api
        docker compose up -d whisper-api
        ```

4.  **Access API:**
    * **API Docs (Swagger UI):** `http://localhost:8000/docs`
    * **Health Check:** `http://localhost:8000/health/`

</details>

### Local Development

<details>
<summary><b>View local setup instructions</b></summary>

1.  **Prerequisites:**
    * Python 3.10+
    * [uv](https://astral.sh/uv) (or pip + venv)
    * FFmpeg: `sudo apt update && sudo apt install ffmpeg` (Debian/Ubuntu) or equivalent for your OS.
    * CUDA Toolkit (if using GPU) compatible with PyTorch.

2.  **Clone & Setup:**
    ```bash
    git clone https://github.com/CCimen/whisper-api.git
    cd whisper-api

    # Create virtual environment
    uv venv
    source .venv/bin/activate  # Or `.\.venv\Scripts\activate` on Windows

    # Install base dependencies
    uv pip install -e .

    # Install PyTorch with CUDA (adjust 'cu118' for your CUDA version)
    # See [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/)
    uv pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118

    # Install dependencies for diarization (optional)
    uv pip install -e ".[diarization]"
    ```

3.  **Configure & Run:**
    * Copy and edit the environment file:
        ```bash
        cp .env.example .env
        # Edit .env file (e.g., set HUGGINGFACE_TOKEN if using diarization)
        ```
    * Run the API server:
        ```bash
        python app/cli.py --port 8000 --reload  # --reload enables auto-reloading on code changes
        ```

</details>

## 🔧 Configuration

<details>
<summary><b>View configuration options</b></summary>

Configuration is managed via environment variables, typically set in the `.env` file.

**Key Settings:**

| Variable                       | Description                                     | Example              | Default           |
| :----------------------------- | :---------------------------------------------- | :------------------- | :---------------- |
| `DEFAULT_MODEL`                | Default model ID to load on startup           | `kblab-large`        | `openai-base`     |
| `USE_CUDA`                     | Use GPU acceleration if available               | `true`               | `true`            |
| `MAX_CONCURRENT_TASKS`         | Max simultaneous transcription/diarization tasks | `1`                  | `1`               |
| `DIARIZATION_ENABLED`          | Enable speaker identification feature           | `true`               | `true`            |
| `HUGGINGFACE_TOKEN`            | HF token for downloading diarization models     | `hf_YourTokenHere`   | `""`              |
| `API_AUTH_REQUIRED`            | Require `X-API-Key` header for requests         | `true`               | `false`           |
| `API_KEY`                      | The secret API key if auth is required        | `your-secret-key`    | `""`              |
| `AUTO_DELETE_AFTER_COMPLETION` | Delete audio files after task completion      | `true`               | `true`            |
| `MODELS_CACHE_DIR`             | Directory to store downloaded models          | `/app/models`        | `/app/models`     |
| `UPLOAD_DIR`                   | Directory for temporary audio file uploads      | `/app/uploads`       | `/app/uploads`    |
| `LOG_LEVEL`                    | Logging level (e.g., INFO, DEBUG)             | `INFO`               | `INFO`            |

See `.env.example` for all available options and their default values.

**Authentication:**

If `API_AUTH_REQUIRED=true`, clients must include the API key in the `X-API-Key` header with each request:

```bash
curl -H "X-API-Key: your_secret_api_key" http://localhost:8000/system/status
```

</details>

## 📊 API Endpoints

API documentation is available via Swagger UI at `/docs` and ReDoc at `/redoc` when the server is running.

<details>
<summary><b>View main endpoint categories</b></summary>

**Health Checks:**

| Method | Endpoint   | Description       |
| :----- | :--------- | :---------------- |
| `GET`  | `/health/` | API Health Check  |

**System & Monitoring:**

| Method | Endpoint                             | Description               |
| :----- | :----------------------------------- | :------------------------ |
| `GET`  | `/system/status`                     | Get Overall System Status |
| `GET`  | `/system/gpu`                        | Get Detailed GPU Status   |
| `GET`  | `/system/models`                     | List Available Models     |
| `POST` | `/system/models/{model_name}/load`   | Load a Model into memory  |
| `POST` | `/system/models/{model_name}/unload` | Unload a Model from memory|
| `GET`  | `/system/queue`                      | Get Task Queue Status     |

**Transcription:**

| Method   | Endpoint                         | Description                  |
| :------- | :------------------------------- | :--------------------------- |
| `POST`   | `/transcriptions/`               | Submit Transcription Job     |
| `GET`    | `/transcriptions/{task_id}/status` | Get Transcription Job Status |
| `GET`    | `/transcriptions/{task_id}`      | Get Transcription Job Result |
| `DELETE` | `/transcriptions/{task_id}`      | Delete Transcription Job     |

**Diarization (Speaker Identification Only):**

| Method   | Endpoint                   | Description                 |
| :------- | :------------------------- | :-------------------------- |
| `POST`   | `/diarize/`                | Submit Diarization Only Job |
| `GET`    | `/diarize/{task_id}/status`  | Get Diarization Task Status |
| `GET`    | `/diarize/{task_id}`       | Get Diarization Task Result |
| `DELETE` | `/diarize/{task_id}`       | Delete Diarization Task     |

</details>

## 📖 Basic Usage

<details>
<summary><b>Example API requests (using cURL)</b></summary>

Replace `your_secret_api_key` with your actual key if authentication is enabled.

**1. Submit Transcription Job (with Diarization):**

```bash
curl -X POST "http://localhost:8000/transcriptions/" \
  -H "accept: application/json" \
  -H "X-API-Key: your_secret_api_key" \
  -F "audio_file=@/path/to/your/audio.mp3" \
  -F "language=sv" \
  -F "model_size=kblab-large" \
  -F "diarization=true"
```
*This will return a JSON response containing the `task_id`.*

**2. Check Job Status:**

```bash
curl -H "accept: application/json" \
  -H "X-API-Key: your_secret_api_key" \
  "http://localhost:8000/transcriptions/{your-task-id}/status"
```

**3. Get Job Results:**

Once the status is `completed`:
```bash
curl -H "accept: application/json" \
  -H "X-API-Key: your_secret_api_key" \
  "http://localhost:8000/transcriptions/{your-task-id}"
```

**Example Status Response:**

```json
{
  "id": "generated-uuid-task-id",
  "status": "processing", // Can be queued, processing, completed, failed
  "progress": 50.0,
  "queue_position": 0,    // 0 if currently processing, >0 if queued
  "error": null,
  "model": "kblab-large" // Model used for this task
}
```

**Example Result Response (Simplified):**
```json
{
  "id": "generated-uuid-task-id",
  "status": "completed",
  "result": {
    "text": "Det här är en transkription...",
    "language": "sv",
    "segments": [
      {
        "start": 0.5,
        "end": 3.2,
        "text": "Det här är en transkription.",
        "speaker": "SPEAKER_00" // Added if diarization=true
      },
      // ... more segments
    ],
    "diarization": [ // Added if diarization=true
       {"speaker": "SPEAKER_00", "start": 0.5, "end": 3.2},
       {"speaker": "SPEAKER_01", "start": 4.1, "end": 6.8},
       // ... more speaker turns
    ]
  },
  "model": "kblab-large",
  "completed_at": "2023-10-27T10:30:00Z"
}
```

</details>

## 🔒 Privacy Considerations

<details>
<summary><b>View privacy features and recommendations</b></summary>

This API includes several features to enhance privacy:

* **Automatic File Deletion:**
    * Enabled by default (`AUTO_DELETE_AFTER_COMPLETION=true`).
    * Uploaded audio files are securely deleted from the `UPLOAD_DIR` immediately after the transcription/diarization task finishes (successfully or unsuccessfully).
* **Secure Storage Options:**
    * **Memory Storage (`tmpfs` - Recommended in Docker):** Configure `tmpfs` volumes in `docker-compose.yml` for the `UPLOAD_DIR` to store temporary files only in RAM, ensuring they vanish when the container stops. Example:
        ```yaml
        volumes:
          whisper_uploads:
            driver: local
            driver_opts:
              type: tmpfs
              device: tmpfs
              o: size=1g # Adjust size as needed
        services:
          whisper-api:
            volumes:
              - whisper_uploads:/app/uploads
        ```
    * **Persistent Storage:** If not using `tmpfs`, files are stored on disk in `UPLOAD_DIR`. Ensure this directory has restrictive permissions.
* **Anonymous Task IDs:** Uses non-sequential UUIDs for task identification.
* **Configurable Logging:** Set `LOG_LEVEL` (e.g., `INFO`, `WARNING`, `ERROR`) to control verbosity and minimize potentially sensitive data in logs.
* **API Authentication:** Use `API_AUTH_REQUIRED=true` and a strong `API_KEY` to prevent unauthorized access.
* **HTTPS:** Deploy behind a reverse proxy (like Nginx or Traefik) configured with TLS/SSL certificates to encrypt API traffic in production.

</details>

## ⚡ Performance Tuning

<details>
<summary><b>View performance optimization options</b></summary>

* **GPU Acceleration:** Ensure `USE_CUDA=true` (default) and a compatible NVIDIA GPU + CUDA Toolkit are available. This provides the most significant speedup.
* **Model Selection:** Smaller models (`tiny`, `base`, `small`) are faster but less accurate. Larger models (`medium`, `large`, `kblab-*`) are more accurate but slower and require more VRAM. Choose based on your needs.
* **Flash Attention 2:**
    * For compatible hardware (NVIDIA Ampere, Hopper, Ada Lovelace architecture GPUs) and CUDA 11.6+, enabling Flash Attention 2 can significantly accelerate transcription, especially for large models.
    * **Requirements:** `flash-attn` package.
    * **Installation:**
        ```bash
        # Ensure PyTorch is installed first
        uv pip install flash-attn --no-build-isolation
        ```
        *(Note: Installation might take time as it often compiles CUDA kernels.)*
    * **Configuration:** Uncomment (or add) this line in `app/models/whisper_model.py` within the `load_model` function:
        ```python
        # Inside the load_model function:
        model_kwargs = {}
        if device == "cuda" and os.getenv("ENABLE_FLASH_ATTENTION", "false").lower() == "true":
             # Check for flash-attn availability and compatibility if needed
             model_kwargs["attn_implementation"] = "flash_attention_2"
             print("INFO: Using Flash Attention 2")

        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id,
            cache_dir=cache_dir,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
            use_safetensors=True,
            **model_kwargs # Pass the kwargs here
        )
        ```
        *You might need to set an environment variable like `ENABLE_FLASH_ATTENTION=true` to control this.*
* **Concurrent Tasks:** Adjust `MAX_CONCURRENT_TASKS`. Increasing this allows processing multiple requests simultaneously but requires sufficient CPU/GPU resources. Start with `1` and increase cautiously based on monitoring.
* **Batch Processing (Advanced):** For very high throughput scenarios, consider modifying the code to support batching multiple audio files together (not implemented by default).

</details>

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
