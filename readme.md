# 🎙️ Whisper Transcription API

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-green?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![PyTorch](https://img.shields.io/badge/PyTorch-CUDA-red?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)](https://opensource.org/licenses/MIT)
[![Docker](https://img.shields.io/badge/Docker-Supported-blue?style=for-the-badge&logo=docker&logoColor=white)](https://www.docker.com/)

</div>

<p align="center">
A high-performance API for audio transcription with optional speaker diarization, built with FastAPI and optimized for GPU acceleration and privacy.
</p>

<p align="center">
  <a href="#-features">Features</a> •
  <a href="#-quick-start">Quick Start</a> •
  <a href="#-api-endpoints">API Endpoints</a> •
  <a href="#-basic-usage">Usage</a> •
  <a href="#-privacy-considerations">Privacy</a> •
  <a href="architecture.md">Architecture</a>
</p>

---

## ✨ Features

- **🚀 High-Performance Transcription**
  - GPU-accelerated Whisper models via `transformers`
  - Efficient task queue for handling concurrent requests

- **🇸🇪 Optimized for Swedish**
  - Support for KB-Whisper models by [KBLab](https://huggingface.co/KBLab)
  - Superior Swedish transcription accuracy

- **🔊 Speaker Identification**
  - Optional speaker diarization using `pyannote.audio`
  - Identify who spoke when in multi-speaker audio

- **🔒 Privacy-Focused**
  - Automatic file deletion after processing
  - Secure storage options (memory-based or encrypted volumes)
  - Anonymized task tracking with UUIDs
  - Optional audit logging

- **⚙️ Configurable & Scalable**
  - Settings managed via `.env` file and environment variables
  - Smart defaults based on detected system capabilities
  - Built with FastAPI and `asyncio` for async processing

- **🐳 Containerized**
  - Optimized Docker setup with GPU support
  - Managed dependencies using `uv`
  - Ready-to-use docker-compose configuration

## 🙏 Special Thanks

We extend our gratitude to **Kungliga biblioteket (The National Library of Sweden)** for their outstanding work on the **KB-Whisper** models, which significantly outperform standard Whisper models on Swedish tasks.

Visit [KBLab's Whisper models on Hugging Face](https://huggingface.co/KBLab) to learn more.

## 🚀 Quick Start

### Using Docker (Recommended)

1. **Clone Repository**

   ```bash
   git clone https://github.com/CCimen/whisper-api.git
   cd openai-transcription-api
   ```

2. **Configure Environment**

   ```bash
   cp .env.example .env
   # Edit .env file with your preferred settings
   ```

   <details>
   <summary>Key settings to review</summary>

   - `API_AUTH_REQUIRED=true` (Recommended) 
   - `API_KEY=<your-secret-key>` (Generate with: `python -c 'import secrets; print(secrets.token_urlsafe(32))'`)
   - `HUGGINGFACE_TOKEN=<your-token>` (Required for diarization)
   - `DEFAULT_MODEL=kblab-large` (or other model key)
   - `AUTO_DELETE_AFTER_COMPLETION=true` (Privacy recommendation)
   - `MAX_CONCURRENT_TASKS=1` (Adjust based on GPU VRAM)

   </details>

3. **Build & Run**

   **GPU Version:**
   ```bash
   docker compose build whisper-api
   docker compose up -d whisper-api
   ```

   **CPU Version:**
   ```bash
   docker compose build --build-arg BASE_IMAGE=python:3.10-slim whisper-api
   docker compose up -d whisper-api
   ```

4. **Access API**
   - Documentation: `http://localhost:8000/docs`
   - Health check: `http://localhost:8000/health/`

### Local Development

<details>
<summary>Setup instructions for local development</summary>

1. **Clone Repository & Install Dependencies**

   ```bash
   git clone https://github.com/CCimen/openai-transcription-api.git
   cd openai-transcription-api
   
   # Install uv (if not already installed)
   # See https://astral.sh/uv for installation instructions
   
   # Create virtual environment
   uv venv
   source .venv/bin/activate
   
   # Install dependencies (CPU-only + Diarization)
   uv pip install -e ".[diarization]"
   
   # For GPU support (example with CUDA 11.8)
   uv pip install torch==2.1.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu118
   uv pip install -e ".[diarization]"
   ```

2. **Configure Environment**
   ```bash
   cp .env.example .env
   # Edit .env file with your preferred settings
   ```

3. **Install FFmpeg** (System dependency)
   ```bash
   # Ubuntu/Debian
   sudo apt install ffmpeg
   
   # macOS
   brew install ffmpeg
   ```

4. **Run API**
   ```bash
   python app/cli.py --port 8000 # Add --reload for development
   ```

5. **Access API**
   - Browse to `http://localhost:8000/docs`

</details>

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

For complete API documentation, see the [Architecture Document](architecture.md).

## 📖 Basic Usage

**Submit Transcription Job:**

```bash
curl -X POST "http://localhost:8000/transcriptions/" \
  -H "accept: application/json" \
  -H "X-API-Key: your_secret_api_key" \
  -F "audio_file=@/path/to/audio.mp3" \
  -F "language=sv" \
  -F "model_size=kblab-large" \
  -F "diarization=true"
```

**Check Status:**

```bash
curl -H "X-API-Key: your_secret_api_key" \
  "http://localhost:8000/transcriptions/your-task-id/status"
```

**Get Results:**

```bash
curl -H "X-API-Key: your_secret_api_key" \
  "http://localhost:8000/transcriptions/your-task-id"
```

## ⚙️ Configuration

<details>
<summary>Key configuration options (expand for details)</summary>

| Variable | Description | Example |
|----------|-------------|---------|
| `DEFAULT_MODEL` | Default model to use | `kblab-large` |
| `USE_CUDA` | Use GPU if available | `true` |
| `MAX_CONCURRENT_TASKS` | Max simultaneous tasks | `1` |
| `DIARIZATION_ENABLED` | Enable speaker identification | `true` |
| `HUGGINGFACE_TOKEN` | Token for diarization models | `hf_YourTokenHere` |
| `API_AUTH_REQUIRED` | Enable API key authentication | `true` |
| `AUTO_DELETE_AFTER_COMPLETION` | Delete files after processing | `true` |
| `JOB_CLEANUP_HOURS` | How long to keep task metadata | `24` |

</details>

## 🔒 Privacy Considerations

This API is designed with privacy in mind:

- **File Deletion**: Automatically deletes audio files after processing (when enabled)
- **Memory Storage**: Option to store files only in RAM using tmpfs
- **Authentication**: API key support for restricted access
- **Anonymized IDs**: Uses UUIDs for task tracking
- **Minimal Logging**: Configurable log levels to prevent exposing sensitive data

For detailed security recommendations, see the [Architecture Document](architecture.md#-privacy-considerations).

## 📄 License

This project is licensed under the [MIT License](LICENSE).
