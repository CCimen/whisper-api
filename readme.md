# Whisper Transcription API

A high-performance API for audio transcription with speaker identification, built with FastAPI and optimized for GPU acceleration.

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.95%2B-green)](https://fastapi.tiangolo.com/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

</div>

## Features

- **🎯 High-Performance Transcription**: Uses GPU-accelerated Whisper models for ~5-10x realtime processing
- **🔊 Speaker Identification**: Optional diarization to identify who said what (requires Hugging Face token)
- **📊 Real-time Progress**: Track processing status with detailed progress updates
- **🔄 Asynchronous Processing**: Handle large files without timeouts
- **🎛️ Flexible Model Selection**: Choose from tiny, small, medium, or large models
- **🚀 Production-Ready**: Optimized memory management, error handling, and resource monitoring
- **📱 REST API**: Simple integration with any frontend or service
- **📄 OpenAPI Documentation**: Auto-generated Swagger UI documentation

## Requirements

- **Python 3.8+**
- **CUDA-compatible GPU** (NVIDIA) with 4GB+ VRAM (8GB+ recommended)
- **FFmpeg** (for audio processing)

## Quick Start

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/username/whisper-api.git
   cd whisper-api
   ```

2. Set up a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Configure environment variables:
   ```bash
   cp .env.example .env
   # Edit .env file with your settings
   ```

### Running in Development Mode

```bash
python run.py --debug
```

Visit http://localhost:8000/docs to see the API documentation.

### Running in Production

```bash
# Basic production deployment
python run.py --host 0.0.0.0 --port 8000

# With model preloading
python run.py --preload-model medium

# With memory limits
python run.py --memory-limit 6.0  # Limit to 6GB VRAM
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Health check |
| `/status` | GET | System status, GPU info, and capabilities |
| `/gpu-status` | GET | Detailed GPU information |
| `/transcriptions` | POST | Start a new transcription job |
| `/transcriptions/{job_id}/status` | GET | Check job status and progress |
| `/transcriptions/{job_id}` | GET | Get transcription results |
| `/transcriptions/{job_id}` | DELETE | Delete a job |
| `/docs` | GET | OpenAPI documentation |

## Transcription Workflow

1. **Submit an audio file** with `POST /transcriptions`
   ```bash
   curl -X POST -F "audio_file=@your-audio.mp3" \
        -F "model_size=medium" \
        -F "diarization=false" \
        http://localhost:8000/transcriptions
   ```

2. **Get the job ID** from the response
   ```json
   {"id": "550e8400-e29b-41d4-a716-446655440000", "status": "pending", "progress": 0.0}
   ```

3. **Check job status** with `GET /transcriptions/{job_id}/status`
   ```bash
   curl http://localhost:8000/transcriptions/550e8400-e29b-41d4-a716-446655440000/status
   ```
   
   Response:
   ```json
   {
     "id": "550e8400-e29b-41d4-a716-446655440000",
     "status": "transcribing", 
     "progress": 0.45
   }
   ```

4. **Get results** when status is "completed"
   ```bash
   curl http://localhost:8000/transcriptions/550e8400-e29b-41d4-a716-446655440000
   ```
   
   Response:
   ```json
   {
     "id": "550e8400-e29b-41d4-a716-446655440000",
     "status": "completed",
     "progress": 1.0,
     "transcription": "This is the full transcription text...",
     "segments": [
       {
         "start": 0.0,
         "end": 2.5,
         "text": "This is the",
         "speaker": "SPEAKER_0"
       },
       {
         "start": 2.5,
         "end": 5.0,
         "text": "full transcription text",
         "speaker": "SPEAKER_0"
       }
     ],
     "speakers": ["SPEAKER_0", "SPEAKER_1"],
     "duration": 5.0,
     "processing_time": 1.2
   }
   ```

## Speaker Diarization

Speaker diarization (identifying who said what) is available as an optional feature. To enable it:

1. Get a Hugging Face token with access to `pyannote/speaker-diarization-3.1`
2. Set these environment variables:
   ```
   DIARIZATION_ENABLED=True
   HUGGINGFACE_TOKEN=your_huggingface_token
   ```
3. Install additional dependencies:
   ```bash
   pip install pyannote.audio==3.1.1 pydub==0.25.1 numpy<2.0.0
   ```
4. Add `diarization=true` parameter when submitting transcription jobs

## Performance Tuning

### Model Selection

Choose the appropriate model size based on your needs:

| Model | VRAM Required | Speed | Accuracy |
|-------|---------------|-------|----------|
| tiny | ~1 GB | Fastest | Lowest |
| small | ~2 GB | Fast | Good |
| medium | ~4 GB | Medium | Better |
| large | ~8 GB | Slowest | Best |

### GPU Memory Management

Set these environment variables to control GPU memory usage:

```
# Keep only the current model in memory (unload others)
KEEP_MULTIPLE_MODELS_IN_MEMORY=False

# Minimum free VRAM to maintain (GB)
MIN_FREE_MEMORY_GB=3.0
```

Or use command-line options:

```bash
python run.py --memory-limit 6.0
```

### Processing Long Files

For very long audio files:

1. Use the `tiny` or `small` model to conserve memory
2. Disable diarization or use a smaller chunk size:
   ```
   DIARIZATION_CHUNK_DURATION=180  # 3 minutes
   ```

## Frontend Integration

This API is designed to work seamlessly with frontend applications. For React applications:

1. Use `fetch` or `axios` to submit audio files
2. Poll the status endpoint at regular intervals (e.g., every 1-2 seconds)
3. Update a progress bar based on the `progress` value
4. Display results when status is `completed`

Example React component available in the `/examples` folder.

## Monitoring and Maintenance

### Health Checks

Monitor the root endpoint for basic availability:

```bash
curl http://localhost:8000/
```

For detailed system status:

```bash
curl http://localhost:8000/status
```

### Log Files

Logs are written to standard output and can be redirected:

```bash
python run.py > whisper-api.log 2>&1
```

### Memory Usage

Monitor GPU memory usage with the included script:

```bash
python tools/gpu_monitor.py
```

## Production Deployment

### Running as a Service

Use the included systemd service file:

```bash
sudo cp deployment/whisper-api.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable whisper-api
sudo systemctl start whisper-api
```

### Using Supervisor

Sample supervisor configuration included in `deployment/supervisor.conf`.

### Docker (Limited GPU Support)

Docker deployment is possible but with limited GPU support. See `deployment/docker` for details.

## Technical Architecture

- **FastAPI**: Web framework for API endpoints
- **PyTorch**: Deep learning framework for Whisper models
- **Uvicorn**: ASGI server for production deployment
- **Whisper Models**: Four model sizes (tiny, small, medium, large)
- **Pyannote.Audio**: Speaker diarization (optional)

### Memory Optimization Techniques

1. **Model Caching**: Intelligent model caching based on usage
2. **Memory Management**: Proactive GPU memory cleanup
3. **Chunked Processing**: Large files processed in manageable chunks
4. **Optimized FFT**: CPU-forced FFT operations to avoid cuFFT errors
5. **Progress Tracking**: Detailed progress tracking with minimal overhead

## Contributing

Contributions are welcome! Please check out our [Contributing Guidelines](CONTRIBUTING.md).

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [OpenAI Whisper](https://github.com/openai/whisper)
- [KBLab Whisper Models](https://huggingface.co/KBLab)
- [Pyannote.Audio](https://github.com/pyannote/pyannote-audio)
- [FastAPI](https://fastapi.tiangolo.com/)