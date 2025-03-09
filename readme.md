# Whisper Transcription API

A FastAPI-based service for audio transcription using KB-Whisper models on GPU, with optional speaker diarization.

## Features

- **Audio Transcription**: Transcribe audio files to text with timestamps
- **Model Selection**: Choose from multiple model sizes (tiny, small, medium, large)
- **Model Caching**: Smart caching of models for better performance
- **Optional Speaker Diarization**: Identify different speakers in audio (requires Hugging Face token)
- **Parallel Processing**: Option to run transcription and diarization in parallel
- **API with Swagger Documentation**: Easy-to-use REST API
- **Asynchronous Processing**: Handle long files efficiently

## Requirements

- CUDA-compatible GPU
- Python 3.8+
- FFmpeg
- Dependencies listed in requirements.txt

## Quick Development Setup

1. Clone this repository:
   ```bash
   git clone <repository-url>
   cd whisper-api
   ```

2. Create a virtual environment and activate it:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Copy environment template and set your values:
   ```bash
   cp .env.example .env
   # Edit .env with your preferred settings
   ```

5. Run the application:
   ```bash
   uvicorn app.main:app --reload
   ```

6. Access the API documentation:
   - Open your browser and go to http://localhost:8000/docs

## Production Setup

For production deployment, you can use the provided setup script:

```bash
sudo bash setup_production.sh
```

This will:
- Install necessary system dependencies
- Set up a Python virtual environment
- Install application dependencies
- Create a systemd service for automatic startup
- Start the service

To customize installation:
```bash
sudo bash setup_production.sh --dir=/custom/path --port=9000
```

## API Endpoints

### Check Status
```
GET /
```
Returns a simple status message indicating the API is running.

### GPU Information
```
GET /gpu-status
```
Returns information about available GPUs.

### Start Transcription
```
POST /transcriptions
```
Upload an audio file to start transcription. Parameters:
- `audio_file`: The audio file to transcribe
- `language`: Language code (default: "sv" for Swedish)
- `model_size`: Model size to use (tiny, small, medium, large)
- `diarization`: Enable speaker diarization (default: false)
- `num_speakers`: Fixed number of speakers (optional)
- `min_speakers`: Minimum number of speakers (optional)
- `max_speakers`: Maximum number of speakers (optional)

Returns a job ID that can be used to check status and retrieve results.

### Check Transcription Status
```
GET /transcriptions/{job_id}/status
```
Check the status of a transcription job.

### Get Transcription Results
```
GET /transcriptions/{job_id}
```
Get the results of a completed transcription job, including:
- Full transcription text
- Timestamped segments with speaker information (if diarization enabled)
- Audio duration
- Processing time

## Speaker Diarization Setup

To enable speaker diarization:

1. Uncomment the diarization dependencies in requirements.txt and install them:
   ```bash
   pip install pyannote.audio==3.1.1 pydub==0.25.1
   ```

2. Obtain a Hugging Face token with access to pyannote/speaker-diarization-3.1

3. Edit your .env file:
   ```
   DIARIZATION_ENABLED=True
   HUGGINGFACE_TOKEN=your-token-here
   ```

## Example Usage

1. Start a transcription:
   ```bash
   curl -X POST -F "audio_file=@audio.mp3" -F "model_size=medium" -F "diarization=true" http://localhost:8000/transcriptions
   ```

2. Check status:
   ```bash
   curl http://localhost:8000/transcriptions/your-job-id/status
   ```

3. Get results:
   ```bash
   curl http://localhost:8000/transcriptions/your-job-id
   ```

## Troubleshooting

If you encounter issues:

- Check the logs: `journalctl -u whisper-api -f` (if using systemd)
- Ensure your GPU is properly configured with CUDA
- Verify you have enough GPU memory for your selected model size
- For diarization issues, ensure your Hugging Face token has proper permissions

## Advanced Configuration

See the `.env.example` file for all available configuration options:

- Model caching behavior
- Memory management
- Diarization settings
- Parallel processing
- Job cleanup intervals