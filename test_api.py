#!/usr/bin/env python3
"""
Test script for whisper-api. Upload an audio file and get transcription with optional diarization.
Usage: python test_api.py audio.mp3 [--model tiny|small|medium|large] [--language sv] [--diarization]
"""

import sys
import time
import json
import argparse
import requests
from pprint import pprint

def test_transcription(file_path, model="medium", language="sv", diarization=False, api_url="http://localhost:8000"):
    """Test the transcription API with a file"""
    
    # Check if API is running
    try:
        response = requests.get(f"{api_url}/")
        print(f"API status: {response.json()['message']}")
    except Exception as e:
        print(f"Error connecting to API: {e}")
        sys.exit(1)
    
    # Check GPU status
    response = requests.get(f"{api_url}/gpu-status")
    gpu_info = response.json()
    print("\nGPU Information:")
    pprint(gpu_info)
    
    if not gpu_info.get("available", False):
        print("GPU not available. Cannot proceed.")
        sys.exit(1)
    
    # Start transcription
    print(f"\nUploading file: {file_path}")
    with open(file_path, "rb") as f:
        files = {"audio_file": f}
        params = {
            "model_size": model,
            "language": language,
            "diarization": diarization
        }
        
        if diarization:
            print("Speaker diarization enabled")
            # Add diarization parameters if you have specific speaker counts
            # params["num_speakers"] = 2  # Uncomment to specify exact number of speakers
            
        response = requests.post(f"{api_url}/transcriptions", files=files, params=params)
    
    if response.status_code != 200:
        print(f"Error starting transcription: {response.text}")
        sys.exit(1)
    
    job_id = response.json().get("id")
    print(f"Transcription started. Job ID: {job_id}")
    
    # Poll for status until complete
    while True:
        response = requests.get(f"{api_url}/transcriptions/{job_id}/status")
        status = response.json().get("status")
        print(f"Status: {status}")
        
        if status == "completed" or status == "error":
            break
            
        time.sleep(5)  # Wait 5 seconds before checking again
    
    if status == "error":
        print("Transcription failed:")
        response = requests.get(f"{api_url}/transcriptions/{job_id}")
        print(f"Error: {response.json().get('error', 'Unknown error')}")
        sys.exit(1)
    
    # Get results
    response = requests.get(f"{api_url}/transcriptions/{job_id}")
    result = response.json()
    
    # Print summary
    print("\nTranscription completed:")
    print(f"Duration: {result.get('duration', 0):.2f} seconds")
    print(f"Processing time: {result.get('processing_time', 0):.2f} seconds")
    
    # Print speakers if diarization was enabled
    if diarization and "speakers" in result and result["speakers"]:
        print("\nDetected speakers:")
        for speaker in result.get("speakers", []):
            print(f"- {speaker}")
    
    # Print full text
    print("\nFull transcription:")
    print("=" * 60)
    print(result.get("transcription", ""))
    
    # Print segments with timestamps and speakers
    print("\nSegments with timestamps:")
    print("=" * 60)
    for segment in result.get("segments", []):
        speaker_info = f" ({segment.get('speaker', '')})" if "speaker" in segment else ""
        print(f"[{segment['start']:.2f}s -> {segment['end']:.2f}s]{speaker_info} {segment['text']}")
    
    # Save to file
    output_file = f"{file_path}.transcription.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    
    print(f"\nFull results saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test the Whisper Transcription API")
    parser.add_argument("audio_file", help="Path to the audio file to transcribe")
    parser.add_argument("--model", "-m", choices=["tiny", "small", "medium", "large"], 
                        default="medium", help="Model size to use (default: medium)")
    parser.add_argument("--language", "-l", default="sv", 
                        help="Language code (default: sv for Swedish)")
    parser.add_argument("--diarization", "-d", action="store_true",
                        help="Enable speaker diarization")
    parser.add_argument("--api-url", default="http://localhost:8000",
                        help="API URL (default: http://localhost:8000)")
    
    args = parser.parse_args()
    
    test_transcription(
        args.audio_file, 
        model=args.model, 
        language=args.language,
        diarization=args.diarization,
        api_url=args.api_url
    )