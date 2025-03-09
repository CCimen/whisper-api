#!/usr/bin/env python3
"""
Performance benchmarking tool for Whisper Transcription API.

This script tests the API with various models and file sizes
to measure performance and resource usage.
"""

import os
import sys
import time
import json
import argparse
import asyncio
import aiohttp
import statistics
from datetime import datetime
import concurrent.futures

try:
    import psutil
    import numpy as np
    from tabulate import tabulate
    from tqdm import tqdm
except ImportError:
    print("This tool requires additional packages. Install with:")
    print("pip install psutil numpy tabulate tqdm")
    sys.exit(1)

# ANSI color codes
RESET = "\033[0m"
BOLD = "\033[1m"
RED = "\033[91m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
MAGENTA = "\033[95m"
CYAN = "\033[96m"

class PerformanceTester:
    """Performance testing framework for Whisper API."""
    
    def __init__(self, api_url, test_files, models):
        self.api_url = api_url
        self.test_files = test_files
        self.models = models
        self.results = {}
    
    async def test_file(self, session, file_path, model, diarization=False):
        """Test a single file with specific model."""
        file_name = os.path.basename(file_path)
        file_size = os.path.getsize(file_path) / (1024 * 1024)  # MB
        
        print(f"\n{BLUE}Testing {file_name} ({file_size:.2f} MB) with {model} model{RESET}")
        print(f"Diarization: {diarization}")
        
        start_time = time.time()
        
        try:
            # Upload file
            print(f"Uploading file...")
            async with session.post(
                f"{self.api_url}/transcriptions",
                data={
                    "model_size": model,
                    "language": "en",
                    "diarization": str(diarization).lower()
                },
                files={"audio_file": open(file_path, "rb")}
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    print(f"{RED}Error: {response.status} - {error_text}{RESET}")
                    return None
                
                upload_response = await response.json()
                job_id = upload_response.get("id")
                
                if not job_id:
                    print(f"{RED}Error: No job ID returned{RESET}")
                    return None
                
                print(f"Job ID: {job_id}")
            
            # Poll for status
            print(f"Processing...")
            completed = False
            status_checks = 0
            max_checks = 300  # 5 minutes with 1s interval
            progress_bar = tqdm(total=100, desc="Progress")
            last_progress = 0
            
            while not completed and status_checks < max_checks:
                await asyncio.sleep(1)
                status_checks += 1
                
                async with session.get(f"{self.api_url}/transcriptions/{job_id}/status") as response:
                    if response.status != 200:
                        error_text = await response.text()
                        print(f"{RED}Error checking status: {response.status} - {error_text}{RESET}")
                        return None
                    
                    status_response = await response.json()
                    status = status_response.get("status")
                    progress = status_response.get("progress", 0) * 100
                    
                    # Update progress bar
                    if progress > last_progress:
                        progress_diff = progress - last_progress
                        progress_bar.update(progress_diff)
                        last_progress = progress
                    
                    if status == "completed" or status == "error":
                        completed = True
                        progress_bar.update(100 - last_progress)
                        progress_bar.close()
                        
                        if status == "error":
                            print(f"{RED}Job failed: {status_response.get('error', 'Unknown error')}{RESET}")
                            return None
            
            if not completed:
                print(f"{RED}Timeout waiting for job to complete{RESET}")
                return None
            
            # Get results
            async with session.get(f"{self.api_url}/transcriptions/{job_id}") as response:
                if response.status != 200:
                    error_text = await response.text()
                    print(f"{RED}Error getting results: {response.status} - {error_text}{RESET}")
                    return None
                
                result = await response.json()
            
            processing_time = time.time() - start_time
            
            # Extract metrics
            duration = result.get("duration", 0)
            transcription_time = result.get("processing_time", 0)
            transcription_length = len(result.get("transcription", ""))
            segment_count = len(result.get("segments", []))
            
            # Calculate metrics
            realtime_factor = duration / transcription_time if transcription_time > 0 else 0
            chars_per_second = transcription_length / transcription_time if transcription_time > 0 else 0
            
            print(f"\n{GREEN}Test completed in {processing_time:.2f}s{RESET}")
            print(f"Audio duration: {duration:.2f}s")
            print(f"Processing time: {transcription_time:.2f}s")
            print(f"Realtime factor: {realtime_factor:.2f}x")
            print(f"Characters: {transcription_length}")
            print(f"Segments: {segment_count}")
            
            return {
                "file_name": file_name,
                "file_size_mb": file_size,
                "model": model,
                "diarization": diarization,
                "duration": duration,
                "processing_time": transcription_time,
                "total_time": processing_time,
                "realtime_factor": realtime_factor,
                "transcription_length": transcription_length,
                "segment_count": segment_count,
                "chars_per_second": chars_per_second
            }
            
        except Exception as e:
            print(f"{RED}Error: {str(e)}{RESET}")
            return None
    
    async def run_tests(self):
        """Run all tests."""
        print(f"{BOLD}{BLUE}Starting performance tests for Whisper API at {self.api_url}{RESET}\n")
        
        # Check API status
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.api_url}/status") as response:
                    if response.status != 200:
                        print(f"{RED}Error: API not available ({response.status}){RESET}")
                        return False
                    
                    status = await response.json()
                    print(f"{GREEN}API is available{RESET}")
                    print(f"Version: {status.get('version', 'Unknown')}")
                    print(f"Default model: {status.get('default_model', 'Unknown')}")
                    print(f"Models in memory: {', '.join(status.get('models_in_memory', []))}")
                    print(f"Diarization enabled: {status.get('diarization', {}).get('enabled', False)}")
                    print(f"Diarization available: {status.get('diarization', {}).get('available', False)}")
                    
                    # Determine if diarization tests should be run
                    diarization_available = status.get('diarization', {}).get('available', False)
                    if not diarization_available:
                        print(f"{YELLOW}Warning: Diarization is not available, skipping diarization tests{RESET}")
                
                # Validate test files exist
                valid_files = []
                for file_path in self.test_files:
                    if os.path.exists(file_path):
                        valid_files.append(file_path)
                    else:
                        print(f"{YELLOW}Warning: File not found: {file_path}{RESET}")
                
                if not valid_files:
                    print(f"{RED}Error: No valid test files{RESET}")
                    return False
                
                self.test_files = valid_files
                
                # Run tests for each file and model
                all_results = []
                
                for file_path in self.test_files:
                    for model in self.models:
                        # Test without diarization
                        result = await self.test_file(session, file_path, model, diarization=False)
                        if result:
                            all_results.append(result)
                        
                        # Test with diarization if available
                        if diarization_available:
                            result = await self.test_file(session, file_path, model, diarization=True)
                            if result:
                                all_results.append(result)
                
                # Save and display results
                self.results = all_results
                self.display_results()
                self.save_results()
                
                return True
        
        except Exception as e:
            print(f"{RED}Error in test run: {str(e)}{RESET}")
            return False
    
    def display_results(self):
        """Display test results in a table."""
        if not self.results:
            print(f"{YELLOW}No results to display{RESET}")
            return
        
        print(f"\n{BOLD}{BLUE}Test Results Summary{RESET}\n")
        
        # Group results by whether diarization was used
        no_diarization = [r for r in self.results if not r["diarization"]]
        with_diarization = [r for r in self.results if r["diarization"]]
        
        # Display results without diarization
        if no_diarization:
            print(f"\n{BOLD}Without Diarization:{RESET}")
            self._display_result_table(no_diarization)
        
        # Display results with diarization
        if with_diarization:
            print(f"\n{BOLD}With Diarization:{RESET}")
            self._display_result_table(with_diarization)
        
        # Display performance by model
        print(f"\n{BOLD}Performance by Model:{RESET}")
        model_stats = {}
        
        for result in self.results:
            model = result["model"]
            if model not in model_stats:
                model_stats[model] = []
            
            model_stats[model].append(result["realtime_factor"])
        
        model_table = []
        for model, factors in model_stats.items():
            avg_factor = statistics.mean(factors)
            min_factor = min(factors)
            max_factor = max(factors)
            
            model_table.append([
                model,
                f"{avg_factor:.2f}x",
                f"{min_factor:.2f}x",
                f"{max_factor:.2f}x"
            ])
        
        print(tabulate(
            model_table,
            headers=["Model", "Avg Realtime", "Min Realtime", "Max Realtime"],
            tablefmt="psql"
        ))
    
    def _display_result_table(self, results):
        """Display a table of results."""
        table = []
        
        for result in results:
            table.append([
                result["file_name"],
                f"{result['file_size_mb']:.2f}",
                result["model"],
                f"{result['duration']:.2f}",
                f"{result['processing_time']:.2f}",
                f"{result['realtime_factor']:.2f}x",
                result["segment_count"],
                f"{result['chars_per_second']:.2f}"
            ])
        
        print(tabulate(
            table,
            headers=["File", "Size (MB)", "Model", "Duration (s)", "Processing (s)", "Realtime", "Segments", "Chars/sec"],
            tablefmt="psql"
        ))
    
    def save_results(self):
        """Save results to a JSON file."""
        if not self.results:
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"benchmark_results_{timestamp}.json"
        
        with open(filename, "w") as f:
            json.dump({
                "timestamp": timestamp,
                "api_url": self.api_url,
                "results": self.results
            }, f, indent=2)
        
        print(f"\n{GREEN}Results saved to {filename}{RESET}")

async def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Performance testing for Whisper API")
    parser.add_argument(
        "--api-url",
        type=str,
        default="http://localhost:8000",
        help="API URL to test against"
    )
    parser.add_argument(
        "--files",
        type=str,
        nargs="+",
        required=True,
        help="Audio files to test with"
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        default=["tiny", "small", "medium"],
        choices=["tiny", "small", "medium", "large"],
        help="Models to test with"
    )
    args = parser.parse_args()
    
    tester = PerformanceTester(args.api_url, args.files, args.models)
    await tester.run_tests()

if __name__ == "__main__":
    asyncio.run(main())