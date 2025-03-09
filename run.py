#!/usr/bin/env python3
"""
Production runner for Whisper Transcription API.

This script provides a robust way to run the API in production with:
- Proper signal handling
- Graceful shutdown
- Process monitoring
- Resource management
"""

import os
import sys
import argparse
import logging
import signal
import time
import psutil
import torch
import uvicorn
from contextlib import contextmanager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("whisper-api")

def get_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run Whisper Transcription API")
    
    parser.add_argument(
        "--host", 
        type=str, 
        default="0.0.0.0",
        help="Host to bind the server to"
    )
    
    parser.add_argument(
        "--port", 
        type=int, 
        default=8000,
        help="Port to bind the server to"
    )
    
    parser.add_argument(
        "--workers", 
        type=int, 
        default=1,
        help="Number of worker processes (use 1 for GPU workloads)"
    )
    
    parser.add_argument(
        "--debug", 
        action="store_true",
        help="Enable debug mode with auto-reload"
    )
    
    parser.add_argument(
        "--preload-model", 
        type=str, 
        default=None,
        help="Preload a specific model at startup (tiny, small, medium, large)"
    )
    
    parser.add_argument(
        "--log-level", 
        type=str, 
        default="info",
        choices=["debug", "info", "warning", "error", "critical"],
        help="Logging level"
    )
    
    parser.add_argument(
        "--memory-limit", 
        type=float, 
        default=0,
        help="GPU memory limit in GB (0 = no limit)"
    )
    
    return parser.parse_args()

def check_gpu():
    """Check if GPU is available and print info."""
    if not torch.cuda.is_available():
        logger.error("No CUDA-capable GPU found. This application requires GPU acceleration.")
        logger.error("Please ensure you have a compatible GPU and CUDA installed.")
        return False
    
    # Get GPU info
    device_count = torch.cuda.device_count()
    if device_count == 0:
        logger.error("PyTorch reports CUDA is available, but no devices found.")
        return False
    
    device_name = torch.cuda.get_device_name(0)
    logger.info(f"Found GPU: {device_name}")
    
    # Check memory
    if hasattr(torch.cuda, 'get_device_properties'):
        prop = torch.cuda.get_device_properties(0)
        total_memory_gb = prop.total_memory / (1024**3)
        logger.info(f"GPU Memory: {total_memory_gb:.2f} GB")
        
        # Check if memory is sufficient
        if total_memory_gb < 4:
            logger.warning(f"GPU memory may be insufficient for larger models: {total_memory_gb:.2f}GB")
            logger.warning("Recommend at least 8GB for optimal performance.")
    
    return True

def setup_memory_limit(limit_gb):
    """Set GPU memory limit if specified."""
    if limit_gb <= 0:
        return
        
    if not torch.cuda.is_available():
        logger.warning("Cannot set memory limit: CUDA not available")
        return
    
    try:
        # Get total memory
        if hasattr(torch.cuda, 'get_device_properties'):
            prop = torch.cuda.get_device_properties(0)
            total_memory_gb = prop.total_memory / (1024**3)
            
            # Validate limit
            if limit_gb > total_memory_gb:
                logger.warning(f"Memory limit {limit_gb}GB exceeds available GPU memory {total_memory_gb:.2f}GB")
                logger.warning("Using 80% of available memory instead")
                limit_gb = total_memory_gb * 0.8
        
        # Convert to bytes and set limit
        limit_bytes = int(limit_gb * 1024 * 1024 * 1024)
        
        # Try different methods based on PyTorch version
        if hasattr(torch.cuda, 'set_per_process_memory_fraction'):
            fraction = limit_gb / total_memory_gb
            torch.cuda.set_per_process_memory_fraction(fraction)
            logger.info(f"Set GPU memory limit to {limit_gb:.2f}GB ({fraction:.2%} of total)")
        elif hasattr(torch.cuda, 'max_memory_allocated'):
            # This doesn't actually limit memory, just track it
            logger.warning("This PyTorch version doesn't support memory limiting")
            logger.info(f"Will monitor usage instead (target: {limit_gb:.2f}GB)")
        else:
            logger.warning("Memory limiting not supported in this PyTorch version")
    except Exception as e:
        logger.error(f"Failed to set memory limit: {e}")

def preload_model(model_name):
    """Preload a model at startup."""
    if not model_name:
        return
        
    logger.info(f"Preloading model: {model_name}")
    try:
        # Import here to avoid circular imports
        from app.transcriber import model_manager
        _ = model_manager.get_pipeline(model_name)
        logger.info(f"Successfully preloaded {model_name} model")
    except Exception as e:
        logger.error(f"Failed to preload model {model_name}: {e}")

@contextmanager
def monitor_resources():
    """Monitor system resources during execution."""
    process = psutil.Process(os.getpid())
    
    # Start monitoring in a separate thread
    import threading
    import time
    
    stop_monitoring = threading.Event()
    
    def monitor_thread():
        peak_memory_percent = 0
        peak_cpu_percent = 0
        start_time = time.time()
        
        while not stop_monitoring.is_set():
            try:
                # Get memory usage
                memory_percent = process.memory_percent()
                peak_memory_percent = max(peak_memory_percent, memory_percent)
                
                # Get CPU usage
                cpu_percent = process.cpu_percent()
                peak_cpu_percent = max(peak_cpu_percent, cpu_percent)
                
                # Sleep to reduce overhead
                time.sleep(5)
            except Exception:
                # Ignore errors in monitoring
                pass
        
        # Final report
        elapsed = time.time() - start_time
        logger.info(f"Resource usage - Runtime: {elapsed:.1f}s, Peak Memory: {peak_memory_percent:.1f}%, Peak CPU: {peak_cpu_percent:.1f}%")
    
    # Start monitoring thread
    monitor_thread = threading.Thread(target=monitor_thread, daemon=True)
    monitor_thread.start()
    
    try:
        yield
    finally:
        # Stop monitoring
        stop_monitoring.set()
        monitor_thread.join(timeout=1.0)

def run_server(args):
    """Run the FastAPI server with uvicorn."""
    # Set log level
    log_level = args.log_level.upper()
    
    # Configure Uvicorn
    config = uvicorn.Config(
        "app.main:app",
        host=args.host,
        port=args.port,
        log_level=log_level.lower(),
        reload=args.debug,
        workers=args.workers,
        timeout_keep_alive=120  # Longer keep-alive for large file uploads
    )
    
    # Create and run server
    server = uvicorn.Server(config)
    
    # Handle signals for graceful shutdown
    def handle_exit(signum, frame):
        logger.info(f"Received signal {signum}, shutting down...")
        # Clean up GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        sys.exit(0)
    
    # Register signal handlers
    signal.signal(signal.SIGINT, handle_exit)
    signal.signal(signal.SIGTERM, handle_exit)
    
    # Run the server
    with monitor_resources():
        server.run()

def main():
    """Main entry point."""
    args = get_args()
    
    # Check GPU availability
    if not check_gpu():
        sys.exit(1)
    
    # Set memory limit if specified
    if args.memory_limit > 0:
        setup_memory_limit(args.memory_limit)
    
    # Preload model if specified
    if args.preload_model:
        preload_model(args.preload_model)
    
    # Run the server
    logger.info(f"Starting Whisper Transcription API on {args.host}:{args.port}")
    logger.info(f"Workers: {args.workers}, Debug: {args.debug}")
    
    try:
        run_server(args)
    except KeyboardInterrupt:
        logger.info("Shutting down...")
    except Exception as e:
        logger.error(f"Error running server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()