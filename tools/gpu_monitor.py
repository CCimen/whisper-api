#!/usr/bin/env python3
"""
GPU monitoring tool for Whisper API.

This script provides real-time monitoring of GPU usage, memory, and processes.
Useful for tracking performance and debugging memory issues.
"""

import time
import os
import sys
import argparse
import signal
import subprocess
import platform
from datetime import datetime

try:
    import psutil
    import torch
    import GPUtil
    from tabulate import tabulate
except ImportError:
    print("This tool requires additional packages. Install with:")
    print("pip install psutil GPUtil tabulate")
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

def clear_screen():
    """Clear the console screen."""
    os.system('cls' if platform.system() == 'Windows' else 'clear')

def bytes_to_gb(bytes_value):
    """Convert bytes to gigabytes."""
    return bytes_value / (1024 ** 3)

def get_gpu_processes():
    """Get list of processes using GPU."""
    try:
        if platform.system() == 'Windows':
            # Windows uses nvidia-smi
            result = subprocess.run(
                ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                check=True
            )
            processes = []
            for line in result.stdout.strip().split("\n"):
                if line:
                    parts = line.split(", ")
                    if len(parts) >= 2:
                        pid = int(parts[0])
                        memory = parts[1]
                        try:
                            proc = psutil.Process(pid)
                            processes.append({
                                "pid": pid,
                                "name": proc.name(),
                                "memory": memory,
                                "command": " ".join(proc.cmdline()[:3]) + "..." if len(proc.cmdline()) > 3 else " ".join(proc.cmdline())
                            })
                        except (psutil.NoSuchProcess, psutil.AccessDenied):
                            pass
            return processes
        else:
            # Linux uses nvidia-smi in a different format
            result = subprocess.run(
                ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                check=True
            )
            processes = []
            for line in result.stdout.strip().split("\n"):
                if line:
                    parts = line.split(", ")
                    if len(parts) >= 2:
                        pid = int(parts[0])
                        memory = parts[1]
                        try:
                            proc = psutil.Process(pid)
                            processes.append({
                                "pid": pid,
                                "name": proc.name(),
                                "memory": memory,
                                "command": " ".join(proc.cmdline()[:3]) + "..." if len(proc.cmdline()) > 3 else " ".join(proc.cmdline())
                            })
                        except (psutil.NoSuchProcess, psutil.AccessDenied):
                            pass
            return processes
    except Exception as e:
        print(f"Error getting GPU processes: {e}")
        return []

def get_system_info():
    """Get system information."""
    cpu_percent = psutil.cpu_percent(interval=0.1)
    memory = psutil.virtual_memory()
    memory_used_gb = bytes_to_gb(memory.used)
    memory_total_gb = bytes_to_gb(memory.total)
    memory_percent = memory.percent
    
    return {
        "cpu_percent": cpu_percent,
        "memory_used_gb": memory_used_gb,
        "memory_total_gb": memory_total_gb,
        "memory_percent": memory_percent
    }

def get_gpu_info():
    """Get detailed GPU information."""
    if not torch.cuda.is_available():
        return None
    
    try:
        gpus = GPUtil.getGPUs()
        gpu_info = []
        
        for i, gpu in enumerate(gpus):
            info = {
                "id": i,
                "name": gpu.name,
                "load": gpu.load * 100,
                "memory_used": gpu.memoryUsed,
                "memory_total": gpu.memoryTotal,
                "memory_percent": (gpu.memoryUsed / gpu.memoryTotal) * 100,
                "temperature": gpu.temperature
            }
            gpu_info.append(info)
        
        return gpu_info
    except Exception as e:
        print(f"Error getting GPU info: {e}")
        return None

def print_dash_line(length=80):
    """Print a dashed line separator."""
    print("-" * length)

def print_header(text, width=80):
    """Print a centered header."""
    print_dash_line(width)
    print(f"{BOLD}{text.center(width)}{RESET}")
    print_dash_line(width)

def display_info(args):
    """Display GPU and system information."""
    clear_screen()
    
    # Print title
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"\n{BOLD}{BLUE}GPU MONITOR - {now}{RESET}\n")
    
    # Get system info
    sys_info = get_system_info()
    
    # Print system info
    print_header("SYSTEM INFO")
    cpu_percent = sys_info["cpu_percent"]
    cpu_color = GREEN if cpu_percent < 70 else YELLOW if cpu_percent < 90 else RED
    memory_percent = sys_info["memory_percent"]
    memory_color = GREEN if memory_percent < 70 else YELLOW if memory_percent < 90 else RED
    
    print(f"CPU Usage:     {cpu_color}{cpu_percent:.1f}%{RESET}")
    print(f"Memory Usage:  {memory_color}{sys_info['memory_used_gb']:.2f} GB / {sys_info['memory_total_gb']:.2f} GB ({memory_percent:.1f}%){RESET}")
    
    # Get GPU info
    gpu_info = get_gpu_info()
    
    if gpu_info:
        # Print GPU info
        print_header("GPU INFO")
        
        for gpu in gpu_info:
            gpu_id = gpu["id"]
            gpu_name = gpu["name"]
            gpu_load = gpu["load"]
            gpu_load_color = GREEN if gpu_load < 70 else YELLOW if gpu_load < 90 else RED
            gpu_mem_percent = gpu["memory_percent"]
            gpu_mem_color = GREEN if gpu_mem_percent < 70 else YELLOW if gpu_mem_percent < 90 else RED
            gpu_temp = gpu["temperature"]
            gpu_temp_color = GREEN if gpu_temp < 70 else YELLOW if gpu_temp < 80 else RED
            
            print(f"GPU {gpu_id}: {BOLD}{gpu_name}{RESET}")
            print(f"  Load:         {gpu_load_color}{gpu_load:.1f}%{RESET}")
            print(f"  Memory:       {gpu_mem_color}{gpu['memory_used']:.2f} MB / {gpu['memory_total']:.2f} MB ({gpu_mem_percent:.1f}%){RESET}")
            print(f"  Temperature:  {gpu_temp_color}{gpu_temp}°C{RESET}")
        
        # Get GPU processes
        processes = get_gpu_processes()
        
        if processes:
            # Print GPU processes
            print_header("GPU PROCESSES")
            
            headers = ["PID", "Name", "GPU Memory", "Command"]
            table_data = [[proc["pid"], proc["name"], proc["memory"], proc["command"]] for proc in processes]
            
            print(tabulate(table_data, headers=headers, tablefmt="psql"))
        else:
            print(f"\n{YELLOW}No processes using GPU{RESET}")
    else:
        print(f"\n{RED}No GPU available or error getting GPU info{RESET}")
    
    # Print refresh info
    print_dash_line()
    print(f"{BOLD}Refreshing every {args.interval} seconds. Press Ctrl+C to exit.{RESET}")

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Monitor GPU usage for Whisper API")
    parser.add_argument(
        "--interval", 
        type=float, 
        default=2.0,
        help="Refresh interval in seconds"
    )
    parser.add_argument(
        "--log-file", 
        type=str, 
        default=None,
        help="Log file to write GPU stats (optional)"
    )
    args = parser.parse_args()
    
    # Handle Ctrl+C gracefully
    def signal_handler(sig, frame):
        print("\nExiting GPU monitor...")
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    
    # Setup logging if enabled
    log_file = None
    if args.log_file:
        try:
            log_file = open(args.log_file, "a")
            log_file.write(f"\n--- GPU Monitor Session: {datetime.now()} ---\n")
            log_file.write("timestamp,cpu_percent,memory_percent,gpu_id,gpu_load,gpu_memory_percent,gpu_temperature\n")
        except Exception as e:
            print(f"Error opening log file: {e}")
            log_file = None
    
    # Check GPU availability
    if not torch.cuda.is_available():
        print(f"{RED}No CUDA-capable GPU found. This tool requires NVIDIA GPU with CUDA.{RESET}")
        sys.exit(1)
    
    # Main loop
    try:
        while True:
            # Display info
            display_info(args)
            
            # Log info if enabled
            if log_file:
                now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                sys_info = get_system_info()
                gpu_info = get_gpu_info()
                
                for gpu in gpu_info:
                    log_file.write(f"{now},{sys_info['cpu_percent']:.1f},{sys_info['memory_percent']:.1f},{gpu['id']},"
                                  f"{gpu['load']:.1f},{gpu['memory_percent']:.1f},{gpu['temperature']}\n")
                log_file.flush()
            
            # Wait for next refresh
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\nExiting GPU monitor...")
    finally:
        if log_file:
            log_file.close()

if __name__ == "__main__":
    main()