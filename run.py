#!/usr/bin/env python3
"""
Main entrypoint script for running the Whisper Transcription API.

This script simply imports and calls the main function from the CLI module.
It exists for backward compatibility and potentially simpler run commands.
"""

import sys
import os

# Ensure the app directory is in the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

try:
    from app.cli import main
except ImportError as e:
    print(f"Error: Could not import the application. Ensure necessary modules are installed and paths are correct.")
    print(f"ImportError: {e}")
    sys.exit(1)
except Exception as e:
    print(f"An unexpected error occurred during import: {e}")
    sys.exit(1)


if __name__ == "__main__":
    # Execute the main function from the CLI module
    main()