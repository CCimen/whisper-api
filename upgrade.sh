#!/bin/bash
# Quick upgrade script to fix NumPy/pyannote compatibility issue

echo "Whisper API Upgrade Script"
echo "=========================="
echo "This script will fix the NumPy compatibility issue with pyannote.audio"
echo

# Check for virtual environment
if [ -d "venv" ]; then
  echo "Found virtual environment, activating..."
  source venv/bin/activate
else
  echo "No 'venv' directory found. Please run this script from your project root"
  echo "or create a virtual environment first with: python -m venv venv"
  exit 1
fi

# Check if numpy is installed and get version
NUMPY_VERSION=$(pip freeze | grep numpy | sed 's/numpy==//g')
echo "Current NumPy version: $NUMPY_VERSION"

# Downgrade NumPy if needed
if [[ "$NUMPY_VERSION" == 2.* ]]; then
  echo "NumPy 2.x detected. Downgrading to compatible version..."
  pip install numpy==1.24.3
  
  # Reinstall pyannote if it's installed
  if pip freeze | grep -q pyannote.audio; then
    echo "Reinstalling pyannote.audio to ensure compatibility..."
    pip uninstall -y pyannote.audio
    pip install pyannote.audio==3.1.1
  fi
  
  echo "NumPy downgraded successfully!"
else
  echo "NumPy version is already compatible, no downgrade needed."
fi

# Restart service if running as systemd
if [ -f "/etc/systemd/system/whisper-api.service" ]; then
  echo "Found systemd service, would you like to restart it? (y/n)"
  read -r restart
  if [[ "$restart" =~ ^[Yy]$ ]]; then
    sudo systemctl restart whisper-api
    echo "Service restarted!"
  fi
fi

echo "Upgrade complete! You can now run your application again."