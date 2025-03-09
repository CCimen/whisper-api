#!/bin/bash
# Setup script for Whisper Transcription API in production

# Exit on error
set -e

# Default installation directory
INSTALL_DIR="/opt/whisper-api"
SERVICE_NAME="whisper-api"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --dir=*)
      INSTALL_DIR="${1#*=}"
      shift
      ;;
    --port=*)
      PORT="${1#*=}"
      shift
      ;;
    --help)
      echo "Usage: setup_production.sh [OPTIONS]"
      echo "Options:"
      echo "  --dir=PATH    Installation directory (default: /opt/whisper-api)"
      echo "  --port=PORT   Port to run the API on (default: 8000)"
      echo "  --help        Show this help message"
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

# Set default port if not specified
PORT=${PORT:-8000}

# Check if running as root
if [ "$EUID" -ne 0 ]; then
  echo "Please run as root or using sudo"
  exit 1
fi

# Check if CUDA is available
if ! command -v nvidia-smi &> /dev/null; then
  echo "WARNING: NVIDIA drivers not found. This application requires a CUDA-capable GPU."
  echo "Continue anyway? (y/n)"
  read -r continue
  if [[ ! "$continue" =~ ^[Yy]$ ]]; then
    exit 1
  fi
fi

echo "Setting up Whisper Transcription API in $INSTALL_DIR"

# Create installation directory
mkdir -p "$INSTALL_DIR"
cd "$INSTALL_DIR"

# Create virtual environment
echo "Creating Python virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install system dependencies for ffmpeg
echo "Installing system dependencies..."
if command -v apt-get &> /dev/null; then
  apt-get update
  apt-get install -y ffmpeg
elif command -v yum &> /dev/null; then
  yum install -y epel-release
  yum install -y ffmpeg
else
  echo "WARNING: Could not install ffmpeg automatically. Please install it manually."
fi

# Clone repository or copy files
if [ -d "whisper-api" ]; then
  echo "Using existing whisper-api directory..."
  cd whisper-api
else
  echo "Copying application files..."
  # Create directories
  mkdir -p app models
  
  # Copy application files from current directory to install dir
  cp -r app/* "$INSTALL_DIR/app/"
  cp requirements.txt "$INSTALL_DIR/"
  cp *.py "$INSTALL_DIR/"
  
  # Create .env file
  echo "Creating .env file..."
  cat > "$INSTALL_DIR/.env" << EOF
DEBUG=False
DEFAULT_MODEL=medium
DEFAULT_LANGUAGE=sv
MAX_MODELS_IN_MEMORY=1
PRELOAD_DEFAULT_MODEL=True
DIARIZATION_ENABLED=False
# HUGGINGFACE_TOKEN=your-token-here  # Uncomment and add your token if using diarization
EOF
fi

# Install Python dependencies
echo "Installing Python dependencies..."
pip install -r requirements.txt

# Create systemd service
echo "Creating systemd service..."
cat > "/etc/systemd/system/$SERVICE_NAME.service" << EOF
[Unit]
Description=Whisper Transcription API
After=network.target

[Service]
User=$(whoami)
Group=$(id -gn)
WorkingDirectory=$INSTALL_DIR
ExecStart=$INSTALL_DIR/venv/bin/uvicorn app.main:app --host 0.0.0.0 --port $PORT --workers 1
Restart=on-failure
RestartSec=5
SyslogIdentifier=$SERVICE_NAME
Environment="PATH=$INSTALL_DIR/venv/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"

# Load environment from .env file
EnvironmentFile=-$INSTALL_DIR/.env

[Install]
WantedBy=multi-user.target
EOF

# Reload systemd
systemctl daemon-reload

# Enable and start service
echo "Enabling and starting service..."
systemctl enable "$SERVICE_NAME"
systemctl start "$SERVICE_NAME"

# Check service status
echo "Service status:"
systemctl status "$SERVICE_NAME"

echo ""
echo "Whisper Transcription API has been installed!"
echo "API is running at: http://localhost:$PORT"
echo "API documentation: http://localhost:$PORT/docs"
echo ""
echo "To check logs: journalctl -u $SERVICE_NAME -f"
echo "To restart: systemctl restart $SERVICE_NAME"
echo "To stop: systemctl stop $SERVICE_NAME"