# Stage 1: Base Image and System Dependencies
# Use an official Python image. Choose one that matches your target CUDA version.
# Defaulting to CUDA 11.8 as a common baseline. Adjust ARG as needed.
ARG CUDA_VERSION=11.8.0
ARG CUDNN_VERSION=8
ARG OS_VERSION=ubuntu22.04
# Match requires-python in pyproject.toml
ARG PYTHON_VERSION=3.10
ARG BASE_IMAGE=nvidia/cuda:${CUDA_VERSION}-cudnn${CUDNN_VERSION}-runtime-${OS_VERSION}
# For CPU-only builds (use --build-arg BASE_IMAGE=...):
# ARG BASE_IMAGE=python:${PYTHON_VERSION}-slim
FROM ${BASE_IMAGE} AS base

# Set environment variables consistently
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=off \
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    PIP_DEFAULT_TIMEOUT=100 \
    PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/root/.local/bin" \
    APP_HOME=/app \
    PYTHONPATH=/app \
    DEBIAN_FRONTEND=noninteractive \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1

WORKDIR $APP_HOME

# Install system dependencies: python3, python3-venv, ffmpeg, git, curl
# Using --fix-missing just in case apt has issues
RUN apt-get update --fix-missing && \
    apt-get install -y --no-install-recommends --fix-missing \
    python3 \
    python3-venv \
    ffmpeg \
    git \
    curl \
    ca-certificates && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install uv using the recommended method (copying binary)
# Explicitly copy the /uv binary from the source image to /usr/local/bin/uv in this stage
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
# Verify installation by running the copied binary
RUN uv --version


# Stage 2: Install Dependencies using uv
FROM base AS builder

WORKDIR $APP_HOME

# Create a virtual environment within the app directory
# Use python3 explicitly now that we know it's installed
ENV VIRTUAL_ENV=${APP_HOME}/.venv
RUN python3 -m venv ${VIRTUAL_ENV}
# Activate the venv for subsequent RUN commands in this stage
ENV PATH="${VIRTUAL_ENV}/bin:${PATH}"

# Copy project definition file
COPY pyproject.toml ./

# Install dependencies AND the project with extras using uv pip install
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install -e ".[diarization]"


# Stage 3: Application Stage
FROM base AS final

WORKDIR $APP_HOME

# Create a non-root user and group for security
ARG UID=1001
ARG GID=1001
# -m creates home dir /home/appuser
RUN groupadd --gid ${GID} appuser && \
    useradd --uid ${UID} --gid ${GID} -m appuser --shell /bin/bash

# Copy the virtual environment with installed dependencies from the builder stage
COPY --from=builder ${APP_HOME}/.venv ${APP_HOME}/.venv

# Copy the application code BEFORE changing user, owned by root initially
COPY ./app ./app
COPY ./run.py .

# Create necessary APP directories as ROOT first
RUN mkdir -p ${APP_HOME}/uploads ${APP_HOME}/results ${APP_HOME}/logs

# Define Hugging Face cache directory within user's home
ENV HF_HOME=/home/appuser/.cache/huggingface
# Often used by transformers
ENV TRANSFORMERS_CACHE=${HF_HOME}/hub

# Create cache directory structure owned by appuser
# We create it here because the user exists now. chown is important.
RUN mkdir -p ${HF_HOME} && chown -R appuser:appuser /home/appuser/.cache

# Change ownership of application code, uploads, results, logs directories
# Leave .venv owned by root, as only execution is needed
RUN chown -R appuser:appuser ${APP_HOME}/app ${APP_HOME}/run.py ${APP_HOME}/uploads ${APP_HOME}/results ${APP_HOME}/logs
# Ownership of /home/appuser/.cache was set earlier

# Update PATH for the final non-root user environment AFTER venv is copied
ENV PATH="${APP_HOME}/.venv/bin:${PATH}"

# Switch to the non-root user
USER appuser

# Expose the port the app runs on
EXPOSE 8000

# Define the command to run the application using uvicorn via the activated venv
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]