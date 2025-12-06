# Base image with Python 3.12
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    portaudio19-dev \
    libsdl-pango-dev \
    libcairo2-dev \
    libpango1.0-dev \
    build-essential \
    texlive-full \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Copy Python files
COPY requirements.txt .
COPY main.py .
COPY src/ ./src/

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Download ONNX model if missing
RUN mkdir -p src/tts && \
    [ ! -f src/tts/kokoro-v1.0.onnx ] && \
    wget -O src/tts/kokoro-v1.0.onnx https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx

# Set PYTHONPATH
ENV PYTHONPATH=/app:$PYTHONPATH

# Start your Python script
CMD ["python", "main.py"]
