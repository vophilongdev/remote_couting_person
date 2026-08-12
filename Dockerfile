FROM python:3.10-slim

# Install system dependencies required for OpenCV, FFmpeg and Dahua/Hikvision SDK libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu

# Copy project source code
COPY . /app

# Set Environment Variables
ENV PYTHONUNBUFFERED=1
ENV HOST=0.0.0.0
ENV PORT=8000

# Expose Gateway Port
EXPOSE 8000

# Run FastAPI Gateway Uvicorn Server
CMD ["uvicorn", "ai_core.server:app", "--host", "0.0.0.0", "--port", "8000"]
