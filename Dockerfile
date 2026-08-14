FROM python:3.10-slim

# Install system dependencies required for OpenCV, FFmpeg and Dahua/Hikvision SDK libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    curl \
    libx11-6 \
    libxv1 \
    libxext6 \
    libasound2 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir torch torchvision --extra-index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu

# Copy project source code
COPY . /app

# Install Dahua NetSDK python wheel
RUN pip install --no-cache-dir /app/ai_core/services/sdk/dist/NetSDK-2.0.0.1-py3-none-linux_x86_64.whl

# Copy legacy OpenSSL 1.1 libraries to system path for Dahua NetSDK dependency
RUN cp /app/ai_core/services/sdk/dist/libcrypto.so.1.1 /usr/lib/ && \
    cp /app/ai_core/services/sdk/dist/libssl.so.1.1 /usr/lib/

# Add NetSDK library path and local SDK path to system dynamic linker config
RUN for dir in /usr/local/lib/python3.10/site-packages/NetSDK/Libs/linux64 /usr/local/lib/python3.10/dist-packages/NetSDK/Libs/linux64; do \
        if [ -d "$dir" ]; then echo "$dir" >> /etc/ld.so.conf.d/netsdk.conf; fi; \
    done && \
    echo "/app/ai_core/services/sdk/dist" >> /etc/ld.so.conf.d/netsdk.conf && \
    ldconfig

# Debug: Print shared library resolutions to verify which dependencies are not found
RUN echo "=== Registered LD Paths ===" && cat /etc/ld.so.conf.d/netsdk.conf && \
    for dir in /usr/local/lib/python3.10/site-packages/NetSDK/Libs/linux64 /usr/local/lib/python3.10/dist-packages/NetSDK/Libs/linux64; do \
        if [ -d "$dir" ]; then \
            echo "=== Checking libraries in $dir ==="; \
            ldd $dir/libdhnetsdk.so || true; \
            ldd $dir/libplay.so || true; \
        fi; \
    done

# Set Environment Variables
ENV PYTHONUNBUFFERED=1
ENV HOST=0.0.0.0
ENV PORT=8000

# Expose Gateway Port
EXPOSE 8000

# Run FastAPI Gateway Uvicorn Server
CMD ["uvicorn", "ai_core.server:app", "--host", "0.0.0.0", "--port", "8000"]
