# -*- coding: utf-8 -*-
"""
Centralized Configuration — 12-Factor App Pattern
All constants, paths, and environment variable defaults in one place.
"""

import os

# ─── Directory Paths ───────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # app/
PROJECT_ROOT = os.path.dirname(BASE_DIR)  # LONG_CORE/

# Load .env file if present (zero-dependency)
_env_path = os.path.join(PROJECT_ROOT, ".env")
if os.path.exists(_env_path):
    with open(_env_path, "r", encoding="utf-8") as _f:
        for _line in _f:
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _k, _v = _line.split("=", 1)
                os.environ.setdefault(_k.strip(), _v.strip().strip("'\""))

# ─── API Configuration ────────────────────────────────────────
DEFAULT_API_URL = os.environ.get("API_URL", "https://api-dev.tado.vn/api/camera-statistics")
DEFAULT_SESSION_TOKEN = os.environ.get("SESSION_TOKEN", "")
CAMERAS_API_URL = "https://api-dev.tado.vn/api/cameras"
RULES_API_URL = "https://api-dev.tado.vn/api/rules"

# ─── YOLO Model Configuration ─────────────────────────────────
DEFAULT_MODEL_PATH = os.environ.get(
    "MODEL_PATH",
    os.path.join(BASE_DIR, "weights", "yolo_best3.pt"),
)

DEFAULT_CONF = float(os.environ.get("CONF", "0.8"))

# ─── gRPC Configuration ───────────────────────────────────────
GRPC_ADDR = os.environ.get("COUNT_PEOPLE_ADDR", "localhost:50051")
USE_GRPC = os.environ.get("USE_GRPC", "false").lower() in ["true", "1", "yes"]

# ─── Camera Configuration ─────────────────────────────────────
DEFAULT_CAMERA_TYPE = os.environ.get("CAMERA_TYPE", "rtsp")
DEFAULT_STORAGE_URL = os.environ.get("STORAGE_URL", "")
DEFAULT_STORAGE_PORT = int(os.environ.get("STORAGE_PORT", "0"))
DEFAULT_STORAGE_USERNAME = os.environ.get("STORAGE_USERNAME", "")
DEFAULT_STORAGE_PASSWORD = os.environ.get("STORAGE_PASSWORD", "")
DEFAULT_STORAGE_CHANNEL = int(os.environ.get("STORAGE_CHANNEL", "1"))

# ─── Server Configuration ─────────────────────────────────────
HOST = os.environ.get("HOST", "0.0.0.0")
PORT = int(os.environ.get("PORT", "8000"))

# ─── License ───────────────────────────────────────────────────
SYSTEM_LICENSE_KEY = os.environ.get("LICENSE_KEY", "")
