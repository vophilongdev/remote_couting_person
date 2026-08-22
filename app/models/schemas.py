# -*- coding: utf-8 -*-
"""
Domain Models / Data Transfer Objects
All data containers and value objects used across the microservice.
"""

import os
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import numpy as np

from app.config.settings import (
    DEFAULT_API_URL,
    DEFAULT_CAMERA_TYPE,
    DEFAULT_CONF,
    DEFAULT_MODEL_PATH,
    DEFAULT_STORAGE_CHANNEL,
    DEFAULT_STORAGE_PASSWORD,
    DEFAULT_STORAGE_PORT,
    DEFAULT_STORAGE_URL,
    DEFAULT_STORAGE_USERNAME,
    GRPC_ADDR,
    SYSTEM_LICENSE_KEY,
    USE_GRPC,
)


@dataclass
class LineConfig:
    name: str
    p1_ratio: Tuple[float, float]  # (x1_ratio, y1_ratio) e.g. (0.0, 0.35)
    p2_ratio: Tuple[float, float]  # (x2_ratio, y2_ratio) e.g. (1.0, 0.35)
    color: Tuple[int, int, int]    # BGR color tuple e.g. (255, 255, 0)
    filter_type: str = "line"      # "line_in" | "line_out" | "line"


class StreamState:
    def __init__(self, stream_id: str):
        self.stream_id: str = stream_id
        self.camera_reader = None
        self.line_counter = None
        self.yolo_model = None
        self.is_running = False
        self.current_frame: Optional[np.ndarray] = None
        self.current_fps: float = 0.0
        self.last_stats: Dict[str, Any] = {"total_in": 0, "total_out": 0, "total_sum": 0, "lines": {}}
        self.conf_threshold: float = DEFAULT_CONF
        self.model_path: str = DEFAULT_MODEL_PATH
        self.api_url: str = DEFAULT_API_URL
        self.session_token: str = os.environ.get("SESSION_TOKEN", "")
        self.camera_type: str = DEFAULT_CAMERA_TYPE
        self.storage_url: str = DEFAULT_STORAGE_URL
        self.storage_port: int = DEFAULT_STORAGE_PORT
        self.storage_password: str = DEFAULT_STORAGE_PASSWORD
        self.storage_username: str = DEFAULT_STORAGE_USERNAME
        self.storage_channel: int = DEFAULT_STORAGE_CHANNEL
        self.use_grpc: bool = USE_GRPC
        self.grpc_addr: str = GRPC_ADDR
        self.license_key: str = SYSTEM_LICENSE_KEY
