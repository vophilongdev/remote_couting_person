# coding=utf-8
"""
FastAPI Server — Inbound Adapter (Transport Layer)
Multi-Camera Gateway Server supporting License Key verification,
H.264/MJPEG streams, and Realtime WebSocket Stats.
"""

import asyncio
import json
import logging
import os
import time
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
import uvicorn
from fastapi import BackgroundTasks, FastAPI, Header, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
from ultralytics import YOLO

from app.clients.http_client import get_backend_cameras, get_backend_rules, post_camera_statistic
from app.clients.rule_parser import parse_lines_from_rules
from app.clients.camera_reader import CameraReader
from app.config.settings import DEFAULT_MODEL_PATH, SYSTEM_LICENSE_KEY
from app.core.line_counter import MultiLineCounter
from app.core.person_tracker import PersonTracker
from app.core.pipeline import build_line_configs, pipeline_loop, process_crossing_events
from app.models.schemas import LineConfig, StreamState

try:
    from app.clients.grpc_client import GRPCClient
except ImportError:
    GRPCClient = None

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("AICoreServer")

app = FastAPI(
    title="AI CORE Gateway & Stream Server",
    description="Multi-Camera Gateway Server supporting License Key verification, H.264/MJPEG streams, and Realtime WebSocket Stats",
    version="2.0.0"
)


# Allow CORS for Frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


async def auto_sync_backend_cameras_loop():
    """Background worker that continuously syncs cameras from Backend API (GET /api/cameras)."""
    logger.info("[Auto-Sync AI Worker] Polling Backend API (GET /api/cameras) for 'people_counting' cameras...")
    while True:
        try:
            cameras = get_backend_cameras()
            for cam in cameras:
                cam_id = str(cam.get("id") or cam.get("stream_id", ""))
                services = cam.get("service_type") or []
                if isinstance(services, str):
                    services = [services]
                
                # Automatically start AI counting if camera has 'people_counting' service
                if cam_id and "people_counting" in services:
                    new_url = cam.get("storage_url") or cam.get("url") or ""
                    
                    should_restart = False
                    if cam_id not in active_streams:
                        should_restart = True
                    else:
                        existing_stream = active_streams[cam_id]
                        if not existing_stream.is_running:
                            should_restart = True
                        elif existing_stream.camera_reader and not existing_stream.camera_reader.is_connected:
                            logger.warning(f"[Auto-Sync] Stream '{cam_id}' camera_reader is disconnected. Restarting stream...")
                            should_restart = True
                        elif new_url != existing_stream.storage_url:
                            logger.info(f"[Auto-Sync] Stream '{cam_id}' RTSP URL changed from '{existing_stream.storage_url}' to '{new_url}'. Restarting stream...")
                            should_restart = True

                    if should_restart:
                        if cam_id in active_streams:
                            old_stream = active_streams[cam_id]
                            old_stream.is_running = False
                            if old_stream.camera_reader:
                                old_stream.camera_reader.stop()

                        logger.info(f"[Auto-Sync] Starting/Restarting AI Worker for camera '{cam.get('name', cam_id)}' (ID: {cam_id})...")
                        stream = StreamState(stream_id=cam_id)
                        use_sdk = cam.get("use_sdk", 0)
                        brand = str(cam.get("brand") or cam.get("manufacturer") or "").lower()
                        if use_sdk == 1 or "dahua" in brand:
                            stream.camera_type = "dahua"
                        elif "hik" in brand:
                            stream.camera_type = "hik"
                        else:
                            stream.camera_type = "rtsp"
                        
                        stream.storage_url = new_url
                        stream.storage_port = int(cam.get("storage_port") or 8888)
                        stream.storage_username = cam.get("storage_username") or "admin"
                        stream.storage_password = cam.get("storage_password") or ""
                        stream.storage_channel = int(cam.get("storage_channel") or 1)

                        active_streams[cam_id] = stream
                        asyncio.create_task(asyncio.to_thread(pipeline_loop, stream))
        except Exception as e:
            logger.error(f"[Auto-Sync Worker Exception] {e}")
        await asyncio.sleep(15)

@app.on_event("startup")
async def startup_event():
    asyncio.create_task(auto_sync_backend_cameras_loop())

# Multi-Camera Manager (Dictionary storing active streams by stream_id)
active_streams: Dict[str, StreamState] = {}
shared_yolo_model = None

def get_shared_yolo_model(model_path: str):
    global shared_yolo_model
    if shared_yolo_model is None and os.path.exists(model_path):
        logger.info(f"Loading shared YOLO Model from {model_path}")
        shared_yolo_model = YOLO(model_path)
    return shared_yolo_model

@app.get("/")
def root():
    return {
        "service": "AI CORE Background Worker",
        "status": "running",
        "active_cameras_counting": len([s for s in active_streams.values() if s.is_running]),
        "camera_ids": [s_id for s_id, s in active_streams.items() if s.is_running]
    }

if __name__ == "__main__":
    from app.config.settings import HOST, PORT
    uvicorn.run(app, host=HOST, port=PORT)
