# coding=utf-8
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

from ai_core.api_client import BaseReportStrategy, CameraStatisticPayload, CameraStatisticsAPIClient, ReportStrategyFactory
from ai_core.camera_reader import CameraReader
from ai_core.line_counter import MultiLineCounter, PersonTracker

try:
    from ai_core.grpc_clients.grpc_clients import GRPCClient
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
    client = CameraStatisticsAPIClient()
    logger.info("[Auto-Sync AI Worker] Polling Backend API (GET /api/cameras) for 'people_counting' cameras...")
    while True:
        try:
            cameras = client.get_cameras()
            for cam in cameras:
                cam_id = str(cam.get("id") or cam.get("stream_id", ""))
                services = cam.get("service_type", [])
                if isinstance(services, str):
                    services = [services]
                
                # Automatically start AI counting if camera has 'people_counting' service
                if cam_id and "people_counting" in services:
                    if cam_id not in active_streams or not active_streams[cam_id].is_running:
                        logger.info(f"[Auto-Sync] Found active camera '{cam.get('name', cam_id)}' (ID: {cam_id}) with 'people_counting'. Starting AI Worker...")
                        stream = StreamState(stream_id=cam_id)
                        use_sdk = cam.get("use_sdk", 0)
                        brand = str(cam.get("brand") or cam.get("manufacturer") or "").lower()
                        if use_sdk == 1 or "dahua" in brand:
                            stream.camera_type = "dahua"
                        elif "hik" in brand:
                            stream.camera_type = "hik"
                        else:
                            stream.camera_type = "rtsp"
                        
                        stream.storage_url = cam.get("storage_url") or cam.get("url") or ""
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

# Global Settings
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_MODEL = os.path.join(BASE_DIR, "ai_core/weights/yolo_best2.pt")
if not os.path.exists(DEFAULT_MODEL):
    DEFAULT_MODEL = os.path.join(BASE_DIR, "weights/yolo_best2.pt")

SYSTEM_LICENSE_KEY = os.environ.get("LICENSE_KEY", "")

class StreamState:
    def __init__(self, stream_id: str):
        self.stream_id: str = stream_id
        self.camera_reader: Optional[CameraReader] = None
        self.line_counter: Optional[MultiLineCounter] = None
        self.api_reporter: Optional[BaseReportStrategy] = None
        self.yolo_model = None
        self.is_running = False
        self.current_frame: Optional[np.ndarray] = None
        self.current_fps: float = 0.0
        self.last_stats: Dict[str, Any] = {"total_in": 0, "total_out": 0, "total_sum": 0, "lines": {}}
        self.conf_threshold: float = float(os.environ.get("CONF", "0.8"))
        self.model_path: str = os.environ.get("MODEL_PATH", DEFAULT_MODEL)
        self.api_url: str = os.environ.get("API_URL", "https://api-dev.tado.vn/api/camera-statistics")
        self.session_token: str = os.environ.get("SESSION_TOKEN", "")
        self.camera_type: str = os.environ.get("CAMERA_TYPE", "rtsp")
        self.storage_url: str = os.environ.get("STORAGE_URL", "")
        self.storage_port: int = int(os.environ.get("STORAGE_PORT", "0"))
        self.storage_username: str = os.environ.get("STORAGE_USERNAME", "")
        self.storage_password: str = os.environ.get("STORAGE_PASSWORD", "")
        self.storage_channel: int = int(os.environ.get("STORAGE_CHANNEL", "1"))
        self.use_grpc: bool = os.environ.get("USE_GRPC", "false").lower() in ["true", "1", "yes"]
        self.grpc_addr: str = os.environ.get("COUNT_PEOPLE_ADDR", "localhost:50051")
        self.license_key: str = SYSTEM_LICENSE_KEY

# Multi-Camera Manager (Dictionary storing active streams by stream_id)
active_streams: Dict[str, StreamState] = {}
shared_yolo_model = None

def get_shared_yolo_model(model_path: str):
    global shared_yolo_model
    if shared_yolo_model is None and os.path.exists(model_path):
        logger.info(f"Loading shared YOLO Model from {model_path}")
        shared_yolo_model = YOLO(model_path)
    return shared_yolo_model

def pipeline_loop(stream: StreamState):
    """Background thread running AI processing loop for a specific stream."""
    grpc_client = None
    tracker = None
    if stream.use_grpc:
        if GRPCClient is None:
            logger.error("gRPC Client module unavailable. Falling back to local YOLO.")
            stream.use_grpc = False
        else:
            logger.info(f"Connecting to Remote gRPC GPU Server at: {stream.grpc_addr}")
            try:
                grpc_client = GRPCClient(server_addr=stream.grpc_addr)
                grpc_client.connect()
                tracker = PersonTracker(max_disappeared=15)
            except Exception as e:
                logger.error(f"Failed to connect to gRPC server ({e}). Falling back to local YOLO.")
                stream.use_grpc = False

    model = None if stream.use_grpc else get_shared_yolo_model(stream.model_path)
    
    stream.api_reporter = ReportStrategyFactory.create_strategy(
        stream_id=stream.stream_id,
        api_url=stream.api_url,
        session_token=stream.session_token,
        batch_mode=False
    )
    
    line_configs = [
        {"name": "COUNTING LINE", "p1": (0, 400), "p2": (1920, 400), "dir_in": "down"},
    ]
    stream.line_counter = MultiLineCounter(lines=line_configs)
    
    stream.camera_reader = CameraReader(
        source=stream.storage_url,
        driver_type=stream.camera_type,
        storage_url=stream.storage_url,
        storage_port=stream.storage_port,
        storage_username=stream.storage_username,
        storage_password=stream.storage_password,
        storage_channel=stream.storage_channel
    )
    stream.camera_reader.start()
    stream.is_running = True
    logger.info(f"Stream {stream.stream_id} pipeline loop started successfully.")

    frame_times = []
    try:
        while stream.is_running:
            frame, frame_idx, timestamp = stream.camera_reader.get_frame(timeout=0.5)
            if frame is None:
                time.sleep(0.01)
                continue

            t_start = time.time()
            tracks = []

            if stream.use_grpc and grpc_client is not None and tracker is not None:
                try:
                    detections = grpc_client.detect(frame)
                    tracks = tracker.update(detections)
                except Exception as e:
                    logger.error(f"gRPC detection error: {e}")
            elif model is not None:
                results = model.track(
                    source=frame,
                    persist=True,
                    classes=[0],
                    conf=stream.conf_threshold,
                    tracker="bytetrack.yaml",
                    verbose=False
                )
                if results and len(results) > 0:
                    r_boxes = results[0].boxes
                    if r_boxes is not None and len(r_boxes) > 0 and r_boxes.id is not None:
                        boxes = r_boxes.xyxy.cpu().numpy()
                        clss = r_boxes.cls.cpu().numpy()
                        confs = r_boxes.conf.cpu().numpy()
                        track_ids = r_boxes.id.int().cpu().numpy()

                        for box, cls, conf, p_id in zip(boxes, clss, confs, track_ids):
                            if cls == 0 and conf >= stream.conf_threshold:
                                tracks.append((box, conf, p_id))

            # Update MultiLineCounter Tracker
            events, stats = stream.line_counter.update(tracks)
            stream.last_stats = stats

            # Report stats payload if line crossed
            if events:
                payload = CameraStatisticPayload(
                    stream_id=stream.stream_id,
                    metric_type="people_counting",
                    data={
                        "count": len(tracks),
                        "person": len(tracks),
                        "in": stats["total_in"],
                        "out": stats["total_out"],
                    },
                    time=time.time(),
                )
                stream.api_reporter.report(payload)

            # Calculate FPS
            t_end = time.time()
            frame_times.append(t_end - t_start)
            if len(frame_times) > 30:
                frame_times.pop(0)
            stream.current_fps = 1.0 / (sum(frame_times) / len(frame_times)) if frame_times else 0.0

            # Draw AI HUD Overlay onto Frame
            stream.line_counter.draw_hud(frame, tracks, fps=stream.current_fps, driver_info=stream.camera_type.upper())
            stream.current_frame = frame

            time.sleep(0.01)
    except Exception as e:
        logger.error(f"Pipeline loop error for stream {stream.stream_id}: {e}")
    finally:
        if stream.camera_reader:
            stream.camera_reader.stop()
        if stream.api_reporter:
            stream.api_reporter.flush()
        stream.is_running = False
        logger.info(f"Stream {stream.stream_id} pipeline loop finished.")

@app.get("/")
def root():
    return {
        "service": "AI CORE Background Worker",
        "status": "running",
        "active_cameras_counting": len([s for s in active_streams.values() if s.is_running]),
        "camera_ids": [s_id for s_id, s in active_streams.items() if s.is_running]
    }

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)
