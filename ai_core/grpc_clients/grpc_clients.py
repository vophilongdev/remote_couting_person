# -*- coding: utf-8 -*-
"""
gRPC Client for Remote Frame Object Detection Service
Allows offloading YOLO object detection to a high-speed GPU gRPC microservice.
"""

import logging
import threading
import time
import cv2
import grpc
from typing import List, Tuple

try:
    from ai_core.grpc_clients.services import services_pb2, services_pb2_grpc
except ImportError:
    try:
        from grpc_clients.services import services_pb2, services_pb2_grpc
    except ImportError:
        from services import services_pb2, services_pb2_grpc

logger = logging.getLogger(__name__)


class GRPCClient:
    """
    Optimized gRPC client with connection pooling, retries, and backoff.
    """

    def __init__(
        self,
        server_addr: str = "localhost:50051",
        target_size: int = 640,
        max_msg_mb: int = 50,
        timeout_s: float = 5.0,
        connect_timeout_s: float = 10.0,
        max_retries: int = 2,
        retry_delay: float = 0.5,
    ):
        self.server_addr = server_addr
        self.target_size = target_size
        self.timeout = timeout_s
        self.connect_timeout = connect_timeout_s
        self.max_retries = max_retries
        self.retry_delay = retry_delay

        self.opts = [
            ("grpc.max_send_message_length", max_msg_mb << 20),
            ("grpc.max_receive_message_length", max_msg_mb << 20),
            ("grpc.keepalive_time_ms", 120000),
            ("grpc.keepalive_timeout_ms", 5000),
            ("grpc.keepalive_permit_without_calls", True),
        ]

        self._lock = threading.Lock()
        self.stub: services_pb2_grpc.YoloServiceStub = None
        self.is_connected = False

    def connect(self) -> bool:
        """Establish gRPC connection to remote YOLO service."""
        try:
            ch = grpc.insecure_channel(self.server_addr, options=self.opts)
            grpc.channel_ready_future(ch).result(timeout=self.connect_timeout)
            with self._lock:
                self.stub = services_pb2_grpc.YoloServiceStub(ch)
                self.is_connected = True
            logger.info(f"[gRPC] Successfully connected to YOLO service at {self.server_addr}")
            return True
        except Exception as e:
            logger.warning(f"[gRPC] Could not connect to {self.server_addr}: {e}")
            self.is_connected = False
            return False

    def detect_yolo(self, frame: cv2.Mat) -> List[Tuple[int, int, int, int, float, str]]:
        """
        Send image frame over gRPC for remote detection.
        Returns list of (x1, y1, x2, y2, confidence, class_name)
        """
        if not self.is_connected or self.stub is None:
            if not self.connect():
                return []

        size = self.target_size
        rs = cv2.resize(frame, (size, size))
        _, buf = cv2.imencode('.jpg', rs, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        req = services_pb2.DetectRequest(
            image=buf.tobytes(),
            resized_width=size,
            resized_height=size,
            orig_width=frame.shape[1],
            orig_height=frame.shape[0],
            allowed_classes=[],
        )

        for attempt in range(self.max_retries):
            try:
                resp = self.stub.DetectFrame(req, timeout=self.timeout, wait_for_ready=True)
                return [(b.x1, b.y1, b.x2, b.y2, b.confidence, b.class_name) for b in resp.bboxes]
            except grpc.RpcError as e:
                logger.warning(f"[gRPC] Attempt {attempt+1}/{self.max_retries} failed: {e.details()}")
                time.sleep(self.retry_delay * (2 ** attempt))

        logger.error("[gRPC] All RPC retries failed.")
        return []