# -*- coding: utf-8 -*-
"""
gRPC Services Protobuf Wrapper for Person Detection
Re-exports from yolo_det_person_pb2 for backward compatibility.
"""
from app.clients.grpc_services.yolo_det_person_pb2 import *  # noqa: F401, F403
from app.clients.grpc_services import yolo_det_person_pb2 as _pb2

# Preserve module-level attributes expected by gRPC code
DetectRequest = _pb2.DetectRequest
BoundingBox = _pb2.BoundingBox
DetectResponse = _pb2.DetectResponse
DESCRIPTOR = _pb2.DESCRIPTOR
