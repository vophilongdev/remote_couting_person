# -*- coding: utf-8 -*-
"""
gRPC Services Protobuf GRPC Wrapper for Person Detection
Re-exports from yolo_det_person_pb2_grpc.
"""
from app.clients.grpc_services.yolo_det_person_pb2_grpc import (  # noqa: F401
    YoloService,
    YoloServiceServicer,
    YoloServiceStub,
    add_YoloServiceServicer_to_server,
)
