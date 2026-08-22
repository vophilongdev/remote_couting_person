# -*- coding: utf-8 -*-
"""
Pipeline Orchestrator — Shared AI Processing Logic
Extracted from app.py and server.py to avoid code duplication.
Both CLI entrypoint and Server entrypoint reuse these functions.
"""

import logging
import time

import numpy as np

from app.clients.http_client import get_backend_rules, post_camera_statistic
from app.clients.rule_parser import parse_lines_from_rules
from app.core.line_counter import MultiLineCounter
from app.core.person_tracker import PersonTracker
from app.models.schemas import LineConfig, StreamState

logger = logging.getLogger("Pipeline")


def build_line_configs(stream_id: str = None, session_token: str = None):
    """
    Fetch counting line configs from Backend API rules.
    Falls back to a default horizontal line if no rules are found.
    Returns a list of LineConfig objects.
    """
    raw_rules = get_backend_rules(
        stream_id=stream_id if stream_id else None,
        rule_type="people_counting",
        session_token=session_token,
    )
    parsed_lines = parse_lines_from_rules(raw_rules)

    if parsed_lines:
        line_configs = [
            LineConfig(
                name=ld["name"],
                p1_ratio=ld["p1_ratio"],
                p2_ratio=ld["p2_ratio"],
                color=ld["color"],
                filter_type=ld.get("type", "line"),
            )
            for ld in parsed_lines
        ]
        logger.info(
            f"[Pipeline] Loaded {len(line_configs)} line config(s) from Backend API: {[l.name for l in line_configs]}"
        )
        return line_configs

    logger.info("[Pipeline] No custom line rules found in API. Using default horizontal line.")
    return [
        LineConfig("COUNTING LINE", (0.0, 0.55), (1.0, 0.55), (0, 255, 255)),  # Yellow
    ]


def run_local_inference(model, frame, imgsz, conf_threshold, tracker=None):
    """
    Run local YOLO inference and tracking on a single frame.
    Returns list of (box, conf, track_id) tuples.
    """
    results = model.track(
        frame,
        imgsz=imgsz,
        conf=conf_threshold,
        persist=True,
        tracker="bytetrack.yaml",
        verbose=False,
    )

    tracks = []
    person_boxes = []

    if results and len(results) > 0 and results[0].boxes is not None:
        r_boxes = results[0].boxes
        if len(r_boxes) > 0:
            boxes = r_boxes.xyxy.cpu().numpy()
            clss = r_boxes.cls.cpu().numpy().astype(int)
            confs = r_boxes.conf.cpu().numpy()

            has_ids = r_boxes.id is not None
            track_ids = r_boxes.id.cpu().numpy().astype(int) if has_ids else None

            for i, (box, cls, conf) in enumerate(zip(boxes, clss, confs)):
                if (cls == 0 or str(cls) == "0") and conf >= conf_threshold:
                    if has_ids and track_ids is not None:
                        tracks.append((box, conf, int(track_ids[i])))
                    else:
                        person_boxes.append((box[0], box[1], box[2], box[3], conf))

    if person_boxes and tracker is not None:
        fallback_tracks = tracker.update(person_boxes)
        tracks.extend(fallback_tracks)
    elif not tracks and tracker is not None:
        tracker.update([])

    return tracks


def run_grpc_inference(grpc_client, frame, conf_threshold, tracker):
    """
    Run remote gRPC inference and tracking on a single frame.
    Returns list of (box, conf, track_id) tuples.
    """
    detections = grpc_client.detect_yolo(frame)
    person_boxes = []
    for x1, y1, x2, y2, conf, cls_name in detections:
        if (cls_name == "person" or str(cls_name) == "0") and conf >= conf_threshold:
            person_boxes.append((x1, y1, x2, y2, conf))

    return tracker.update(person_boxes)


def process_crossing_events(events, stream_id, api_url=None, session_token=None):
    """
    Process line crossing events and report aggregated counts to Backend API.
    """
    if not events:
        return

    inc_in = sum(1 for ev in events if ev.get("direction") == "IN")
    inc_out = sum(1 for ev in events if ev.get("direction") == "OUT")
    event_count = inc_in + inc_out

    if event_count > 0:
        logger.info(f"[Batch Crossing Event] Stream '{stream_id}': IN={inc_in}, OUT={inc_out}, TOTAL={event_count}")
        post_camera_statistic(
            stream_id=stream_id,
            total_in=inc_in,
            total_out=inc_out,
            current_count=event_count,
            api_url=api_url,
            session_token=session_token,
        )


def pipeline_loop(stream: StreamState):
    """
    Background thread running AI processing loop for a specific stream.
    Used by the Server entrypoint.
    """
    grpc_client = None
    tracker = None

    if stream.use_grpc:
        try:
            from app.clients.grpc_client import GRPCClient
            grpc_client = GRPCClient(server_addr=stream.grpc_addr)
            grpc_client.connect()
            tracker = PersonTracker(max_disappeared=30)
        except Exception as e:
            logger.error(f"Failed to connect to gRPC server ({e}). Falling back to local YOLO.")
            stream.use_grpc = False

    model = None
    if not stream.use_grpc:
        from app.api.server import get_shared_yolo_model
        model = get_shared_yolo_model(stream.model_path)

    # Fetch counting lines dynamically from Backend API rules
    line_configs = build_line_configs(
        stream_id=stream.stream_id,
        session_token=stream.session_token,
    )
    stream.line_counter = MultiLineCounter(lines=line_configs)

    from app.clients.camera_reader import CameraReader
    stream.camera_reader = CameraReader(
        source=stream.storage_url,
        driver_type=stream.camera_type,
        storage_url=stream.storage_url,
        storage_port=stream.storage_port,
        storage_username=stream.storage_username,
        storage_password=stream.storage_password,
        storage_channel=stream.storage_channel,
    )
    stream.camera_reader.start()
    stream.is_running = True
    logger.info(f"Stream {stream.stream_id} pipeline loop started successfully.")

    frame_times = []
    last_rule_check_time = time.time()
    try:
        while stream.is_running:
            # Dynamically reload rules from API every 10 seconds without stopping stream
            if time.time() - last_rule_check_time > 10.0:
                last_rule_check_time = time.time()
                try:
                    new_line_configs = build_line_configs(
                        stream_id=stream.stream_id,
                        session_token=stream.session_token,
                    )
                    if new_line_configs and stream.line_counter:
                        old_names = [l.name for l in stream.line_counter.lines_config]
                        new_names = [l.name for l in new_line_configs]
                        if old_names != new_names or len(stream.line_counter.lines_config) != len(new_line_configs):
                            logger.info(f"[Dynamic Rule Refresh] Updating {len(new_line_configs)} line config(s) for stream '{stream.stream_id}'")
                            stream.line_counter = MultiLineCounter(lines=new_line_configs)
                except Exception as err:
                    logger.error(f"[Dynamic Rule Refresh Error] {err}")

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
                    imgsz=640,
                    tracker="bytetrack.yaml",
                    verbose=False,
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

            # Get frame size
            h, w = frame.shape[:2]
            # Update MultiLineCounter Tracker
            events = stream.line_counter.update(
                frame_idx=frame_idx,
                tracks=tracks,
                frame_size=(w, h),
            )
            stats = stream.line_counter.get_stats()
            stream.last_stats = stats

            # Report stats payload if line crossed
            process_crossing_events(
                events,
                stream_id=stream.stream_id,
                api_url=stream.api_url,
                session_token=stream.session_token,
            )

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
        stream.is_running = False
        logger.info(f"Stream {stream.stream_id} pipeline loop finished.")
