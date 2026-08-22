# -*- coding: utf-8 -*-
"""
AI CORE - High-Performance Person Line Counter Engine (3-Line & Multi-Camera Driver Support)
Architecture: Producer-Consumer Threading + Frame Queue Drop + Multi-Line Tracking + Async API Reporting + gRPC Support

CLI Entrypoint — run directly: python -m app.main [args]
"""

import argparse
import gc
import os
import signal
import sys
import time
import traceback

import cv2
import numpy as np
from ultralytics import YOLO

from app.clients.http_client import flush_pending_requests, get_backend_rules, post_camera_statistic
from app.clients.rule_parser import parse_lines_from_rules
from app.clients.camera_reader import CameraReader
from app.config.settings import BASE_DIR, PROJECT_ROOT
from app.core.line_counter import MultiLineCounter
from app.core.person_tracker import PersonTracker
from app.models.schemas import LineConfig


def parse_args():
    parser = argparse.ArgumentParser(description="AI CORE - Person 3-Line Counter Engine")

    # Camera Source & Drivers
    parser.add_argument(
        "--source",
        type=str,
        default="images/test_couting_people.mp4",
        help="Path to video file, RTSP stream URL, or webcam ID (e.g. 0)",
    )
    parser.add_argument(
        "--camera-type",
        type=str,
        default="rtsp",
        choices=["rtsp", "0", "dahua", "1", "hik", "2"],
        help="Camera driver type: 'rtsp' (0), 'dahua' (1), or 'hik' (2)",
    )

    # SDK Credentials (for Dahua & Hikvision)
    parser.add_argument("--storage-url", type=str, default="", help="Camera IP/Domain for SDK login")
    parser.add_argument("--storage-port", type=int, default=37777, help="Camera SDK Port (Dahua default 37777, Hik default 8000)")
    parser.add_argument("--storage-username", type=str, default="admin", help="Camera SDK Username")
    parser.add_argument("--storage-password", type=str, default="", help="Camera SDK Password")
    parser.add_argument("--storage-channel", type=int, default=1, help="Camera SDK Channel number (1-based)")

    # Inference Mode: Local vs gRPC
    parser.add_argument(
        "--use-grpc",
        action="store_true",
        help="Offload YOLO detection to remote gRPC GPU microservice",
    )
    parser.add_argument(
        "--grpc-addr",
        type=str,
        default=os.environ.get("COUNT_PEOPLE_ADDR", "localhost:50051"),
        help="gRPC YoloService server address (e.g. 127.0.0.1:50051)",
    )

    # Model & Local Inference Settings
    parser.add_argument(
        "--model",
        type=str,
        default=os.path.join(BASE_DIR, "weights/yolo_best3.pt"),
        help="Path to YOLO model weights file (for local inference)",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.5,
        help="Confidence threshold for person detection",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Inference image size (e.g. 640)",
    )

    # API Reporting Settings
    parser.add_argument(
        "--stream-id",
        type=str,
        default=os.environ.get("STREAM_ID", ""),
        help="UUID string for Stream ID to send to camera statistics API",
    )
    parser.add_argument(
        "--api-url",
        type=str,
        default=os.environ.get("API_URL", "https://api-dev.tado.vn/api/camera-statistics"),
        help="API endpoint for POSTing camera statistics",
    )
    parser.add_argument(
        "--session",
        type=str,
        default=os.environ.get("SESSION_TOKEN", ""),
        help="Session token string for API header authentication",
    )
    parser.add_argument(
        "--batch-mode",
        action="store_true",
        help="Use Batch API endpoint (/api/camera-statistics/batch)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10,
        help="Number of records to accumulate before flushing batch POST",
    )

    # Execution Flags
    parser.add_argument(
        "--no-loop",
        action="store_true",
        help="Disable continuous video looping for video files",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Disable GUI window (headless mode for Docker / Server)",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # 1. Resolve Video Source Path
    source = args.source
    if isinstance(source, str) and not source.isdigit() and not source.startswith(("rtsp://", "rtmp://", "http://")):
        if not os.path.exists(source):
            alt_source = os.path.join(PROJECT_ROOT, source)
            if os.path.exists(alt_source):
                source = alt_source

    # 2. Setup Inference Mode (Local YOLO vs gRPC Client)
    model = None
    grpc_client = None
    tracker = PersonTracker(max_disappeared=15)

    if args.use_grpc:
        print(f"[AI CORE] Using Remote gRPC GPU Inference Service at: {args.grpc_addr}")
        try:
            from app.clients.grpc_client import GRPCClient
            grpc_client = GRPCClient(server_addr=args.grpc_addr, target_size=args.imgsz)
            grpc_client.connect()
        except Exception as e:
            print(f"[AI CORE Warning] Could not initialize gRPC Client ({e}). Falling back to Local YOLO.")
            args.use_grpc = False

    if not args.use_grpc:
        model_path = args.model
        print(f"[AI CORE] Initializing Local YOLO Model: {model_path}")
        model = YOLO(model_path)

    # 4. Configure Counting Lines (Fetch dynamically from Backend API or fallback to default)
    stream_id = getattr(args, "stream_id", None) or os.environ.get("STREAM_ID", "")
    raw_rules = get_backend_rules(stream_id=stream_id if stream_id else None, rule_type="people_counting")
    parsed_lines = parse_lines_from_rules(raw_rules)
    if parsed_lines:
        line_configs = [
            LineConfig(name=ld["name"], p1_ratio=ld["p1_ratio"], p2_ratio=ld["p2_ratio"], color=ld["color"], filter_type=ld.get("type", "line"))
            for ld in parsed_lines
        ]
        print(f"[AI CORE] Loaded {len(line_configs)} line config(s) from Backend API: {[l.name for l in line_configs]}")
    else:
        print("[AI CORE] No custom line rules found in API. Using default horizontal line.")
        line_configs = [
            LineConfig("COUNTING LINE", (0.0, 0.55), (1.0, 0.55), (0, 255, 255)),  # Yellow
        ]
    line_counter = MultiLineCounter(lines=line_configs)

    # 5. Initialize Multi-Threaded Camera Reader (RTSP / Dahua / Hikvision)
    storage_port = args.storage_port
    if args.camera_type in ["hik", "2"] and storage_port == 37777:
        storage_port = 8000  # Default Hikvision SDK Port

    camera_reader = CameraReader(
        source=source,
        driver_type=args.camera_type,
        storage_url=args.storage_url,
        storage_port=storage_port,
        storage_username=args.storage_username,
        storage_password=args.storage_password,
        storage_channel=args.storage_channel,
        queue_size=16,
    )
    camera_reader.start()

    # 6. GUI Window Setup
    has_display = bool(os.environ.get("DISPLAY"))
    show_gui = (not args.no_show) and has_display
    window_name = "AI CORE - Person 3-Line Counter"

    if show_gui:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 1280, 720)
        print("GUI Active: [+] Zoom In | [-] Zoom Out | [R] Reset | [Q] Quit")
    else:
        print("Running AI CORE in Headless Mode (No GUI Window)")

    # Signal Handling for Graceful Shutdown
    shutdown_requested = False

    def _sig_handler(sig, frame):
        nonlocal shutdown_requested
        print("\n[AI CORE] Termination signal received. Stopping pipeline...")
        shutdown_requested = True

    signal.signal(signal.SIGINT, _sig_handler)
    signal.signal(signal.SIGTERM, _sig_handler)

    # FPS & Performance Metrics
    fps = 0.0
    frame_times = []
    zoom_scale = 1.0

    print(f"\n=======================================================")
    print(f" AI CORE Person 3-Line Counter Pipeline Running")
    print(f" Driver Mode    : {camera_reader.driver_type.upper()}")
    print(f" Inference Mode : {'gRPC Service (' + args.grpc_addr + ')' if args.use_grpc else 'Local PyTorch YOLO'}")
    print(f" Source         : {source}")
    print(f" Stream ID      : {args.stream_id}")
    print(f"=======================================================\n")

    try:
        while not shutdown_requested:
            t_start = time.time()

            frame, frame_idx, timestamp = camera_reader.get_frame(timeout=0.5)
            if frame is None:
                if show_gui:
                    blank = np.zeros((720, 1280, 3), dtype=np.uint8)
                    if camera_reader.driver_type in ["dahua", "hik"]:
                        cam_info = f"{args.storage_url}:{storage_port}"
                    else:
                        cam_info = source
                    cv2.putText(
                        blank,
                        f"CAMERA OFFLINE / CONNECTING FAILED",
                        (50, 330),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.0,
                        (0, 0, 255),
                        2,
                        cv2.LINE_AA,
                    )
                    cv2.putText(
                        blank,
                        f"Host: {cam_info} (Driver: {camera_reader.driver_type.upper()})",
                        (50, 380),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (200, 200, 200),
                        1,
                        cv2.LINE_AA,
                    )
                    cv2.putText(
                        blank,
                        "Press [Q] or [ESC] to Quit",
                        (50, 440),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 255, 255),
                        1,
                        cv2.LINE_AA,
                    )
                    cv2.imshow(window_name, blank)
                    key = cv2.waitKey(50) & 0xFF
                    if key in [ord("q"), ord("Q"), 27]:
                        break

                if not camera_reader.is_connected and not args.no_loop:
                    time.sleep(0.05)
                    continue
                elif args.no_loop:
                    print("[AI CORE] Stream finished.")
                    break
                else:
                    time.sleep(0.01)
                    continue

            height, width = frame.shape[:2]
            tracks = []

            # Option A: Remote gRPC Service Inference
            if args.use_grpc and grpc_client is not None:
                detections = grpc_client.detect_yolo(frame)
                person_boxes = []
                for x1, y1, x2, y2, conf, cls_name in detections:
                    if (cls_name == "person" or str(cls_name) == "0") and conf >= args.conf:
                        person_boxes.append((x1, y1, x2, y2, conf))

                # Track IDs using IoU PersonTracker
                tracks = tracker.update(person_boxes)

            # Option B: Local PyTorch YOLO Inference & Tracking
            else:
                results = model.track(
                    frame,
                    imgsz=args.imgsz,
                    conf=args.conf,
                    persist=True,
                    tracker="bytetrack.yaml",
                    verbose=False,
                )

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
                            if (cls == 0 or str(cls) == "0") and conf >= args.conf:  # Class 0: Person
                                if has_ids and track_ids is not None:
                                    tracks.append((box, conf, int(track_ids[i])))
                                else:
                                    person_boxes.append((box[0], box[1], box[2], box[3], conf))

                if person_boxes:
                    fallback_tracks = tracker.update(person_boxes)
                    tracks.extend(fallback_tracks)
                elif not tracks:
                    tracker.update([])

            # Update MultiLineCounter & Detect Crossing Events
            events = line_counter.update(
                frame_idx=frame_idx,
                tracks=tracks,
                frame_size=(width, height),
            )

            # Process & Report Line Crossing Events
            stats = line_counter.get_stats()
            for ev in events:
                p_id = ev["person_id"]
                line_name = ev["line_name"]
                direction = ev["direction"]
                print(f"[Frame {frame_idx}] Person ID:{p_id} crossed {line_name} ({direction})!")

                inc_in = 1 if direction == "IN" else 0
                inc_out = 1 if direction == "OUT" else 0
                event_count = inc_in + inc_out

                # Post camera stats directly to Backend API (count & person = in + out)
                post_camera_statistic(
                    stream_id=args.stream_id,
                    total_in=inc_in,
                    total_out=inc_out,
                    current_count=event_count,
                    api_url=args.api_url,
                    session_token=args.session,
                )

            # Calculate Rolling Average FPS
            t_end = time.time()
            frame_times.append(t_end - t_start)
            if len(frame_times) > 30:
                frame_times.pop(0)
            fps = 1.0 / (sum(frame_times) / len(frame_times)) if frame_times else 0.0

            # Render Lines, Tracked Bounding Boxes & HUD Dashboard Overlay
            mode_str = f"{camera_reader.driver_type.upper()} | gRPC" if args.use_grpc else camera_reader.driver_type.upper()
            line_counter.draw_hud(frame, tracks, fps=fps, driver_info=mode_str)

            # Display GUI Window
            if show_gui:
                cv2.putText(
                    frame,
                    f"Zoom: {zoom_scale:.1f}x | Mode: {mode_str} | [+] Zoom In [-] Zoom Out [Q] Quit",
                    (12, height - 15),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )
                disp_frame = cv2.resize(frame, (max(100, int(width * zoom_scale)), max(100, int(height * zoom_scale)))) if zoom_scale != 1.0 else frame
                cv2.imshow(window_name, disp_frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                elif key in [ord("+"), ord("=")]:
                    zoom_scale = min(3.0, zoom_scale + 0.2)
                elif key in [ord("-"), ord("_")]:
                    zoom_scale = max(0.4, zoom_scale - 0.2)
                elif key in [ord("r"), ord("R")]:
                    zoom_scale = 1.0

            # Memory Garbage Collection every 300 frames
            if frame_idx % 300 == 0:
                gc.collect()

    except Exception as e:
        print(f"[AI CORE Error] Pipeline exception: {e}")
        traceback.print_exc()

    finally:
        print("\n[AI CORE] Cleaning up resources...")
        camera_reader.stop()
        flush_pending_requests()

        if show_gui:
            cv2.destroyAllWindows()

        final_stats = line_counter.get_stats()
        print(f"\n=======================================================")
        print(f" AI CORE Pipeline Finished!")
        print(f" Total IN  : {final_stats['total_in']}")
        print(f" Total OUT : {final_stats['total_out']}")
        print(f" Total SUM : {final_stats['total_count']}")
        print(f"=======================================================\n")


if __name__ == "__main__":
    main()
