# -*- coding: utf-8 -*-
"""
Standalone test script to verify line configuration from Backend UI API for People Counting.
Renders OpenCV GUI window with live line visualization.
"""
import argparse
import os
import sys
import time
import cv2
import numpy as np

# Ensure workspace root is in sys.path
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, BASE_DIR)

from ultralytics import YOLO
from ai_core.api_client import get_backend_rules, parse_lines_from_rules
from ai_core.line_counter import LineConfig, MultiLineCounter


def main():
    parser = argparse.ArgumentParser(description="Test UI Line Configuration & Person Line Counter")
    parser.add_argument(
        "--stream-id",
        type=str,
        default="fe157ce6-6f62-4dba-81d1-218b161f9d6f",
        help="Stream ID configured on UI Backend",
    )
    parser.add_argument(
        "--source",
        type=str,
        default=os.path.join(BASE_DIR, "images/test_couting_people.mp4"),
        help="Path to video file or camera stream URL",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.5,
        help="Confidence threshold for YOLO person detection",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Disable GUI window for headless test run",
    )
    args = parser.parse_args()

    print("\n=======================================================")
    print("  TESTING UI LINE CONFIGURATION FOR PEOPLE COUNTING")
    print(f"  Stream ID : {args.stream_id}")
    print(f"  Source    : {args.source}")
    print("=======================================================\n")

    # 1. Fetch rules from Backend API
    print(f"[API Query] Fetching rules from Backend (GET /api/rules) for stream {args.stream_id}...")
    raw_rules = get_backend_rules(stream_id=args.stream_id, rule_type="people_counting")
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
        print(f" SUCCESS: Loaded {len(line_configs)} line(s) from UI API for People Counting:")
        for idx, lc in enumerate(line_configs, 1):
            print(f"   Line {idx}: '{lc.name}' | Point 1: {lc.p1_ratio} | Point 2: {lc.p2_ratio}")
    else:
        print(" WARNING: No Line In / Line Out filters found on UI API for this stream_id.")
        print("   Using fallback horizontal line at y=55% height.")
        line_configs = [
            LineConfig("DEFAULT LINE (y=55%)", (0.0, 0.55), (1.0, 0.55), (0, 255, 255)),
        ]

    line_counter = MultiLineCounter(lines=line_configs)

    # 2. Load YOLO Model
    weights_path = os.path.join(BASE_DIR, "ai_core/weights/yolo_best2.pt")
    if not os.path.exists(weights_path):
        weights_path = os.path.join(BASE_DIR, "ai_core/weights/yolo_best3.pt")

    print(f"\n[YOLO] Loading model from: {weights_path}")
    model = YOLO(weights_path)

    # 3. Open Video Source
    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video source: {args.source}")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_target = cap.get(cv2.CAP_PROP_FPS) or 30.0
    delay_ms = int(1000.0 / fps_target)

    window_name = f"TEST STREAM LINE - UI RULE ({args.stream_id[:8]})"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1280, 720)

    print("\nStarting GUI Display Window. Press [Q] or [ESC] to stop.\n")

    frame_idx = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                print("\n[Video] Reached end of video file.")
                break

            frame_idx += 1
            t_start = time.time()

            # Run YOLO Person tracking
            results = model.track(
                source=frame,
                persist=True,
                classes=[0],
                conf=args.conf,
                tracker="bytetrack.yaml",
                verbose=False,
            )

            tracks = []
            if results and len(results) > 0:
                r_boxes = results[0].boxes
                if r_boxes is not None and len(r_boxes) > 0 and r_boxes.id is not None:
                    boxes = r_boxes.xyxy.cpu().numpy()
                    clss = r_boxes.cls.cpu().numpy()
                    confs = r_boxes.conf.cpu().numpy()
                    track_ids = r_boxes.id.int().cpu().numpy()

                    for box, cls, conf, p_id in zip(boxes, clss, confs, track_ids):
                        if cls == 0 and conf >= args.conf:
                            tracks.append((box, conf, p_id))

            # Update line counter
            events = line_counter.update(
                frame_idx=frame_idx,
                tracks=tracks,
                frame_size=(width, height),
            )

            for ev in events:
                print(f" [Frame {frame_idx}] Person ID:{ev['person_id']} crossed line '{ev['line_name']}' ({ev['direction']})")

            # Draw HUD Overlay and line
            t_elapsed = time.time() - t_start
            current_fps = 1.0 / t_elapsed if t_elapsed > 0 else fps_target
            line_counter.draw_hud(frame, tracks, fps=current_fps, driver_info="UI RULE TEST")

            # Display frame
            if not args.no_show:
                cv2.imshow(window_name, frame)

                key = cv2.waitKey(max(1, delay_ms)) & 0xFF
                if key in [ord("q"), ord("Q"), 27]:
                    print("\n[User] Terminated test by pressing 'Q'.")
                    break

    except KeyboardInterrupt:
        print("\n[User] Interrupted test.")
    finally:
        cap.release()
        cv2.destroyAllWindows()

    stats = line_counter.get_stats()
    print("\n=======================================================")
    print(" TEST COMPLETED - SUMMARY STATISTICS")
    print(f"   - Total Persons IN  : {stats['total_in']}")
    print(f"   - Total Persons OUT : {stats['total_out']}")
    print(f"   - Total Crossings   : {stats['total_count']}")
    print("=======================================================\n")


if __name__ == "__main__":
    main()
