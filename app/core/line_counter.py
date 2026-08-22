# -*- coding: utf-8 -*-
"""
Multi-Line Person Counter & Visualizer Engine
Supports 3 Counting Lines (Line 1, Line 2, Line 3) with custom line coordinates,
crossing detection, state tracking, and HUD dashboard overlay rendering.
"""

import logging
from typing import Dict, List, Set, Tuple

import cv2
import numpy as np

from app.models.schemas import LineConfig

logger = logging.getLogger("LineCounter")


class MultiLineCounter:
    """
    Manages Counting Lines and tracks line crossing events based on UI rules.
    """

    def __init__(self, lines: List[LineConfig] = None):
        if lines is None or len(lines) == 0:
            # Default to 1 single horizontal counting line at 55% height
            lines = [
                LineConfig("COUNTING LINE", (0.0, 0.55), (1.0, 0.55), (0, 255, 255)),  # Yellow
            ]
        self.lines_config = lines

        # Tracking state maps
        self.track_history: Dict[int, Tuple[int, int]] = {}
        
        # Per-line crossed IDs sets
        self.crossed_in: Dict[str, Set[int]] = {line.name: set() for line in lines}
        self.crossed_out: Dict[str, Set[int]] = {line.name: set() for line in lines}

        # Global crossed IDs across all lines
        self.global_crossed_in: Set[int] = set()
        self.global_crossed_out: Set[int] = set()

    def reset(self):
        """Reset all tracking history and counters."""
        self.track_history.clear()
        for name in self.crossed_in:
            self.crossed_in[name].clear()
            self.crossed_out[name].clear()
        self.global_crossed_in.clear()
        self.global_crossed_out.clear()

    def update(
        self,
        frame_idx: int,
        tracks: List[Tuple[np.ndarray, float, int]],  # List of (box [x1,y1,x2,y2], conf, track_id)
        frame_size: Tuple[int, int],                  # (width, height)
    ) -> List[Dict]:
        """
        Process current frame tracks and detect line crossings across all lines.
        Returns list of new crossing event dicts.
        """
        width, height = frame_size
        new_events = []

        # Convert line ratio coordinates to absolute pixel values
        abs_lines = []
        for line in self.lines_config:
            x1 = int(line.p1_ratio[0] * width)
            y1 = int(line.p1_ratio[1] * height)
            x2 = int(line.p2_ratio[0] * width)
            y2 = int(line.p2_ratio[1] * height)
            abs_lines.append((line, (x1, y1), (x2, y2), line.color))

        for box, conf, p_id in tracks:
            # Use bottom-center (feet position) for accurate ground-plane line crossing
            cx = int((box[0] + box[2]) / 2)
            cy = int(box[3])
            curr_center = (cx, cy)

            if p_id in self.track_history:
                px, py = self.track_history[p_id]
            else:
                px, py = curr_center

            self.track_history[p_id] = curr_center

            # Check crossing for each line
            for line_cfg, p1, p2, color in abs_lines:
                name = line_cfg.name
                ftype = getattr(line_cfg, "filter_type", "line").lower()

                crossed, raw_direction = self._check_line_crossing(p1, p2, (px, py), curr_center)
                if crossed:
                    # Respect UI filter_type rules:
                    # line_in filter specifically counts IN traffic
                    # line_out filter specifically counts OUT traffic
                    if ftype == "line_in":
                        event_dir = "IN"
                    elif ftype == "line_out":
                        event_dir = "OUT"
                    else:
                        event_dir = raw_direction

                    if event_dir == "IN" and p_id not in self.crossed_in[name]:
                        self.crossed_in[name].add(p_id)
                        self.global_crossed_in.add(p_id)
                        event = {
                            "frame_idx": frame_idx,
                            "person_id": p_id,
                            "line_name": name,
                            "direction": "IN",
                            "center": curr_center,
                        }
                        new_events.append(event)
                        logger.info(
                            f"[Line Crossing] Object ID:{p_id} crossed line '{name}' (IN) at frame pixel line P1={p1}, P2={p2}, center={curr_center}"
                        )
                    elif event_dir == "OUT" and p_id not in self.crossed_out[name]:
                        self.crossed_out[name].add(p_id)
                        self.global_crossed_out.add(p_id)
                        event = {
                            "frame_idx": frame_idx,
                            "person_id": p_id,
                            "line_name": name,
                            "direction": "OUT",
                            "center": curr_center,
                        }
                        new_events.append(event)
                        logger.info(
                            f"[Line Crossing] Object ID:{p_id} crossed line '{name}' (OUT) at frame pixel line P1={p1}, P2={p2}, center={curr_center}"
                        )

        return new_events

    @staticmethod
    def _ccw(A: Tuple[int, int], B: Tuple[int, int], C: Tuple[int, int]) -> float:
        """2D Orientation test (cross product of vectors AB and AC)."""
        return float((B[0] - A[0]) * (C[1] - A[1]) - (B[1] - A[1]) * (C[0] - A[0]))

    @classmethod
    def _check_line_crossing(
        cls,
        p1: Tuple[int, int],
        p2: Tuple[int, int],
        prev_pt: Tuple[int, int],
        curr_pt: Tuple[int, int],
    ) -> Tuple[bool, str]:
        """
        Check line crossing using robust orientation test and side-change detection.
        """
        if prev_pt == curr_pt:
            return False, ""

        ccw_prev = cls._ccw(p1, p2, prev_pt)
        ccw_curr = cls._ccw(p1, p2, curr_pt)
        ccw_p1 = cls._ccw(prev_pt, curr_pt, p1)
        ccw_p2 = cls._ccw(prev_pt, curr_pt, p2)

        # 1. Standard Segment Intersection
        if (ccw_prev * ccw_curr <= 0) and (ccw_p1 * ccw_p2 <= 0):
            if ccw_prev < 0 and ccw_curr >= 0:
                return True, "IN"
            elif ccw_prev > 0 and ccw_curr <= 0:
                return True, "OUT"
            elif ccw_prev <= 0 and ccw_curr > 0:
                return True, "IN"
            elif ccw_prev >= 0 and ccw_curr < 0:
                return True, "OUT"

        # 2. Side Sign Flip Test (Fallback for fast motion / wide lines)
        if ccw_prev * ccw_curr < 0:
            direction = "IN" if ccw_prev < 0 else "OUT"
            return True, direction

        return False, ""

    def get_stats(self) -> Dict:
        """Get summary dictionary of counts."""
        total_in = len(self.global_crossed_in)
        total_out = len(self.global_crossed_out)
        stats = {
            "total_in": total_in,
            "total_out": total_out,
            "total_count": total_in + total_out,
            "lines": {},
        }
        for name in self.crossed_in:
            stats["lines"][name] = {
                "in": len(self.crossed_in[name]),
                "out": len(self.crossed_out[name]),
            }
        return stats

    def draw_hud(
        self,
        frame: np.ndarray,
        tracks: List[Tuple[np.ndarray, float, int]],
        fps: float = 0.0,
        driver_info: str = "RTSP",
    ):
        """
        Draw lines, tracked bounding boxes, trajectories, and live HUD overlay on the frame.
        """
        height, width = frame.shape[:2]

        # 1. Draw Counting Lines
        for line in self.lines_config:
            x1 = int(line.p1_ratio[0] * width)
            y1 = int(line.p1_ratio[1] * height)
            x2 = int(line.p2_ratio[0] * width)
            y2 = int(line.p2_ratio[1] * height)

            # Draw glowing line
            cv2.line(frame, (x1, y1), (x2, y2), line.color, 3, cv2.LINE_AA)
            cv2.circle(frame, (x1, y1), 6, line.color, -1)
            cv2.circle(frame, (x2, y2), 6, line.color, -1)

            # Label text for each line at segment midpoint
            line_in_cnt = len(self.crossed_in[line.name])
            line_out_cnt = len(self.crossed_out[line.name])
            label = f"{line.name} | IN:{line_in_cnt} OUT:{line_out_cnt}"
            mid_x = (x1 + x2) // 2
            mid_y = (y1 + y2) // 2
            cv2.putText(
                frame,
                label,
                (mid_x + 10, mid_y - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                line.color,
                2,
                cv2.LINE_AA,
            )

        # 2. Draw Tracked Objects
        for box, conf, p_id in tracks:
            x1, y1, x2, y2 = map(int, box)
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)

            # Determine box color based on line crossing status
            box_color = (0, 255, 0)
            tag = ""
            if p_id in self.global_crossed_in and p_id in self.global_crossed_out:
                box_color = (0, 165, 255)  # Orange
                tag = " [IN/OUT]"
            elif p_id in self.global_crossed_in:
                box_color = (0, 255, 255)  # Yellow
                tag = " [IN]"
            elif p_id in self.global_crossed_out:
                box_color = (255, 0, 255)  # Magenta
                tag = " [OUT]"

            cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
            cv2.circle(frame, (cx, cy), 4, box_color, -1)

            label_text = f"ID:{p_id}{tag}"
            (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 4, y1), box_color, cv2.FILLED)
            cv2.putText(
                frame,
                label_text,
                (x1 + 2, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),
                1,
                cv2.LINE_AA,
            )

        # 3. Draw HUD Dashboard Panel
        hud_w, hud_h = 340, 150
        hud_x, hud_y = 12, 12

        # Transparent dark overlay background
        sub_img = frame[hud_y : hud_y + hud_h, hud_x : hud_x + hud_w]
        black_rect = np.zeros(sub_img.shape, dtype=np.uint8)
        res = cv2.addWeighted(sub_img, 0.3, black_rect, 0.7, 0)
        frame[hud_y : hud_y + hud_h, hud_x : hud_x + hud_w] = res
        cv2.rectangle(frame, (hud_x, hud_y), (hud_x + hud_w, hud_y + hud_h), (255, 255, 255), 1, cv2.LINE_AA)

        # Dashboard Text
        total_in = len(self.global_crossed_in)
        total_out = len(self.global_crossed_out)
        total = total_in + total_out

        cv2.putText(frame, f"AI CORE 3-LINE COUNTER ({driver_info.upper()})", (hud_x + 10, hud_y + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, f"TOTAL IN: {total_in:<4} | TOTAL OUT: {total_out:<4} | SUM: {total}", (hud_x + 10, hud_y + 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        # Per-line details
        line_offset = 65
        for i, line in enumerate(self.lines_config):
            in_c = len(self.crossed_in[line.name])
            out_c = len(self.crossed_out[line.name])
            cv2.putText(
                frame,
                f"L{i+1}: {line.name.split(' ')[0]} -> IN: {in_c:<3} | OUT: {out_c:<3}",
                (hud_x + 10, hud_y + line_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                line.color,
                1,
                cv2.LINE_AA,
            )
            line_offset += 20

        cv2.putText(frame, f"Active Tracks: {len(tracks)} | FPS: {fps:.1f}", (hud_x + 10, hud_y + 138), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
