# -*- coding: utf-8 -*-
"""
Lightweight IoU Tracker for tracking detected person bounding boxes.
Used when detection is offloaded via gRPC or external model.
"""

from typing import List, Tuple

import numpy as np


def calculate_iou(box1, box2):
    x1_1, y1_1, x2_1, y2_1 = box1[:4]
    x1_2, y1_2, x2_2, y2_2 = box2[:4]

    xi1 = max(x1_1, x1_2)
    yi1 = max(y1_1, y1_2)
    xi2 = min(x2_1, x2_2)
    yi2 = min(y2_1, y2_2)

    if xi1 < xi2 and yi1 < yi2:
        inter_area = (xi2 - xi1) * (yi2 - yi1)
        box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
        box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
        union_area = box1_area + box2_area - inter_area
        return inter_area / union_area if union_area > 0 else 0
    return 0.0


class PersonTracker:
    """
    Lightweight IoU Tracker for tracking detected person bounding boxes.
    Used when detection is offloaded via gRPC or external model.
    """

    def __init__(self, max_disappeared: int = 15):
        self.next_id = 1
        self.tracked_objects = {}  # id -> {'box': box, 'conf': conf, 'disappeared': int}
        self.max_disappeared = max_disappeared

    def update(self, detected_boxes: List[Tuple]) -> List[Tuple[np.ndarray, float, int]]:
        """
        Input: list of (x1, y1, x2, y2, conf) or [x1, y1, x2, y2]
        Output: list of (box_np, conf, track_id)
        """
        if len(detected_boxes) == 0:
            to_delete = []
            for p_id in self.tracked_objects:
                self.tracked_objects[p_id]['disappeared'] += 1
                if self.tracked_objects[p_id]['disappeared'] > self.max_disappeared:
                    to_delete.append(p_id)
            for p_id in to_delete:
                del self.tracked_objects[p_id]
            return []

        active_tracks = []
        updated_ids = set()

        for det in detected_boxes:
            box = np.array(det[:4], dtype=float)
            conf = float(det[4]) if len(det) > 4 else 1.0

            best_id = None
            best_iou = 0.25

            for p_id, data in self.tracked_objects.items():
                if p_id in updated_ids:
                    continue
                iou = calculate_iou(box, data['box'])
                if iou > best_iou:
                    best_iou = iou
                    best_id = p_id

            if best_id is not None:
                self.tracked_objects[best_id] = {'box': box, 'conf': conf, 'disappeared': 0}
                updated_ids.add(best_id)
                active_tracks.append((box, conf, best_id))
            else:
                p_id = self.next_id
                self.next_id += 1
                self.tracked_objects[p_id] = {'box': box, 'conf': conf, 'disappeared': 0}
                updated_ids.add(p_id)
                active_tracks.append((box, conf, p_id))

        return active_tracks
