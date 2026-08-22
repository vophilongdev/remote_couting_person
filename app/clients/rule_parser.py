# -*- coding: utf-8 -*-
"""
Rule Parser — Parses Backend API rules JSON into line configuration data.
Extracted from api_client.py for Single Responsibility.
"""

import logging
from typing import Any, Dict, List

logger = logging.getLogger("RuleParser")


def parse_lines_from_rules(rules: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Parse rules JSON into line configuration data dictionaries.
    Extracts name, condition points [[x1, y1], [x2, y2]] from filters where type=='direction'.
    """
    lines = []
    default_colors = [
        (0, 255, 255),  # Yellow
        (0, 255, 0),    # Green
        (255, 0, 255),  # Magenta
        (0, 165, 255),  # Orange
        (255, 255, 0),  # Cyan
    ]

    allowed_types = ["line_in", "line_out", "line", "direction"]

    for idx, rule in enumerate(rules):
        rule_name = rule.get("name") or f"LINE {idx+1}"
        filters = rule.get("filters") or (rule.get("param") or {}).get("filters") or []
        for filt in filters:
            filt_type = str(filt.get("type") or "").strip().lower()
            status = filt.get("status", 1)
            condition = filt.get("condition")
            
            # Match Line In, Line Out, Line, and Direction filters with valid 2D float coordinate pairs
            if status == 1 and condition and isinstance(condition, list) and len(condition) >= 2:
                if filt_type in allowed_types:
                    try:
                        x1, y1 = float(condition[0][0]), float(condition[0][1])
                        x2, y2 = float(condition[1][0]), float(condition[1][1])
                        color = default_colors[len(lines) % len(default_colors)]
                        lines.append({
                            "name": rule_name,
                            "p1_ratio": (x1, y1),
                            "p2_ratio": (x2, y2),
                            "color": color,
                            "type": filt_type
                        })
                        logger.info(
                            f"[Rule Parser] Parsed '{rule_name}' ({filt_type}) normalized coords: P1=({x1:.4f}, {y1:.4f}), P2=({x2:.4f}, {y2:.4f})"
                        )
                    except (ValueError, TypeError, IndexError) as err:
                        logger.warning(f"[Rule Parser Warning] Could not parse condition {condition}: {err}")

    return lines
