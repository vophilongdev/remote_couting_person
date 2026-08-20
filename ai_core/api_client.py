import logging
import os
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional
import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("CameraStatsAPI")

DEFAULT_API_URL = "https://api-dev.tado.vn/api/camera-statistics"
DEFAULT_SESSION_TOKEN = "359c8b594fe339f1d7b8337ad22f4ac1"

# ThreadPoolExecutor to run POST requests asynchronously in background without lagging video frames
_executor = ThreadPoolExecutor(max_workers=5)


def flush_pending_requests():
    """Wait for all pending POST requests to finish before exiting."""
    _executor.shutdown(wait=True)
    logger.info("[API] All pending POST requests completed.")


def _send_post_request(url: str, headers: dict, payload_data: dict, max_retries: int = 3):
    """Internal worker function executing HTTP POST request with auto-retry."""
    for attempt in range(1, max_retries + 1):
        try:
            response = requests.post(url, json=payload_data, headers=headers, timeout=5)
            if response.status_code in [200, 201]:
                logger.info(f"[POST Success {response.status_code} OK] Stream:{payload_data.get('stream_id')} | Data:{payload_data.get('data')}")
                return
            else:
                logger.warning(f"[POST Error {response.status_code}] Attempt {attempt}/{max_retries} | Response: {response.text}")
        except Exception as e:
            logger.error(f"[POST Exception] Attempt {attempt}/{max_retries} | {e}")

        if attempt < max_retries:
            import time
            time.sleep(1)

    logger.error(f"[POST FAILED] All {max_retries} attempts failed for Stream:{payload_data.get('stream_id')}")


def post_camera_statistic(
    stream_id: str,
    total_in: int,
    total_out: int,
    current_count: int,
    api_url: Optional[str] = None,
    session_token: Optional[str] = None,
    async_send: bool = True,
):
    """
    Direct function to post camera counting statistics to Backend API.
    No intermediate classes, strategy pattern, or factory required.
    """
    if not stream_id:
        return

    url = (api_url or os.environ.get("API_URL", DEFAULT_API_URL)).rstrip("/")
    token = session_token or os.environ.get("SESSION_TOKEN", DEFAULT_SESSION_TOKEN)

    dt = datetime.now()
    payload = {
        "stream_id": stream_id,
        "metric_type": "people_counting",
        "data": {
            "count": current_count,
            "person": current_count,
            "in": total_in,
            "out": total_out,
            "enter": total_in,
            "exit": total_out,
        },
        "year": dt.year,
        "month": dt.month,
        "day": dt.day,
        "hour": dt.hour,
        "minute": dt.minute,
    }

    headers = {"Content-Type": "application/json"}
    if token:
        headers["session"] = token

    if async_send:
        _executor.submit(_send_post_request, url, headers, payload)
    else:
        _send_post_request(url, headers, payload)


def get_backend_cameras(session_token: Optional[str] = None) -> List[Dict[str, Any]]:
    """Direct function to fetch list of active cameras from Backend API."""
    url = "https://api-dev.tado.vn/api/cameras"
    token = session_token or os.environ.get("SESSION_TOKEN", DEFAULT_SESSION_TOKEN)

    headers = {"Content-Type": "application/json"}
    if token:
        headers["session"] = token

    try:
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code == 200:
            res = response.json()
            if isinstance(res, dict) and "data" in res:
                return res["data"]
            elif isinstance(res, list):
                return res
        logger.warning(f"[GET Cameras Error {response.status_code}] {response.text}")
        return []
    except Exception as e:
        logger.error(f"[GET Cameras Exception] {e}")
        return []


def get_backend_rules(
    stream_id: Optional[str] = None,
    rule_type: Optional[str] = "people_counting",
    session_token: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Direct function to fetch list of rules (GET /api/rules) from Backend API.
    Strictly filters by stream_id, rule_type (default: 'people_counting'), and active status.
    """
    url = "https://api-dev.tado.vn/api/rules"
    token = session_token or os.environ.get("SESSION_TOKEN", DEFAULT_SESSION_TOKEN)

    params = {"skip": 0, "limit": 100}
    if stream_id:
        params["stream_ids"] = stream_id
    if rule_type:
        params["type"] = rule_type

    headers = {"Content-Type": "application/json"}
    if token:
        headers["session"] = token

    try:
        response = requests.get(url, headers=headers, params=params, timeout=10)
        if response.status_code == 200:
            res = response.json()
            rules = []
            if isinstance(res, dict) and "data" in res:
                rules = res["data"]
            elif isinstance(res, list):
                rules = res
            
            filtered_rules = []
            for r in rules:
                # 1. Ignore deleted or inactive rules
                if r.get("deleted_at") is not None or r.get("status") == 0:
                    continue

                # 2. Filter rules matching stream_id
                if stream_id:
                    st_ids = r.get("stream_ids") or []
                    if stream_id not in st_ids and str(r.get("stream_id")) != stream_id:
                        continue

                # 3. Strictly match rule_type if specified
                if rule_type:
                    r_type = str(r.get("type") or r.get("service_type") or "").strip().lower()
                    if r_type != rule_type.strip().lower():
                        continue

                filtered_rules.append(r)
            return filtered_rules
        logger.warning(f"[GET Rules Error {response.status_code}] {response.text}")
        return []
    except Exception as e:
        logger.error(f"[GET Rules Exception] {e}")
        return []


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

