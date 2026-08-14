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
