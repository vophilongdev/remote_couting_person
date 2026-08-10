import logging
import os
import time
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("CameraStatsAPI")


@dataclass
class CameraStatisticData:
    count: int
    person: int
    in_count: int = 0
    out_count: int = 0
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        res = {
            "count": self.count,
            "person": self.person,
            "in": self.in_count,
            "out": self.out_count,
        }
        res.update(self.extra)
        return res


@dataclass
class CameraStatisticPayload:
    stream_id: str
    metric_type: str = "people_counting"
    data: Dict[str, Any] = field(default_factory=dict)
    time: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "stream_id": self.stream_id,
            "metric_type": self.metric_type,
            "data": self.data,
            "time": self.time,
        }


class CameraStatisticsAPIClient:
    """REST Client for CMS Backend Camera Statistics API."""

    def __init__(self, base_url: str = "https://api-dev.tado.vn/api/camera-statistics", session_token: str = ""):
        self.base_url = base_url.rstrip("/")
        self.session_token = session_token or os.environ.get("SESSION_TOKEN", "")

    def _get_headers(self) -> Dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.session_token:
            headers["session"] = self.session_token
        return headers

    def create_statistic(self, payload: CameraStatisticPayload) -> Optional[Dict[str, Any]]:
        url = self.base_url
        try:
            response = requests.post(url, json=payload.to_dict(), headers=self._get_headers(), timeout=5)
            if response.status_code in [200, 201]:
                logger.info(f"[POST Success {response.status_code} OK] Stream:{payload.stream_id} | Data:{payload.data}")
                try:
                    return response.json()
                except Exception:
                    return {}
            else:
                logger.warning(f"[POST Error {response.status_code}] Response: {response.text}")
                return None
        except Exception as e:
            logger.error(f"[POST Exception] {e}")
            return None

    def create_statistic_batch(self, payloads: List[CameraStatisticPayload]) -> Optional[Dict[str, Any]]:
        url = f"{self.base_url}/batch"
        body = {"records": [p.to_dict() for p in payloads]}
        try:
            response = requests.post(url, json=body, headers=self._get_headers(), timeout=5)
            if response.status_code in [200, 201]:
                logger.info(f"[POST Batch Success {response.status_code} OK] Sent {len(payloads)} records")
                return response.json()
            else:
                logger.warning(f"[POST Batch Error {response.status_code}] Response: {response.text}")
                return None
        except Exception as e:
            logger.error(f"[POST Batch Exception] {e}")
            return None


class BaseReportStrategy(ABC):
    @abstractmethod
    def report(self, payload: CameraStatisticPayload):
        pass

    def flush(self):
        pass


class SyncReportStrategy(BaseReportStrategy):
    def __init__(self, client: CameraStatisticsAPIClient):
        self.client = client
        self.executor = ThreadPoolExecutor(max_workers=3)

    def report(self, payload: CameraStatisticPayload):
        if not payload.stream_id:
            return
        self.executor.submit(self.client.create_statistic, payload)

    def flush(self):
        self.executor.shutdown(wait=True)


class BatchReportStrategy(BaseReportStrategy):
    def __init__(self, client: CameraStatisticsAPIClient, batch_size: int = 10):
        self.client = client
        self.batch_size = batch_size
        self.buffer: List[CameraStatisticPayload] = []
        self.executor = ThreadPoolExecutor(max_workers=2)

    def report(self, payload: CameraStatisticPayload):
        if not payload.stream_id:
            return
        self.buffer.append(payload)
        if len(self.buffer) >= self.batch_size:
            to_send = self.buffer.copy()
            self.buffer.clear()
            self.executor.submit(self.client.create_statistic_batch, to_send)

    def flush(self):
        if self.buffer:
            to_send = self.buffer.copy()
            self.buffer.clear()
            self.client.create_statistic_batch(to_send)
        self.executor.shutdown(wait=True)


class ReportStrategyFactory:
    @staticmethod
    def create_strategy(
        stream_id: str,
        api_url: str,
        session_token: str,
        batch_mode: bool = False,
        batch_size: int = 10,
    ) -> BaseReportStrategy:
        client = CameraStatisticsAPIClient(base_url=api_url, session_token=session_token)

        if batch_mode:
            return BatchReportStrategy(client=client, batch_size=batch_size)
        else:
            return SyncReportStrategy(client=client)
