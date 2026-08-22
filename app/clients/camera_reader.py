# -*- coding: utf-8 -*-
"""
Multi-Source High-Performance Camera Reader — Outbound Adapter
Supports 3 Camera Streaming Drivers/Libraries:
1. RTSP / Video File / Webcam (PyAV with nobuffer + fallback to OpenCV VideoCapture)
2. Dahua SDK (NetSDK via DahuaCameraViewer)
3. Hikvision SDK (HCNetSDK via HIKCameraViewer)
"""

import logging
import queue
import threading
import time

import cv2
import numpy as np

try:
    from app.services.sdk.dha_sdk_realplay import DahuaCameraViewer
except ImportError:
    try:
        from services.sdk.dha_sdk_realplay import DahuaCameraViewer
    except ImportError:
        DahuaCameraViewer = None

try:
    from app.services.sdk.hik_sdk_realplay import HIKCameraViewer
except ImportError:
    try:
        from services.sdk.hik_sdk_realplay import HIKCameraViewer
    except ImportError:
        HIKCameraViewer = None

logger = logging.getLogger(__name__)


class CameraReader:
    """
    Unified Threaded Camera Reader for high-throughput video processing.
    """

    DRIVER_RTSP = "rtsp"
    DRIVER_DAHUA = "dahua"
    DRIVER_HIK = "hik"

    def __init__(
        self,
        source: str,
        driver_type: str = "rtsp",
        storage_url: str = "",
        storage_port: int = 37777,
        storage_username: str = "",
        storage_password: str = "",
        storage_channel: int = 1,
        queue_size: int = 16,
        target_fps: float = 30.0,
    ):
        self.source = str(source)
        self.driver_type = str(driver_type).lower().strip()
        if self.driver_type in ["0", "rtsp", "file"]:
            self.driver_type = self.DRIVER_RTSP
        elif self.driver_type in ["1", "dahua", "dha"]:
            self.driver_type = self.DRIVER_DAHUA
        elif self.driver_type in ["2", "hik", "hikvision"]:
            self.driver_type = self.DRIVER_HIK

        self.storage_url = storage_url or self.source
        self.storage_port = int(storage_port)
        self.storage_username = storage_username
        self.storage_password = storage_password
        self.storage_channel = int(storage_channel)

        self.queue_size = queue_size
        self.target_fps = target_fps
        self.frame_queue = queue.Queue(maxsize=queue_size)
        self.shutdown_flag = threading.Event()
        self.reader_thread = None

        self.is_connected = False
        self.total_frames_read = 0
        self.dropped_frames = 0

    def start(self):
        """Start the background reading thread."""
        self.shutdown_flag.clear()
        self.reader_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.reader_thread.start()
        logger.info(f"CameraReader thread started. Driver: {self.driver_type}")

    def stop(self):
        """Signal thread to stop and wait for completion."""
        self.shutdown_flag.set()
        if self.reader_thread and self.reader_thread.is_alive():
            self.reader_thread.join(timeout=2.0)
        logger.info("CameraReader thread stopped.")

    def get_frame(self, timeout: float = 0.5):
        """
        Fetch latest frame from queue.
        Returns (frame, frame_idx, timestamp) or (None, 0, 0.0)
        """
        try:
            return self.frame_queue.get(timeout=timeout)
        except queue.Empty:
            return None, 0, 0.0

    def _push_frame(self, frame, frame_idx, timestamp):
        """Push frame to queue. Drop oldest frame if full to prevent latency build-up."""
        if frame is None:
            return
        try:
            self.frame_queue.put((frame, frame_idx, timestamp), block=False)
        except queue.Full:
            try:
                self.frame_queue.get_nowait()
                self.dropped_frames += 1
                self.frame_queue.put((frame, frame_idx, timestamp), block=False)
            except (queue.Empty, queue.Full):
                pass
        self.total_frames_read += 1

    def _worker_loop(self):
        """Main reader worker loop handling driver selection."""
        if self.driver_type == self.DRIVER_DAHUA:
            self._read_dahua_sdk()
        elif self.driver_type == self.DRIVER_HIK:
            self._read_hik_sdk()
        else:
            self._read_rtsp_pyav_cv()

    def _read_rtsp_pyav_cv(self):
        """Read frames using PyAV (low latency) with fallback to OpenCV VideoCapture."""
        use_pyav = False
        try:
            import av
            use_pyav = True
        except ImportError:
            logger.warning("PyAV ('av') package not installed. Falling back to OpenCV VideoCapture.")

        if use_pyav and (self.source.startswith("rtsp://") or self.source.startswith("rtmp://") or self.source.startswith("http://")):
            self._read_pyav()
        else:
            self._read_opencv()

    def _read_pyav(self):
        import av
        logger.info(f"Opening RTSP stream with PyAV: {self.source}")
        frame_idx = 0
        while not self.shutdown_flag.is_set():
            container = None
            try:
                container = av.open(
                    self.source,
                    options={
                        'rtsp_transport': 'tcp',
                        'fflags': 'nobuffer+discardcorrupt',
                        'err_detect': 'ignore_err',
                        'max_delay': '500000',
                        'reorder_queue_size': '1'
                    },
                    timeout=10
                )
                stream = container.streams.video[0]
                self.is_connected = True
                logger.info("Connected to RTSP stream via PyAV.")

                for packet in container.demux(stream):
                    if self.shutdown_flag.is_set():
                        break
                    for frame in packet.decode():
                        if self.shutdown_flag.is_set():
                            break
                        frame_idx += 1
                        img = frame.to_ndarray(format='bgr24')
                        self._push_frame(img, frame_idx, time.time())

            except Exception as e:
                logger.error(f"PyAV RTSP error: {e}. Retrying in 2 seconds...")
                self.is_connected = False
                time.sleep(2)
            finally:
                if container:
                    try:
                        container.close()
                    except Exception:
                        pass

    def _read_opencv(self):
        logger.info(f"Opening video source with OpenCV: {self.source}")
        src = int(self.source) if self.source.isdigit() else self.source

        while not self.shutdown_flag.is_set():
            cap = cv2.VideoCapture(src)

            if not cap.isOpened():
                logger.warning(f"RTSP disconnected, retrying in 2 seconds... (Source: {self.source})")
                self.is_connected = False
                time.sleep(2.0)
                continue

            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            self.is_connected = True
            frame_idx = 0
            frame_delay = 1.0 / self.target_fps if isinstance(src, str) and not src.isdigit() else 0.001

            while not self.shutdown_flag.is_set():
                ret, frame = cap.read()
                if not ret or frame is None:
                    is_local_file = isinstance(src, str) and not src.isdigit() and not (
                        self.source.startswith("rtsp://") or self.source.startswith("rtmp://") or self.source.startswith("http://")
                    )
                    if is_local_file:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        time.sleep(0.05)
                        continue
                    else:
                        logger.warning("RTSP disconnected, retrying in 2 seconds...")
                        self.is_connected = False
                        cap.release()
                        time.sleep(2.0)
                        break

                frame_idx += 1
                self._push_frame(frame, frame_idx, time.time())
                if frame_delay > 0:
                    time.sleep(frame_delay)

            cap.release()
            self.is_connected = False

    def _read_dahua_sdk(self):
        logger.info(f"Connecting to Dahua Camera via SDK: {self.storage_url}:{self.storage_port}")
        if DahuaCameraViewer is None:
            logger.error("Dahua SDK viewer module could not be imported.")
            return

        viewer = DahuaCameraViewer()
        if not viewer.connect(self.storage_url, self.storage_port, self.storage_username, self.storage_password):
            logger.error("Dahua SDK connection failed.")
            return

        # Try 0-based channel index first, fallback to 1-based or 0
        ch = self.storage_channel - 1 if self.storage_channel > 0 else 0
        viewer.start_stream(ch, stream_type=0)
        self.is_connected = True
        frame_idx = 0
        retry_ch_count = 0

        while not self.shutdown_flag.is_set() and viewer.is_playing:
            try:
                frame = viewer.get_frame()
                if frame is not None:
                    frame_idx += 1
                    self._push_frame(frame, frame_idx, time.time())
                else:
                    retry_ch_count += 1
                    if frame_idx == 0 and retry_ch_count == 100:  # ~1 second with no frames
                        logger.warning(f"Dahua channel {ch} returned no frames, trying channel {self.storage_channel}...")
                        viewer.stop_stream()
                        ch = self.storage_channel
                        viewer.start_stream(ch, stream_type=0)
                    time.sleep(0.01)
                time.sleep(0.033)
            except Exception as e:
                logger.error(f"Dahua SDK frame reader error: {e}")
                time.sleep(0.5)

        viewer.cleanup()
        self.is_connected = False

    def _read_hik_sdk(self):
        logger.info(f"Connecting to Hikvision Camera via SDK: {self.storage_url}:{self.storage_port}")
        if HIKCameraViewer is None:
            logger.error("Hikvision SDK viewer module could not be imported.")
            return

        viewer = HIKCameraViewer()
        if not viewer.connect(self.storage_url, self.storage_port, self.storage_username, self.storage_password):
            logger.error("Hikvision SDK connection failed.")
            return

        if not viewer.start_stream(self.storage_channel, 0):
            logger.error("Hikvision SDK start_stream failed.")
            return

        self.is_connected = True
        frame_idx = 0
        time.sleep(1.0)

        while not self.shutdown_flag.is_set():
            try:
                frame = viewer.get_frame()
                if frame is not None:
                    frame_idx += 1
                    self._push_frame(frame, frame_idx, time.time())
                else:
                    time.sleep(0.01)
                time.sleep(0.033)
            except Exception as e:
                logger.error(f"Hikvision SDK frame reader error: {e}")
                time.sleep(0.5)

        viewer.stop_stream()
        viewer.cleanup()
        self.is_connected = False
