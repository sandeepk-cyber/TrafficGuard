"""
TrafficGuard - Camera & Live Stream Ingestion Manager
Decoupled continuous capture with latest-frame buffer, connection state tracking (ONLINE/OFFLINE/RECONNECTING),
automatic reconnect backoff, and clean subprocess lifecycle.
Zero synthetic cartoon fallbacks. Zero emojis.
"""
import os
import time
import subprocess
import threading
import logging
import cv2
import yaml
import numpy as np

logger = logging.getLogger("trafficguard.camera")

CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config", "cameras.yaml")
FRAME_WIDTH = 640
FRAME_HEIGHT = 360


class FFmpegStreamReader:
    """Reads live HLS / RTSP / MJPEG streams by piping raw BGR frames from FFmpeg."""
    def __init__(self, url, width=FRAME_WIDTH, height=FRAME_HEIGHT):
        self.url = url
        self.width = width
        self.height = height
        self.frame_size = width * height * 3
        self.process = None
        self.latest_frame = None
        self.frame_lock = threading.Lock()
        self.running = False
        self.thread = None
        self.last_frame_time = 0.0
        self._start()

    def _start(self):
        self.running = True
        self.thread = threading.Thread(target=self._reader_loop, daemon=True)
        self.thread.start()

    def _reader_loop(self):
        cmd = [
            "ffmpeg",
            "-loglevel", "error",
            "-reconnect", "1",
            "-reconnect_at_eof", "1",
            "-reconnect_streamed", "1",
            "-reconnect_delay_max", "3",
            "-i", self.url,
            "-vf", f"scale={self.width}:{self.height}",
            "-f", "rawvideo",
            "-pix_fmt", "bgr24",
            "-"
        ]
        try:
            self.process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
            logger.info("FFmpeg pipe launched for live stream: %s", self.url)

            while self.running and self.process and self.process.poll() is None:
                raw_bytes = self.process.stdout.read(self.frame_size)
                if len(raw_bytes) == self.frame_size:
                    frame = np.frombuffer(raw_bytes, np.uint8).reshape((self.height, self.width, 3))
                    with self.frame_lock:
                        self.latest_frame = frame
                        self.last_frame_time = time.time()
                else:
                    time.sleep(0.02)
        except Exception as e:
            logger.warning("FFmpeg pipe stream error for %s: %s", self.url, e)
        finally:
            self._cleanup_process()

    def read_frame(self):
        with self.frame_lock:
            if self.latest_frame is not None:
                return True, self.latest_frame.copy()
            return False, None

    def _cleanup_process(self):
        if self.process:
            try:
                self.process.terminate()
                self.process.wait(timeout=1.0)
            except Exception:
                try:
                    self.process.kill()
                except Exception:
                    pass
            self.process = None

    def release(self):
        self.running = False
        self._cleanup_process()
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1.0)


class CameraStreamSource:
    """Decoupled continuous frame ingest for a single camera source."""
    def __init__(self, config):
        self.id = config.get("id", "cam1")
        self.name = config.get("name", "Camera")
        self.stream_type = config.get("type", "file").lower()
        self.url = config.get("url", "")
        self.location = config.get("location", "Benchmark clip")
        self.latitude = config.get("latitude")
        self.longitude = config.get("longitude")
        self.loop = config.get("loop", True)

        self.status = "CONNECTING"  # CONNECTING, ONLINE, OFFLINE, RECONNECTING
        self.last_frame_time = 0.0
        self.capture_fps = 0.0

        self.cv_cap = None
        self.ffmpeg_reader = None

        self.latest_frame = None
        self.frame_lock = threading.Lock()
        self.running = False
        self.capture_thread = None

        self._start_capture()

    def _start_capture(self):
        self.running = True
        self.capture_thread = threading.Thread(target=self._capture_worker, daemon=True)
        self.capture_thread.start()

    def _connect_source(self):
        """Initializes OpenCV or FFmpeg backend."""
        if not self.url:
            self.status = "OFFLINE"
            return False

        # Live HLS / RTSP / MJPEG
        if self.stream_type in ["hls", "rtsp", "mjpeg", "iptv"] or ".m3u8" in self.url:
            try:
                self.ffmpeg_reader = FFmpegStreamReader(self.url, FRAME_WIDTH, FRAME_HEIGHT)
                return True
            except Exception as e:
                logger.warning("FFmpeg connection failed for %s: %s", self.url, e)
                return False

        # Local video file or OpenCV RTSP
        if os.path.exists(self.url) or self.stream_type == "file":
            try:
                self.cv_cap = cv2.VideoCapture(self.url)
                if self.cv_cap.isOpened():
                    return True
            except Exception as e:
                logger.warning("OpenCV VideoCapture failed for %s: %s", self.url, e)
                return False

        # Fallback OpenCV network capture
        try:
            self.cv_cap = cv2.VideoCapture(self.url)
            return self.cv_cap.isOpened()
        except Exception:
            return False

    def _capture_worker(self):
        """Dedicated continuous capture thread maintaining a zero-latency latest-frame buffer."""
        reconnect_backoff = 1.0
        frame_counter = 0
        sec_timer = time.time()

        while self.running:
            connected = self._connect_source()
            if not connected:
                self.status = "OFFLINE"
                time.sleep(min(8.0, reconnect_backoff))
                reconnect_backoff *= 1.5
                continue

            self.status = "ONLINE"
            reconnect_backoff = 1.0

            while self.running:
                loop_start = time.time()
                frame = None
                success = False

                if self.ffmpeg_reader:
                    success, frame = self.ffmpeg_reader.read_frame()
                    if not success and (time.time() - self.ffmpeg_reader.last_frame_time > 4.0):
                        # Stream dropped
                        self.status = "RECONNECTING"
                        break

                elif self.cv_cap and self.cv_cap.isOpened():
                    try:
                        ret, raw = self.cv_cap.read()
                        if ret and raw is not None:
                            frame = cv2.resize(raw, (FRAME_WIDTH, FRAME_HEIGHT))
                            success = True
                        else:
                            if self.loop and self.stream_type == "file":
                                self.cv_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                                ret2, raw2 = self.cv_cap.read()
                                if ret2 and raw2 is not None:
                                    frame = cv2.resize(raw2, (FRAME_WIDTH, FRAME_HEIGHT))
                                    success = True
                            else:
                                self.status = "OFFLINE"
                                break
                    except Exception:
                        if not self.running:
                            break
                        time.sleep(0.05)

                if success and frame is not None:
                    with self.frame_lock:
                        self.latest_frame = frame
                        self.last_frame_time = time.time()
                        self.status = "ONLINE"

                    frame_counter += 1
                    now = time.time()
                    if now - sec_timer >= 1.0:
                        self.capture_fps = round(frame_counter / (now - sec_timer), 1)
                        frame_counter = 0
                        sec_timer = now

                    # Pace capture to avoid 100% CPU spinning on local file playback
                    if self.stream_type == "file":
                        time.sleep(0.025)
                else:
                    if self.last_frame_time > 0 and (time.time() - self.last_frame_time > 4.0):
                        self.status = "OFFLINE"
                    time.sleep(0.05)

            # Cleanup before potential reconnect
            try:
                if self.cv_cap:
                    self.cv_cap.release()
                    self.cv_cap = None
            except Exception:
                pass
            if self.ffmpeg_reader:
                self.ffmpeg_reader.release()
                self.ffmpeg_reader = None

    def read_frame(self):
        """Returns the newest frame available from the capture buffer."""
        with self.frame_lock:
            if self.latest_frame is not None and self.status == "ONLINE":
                return True, self.latest_frame.copy()

        # Render clean, restrained OFFLINE placeholder frame
        placeholder = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)
        placeholder[:] = (24, 24, 28)
        msg = f"CAMERA {self.status}" if self.status in ["OFFLINE", "RECONNECTING"] else "CONNECTING..."
        cv2.putText(placeholder, msg, (FRAME_WIDTH // 2 - 110, FRAME_HEIGHT // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (120, 120, 140), 1, cv2.LINE_AA)
        cv2.putText(placeholder, self.name, (FRAME_WIDTH // 2 - 130, FRAME_HEIGHT // 2 + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (80, 80, 95), 1, cv2.LINE_AA)
        return False, placeholder

    def release(self):
        self.running = False
        if self.capture_thread and self.capture_thread.is_alive():
            self.capture_thread.join(timeout=1.5)
        try:
            if self.cv_cap:
                self.cv_cap.release()
                self.cv_cap = None
        except Exception:
            pass
        if self.ffmpeg_reader:
            self.ffmpeg_reader.release()
            self.ffmpeg_reader = None


class CameraManager:
    def __init__(self, config_path=CONFIG_PATH):
        self.config_path = config_path
        self.lock = threading.RLock()
        self.cameras = {}
        self.active_camera_id = None
        self.active_source = None
        self._load_catalog()

    def _load_catalog(self):
        os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, "r", encoding="utf-8") as f:
                    data = yaml.safe_load(f) or {}
                self.cameras = data.get("cameras", {})
            except Exception as e:
                logger.error("Error reading camera catalog: %s", e)
                self.cameras = {}

        if not self.cameras:
            self.cameras = {
                "cctv_kaggle_04_highway": {
                    "name": "CCTV Benchmark: Highway Incident",
                    "type": "file",
                    "url": "data/test/cctv_kaggle_04_highway_highspeed.mp4",
                    "source_type": "benchmark",
                    "location": "Elevated highway benchmark",
                    "latitude": None,
                    "longitude": None,
                    "loop": True
                }
            }
            self.save_catalog()

        first_cam_id = next(iter(self.cameras))
        self.set_active_camera(first_cam_id)

    def save_catalog(self):
        try:
            with open(self.config_path, "w", encoding="utf-8") as f:
                yaml.safe_dump({"cameras": self.cameras}, f, sort_keys=False)
        except Exception as e:
            logger.error("Error saving camera catalog: %s", e)

    def list_cameras(self):
        with self.lock:
            res = []
            for cid, cinfo in self.cameras.items():
                res.append({
                    "id": cid,
                    "name": cinfo.get("name", cid),
                    "type": cinfo.get("type", "file"),
                    "url": cinfo.get("url", ""),
                    "location": cinfo.get("location", "CCTV Feed"),
                    "latitude": cinfo.get("latitude"),
                    "longitude": cinfo.get("longitude"),
                    "is_active": (cid == self.active_camera_id)
                })
            return res

    def set_active_camera(self, camera_id):
        with self.lock:
            if camera_id not in self.cameras:
                logger.warning("Camera ID '%s' not registered in catalog.", camera_id)
                return False

            if self.active_source:
                self.active_source.release()
                self.active_source = None

            cfg = dict(self.cameras[camera_id])
            cfg["id"] = camera_id
            self.active_camera_id = camera_id
            self.active_source = CameraStreamSource(cfg)
            logger.info("Active camera switched to: %s (%s)", self.active_source.name, camera_id)
            return True

    def read_frame(self):
        with self.lock:
            if self.active_source:
                return self.active_source.read_frame()
            blank = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)
            return False, blank

    def get_active_camera_info(self):
        with self.lock:
            if self.active_source:
                return {
                    "id": self.active_source.id,
                    "name": self.active_source.name,
                    "status": self.active_source.status,
                    "capture_fps": self.active_source.capture_fps,
                    "type": self.active_source.stream_type,
                    "url": self.active_source.url,
                    "location": self.active_source.location,
                    "latitude": self.active_source.latitude,
                    "longitude": self.active_source.longitude
                }
            return {
                "id": "none", "name": "No Camera Active", "status": "OFFLINE",
                "capture_fps": 0.0, "type": "none", "url": "",
                "location": "None", "latitude": None, "longitude": None
            }

    def release(self):
        with self.lock:
            if self.active_source:
                self.active_source.release()
                self.active_source = None
