"""
TrafficGuard - Dynamic Camera & Live IPTV Ingestion Manager
Supports on-the-fly camera injection (HLS .m3u8, RTSP, MJPEG, Local Video, Simulation).
"""
import os
import time
import subprocess
import threading
import logging
import cv2
import yaml
import numpy as np

from app.video import SyntheticTrafficGenerator, FRAME_WIDTH, FRAME_HEIGHT

logger = logging.getLogger("trafficguard.camera")

CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config", "cameras.yaml")


class FFmpegStreamReader:
    """Reads live HLS (.m3u8) / RTSP / Web streams by piping raw BGR frames from FFmpeg."""
    def __init__(self, url, width=FRAME_WIDTH, height=FRAME_HEIGHT):
        self.url = url
        self.width = width
        self.height = height
        self.frame_size = width * height * 3
        self.process = None
        self.latest_frame = None
        self.running = False
        self.thread = None
        self.connected = False
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
            "-reconnect_delay_max", "2",
            "-i", self.url,
            "-vf", f"scale={self.width}:{self.height}",
            "-f", "rawvideo",
            "-pix_fmt", "bgr24",
            "-"
        ]
        try:
            self.process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
            self.connected = True
            logger.info("FFmpeg pipe launched for live stream: %s", self.url)

            while self.running and self.process and self.process.poll() is None:
                raw_bytes = self.process.stdout.read(self.frame_size)
                if len(raw_bytes) == self.frame_size:
                    frame = np.frombuffer(raw_bytes, np.uint8).reshape((self.height, self.width, 3))
                    self.latest_frame = frame
                else:
                    time.sleep(0.01)
        except Exception as e:
            logger.warning("FFmpeg pipe stream error: %s", e)
        finally:
            self.connected = False
            if self.process:
                try:
                    self.process.kill()
                except Exception:
                    pass

    def read_frame(self):
        return (self.latest_frame is not None), self.latest_frame

    def release(self):
        self.running = False
        if self.process:
            try:
                self.process.kill()
            except Exception:
                pass
        self.connected = False


class CameraStreamSource:
    def __init__(self, config):
        self.id = config.get("id", "cam1")
        self.name = config.get("name", "Camera")
        self.stream_type = config.get("type", "simulation").lower()
        self.url = config.get("url", "")
        self.location = config.get("location", "Urban Corridor")
        self.latitude = config.get("latitude", 37.7749)
        self.longitude = config.get("longitude", -122.4194)

        self.cv_cap = None
        self.ffmpeg_reader = None
        self.sim_gen = None
        self.connected = False
        self._init_source()

    def _init_source(self):
        # 1. Simulation stream
        if self.stream_type == "simulation" or not self.url:
            self.sim_gen = SyntheticTrafficGenerator()
            self.connected = True
            return

        # 2. Local Video File
        if self.stream_type == "file" or os.path.exists(self.url):
            self.cv_cap = cv2.VideoCapture(self.url)
            self.connected = self.cv_cap.isOpened()
            if not self.connected:
                logger.warning("Failed to open local video file: %s. Using simulation fallback.", self.url)
                self.sim_gen = SyntheticTrafficGenerator()
                self.connected = True
            return

        # 3. Live HLS (.m3u8) or IPTV via FFmpeg
        if self.url.endswith(".m3u8") or "m3u8" in self.url or self.stream_type in ["hls", "iptv"]:
            try:
                self.ffmpeg_reader = FFmpegStreamReader(self.url)
                self.connected = True
                return
            except Exception as e:
                logger.warning("Could not launch FFmpeg reader for %s: %s", self.url, e)

        # 4. Standard OpenCV VideoCapture (RTSP, HTTP MJPEG)
        try:
            self.cv_cap = cv2.VideoCapture(self.url)
            self.connected = self.cv_cap.isOpened()
        except Exception as e:
            logger.warning("OpenCV VideoCapture failed for %s: %s", self.url, e)

        if not self.connected:
            standby_cctv = "data/test/cctv_kaggle_master_feed.mp4"
            if not os.path.exists(standby_cctv):
                standby_cctv = "data/test/accident_cctv.webm"
            logger.warning("Stream %s unavailable. Engaging authentic CCTV standby feed: %s", self.url, standby_cctv)
            if os.path.exists(standby_cctv):
                self.cv_cap = cv2.VideoCapture(standby_cctv)
                self.connected = self.cv_cap.isOpened()
            else:
                self.sim_gen = SyntheticTrafficGenerator()
                self.connected = True

    def read_frame(self):
        # A. Explicit Simulation only
        if self.stream_type == "simulation" and self.sim_gen:
            frame, synthetic_dets = self.sim_gen.get_frame()
            return True, frame, synthetic_dets

        # B. FFmpeg Pipe
        if self.ffmpeg_reader:
            success, frame = self.ffmpeg_reader.read_frame()
            if success and frame is not None:
                return True, frame, None
            return False, None, None

        # C. OpenCV VideoCapture
        if self.cv_cap and self.cv_cap.isOpened():
            ret, frame = self.cv_cap.read()
            if ret:
                frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
                return True, frame, None
            else:
                # Loop local video files
                self.cv_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret2, frame2 = self.cv_cap.read()
                if ret2:
                    frame2 = cv2.resize(frame2, (FRAME_WIDTH, FRAME_HEIGHT))
                    return True, frame2, None
                return False, None, None

        return False, None, None

    def trigger_incident(self):
        if self.sim_gen:
            self.sim_gen.trigger_incident()

    def release(self):
        if self.ffmpeg_reader:
            self.ffmpeg_reader.release()
        if self.cv_cap:
            self.cv_cap.release()
        self.connected = False


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

        # Default catalog if empty
        if not self.cameras:
            self.cameras = {
                "cctv_expressway_accident": {
                    "name": "Metropolitan Expressway Collision (CCTV 1080p)",
                    "type": "file",
                    "url": "data/test/accident_cctv.webm",
                    "location": "Metropolitan Expressway Junction KM 14.2",
                    "latitude": 34.0522,
                    "longitude": -118.2437
                },
                "cctv_intersection_collision": {
                    "name": "Urban Intersection Collision (CCTV 1080p)",
                    "type": "file",
                    "url": "data/test/cctv_junction_accident.webm",
                    "location": "Jones Blvd & Copper Crest Intersection",
                    "latitude": 36.1699,
                    "longitude": -115.1398
                },
                "cctv_highway_interchange": {
                    "name": "Jane Byrne Interstate Interchange (CCTV 1080p)",
                    "type": "file",
                    "url": "data/test/cctv_highway_interchange.webm",
                    "location": "Interstate 90/94 Expressway Corridor",
                    "latitude": 41.8756,
                    "longitude": -87.6441
                },
                "caltrans_live_cctv": {
                    "name": "Caltrans Live Traffic Feed (IPTV HLS)",
                    "type": "hls",
                    "url": "https://cph-p2p-msl.akamaized.net/hls/live/200034/test/master.m3u8",
                    "location": "State Route 99 Traffic Corridor",
                    "latitude": 36.7468,
                    "longitude": -119.7726
                }
            }
            self.save_catalog()

        # Set default active camera
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
                    "location": cinfo.get("location", ""),
                    "latitude": cinfo.get("latitude", 0.0),
                    "longitude": cinfo.get("longitude", 0.0),
                    "is_active": (cid == self.active_camera_id)
                })
            return res

    def add_camera(self, cid, name, stream_type, url, location="City", lat=0.0, lon=0.0):
        with self.lock:
            self.cameras[cid] = {
                "name": name,
                "type": stream_type,
                "url": url,
                "location": location,
                "latitude": float(lat),
                "longitude": float(lon)
            }
            self.save_catalog()
            logger.info("New camera added: %s (%s)", name, url)
            return True

    def remove_camera(self, cid):
        with self.lock:
            if cid in self.cameras:
                del self.cameras[cid]
                self.save_catalog()
                if self.active_camera_id == cid:
                    fallback = next(iter(self.cameras), None)
                    if fallback:
                        self.set_active_camera(fallback)
                return True
            return False

    def set_active_camera(self, cid):
        with self.lock:
            if cid not in self.cameras:
                return False

            if self.active_source:
                self.active_source.release()

            cfg = dict(self.cameras[cid])
            cfg["id"] = cid
            self.active_camera_id = cid
            self.active_source = CameraStreamSource(cfg)
            logger.info("Switched active camera to: %s", cfg.get("name"))
            return True

    def get_active_camera_info(self):
        with self.lock:
            if self.active_camera_id and self.active_camera_id in self.cameras:
                info = dict(self.cameras[self.active_camera_id])
                info["id"] = self.active_camera_id
                info["status"] = "ONLINE" if (self.active_source and self.active_source.connected) else "CONNECTING"
                return info
            return {"name": "No Camera", "location": "Unknown", "status": "OFFLINE", "id": "none"}

    def read_frame(self):
        with self.lock:
            if self.active_source:
                return self.active_source.read_frame()
            return False, None, None

    def trigger_incident(self):
        with self.lock:
            if self.active_source:
                self.active_source.trigger_incident()

    def release(self):
        with self.lock:
            if self.active_source:
                self.active_source.release()
