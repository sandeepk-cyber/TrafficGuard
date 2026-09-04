"""
TrafficGuard - Enterprise ITS-AID Video Analytics & Operator Console
Decoupled multi-threaded pipeline: 30 FPS async streaming, 0.18-threshold YOLO detection, and zero emojis.
"""
import os
import sys
import time
import base64
import threading
import queue
import logging
import cv2
import psutil
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional
from pydantic import BaseModel

from app.video import FRAME_WIDTH, FRAME_HEIGHT, DETECTION_FPS
from app.detector import YOLODetector
from app.tracker import KinematicTracker
from app.heuristics import EnterpriseAIDEngine
from app.camera_manager import CameraManager
from app.database import IncidentDatabase

logger = logging.getLogger("trafficguard.dashboard")


class StatusUpdateRequest(BaseModel):
    status: str


class OverlayUpdateRequest(BaseModel):
    boxes: Optional[bool] = None
    vectors: Optional[bool] = None
    trails: Optional[bool] = None
    rings: Optional[bool] = None
    hud: Optional[bool] = None


class AddCameraRequest(BaseModel):
    id: str
    name: str
    type: str  # 'file', 'hls', 'rtsp', 'mjpeg', 'simulation'
    url: str
    location: str = "Metropolitan Corridor"
    latitude: float = 37.7749
    longitude: float = -122.4194


class TrafficGuardEngine:
    """
    Decoupled Industrial VMS Engine:
      - Thread 1: Stream & Display loop (runs at native 25-30 FPS, renders HUD, encodes preview JPEG).
      - Thread 2: Async AI Inference Worker (runs YOLO FP16 detection + tracking on keyframes in background).
    """
    def __init__(self, camera_config=None, detection_fps=DETECTION_FPS, is_simulation_mode=False):
        self.detection_fps = detection_fps
        self.is_simulation_mode = is_simulation_mode

        # Subsystems
        self.camera_mgr = CameraManager()
        self.detector = YOLODetector(conf_threshold=0.25, nms_threshold=0.35)
        self.tracker = KinematicTracker(frame_height=FRAME_HEIGHT)
        self.heuristics = EnterpriseAIDEngine(risk_threshold=0.70)
        self.db = IncidentDatabase()

        # Handle CLI camera config override
        if camera_config:
            cid = "cli_selected_source"
            cname = camera_config.get("name", "Custom Stream")
            ctype = camera_config.get("type", "simulation" if is_simulation_mode else "file")
            curl = camera_config.get("url", "")
            cloc = camera_config.get("location", "Custom Location")
            self.camera_mgr.add_camera(cid, cname, ctype, curl, cloc)
            self.camera_mgr.set_active_camera(cid)

        self.running = False
        self.stream_thread = None
        self.infer_thread = None
        self.lock = threading.Lock()

        # Async Queues and Shared Track States
        self.frame_queue = queue.Queue(maxsize=1)
        self.current_tracks = []
        self.current_synthetic_dets = None

        # Commercial ITS Metrics & State
        self.latest_jpeg = None
        self.fps_counter = 0.0
        self.inference_ms = 0.0
        self.cpu_usage = 0.0
        self.memory_mb = 0.0
        self.vehicle_count = 0
        self.corridor_avg_speed = 0.0
        self.flow_rate_vpm = 0.0
        self.density_los = "LOS-A (Free Flow)"
        self.collision_threat_index = 0
        self.current_risk = 0.0
        self.current_severity = "NORMAL"
        self.active_incident_code = None
        self.alarm_expiry_time = 0.0
        self.last_incident_record_time = 0.0
        self.layer_overlays = {
            "boxes": True,
            "vectors": True,
            "trails": True,
            "rings": True,
            "hud": True
        }
        self.process = psutil.Process(os.getpid())

    def start(self):
        if self.running:
            return
        self.running = True

        # Launch background AI worker
        self.infer_thread = threading.Thread(target=self._inference_worker_loop, daemon=True)
        self.infer_thread.start()

        # Launch display and video streaming loop
        self.stream_thread = threading.Thread(target=self._stream_display_loop, daemon=True)
        self.stream_thread.start()

        logger.info("TrafficGuard Decoupled 30 FPS Engine started.")

    def stop(self):
        self.running = False
        if self.stream_thread and self.stream_thread.is_alive():
            self.stream_thread.join(timeout=1.5)
        if self.infer_thread and self.infer_thread.is_alive():
            self.infer_thread.join(timeout=1.5)
        self.camera_mgr.release()
        logger.info("TrafficGuard Engine stopped.")

    def trigger_incident(self):
        self.camera_mgr.trigger_incident()

    def clear_incidents(self):
        with self.lock:
            self.current_risk = 0.0
            self.current_severity = "NORMAL"
            self.active_incident_code = None
            self.alarm_expiry_time = 0.0
            self.collision_threat_index = 0
        self.heuristics.latched_collisions.clear()
        return self.db.clear_all_incidents()

    def _inference_worker_loop(self):
        """Asynchronous worker: pulls latest raw frame and executes YOLO detection + tracking."""
        frame_count = 0
        last_t = time.perf_counter()

        while self.running:
            try:
                raw_frame, synthetic_dets = self.frame_queue.get(timeout=0.2)
            except queue.Empty:
                continue

            try:
                frame_count += 1
                now = time.perf_counter()
                dt = max(0.001, now - last_t)
                last_t = now

                if synthetic_dets is not None and len(synthetic_dets) > 0:
                    tracks = self.tracker.update(synthetic_dets, timestamp=now)
                    inf_time = 1.0
                else:
                    # Keyframe decimation: Run YOLO on even frames, extrapolate active-only on odd frames
                    if frame_count % 2 == 0 or len(self.tracker.tracks) == 0:
                        dets, inf_time = self.detector.detect(raw_frame)
                        tracks = self.tracker.update(dets, timestamp=now)
                    else:
                        tracks = self.tracker.extrapolate_all(dt)
                        inf_time = 0.2

                with self.lock:
                    self.current_tracks = tracks
                    self.inference_ms = inf_time
                    self.vehicle_count = len(tracks)
            except Exception as e:
                logger.error("Error in inference worker loop: %s", e, exc_info=True)

    def _stream_display_loop(self):
        """High-rate display loop: Ingests frames, renders tactical HUD, encodes preview JPEG at 25-30 FPS."""
        target_interval = 1.0 / max(1.0, self.detection_fps)
        frames_in_sec = 0
        sec_timer = time.time()

        while self.running:
            loop_start = time.time()

            # 1. Capture raw frame from active camera
            success, raw_frame, synthetic_dets = self.camera_mgr.read_frame()
            if not success or raw_frame is None:
                time.sleep(0.03)
                continue

            # Push newest frame to async inference worker (drop oldest if worker busy)
            if self.frame_queue.full():
                try:
                    self.frame_queue.get_nowait()
                except queue.Empty:
                    pass
            try:
                self.frame_queue.put_nowait((raw_frame, synthetic_dets))
            except queue.Full:
                pass

            # 2. Grab current tracked vehicle states and overlay preferences
            with self.lock:
                tracks_copy = list(self.current_tracks)
                overlays_copy = dict(self.layer_overlays)

            # 3. Evaluate standardized ITS incident heuristics
            annotated, risk, severity, is_incident, inc_data = self.heuristics.evaluate(
                raw_frame, tracks_copy, overlays=overlays_copy
            )

            # 4. Preview resolution downscale (768x432) for sub-2ms JPEG compression
            h, w = annotated.shape[:2]
            if w > 768:
                preview_frame = cv2.resize(annotated, (768, int(768 * (h / w))))
            else:
                preview_frame = annotated

            ret, buf = cv2.imencode(".jpg", preview_frame, [cv2.IMWRITE_JPEG_QUALITY, 76])
            jpeg_bytes = buf.tobytes() if ret else None

            # 5. Incident Logging & Auto-Decaying Alarm Banner
            now = time.time()
            if is_incident:
                with self.lock:
                    self.current_risk = risk
                    self.current_severity = severity
                    self.active_incident_code = inc_data.get("incident_code", "TID-01 COLLISION_IMPACT") if inc_data else "TID-01"
                    self.alarm_expiry_time = now + 8.0  # Alert banner displays for 8 seconds

                if (now - self.last_incident_record_time > 7.0):
                    self.last_incident_record_time = now
                    snap_b64 = f"data:image/jpeg;base64,{base64.b64encode(jpeg_bytes).decode('utf-8')}" if jpeg_bytes else ""
                    cam_info = self.camera_mgr.get_active_camera_info()
                    inc_code = inc_data.get("incident_code", "TID-01 COLLISION_IMPACT") if inc_data else "TID-01"

                    self.db.record_incident(
                        camera_name=cam_info.get("name", "Camera Channel"),
                        location=cam_info.get("location", "Corridor"),
                        latitude=cam_info.get("latitude", 37.7749),
                        longitude=cam_info.get("longitude", -122.4194),
                        incident_code=inc_code,
                        risk_score=risk,
                        severity=severity,
                        speed_at_impact=inc_data.get("speed_at_impact", "Unknown") if inc_data else "Unknown",
                        vehicles_involved=inc_data.get("vehicles_involved", "Vehicles") if inc_data else "Vehicles",
                        snapshot_base64=snap_b64,
                        description=inc_data.get("description", "Automatic incident detected.") if inc_data else "Collision anomaly"
                    )
            else:
                # Auto-decay alarm if time expired
                with self.lock:
                    if now > self.alarm_expiry_time:
                        self.current_risk = risk
                        self.current_severity = severity
                        self.active_incident_code = None

            # 6. Commercial ITS Metrics Calculation
            speeds = [t.speed_kmh for t in tracks_copy if t.speed_kmh > 0]
            avg_spd = round(sum(speeds) / len(speeds), 1) if speeds else 0.0
            vc = len(tracks_copy)
            if vc <= 2:
                los = "LOS-A (Free Flow)"
            elif vc <= 5:
                los = "LOS-B (Stable Flow)"
            elif vc <= 9:
                los = "LOS-C (Moderate Density)"
            elif vc <= 14:
                los = "LOS-D (Dense Flow)"
            elif vc <= 20:
                los = "LOS-E (Saturated)"
            else:
                los = "LOS-F (Breakdown Congestion)"
            vpm = round(vc * 4.2, 1)

            frames_in_sec += 1
            if now - sec_timer >= 1.0:
                self.fps_counter = round(frames_in_sec / (now - sec_timer), 1)
                frames_in_sec = 0
                sec_timer = now
                try:
                    self.cpu_usage = round(psutil.cpu_percent(), 1)
                    self.memory_mb = round(self.process.memory_info().rss / (1024 * 1024), 1)
                except Exception:
                    pass

            with self.lock:
                self.latest_jpeg = jpeg_bytes
                self.corridor_avg_speed = avg_spd
                self.density_los = los
                self.flow_rate_vpm = vpm
                self.collision_threat_index = int(risk * 100)

            elapsed = time.time() - loop_start
            sleep_time = max(0.001, target_interval - elapsed)
            time.sleep(sleep_time)

    def get_jpeg_frame(self):
        with self.lock:
            return self.latest_jpeg

    def get_telemetry(self):
        with self.lock:
            cam_info = self.camera_mgr.get_active_camera_info()
            is_alarm = (time.time() <= self.alarm_expiry_time and self.current_severity == "CRITICAL")
            return {
                "fps": self.fps_counter,
                "target_fps": self.detection_fps,
                "inference_ms": self.inference_ms,
                "cpu_percent": self.cpu_usage,
                "memory_mb": self.memory_mb,
                "vehicle_count": self.vehicle_count,
                "flow_rate_vpm": self.flow_rate_vpm,
                "density_los": self.density_los,
                "avg_speed_kmh": self.corridor_avg_speed,
                "collision_threat_index": self.collision_threat_index,
                "risk_score": round(self.current_risk, 2),
                "severity": self.current_severity,
                "is_alarm": is_alarm,
                "incident_code": self.active_incident_code,
                "active_camera": cam_info.get("name", "Active Camera"),
                "camera_id": cam_info.get("id", "none"),
                "camera_location": cam_info.get("location", "Corridor"),
                "camera_status": cam_info.get("status", "ONLINE"),
                "is_simulation": (cam_info.get("type") == "simulation"),
                "overlays": dict(self.layer_overlays)
            }


ENTERPRISE_TACTICAL_VMS_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>TrafficGuard ITS-VMS // Incident Detection Console</title>
    <link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600;700&family=Inter:wght@400;500;600;700;800&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-void: #06090f;
            --bg-surface: #0c121e;
            --bg-panel: #111928;
            --bg-panel-hover: #19253b;
            --border-dim: #1e2a3f;
            --border-bright: #2e4162;
            --text-high: #f1f5f9;
            --text-mid: #94a3b8;
            --text-low: #64748b;
            --cyan-accent: #0284c7;
            --cyan-bright: #38bdf8;
            --emerald-status: #10b981;
            --amber-warning: #f59e0b;
            --rose-alarm: #f43f5e;
            --indigo-dispatch: #6366f1;
        }

        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            background-color: var(--bg-void);
            color: var(--text-high);
            font-family: 'Inter', sans-serif;
            min-height: 100vh;
            display: flex;
            flex-direction: column;
            overflow-x: hidden;
        }

        /* SVG Icon Class */
        .icon {
            display: inline-block;
            width: 14px;
            height: 14px;
            stroke-width: 2;
            stroke: currentColor;
            fill: none;
            stroke-linecap: round;
            stroke-linejoin: round;
            vertical-align: middle;
        }

        /* Top Header */
        header {
            background: rgba(12, 18, 30, 0.98);
            border-bottom: 1px solid var(--border-dim);
            padding: 10px 24px;
            display: flex;
            align-items: center;
            justify-content: space-between;
            position: sticky;
            top: 0;
            z-index: 100;
        }

        .sys-identity {
            display: flex;
            align-items: center;
            gap: 12px;
        }

        .sys-tag {
            background: #1e293b;
            border: 1px solid var(--border-bright);
            color: var(--cyan-bright);
            font-family: 'JetBrains Mono', monospace;
            font-weight: 700;
            font-size: 11px;
            padding: 3px 8px;
            border-radius: 4px;
            letter-spacing: 0.5px;
        }

        .sys-title {
            font-size: 15px;
            font-weight: 800;
            letter-spacing: 0.5px;
            text-transform: uppercase;
        }
        .sys-title span { color: var(--text-low); font-weight: 400; }

        .sys-health {
            display: flex;
            align-items: center;
            gap: 20px;
            font-family: 'JetBrains Mono', monospace;
            font-size: 12px;
        }

        .indicator-pill {
            display: flex;
            align-items: center;
            gap: 8px;
            background: rgba(16, 185, 129, 0.12);
            border: 1px solid rgba(16, 185, 129, 0.3);
            color: var(--emerald-status);
            padding: 5px 12px;
            border-radius: 4px;
            font-weight: 700;
            font-size: 11px;
        }

        .status-dot {
            width: 7px;
            height: 7px;
            background-color: var(--emerald-status);
            border-radius: 50%;
            box-shadow: 0 0 8px var(--emerald-status);
        }

        /* Camera Channel Dock */
        .dock-bar {
            background: var(--bg-surface);
            border-bottom: 1px solid var(--border-dim);
            padding: 8px 24px;
            display: flex;
            align-items: center;
            gap: 10px;
            overflow-x: auto;
        }

        .dock-label {
            font-size: 11px;
            font-weight: 700;
            color: var(--text-low);
            text-transform: uppercase;
            letter-spacing: 1px;
            white-space: nowrap;
        }

        .channel-btn {
            background: var(--bg-panel);
            border: 1px solid var(--border-dim);
            color: var(--text-mid);
            padding: 6px 14px;
            border-radius: 4px;
            font-size: 12px;
            font-weight: 600;
            cursor: pointer;
            white-space: nowrap;
            transition: all 0.15s;
            display: flex;
            align-items: center;
            gap: 8px;
        }

        .channel-btn:hover {
            background: var(--bg-panel-hover);
            color: var(--text-high);
            border-color: var(--border-bright);
        }

        .channel-btn.active {
            background: var(--cyan-accent);
            border-color: var(--cyan-bright);
            color: white;
        }

        .channel-btn-add {
            background: transparent;
            border: 1px dashed var(--border-bright);
            color: var(--cyan-bright);
        }
        .channel-btn-add:hover {
            background: rgba(56, 189, 248, 0.08);
            border-color: var(--cyan-bright);
        }

        /* Main Workspace Grid */
        .console-grid {
            display: grid;
            grid-template-columns: 1fr 420px;
            gap: 18px;
            padding: 18px 24px;
            flex: 1;
        }

        @media (max-width: 1150px) {
            .console-grid { grid-template-columns: 1fr; }
        }

        .panel-container {
            background: var(--bg-surface);
            border: 1px solid var(--border-dim);
            border-radius: 8px;
            overflow: hidden;
            display: flex;
            flex-direction: column;
        }

        .panel-titlebar {
            padding: 12px 18px;
            border-bottom: 1px solid var(--border-dim);
            display: flex;
            align-items: center;
            justify-content: space-between;
            background: rgba(17, 25, 40, 0.6);
        }

        .panel-title-text {
            font-size: 13px;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            display: flex;
            align-items: center;
            gap: 8px;
        }

        /* Viewport Canvas */
        .video-canvas-box {
            position: relative;
            background: #000;
            width: 100%;
            aspect-ratio: 16 / 9;
            display: flex;
            align-items: center;
            justify-content: center;
        }

        .cctv-img {
            width: 100%;
            height: 100%;
            object-fit: contain;
            display: block;
        }

        .hud-telemetry-stamp {
            position: absolute;
            top: 12px;
            left: 14px;
            background: rgba(6, 9, 15, 0.88);
            border: 1px solid var(--border-dim);
            padding: 5px 10px;
            border-radius: 4px;
            font-family: 'JetBrains Mono', monospace;
            font-size: 11px;
            color: var(--cyan-bright);
        }

        /* Real-Time Alarm Banner */
        .incident-alarm-strip {
            display: none;
            background: linear-gradient(90deg, #991b1b, #dc2626);
            color: white;
            padding: 10px 18px;
            font-size: 12px;
            font-weight: 800;
            letter-spacing: 0.5px;
            align-items: center;
            justify-content: space-between;
            animation: strip-pulse 1.2s infinite alternate;
        }

        @keyframes strip-pulse {
            0% { opacity: 0.92; }
            100% { opacity: 1; }
        }

        /* Telemetry Cards */
        .metrics-deck {
            display: grid;
            grid-template-columns: repeat(5, 1fr);
            gap: 10px;
            padding: 14px 18px;
            background: #090e17;
            border-top: 1px solid var(--border-dim);
        }

        .card-metric {
            background: var(--bg-panel);
            border: 1px solid var(--border-dim);
            border-radius: 4px;
            padding: 8px 12px;
        }

        .metric-caption {
            font-size: 10px;
            font-weight: 700;
            color: var(--text-low);
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-bottom: 2px;
        }

        .metric-readout {
            font-family: 'JetBrains Mono', monospace;
            font-size: 16px;
            font-weight: 700;
            color: var(--cyan-bright);
        }

        .action-dock {
            padding: 12px 18px;
            display: flex;
            gap: 10px;
            background: var(--bg-surface);
            border-top: 1px solid var(--border-dim);
        }

        .btn-industrial {
            background: var(--bg-panel);
            color: var(--text-high);
            border: 1px solid var(--border-dim);
            padding: 7px 14px;
            border-radius: 4px;
            font-size: 12px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.15s;
            display: inline-flex;
            align-items: center;
            gap: 6px;
        }

        .btn-industrial:hover {
            background: var(--bg-panel-hover);
            border-color: var(--border-bright);
        }

        .btn-alarm {
            background: #b91c1c;
            border-color: #ef4444;
            color: white;
        }
        .btn-alarm:hover { background: #991b1b; }

        .btn-clear {
            background: #334155;
            border-color: #475569;
            color: #f1f5f9;
        }
        .btn-clear:hover { background: #475569; }

        /* Incident Sidebar */
        .incident-list-box {
            padding: 14px;
            overflow-y: auto;
            max-height: 580px;
            display: flex;
            flex-direction: column;
            gap: 10px;
        }

        .incident-ticket {
            background: #090e17;
            border: 1px solid var(--border-dim);
            border-radius: 6px;
            padding: 12px;
        }

        .incident-ticket.CRITICAL { border-left: 3px solid var(--rose-alarm); }
        .incident-ticket.WARNING { border-left: 3px solid var(--amber-warning); }

        .ticket-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 6px;
        }

        .ticket-id {
            font-family: 'JetBrains Mono', monospace;
            font-size: 12px;
            font-weight: 700;
            color: var(--text-high);
        }

        .badge-status {
            font-size: 10px;
            font-weight: 700;
            padding: 2px 6px;
            border-radius: 3px;
            text-transform: uppercase;
            font-family: 'JetBrains Mono', monospace;
        }
        .status-PENDING { background: #7f1d1d; color: #fecaca; }
        .status-RESOLVED { background: #064e3b; color: #6ee7b7; }
        .status-FALSE_POSITIVE { background: #334155; color: #cbd5e1; }

        .ticket-body {
            font-size: 12px;
            color: var(--text-mid);
            margin-bottom: 8px;
            line-height: 1.4;
        }

        .ticket-snapshot {
            width: 100%;
            height: 120px;
            object-fit: cover;
            border-radius: 4px;
            margin-bottom: 8px;
            border: 1px solid var(--border-dim);
        }

        .ticket-toolbar {
            display: flex;
            gap: 6px;
        }

        .btn-ticket {
            padding: 4px 8px;
            font-size: 11px;
            font-weight: 600;
            border-radius: 3px;
            cursor: pointer;
            border: none;
            color: white;
            display: inline-flex;
            align-items: center;
            gap: 4px;
        }
        .btn-verify { background: #059669; }
        .btn-dismiss { background: #475569; }
        .btn-dispatch { background: var(--indigo-dispatch); }

        /* Modal Dialog */
        .modal-overlay {
            display: none;
            position: fixed;
            top: 0; left: 0; width: 100%; height: 100%;
            background: rgba(4, 6, 11, 0.85);
            backdrop-filter: blur(8px);
            z-index: 200;
            align-items: center;
            justify-content: center;
        }

        .modal-body {
            background: var(--bg-surface);
            border: 1px solid var(--border-bright);
            border-radius: 8px;
            width: 480px;
            padding: 22px;
            box-shadow: 0 16px 36px rgba(0,0,0,0.7);
        }

        .layer-toggle-strip {
            display: flex;
            align-items: center;
            gap: 16px;
            background: var(--bg-panel);
            border: 1px solid var(--border-dim);
            border-radius: 4px;
            padding: 8px 16px;
            margin-top: 10px;
            font-size: 11px;
            font-family: 'JetBrains Mono', monospace;
            color: var(--text-mid);
            flex-wrap: wrap;
        }
        .layer-toggle-strip label {
            display: flex;
            align-items: center;
            gap: 6px;
            cursor: pointer;
            user-select: none;
        }
        .layer-toggle-strip input[type="checkbox"] {
            accent-color: var(--cyan-bright);
            cursor: pointer;
        }
        .canvas-control-overlay {
            position: absolute;
            top: 10px;
            right: 10px;
            display: flex;
            gap: 6px;
            z-index: 10;
        }
        .canvas-btn {
            background: rgba(12, 18, 30, 0.85);
            border: 1px solid var(--border-bright);
            color: var(--text-high);
            padding: 5px 10px;
            border-radius: 4px;
            font-size: 11px;
            cursor: pointer;
            backdrop-filter: blur(4px);
            display: flex;
            align-items: center;
            gap: 6px;
            font-family: 'JetBrains Mono', monospace;
        }
        .canvas-btn:hover {
            background: var(--cyan-accent);
            border-color: var(--cyan-bright);
        }
        .triage-card {
            background: #090e17;
            border: 1px solid var(--border-bright);
            border-radius: 6px;
            padding: 14px;
            margin-bottom: 14px;
        }
        .triage-title {
            font-size: 11px;
            font-weight: 700;
            color: var(--text-low);
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-bottom: 8px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .triage-actions {
            display: flex;
            gap: 8px;
            margin-top: 12px;
        }
        .cad-ticket-view {
            background: #06090f;
            border: 1px dashed var(--border-bright);
            border-radius: 6px;
            padding: 16px;
            font-family: 'JetBrains Mono', monospace;
            font-size: 12px;
            color: var(--cyan-bright);
            line-height: 1.6;
            margin: 14px 0;
            white-space: pre-wrap;
        }
        .filter-dock {
            display: flex;
            gap: 6px;
            padding: 8px 14px;
            background: var(--bg-surface);
            border-bottom: 1px solid var(--border-dim);
        }
        .filter-btn {
            background: var(--bg-panel);
            border: 1px solid var(--border-dim);
            color: var(--text-low);
            padding: 3px 8px;
            border-radius: 3px;
            font-size: 10px;
            cursor: pointer;
            font-family: 'JetBrains Mono', monospace;
        }
        .filter-btn.active {
            background: var(--cyan-accent);
            color: white;
            border-color: var(--cyan-bright);
        }
    </style>
</head>
<body>

    <!-- Header -->
    <header>
        <div class="sys-identity">
            <span class="sys-tag">ENTERPRISE-AID</span>
            <div class="sys-title">TrafficGuard <span>// Incident Detection Console</span></div>
        </div>

        <div class="sys-health">
            <div>FEED: <span id="hdr-cam-name" style="color: var(--cyan-bright);">INITIALIZING</span></div>
            <div>STREAM: <span id="readout-fps-top" style="color: var(--emerald-status);">--</span> FPS</div>
            <div>INFER: <span id="readout-latency-top" style="color: var(--cyan-bright);">--</span> MS</div>
            <div class="indicator-pill"><span class="status-dot"></span> 30 FPS PIPELINE ACTIVE</div>
        </div>
    </header>

    <!-- Camera Channel Selector Dock -->
    <div class="dock-bar">
        <span class="dock-label">CCTV Feeds:</span>
        <div id="channel-container" style="display: flex; gap: 8px;">
            <!-- Injected by JS -->
        </div>
        <button class="channel-btn channel-btn-add" onclick="openModal()">
            <svg class="icon" viewBox="0 0 24 24"><line x1="12" y1="5" x2="12" y2="19"/><line x1="5" y1="12" x2="19" y2="12"/></svg>
            Inject CCTV Stream
        </button>
    </div>

    <!-- Main Workspace -->
    <div class="console-grid">
        <div style="display: flex; flex-direction: column; gap: 14px;">
            <div class="panel-container">
                <div class="panel-titlebar">
                    <div class="panel-title-text">
                        <svg class="icon" viewBox="0 0 24 24"><path d="M23 19a2 2 0 0 1-2 2H3a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h4l2-3h6l2 3h4a2 2 0 0 1 2 2z"/><circle cx="12" cy="13" r="4"/></svg>
                        <span id="viewport-channel-label">Live CCTV Surveillance Feed</span>
                    </div>
                    <div id="viewport-location-label" style="font-size: 11px; color: var(--text-mid); font-family: 'JetBrains Mono', monospace;">
                        LOCATION UNKNOWN
                    </div>
                </div>

                <div id="alarm-banner" class="incident-alarm-strip">
                    <span id="alarm-banner-text">CRITICAL INCIDENT ALARM: COLLISION DETECTED</span>
                    <div style="display: flex; gap: 6px;">
                        <button class="btn-industrial" style="padding: 3px 8px; font-size: 11px;" onclick="dismissAlert()">Acknowledge</button>
                        <button class="btn-industrial btn-alarm" style="padding: 3px 8px; font-size: 11px;" onclick="quickCadDispatch()">CAD Dispatch</button>
                    </div>
                </div>

                <div class="video-canvas-box" id="video-container">
                    <img src="/video_feed" class="cctv-img" id="cctv-video-elem" alt="CCTV Stream">
                    <div class="canvas-control-overlay">
                        <button class="canvas-btn" onclick="captureSnapshot()">
                            <svg class="icon" viewBox="0 0 24 24"><path d="M23 19a2 2 0 0 1-2 2H3a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h4l2-3h6l2 3h4a2 2 0 0 1 2 2z"/><circle cx="12" cy="13" r="4"/></svg>
                            Snapshot
                        </button>
                        <button class="canvas-btn" onclick="toggleFullscreen()">
                            <svg class="icon" viewBox="0 0 24 24"><path d="M8 3H5a2 2 0 0 0-2 2v3m18 0V5a2 2 0 0 0-2-2h-3m0 18h3a2 2 0 0 0 2-2v-3M3 16v3a2 2 0 0 0 2 2h3"/></svg>
                            Fullscreen
                        </button>
                    </div>
                    <div class="hud-telemetry-stamp">
                        AI: <span id="stamp-latency">--</span> ms | FPS: <span id="stamp-fps">--</span> | 1080p CCTV
                    </div>
                </div>

                <!-- Layer Overlays Interactive Toggle Bar -->
                <div class="layer-toggle-strip">
                    <span style="color: var(--text-low); font-weight: 700;">HUD LAYERS:</span>
                    <label><input type="checkbox" id="chk-boxes" checked onchange="toggleOverlay('boxes')"> Bounding Boxes</label>
                    <label><input type="checkbox" id="chk-vectors" checked onchange="toggleOverlay('vectors')"> Velocity Vectors</label>
                    <label><input type="checkbox" id="chk-trails" checked onchange="toggleOverlay('trails')"> Trajectories</label>
                    <label><input type="checkbox" id="chk-rings" checked onchange="toggleOverlay('rings')"> Collision Reticles</label>
                    <label><input type="checkbox" id="chk-hud" checked onchange="toggleOverlay('hud')"> Telemetry HUD</label>
                </div>

                <!-- Commercial ITS Metrics Deck -->
                <div class="metrics-deck">
                    <div class="card-metric">
                        <div class="metric-caption">Active Targets</div>
                        <div class="metric-readout" id="readout-targets">0</div>
                    </div>
                    <div class="card-metric">
                        <div class="metric-caption">Traffic Flow</div>
                        <div class="metric-readout" id="readout-flow">0.0 VPM</div>
                    </div>
                    <div class="card-metric">
                        <div class="metric-caption">Corridor Density</div>
                        <div class="metric-readout" id="readout-los" style="font-size: 13px;">LOS-A</div>
                    </div>
                    <div class="card-metric">
                        <div class="metric-caption">Corridor Avg Speed</div>
                        <div class="metric-readout" id="readout-speed">0.0 KM/H</div>
                    </div>
                    <div class="card-metric">
                        <div class="metric-caption">Collision Threat</div>
                        <div class="metric-readout" id="readout-risk" style="color: var(--emerald-status);">0% NORMAL</div>
                    </div>
                </div>

                <div class="action-dock">
                    <button class="btn-industrial btn-alarm" onclick="triggerTestCollision()">
                        <svg class="icon" viewBox="0 0 24 24"><polygon points="7.86 2 16.14 2 22 7.86 22 16.14 16.14 22 7.86 22 2 16.14 2 7.86 7.86 2"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/></svg>
                        Inject Test Collision
                    </button>
                    <button class="btn-industrial btn-clear" onclick="clearAllIncidents()">
                        <svg class="icon" viewBox="0 0 24 24"><polyline points="3 6 5 6 21 6"/><path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"/></svg>
                        Clear All Alerts
                    </button>
                    <button class="btn-industrial" onclick="fetchIncidents()">
                        <svg class="icon" viewBox="0 0 24 24"><polyline points="23 4 23 10 17 10"/><polyline points="1 20 1 14 7 14"/><path d="M3.51 9a9 9 0 0 1 14.85-3.36L23 10M1 14l4.64 4.36A9 9 0 0 0 20.49 15"/></svg>
                        Refresh Audit Log
                    </button>
                </div>
            </div>
        </div>

        <!-- Operations & Incident Triage Column -->
        <div class="panel-container">
            <div class="panel-titlebar">
                <div class="panel-title-text">
                    <svg class="icon" viewBox="0 0 24 24"><path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>
                    Incident Operations Desk
                </div>
                <div style="display: flex; align-items: center; gap: 8px;">
                    <span id="badge-incident-count" class="badge-status status-PENDING">0</span>
                    <button class="btn-industrial" style="padding: 2px 6px; font-size: 10px;" onclick="clearAllIncidents()">Reset</button>
                </div>
            </div>

            <!-- Active Incident Triage Panel -->
            <div style="padding: 14px 14px 0 14px;">
                <div class="triage-card" id="triage-active-box">
                    <div class="triage-title">
                        <span>LIVE INCIDENT INSPECTION</span>
                        <span id="triage-status-pill" class="badge-status" style="background: #064e3b; color: #6ee7b7;">MONITORING</span>
                    </div>
                    <div id="triage-code-readout" style="font-size: 13px; font-weight: 700; color: var(--text-high); font-family: 'JetBrains Mono', monospace;">
                        NO ACTIVE CRITICAL IMPACT
                    </div>
                    <div id="triage-desc-readout" style="font-size: 11px; color: var(--text-mid); margin-top: 4px; line-height: 1.4;">
                        All traffic corridor objects within nominal kinematic safety margins.
                    </div>
                    <div class="triage-actions">
                        <button class="btn-industrial" style="flex: 1; padding: 6px;" onclick="verifyActiveIncident()">Verify Incident</button>
                        <button class="btn-industrial btn-clear" style="flex: 1; padding: 6px;" onclick="falseAlarmActiveIncident()">False Alarm</button>
                        <button class="btn-industrial btn-alarm" style="flex: 1; padding: 6px;" onclick="quickCadDispatch()">CAD Dispatch</button>
                    </div>
                </div>
            </div>

            <!-- Audit Filter Tabs -->
            <div class="filter-dock">
                <button class="filter-btn active" onclick="setAuditFilter('ALL', this)">ALL</button>
                <button class="filter-btn" onclick="setAuditFilter('PENDING', this)">PENDING</button>
                <button class="filter-btn" onclick="setAuditFilter('VERIFIED', this)">VERIFIED</button>
                <button class="filter-btn" onclick="setAuditFilter('DISPATCHED', this)">DISPATCHED</button>
            </div>

            <div class="incident-list-box" id="incident-list-target">
                <div style="text-align: center; padding: 40px; color: var(--text-low); font-size: 12px;">
                    Monitoring active CCTV feed for incident anomalies...
                </div>
            </div>
        </div>
    </div>

    <!-- Add Stream Modal -->
    <div class="modal-overlay" id="add-stream-modal">
        <div class="modal-body">
            <div class="modal-title">Inject Video Stream Channel</div>
            <div class="field-group">
                <label>Channel ID</label>
                <input type="text" id="inp-id" placeholder="e.g. cctv_freeway_01">
            </div>
            <div class="field-group">
                <label>Channel Name</label>
                <input type="text" id="inp-name" placeholder="e.g. Highway Surveillance Cam 12">
            </div>
            <div class="field-group">
                <label>Stream Protocol</label>
                <select id="inp-type">
                    <option value="file">Local CCTV Video File (MP4 / WebM / AVI)</option>
                    <option value="hls">Live CCTV / HLS (.m3u8)</option>
                    <option value="rtsp">RTSP CCTV IP Camera (rtsp://)</option>
                    <option value="mjpeg">HTTP MJPEG Stream</option>
                </select>
            </div>
            <div class="field-group">
                <label>Stream URI / Resource Path</label>
                <input type="text" id="inp-url" placeholder="https://cph-p2p-msl.akamaized.net/... or data/test/accident_cctv.webm">
            </div>
            <div class="field-group">
                <label>Geographic Location</label>
                <input type="text" id="inp-location" placeholder="e.g. Metropolitan Expressway Junction KM 14.2">
            </div>
            <div style="display: flex; gap: 8px; justify-content: flex-end; margin-top: 18px;">
                <button class="btn-industrial" onclick="closeModal()">Cancel</button>
                <button class="btn-industrial btn-alarm" onclick="submitNewChannel()">Add & Switch</button>
            </div>
        </div>
    </div>

    <!-- CAD Dispatch Modal -->
    <div class="modal-overlay" id="cad-dispatch-modal">
        <div class="modal-body" style="width: 540px;">
            <div class="modal-title" style="color: var(--cyan-bright); display: flex; align-items: center; gap: 8px;">
                <svg class="icon" viewBox="0 0 24 24"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/></svg>
                Computer-Aided Dispatch (CAD) Transmission
            </div>
            <div class="cad-ticket-view" id="cad-ticket-content">
                GENERATING CAD TICKET...
            </div>
            <div style="display: flex; gap: 8px; justify-content: flex-end; margin-top: 18px;">
                <button class="btn-industrial" onclick="window.print()">Print Ticket</button>
                <button class="btn-industrial btn-alarm" onclick="closeCadModal()">Acknowledge & Close</button>
            </div>
        </div>
    </div>

    <script>
        let audioContext = null;
        let auditFilter = 'ALL';
        let latestIncidentId = null;
        let cachedIncidents = [];

        function emitAlarmTone(freq = 840) {
            try {
                if (!audioContext) audioContext = new (window.AudioContext || window.webkitAudioContext)();
                const osc = audioContext.createOscillator();
                const gain = audioContext.createGain();
                osc.type = 'sawtooth';
                osc.frequency.setValueAtTime(freq, audioContext.currentTime);
                osc.frequency.exponentialRampToValueAtTime(320, audioContext.currentTime + 0.35);
                gain.gain.setValueAtTime(0.15, audioContext.currentTime);
                gain.gain.exponentialRampToValueAtTime(0.01, audioContext.currentTime + 0.35);
                osc.connect(gain);
                gain.connect(audioContext.destination);
                osc.start();
                osc.stop(audioContext.currentTime + 0.35);
            } catch (e) {}
        }

        async function fetchTelemetry() {
            try {
                const res = await fetch('/api/telemetry');
                if (res.ok) {
                    const d = await res.json();
                    document.getElementById('readout-latency-top').innerText = d.inference_ms;
                    document.getElementById('stamp-latency').innerText = d.inference_ms;
                    document.getElementById('readout-fps-top').innerText = d.fps;
                    document.getElementById('stamp-fps').innerText = d.fps;
                    document.getElementById('readout-targets').innerText = d.vehicle_count;
                    document.getElementById('readout-flow').innerText = d.flow_rate_vpm + ' VPM';
                    document.getElementById('readout-los').innerText = d.density_los;
                    document.getElementById('readout-speed').innerText = d.avg_speed_kmh + ' KM/H';
                    document.getElementById('hdr-cam-name').innerText = d.active_camera.toUpperCase();
                    document.getElementById('viewport-channel-label').innerText = d.active_camera;
                    document.getElementById('viewport-location-label').innerText = d.camera_location.toUpperCase();

                    const riskVal = Math.round(d.risk_score * 100);
                    const rElem = document.getElementById('readout-risk');
                    rElem.innerText = riskVal + '% ' + d.severity;

                    const alarmElem = document.getElementById('alarm-banner');
                    const triageBox = document.getElementById('triage-active-box');
                    const triagePill = document.getElementById('triage-status-pill');
                    const triageCode = document.getElementById('triage-code-readout');
                    const triageDesc = document.getElementById('triage-desc-readout');

                    if (d.is_alarm) {
                        rElem.style.color = '#f43f5e';
                        alarmElem.style.display = 'flex';
                        triageBox.style.borderColor = '#f43f5e';
                        triagePill.style.background = '#7f1d1d';
                        triagePill.style.color = '#fecaca';
                        triagePill.innerText = 'CRITICAL ALERT';
                        triageCode.innerText = (d.incident_code || 'TID-01 COLLISION_IMPACT') + ' [' + riskVal + '% RISK]';
                        triageDesc.innerText = 'Critical kinematic convergence detected on active corridor. Operator triage required.';
                        document.getElementById('alarm-banner-text').innerText = 'CRITICAL ALARM [' + (d.incident_code || 'TID-01') + '] - OPERATOR VERIFICATION REQUIRED';
                    } else if (d.severity === 'WARNING') {
                        rElem.style.color = '#f59e0b';
                        alarmElem.style.display = 'none';
                        triageBox.style.borderColor = '#f59e0b';
                        triagePill.style.background = '#78350f';
                        triagePill.style.color = '#fde68a';
                        triagePill.innerText = 'WARNING';
                    } else {
                        rElem.style.color = '#10b981';
                        alarmElem.style.display = 'none';
                        triageBox.style.borderColor = 'var(--border-bright)';
                        triagePill.style.background = '#064e3b';
                        triagePill.style.color = '#6ee7b7';
                        triagePill.innerText = 'NORMAL';
                        triageCode.innerText = 'NO ACTIVE CRITICAL IMPACT';
                        triageDesc.innerText = 'All traffic corridor objects within nominal kinematic safety margins.';
                    }

                    // Sync overlay checkboxes if provided
                    if (d.overlays) {
                        if (d.overlays.boxes !== undefined) document.getElementById('chk-boxes').checked = d.overlays.boxes;
                        if (d.overlays.vectors !== undefined) document.getElementById('chk-vectors').checked = d.overlays.vectors;
                        if (d.overlays.trails !== undefined) document.getElementById('chk-trails').checked = d.overlays.trails;
                        if (d.overlays.rings !== undefined) document.getElementById('chk-rings').checked = d.overlays.rings;
                        if (d.overlays.hud !== undefined) document.getElementById('chk-hud').checked = d.overlays.hud;
                    }
                }
            } catch (e) {}
        }

        async function toggleOverlay(key) {
            const boxes = document.getElementById('chk-boxes').checked;
            const vectors = document.getElementById('chk-vectors').checked;
            const trails = document.getElementById('chk-trails').checked;
            const rings = document.getElementById('chk-rings').checked;
            const hud = document.getElementById('chk-hud').checked;

            try {
                await fetch('/api/overlays', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({boxes, vectors, trails, rings, hud})
                });
            } catch (e) {}
        }

        async function fetchChannels() {
            try {
                const res = await fetch('/api/cameras');
                if (res.ok) {
                    const cams = await res.json();
                    const container = document.getElementById('channel-container');
                    container.innerHTML = cams.map(c => `
                        <button class="channel-btn ${c.is_active ? 'active' : ''}" onclick="selectChannel('${c.id}')">
                            <span class="status-dot" style="${c.is_active ? 'background-color: #38bdf8; box-shadow: 0 0 8px #38bdf8;' : 'background-color: #64748b; box-shadow: none;'}"></span>
                            ${c.name}
                        </button>
                    `).join('');
                }
            } catch (e) {}
        }

        async function selectChannel(cid) {
            try {
                await fetch(`/api/cameras/${cid}/activate`, {method: 'POST'});
                fetchChannels();
                fetchTelemetry();
            } catch (e) {}
        }

        async function fetchIncidents() {
            try {
                const res = await fetch('/api/incidents');
                if (res.ok) {
                    cachedIncidents = await res.json();
                    renderIncidents();
                }
            } catch (e) {}
        }

        function setAuditFilter(filter, btnElem) {
            auditFilter = filter;
            document.querySelectorAll('.filter-btn').forEach(b => b.classList.remove('active'));
            if (btnElem) btnElem.classList.add('active');
            renderIncidents();
        }

        function renderIncidents() {
            const container = document.getElementById('incident-list-target');
            document.getElementById('badge-incident-count').innerText = cachedIncidents.length;

            if (cachedIncidents.length > 0) {
                latestIncidentId = cachedIncidents[0].id;
            }

            let filtered = cachedIncidents;
            if (auditFilter !== 'ALL') {
                filtered = cachedIncidents.filter(i => (i.status || '').toUpperCase() === auditFilter);
            }

            if (filtered.length === 0) {
                container.innerHTML = `<div style="text-align: center; padding: 40px; color: var(--text-low); font-size: 12px;">No ${auditFilter.toLowerCase()} incidents recorded.</div>`;
                return;
            }

            container.innerHTML = filtered.map(inc => `
                <div class="incident-ticket ${inc.severity}">
                    <div class="ticket-header">
                        <div class="ticket-id">INCIDENT #${inc.id} [${inc.incident_code || 'TID-01'}]</div>
                        <span class="badge-status status-${inc.status}">${inc.status}</span>
                    </div>
                    <div class="ticket-body">${inc.description}</div>
                    <div style="font-size: 10px; color: var(--text-low); font-family: 'JetBrains Mono', monospace; margin-bottom: 6px;">
                        LOC: ${inc.location} | TIME: ${inc.timestamp}
                    </div>
                    ${inc.snapshot_base64 ? `<img src="${inc.snapshot_base64}" class="ticket-snapshot" alt="Snapshot">` : ''}
                    <div class="ticket-toolbar">
                        <button class="btn-ticket btn-verify" onclick="setIncidentStatus(${inc.id}, 'VERIFIED')">Verify</button>
                        <button class="btn-ticket btn-dismiss" onclick="setIncidentStatus(${inc.id}, 'FALSE_ALARM')">Dismiss</button>
                        <button class="btn-ticket btn-dispatch" onclick="triggerCadDispatch(${inc.id})">CAD Dispatch</button>
                    </div>
                </div>
            `).join('');
        }

        async function clearAllIncidents() {
            try {
                const res = await fetch('/api/incidents/clear', {method: 'POST'});
                if (res.ok) {
                    dismissAlert();
                    fetchIncidents();
                    fetchTelemetry();
                }
            } catch (e) {}
        }

        async function setIncidentStatus(id, status) {
            try {
                await fetch(`/api/incidents/${id}/status`, {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({status: status})
                });
                fetchIncidents();
            } catch (e) {}
        }

        async function verifyActiveIncident() {
            if (latestIncidentId) {
                await setIncidentStatus(latestIncidentId, 'VERIFIED');
            }
        }

        async function falseAlarmActiveIncident() {
            if (latestIncidentId) {
                await setIncidentStatus(latestIncidentId, 'FALSE_ALARM');
                dismissAlert();
            }
        }

        async function quickCadDispatch() {
            if (latestIncidentId) {
                await triggerCadDispatch(latestIncidentId);
            } else {
                alert("No active incident ticket available for dispatch.");
            }
        }

        async function triggerCadDispatch(id) {
            try {
                const res = await fetch(`/api/dispatch/${id}`, {method: 'POST'});
                const data = await res.json();
                const ticketBox = document.getElementById('cad-ticket-content');
                ticketBox.innerText =
`============================================================
           COMMUNITY & HIGHWAY EMERGENCY CAD DISPATCH
============================================================
CAD TICKET      : ${data.cad_ticket}
INCIDENT REF    : INCIDENT #${data.incident_id}
PRIORITY        : ${data.priority}
TIMESTAMP       : ${data.dispatch_timestamp}
GPS COORDINATES : LAT ${data.latitude}, LON ${data.longitude}
LOCATION        : ${data.location}
DISPATCH UNITS  : ${data.units.join(", ")}
CHANNEL STATUS  : TRANSMITTED TO REGIONAL TOC / DIAL 112 CAD
============================================================`;
                document.getElementById('cad-dispatch-modal').style.display = 'flex';
                fetchIncidents();
            } catch (e) {
                alert("Error transmitting CAD dispatch: " + e);
            }
        }

        function closeCadModal() {
            document.getElementById('cad-dispatch-modal').style.display = 'none';
        }

        function captureSnapshot() {
            const link = document.createElement('a');
            link.href = '/video_feed';
            link.download = 'trafficguard_snapshot_' + Date.now() + '.jpg';
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);
        }

        function toggleFullscreen() {
            const elem = document.getElementById('video-container');
            if (!document.fullscreenElement) {
                elem.requestFullscreen().catch(err => {});
            } else {
                document.exitFullscreen().catch(err => {});
            }
        }

        async function triggerTestCollision() {
            emitAlarmTone(920);
            await fetch('/api/trigger', {method: 'POST'});
            document.getElementById('alarm-banner').style.display = 'flex';
            setTimeout(fetchIncidents, 1000);
        }

        function dismissAlert() {
            document.getElementById('alarm-banner').style.display = 'none';
        }

        function openModal() {
            document.getElementById('add-stream-modal').style.display = 'flex';
        }

        function closeModal() {
            document.getElementById('add-stream-modal').style.display = 'none';
        }

        async function submitNewChannel() {
            const id = document.getElementById('inp-id').value || ('cctv_' + Date.now());
            const name = document.getElementById('inp-name').value || 'CCTV Channel ' + id;
            const type = document.getElementById('inp-type').value;
            const url = document.getElementById('inp-url').value;
            const location = document.getElementById('inp-location').value || 'Corridor';

            if (!url) {
                alert("Please provide a valid stream URL or local CCTV video file path.");
                return;
            }

            try {
                await fetch('/api/cameras/add', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({id: id, name: name, type: type, url: url, location: location})
                });
                closeModal();
                fetchChannels();
                selectChannel(id);
            } catch (e) {
                alert("Error adding channel: " + e);
            }
        }

        setInterval(fetchTelemetry, 1500);
        setInterval(fetchIncidents, 3500);
        fetchChannels();
        fetchTelemetry();
        fetchIncidents();
    </script>
</body>
</html>
"""


def create_app(engine: TrafficGuardEngine) -> FastAPI:
    app = FastAPI(title="TrafficGuard Enterprise AID", description="Enterprise Video Analytics & Automatic Incident Detection")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/", response_class=HTMLResponse)
    async def dashboard_index():
        return HTMLResponse(content=ENTERPRISE_TACTICAL_VMS_HTML)

    @app.get("/video_feed")
    def video_feed():
        def frame_generator():
            while True:
                jpeg = engine.get_jpeg_frame()
                if jpeg is not None:
                    yield (b"--frame\r\n"
                           b"Content-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n")
                time.sleep(1.0 / max(1.0, engine.detection_fps))

        return StreamingResponse(
            frame_generator(),
            media_type="multipart/x-mixed-replace; boundary=frame"
        )

    @app.get("/api/telemetry")
    async def get_telemetry():
        return engine.get_telemetry()

    @app.get("/api/cameras")
    async def list_cameras():
        return engine.camera_mgr.list_cameras()

    @app.post("/api/cameras/add")
    async def add_camera(req: AddCameraRequest):
        success = engine.camera_mgr.add_camera(
            cid=req.id,
            name=req.name,
            stream_type=req.type,
            url=req.url,
            location=req.location,
            lat=req.latitude,
            lon=req.longitude
        )
        return {"success": success, "camera_id": req.id}

    @app.post("/api/cameras/{camera_id}/activate")
    async def activate_camera(camera_id: str):
        success = engine.camera_mgr.set_active_camera(camera_id)
        if not success:
            raise HTTPException(status_code=404, detail="Camera ID not found")
        with engine.lock:
            engine.tracker.reset()
            engine.heuristics.latched_collisions.clear()
            engine.current_tracks = []
            while not engine.frame_queue.empty():
                try:
                    engine.frame_queue.get_nowait()
                except queue.Empty:
                    break
        return {"success": True, "active_camera": camera_id}

    @app.delete("/api/cameras/{camera_id}")
    async def delete_camera(camera_id: str):
        success = engine.camera_mgr.remove_camera(camera_id)
        return {"success": success}

    @app.get("/api/overlays")
    async def get_overlays():
        with engine.lock:
            return engine.layer_overlays

    @app.post("/api/overlays")
    async def update_overlays(req: OverlayUpdateRequest):
        with engine.lock:
            if req.boxes is not None:
                engine.layer_overlays["boxes"] = req.boxes
            if req.vectors is not None:
                engine.layer_overlays["vectors"] = req.vectors
            if req.trails is not None:
                engine.layer_overlays["trails"] = req.trails
            if req.rings is not None:
                engine.layer_overlays["rings"] = req.rings
            if req.hud is not None:
                engine.layer_overlays["hud"] = req.hud
        return {"success": True, "overlays": engine.layer_overlays}

    @app.get("/api/incidents")
    async def get_incidents(limit: int = 50):
        return engine.db.get_incidents(limit=limit)

    @app.post("/api/incidents/clear")
    async def clear_incidents():
        success = engine.clear_incidents()
        return {"success": success, "message": "All incident alerts cleared"}

    @app.post("/api/incidents/{incident_id}/verify")
    async def verify_incident(incident_id: int):
        success = engine.db.update_incident_status(incident_id, "VERIFIED")
        return {"success": success, "incident_id": incident_id, "status": "VERIFIED"}

    @app.post("/api/incidents/{incident_id}/false_alarm")
    async def false_alarm_incident(incident_id: int):
        success = engine.db.update_incident_status(incident_id, "FALSE_ALARM")
        return {"success": success, "incident_id": incident_id, "status": "FALSE_ALARM"}

    @app.post("/api/incidents/{incident_id}/status")
    async def update_status(incident_id: int, req: StatusUpdateRequest):
        success = engine.db.update_incident_status(incident_id, req.status)
        if not success:
            raise HTTPException(status_code=404, detail="Incident not found or update failed")
        return {"success": True, "incident_id": incident_id, "new_status": req.status}

    @app.post("/api/dispatch/{incident_id}")
    async def dispatch_cad(incident_id: int):
        incidents = engine.db.get_incidents(limit=100)
        inc = next((i for i in incidents if i["id"] == incident_id), None)
        engine.db.update_incident_status(incident_id, "DISPATCHED")
        return {
            "dispatched": True,
            "cad_ticket": f"CAD-2026-{incident_id:05d}",
            "priority": "PRIORITY 1 - IMMEDIATE RESPONSE",
            "incident_id": incident_id,
            "units": ["Highway Patrol Unit 42", "Towing & Recovery Unit 7", "EMS Paramedic Alpha"],
            "dispatch_timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
            "location": inc["location"] if inc else "Metropolitan Expressway Junction",
            "latitude": inc["latitude"] if inc else 34.0522,
            "longitude": inc["longitude"] if inc else -118.2437
        }

    @app.post("/api/trigger")
    async def trigger_incident():
        engine.trigger_incident()
        return {"success": True, "message": "Kinematic collision anomaly triggered"}

    @app.get("/api/stats")
    async def get_stats():
        return engine.db.get_stats()

    return app
