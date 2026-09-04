"""
TrafficGuard - Lightweight CCTV Accident Detection MVP
Operator Console & Streaming Server.
Zero emojis, strict surveillance standards.
"""
import os
import sys
import time
import base64
import threading
import collections
import logging
import cv2
import psutil
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional
from pydantic import BaseModel

from app.detector import YOLODetector
from app.tracker import KinematicTracker
from app.heuristics import IncidentEngine
from app.camera_manager import CameraManager, FRAME_WIDTH, FRAME_HEIGHT
from app.database import IncidentDatabase

logger = logging.getLogger("trafficguard.dashboard")

DEFAULT_DETECTION_FPS = 10.0


class StatusUpdateRequest(BaseModel):
    status: str


class AddCameraRequest(BaseModel):
    id: str
    name: str
    type: str  # 'file', 'hls', 'rtsp'
    url: str
    location: str = "Surveillance Corridor"
    latitude: Optional[float] = None
    longitude: Optional[float] = None


class TrafficGuardEngine:
    """
    Decoupled Computer Vision Pipeline:
      - Continuous camera capture thread (CameraManager)
      - Dedicated AI inference thread running at configurable detection_fps (default 10 FPS)
      - Non-blocking MJPEG display stream
    """
    def __init__(self, camera_config=None, detection_fps=DEFAULT_DETECTION_FPS):
        self.detection_fps = float(detection_fps)
        self.lock = threading.Lock()
        self.running = False
        self.infer_thread = None

        # Core CV Subsystems
        self.camera_mgr = CameraManager()
        self.detector = YOLODetector(conf_threshold=0.20, nms_threshold=0.40)
        self.tracker = KinematicTracker()
        self.heuristics = IncidentEngine(risk_threshold=0.65)
        self.db = IncidentDatabase()

        # Handle explicit CLI camera override if provided
        if camera_config:
            cid = "cli_selected_source"
            cname = camera_config.get("name", "Active Source")
            ctype = camera_config.get("type", "file")
            curl = camera_config.get("url", "")
            cloc = camera_config.get("location", "CCTV Benchmark")
            self.camera_mgr.add_camera(cid, cname, ctype, curl, cloc,
                                       lat=camera_config.get("latitude"),
                                       lon=camera_config.get("longitude"))
            self.camera_mgr.set_active_camera(cid)

        # Thread-safe State & Telemetry
        self.latest_jpeg = None
        self.current_tracks = []
        self.current_status = "NORMAL"
        self.current_risk = 0.0
        self.current_incident = None
        self.vehicle_count = 0

        # Empirical Performance Metrics
        self.actual_detection_fps = 0.0
        self.inference_latencies = collections.deque(maxlen=50)
        self.avg_inference_ms = 0.0
        self.p95_inference_ms = 0.0
        self.cpu_percent = 0.0
        self.memory_mb = 0.0
        self.process = psutil.Process(os.getpid())

        # Alarm Auto-decay and Database Throttle
        self.last_incident_record_time = 0.0
        self.alarm_expiry_time = 0.0

    def start(self):
        if self.running:
            return
        self.running = True
        self.infer_thread = threading.Thread(target=self._inference_worker_loop, daemon=True)
        self.infer_thread.start()
        logger.info("TrafficGuard Engine started at target %.1f FPS.", self.detection_fps)

    def stop(self):
        self.running = False
        if self.infer_thread and self.infer_thread.is_alive():
            self.infer_thread.join(timeout=1.5)
        self.camera_mgr.release()
        logger.info("TrafficGuard Engine stopped.")

    def reset_state(self):
        with self.lock:
            self.tracker.reset()
            self.heuristics.reset()
            self.current_tracks = []
            self.current_status = "NORMAL"
            self.current_risk = 0.0
            self.current_incident = None
            self.vehicle_count = 0
            self.alarm_expiry_time = 0.0

    def clear_incidents(self):
        self.reset_state()
        return self.db.clear_all_incidents()

    def _inference_worker_loop(self):
        """Dedicated inference loop running at detection_fps on freshest camera frames."""
        target_interval = 1.0 / max(1.0, self.detection_fps)
        frame_counter = 0
        sec_timer = time.time()
        stat_timer = time.time()

        while self.running:
            loop_start = time.perf_counter()

            # 1. Ingest latest frame from active camera
            success, raw_frame = self.camera_mgr.read_frame()
            now = time.time()

            if not success or raw_frame is None:
                with self.lock:
                    self.current_tracks = []
                    self.current_status = "OFFLINE"
                    self.current_risk = 0.0
                    self.current_incident = None
                    self.vehicle_count = 0

                # Encode placeholder frame for display stream
                if raw_frame is not None:
                    ret, buf = cv2.imencode(".jpg", raw_frame, [cv2.IMWRITE_JPEG_QUALITY, 72])
                    if ret:
                        with self.lock:
                            self.latest_jpeg = buf.tobytes()

                time.sleep(0.08)
                continue

            # 2. Execute YOLO Object Detection
            dets, inf_ms = self.detector.detect(raw_frame)

            # 3. Associate with Hungarian Kinematic Tracker
            tracks = self.tracker.update(dets, timestamp=now)

            # 4. Multi-factor Incident Evaluation & CCTV Annotation
            annotated_frame, risk_score, status, is_incident, inc_data = self.heuristics.evaluate(
                raw_frame, tracks, timestamp=now
            )

            # 5. Fast JPEG preview encoding
            h, w = annotated_frame.shape[:2]
            if w > 720:
                preview = cv2.resize(annotated_frame, (720, int(720 * (h / w))))
            else:
                preview = annotated_frame

            ret, buf = cv2.imencode(".jpg", preview, [cv2.IMWRITE_JPEG_QUALITY, 72])
            jpeg_bytes = buf.tobytes() if ret else None

            # 6. Update shared telemetry and track state
            with self.lock:
                self.latest_jpeg = jpeg_bytes
                self.current_tracks = tracks
                self.vehicle_count = len(tracks)
                self.inference_latencies.append(inf_ms)

                if is_incident:
                    self.current_status = status
                    self.current_risk = risk_score
                    if inc_data:
                        self.current_incident = inc_data
                    self.alarm_expiry_time = now + 5.0  # Latch alarm display for 5s
                elif now > self.alarm_expiry_time:
                    self.current_status = status
                    self.current_risk = risk_score
                    self.current_incident = None

            # 7. Record confirmed incident to SQLite audit database
            if is_incident and (now - self.last_incident_record_time > 5.0):
                self.last_incident_record_time = now
                cam_info = self.camera_mgr.get_active_camera_info()
                snap_b64 = f"data:image/jpeg;base64,{base64.b64encode(jpeg_bytes).decode('utf-8')}" if jpeg_bytes else ""
                inc_code = inc_data.get("incident_code", "TID-01") if inc_data else "TID-01"
                inc_desc = inc_data.get("description", "Collision anomaly") if inc_data else "Collision anomaly"
                inc_veh = inc_data.get("vehicles_involved", "Vehicles") if inc_data else "Vehicles"
                inc_spd = inc_data.get("speed_at_impact", "Unknown") if inc_data else "Unknown"

                self.db.record_incident(
                    camera_name=cam_info.get("name", "Camera Channel"),
                    location=cam_info.get("location", "CCTV Benchmark"),
                    latitude=cam_info.get("latitude") or 0.0,
                    longitude=cam_info.get("longitude") or 0.0,
                    incident_code=inc_code,
                    risk_score=risk_score,
                    severity=status,
                    speed_at_impact=inc_spd,
                    vehicles_involved=inc_veh,
                    snapshot_base64=snap_b64,
                    description=inc_desc
                )

            # 8. Update execution rate & system resource telemetry
            frame_counter += 1
            if now - sec_timer >= 1.0:
                self.actual_detection_fps = round(frame_counter / (now - sec_timer), 1)
                frame_counter = 0
                sec_timer = now

            if now - stat_timer >= 2.0:
                stat_timer = now
                with self.lock:
                    if self.inference_latencies:
                        lats = sorted(self.inference_latencies)
                        self.avg_inference_ms = round(sum(lats) / len(lats), 1)
                        p95_idx = int(len(lats) * 0.95)
                        self.p95_inference_ms = round(lats[min(p95_idx, len(lats) - 1)], 1)
                try:
                    self.cpu_percent = round(psutil.cpu_percent(), 1)
                    self.memory_mb = round(self.process.memory_info().rss / (1024 * 1024), 1)
                except Exception:
                    pass

            # 9. Hardware rate throttling
            elapsed = time.perf_counter() - loop_start
            sleep_time = max(0.001, target_interval - elapsed)
            time.sleep(sleep_time)

    def get_jpeg_frame(self):
        with self.lock:
            return self.latest_jpeg

    def get_telemetry(self):
        with self.lock:
            cam_info = self.camera_mgr.get_active_camera_info()
            return {
                "status": self.current_status,
                "risk_score": round(self.current_risk, 2),
                "incident_code": self.current_incident.get("incident_code") if self.current_incident else None,
                "incident_description": self.current_incident.get("description") if self.current_incident else None,
                "vehicles_involved": self.current_incident.get("vehicles_involved") if self.current_incident else None,
                "vehicle_count": self.vehicle_count,
                "capture_fps": round(cam_info.get("capture_fps", 0.0), 1),
                "detection_fps": self.actual_detection_fps,
                "target_fps": self.detection_fps,
                "inference_avg_ms": self.avg_inference_ms,
                "inference_p95_ms": self.p95_inference_ms,
                "cpu_percent": self.cpu_percent,
                "memory_mb": self.memory_mb,
                "active_camera_id": cam_info.get("id"),
                "active_camera_name": cam_info.get("name"),
                "camera_status": cam_info.get("status", "ONLINE"),
                "camera_type": cam_info.get("type", "file"),
                "camera_location": cam_info.get("location", "CCTV Benchmark")
            }


OPERATOR_DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>TrafficGuard // CCTV Incident Monitoring Console</title>
    <link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600;700&family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-base: #0a0d14;
            --bg-card: #111622;
            --bg-card-hover: #161c2b;
            --border: #1e2638;
            --border-bright: #2d374d;
            --text-primary: #f1f5f9;
            --text-secondary: #94a3b8;
            --text-muted: #64748b;
            --status-online: #10b981;
            --status-warning: #f59e0b;
            --status-critical: #ef4444;
            --accent-blue: #3b82f6;
        }

        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            background-color: var(--bg-base);
            color: var(--text-primary);
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            min-height: 100vh;
            display: flex;
            flex-direction: column;
            overflow-x: hidden;
        }

        /* Top Header Bar */
        header {
            background: #0d121c;
            border-bottom: 1px solid var(--border);
            padding: 10px 24px;
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 20px;
        }

        .brand-section {
            display: flex;
            align-items: center;
            gap: 12px;
        }
        .brand-title {
            font-family: 'JetBrains Mono', monospace;
            font-size: 15px;
            font-weight: 700;
            letter-spacing: 0.5px;
            color: #ffffff;
        }
        .brand-badge {
            font-family: 'JetBrains Mono', monospace;
            font-size: 11px;
            padding: 2px 7px;
            background: #1e293b;
            color: #94a3b8;
            border-radius: 4px;
            border: 1px solid var(--border);
        }

        .controls-section {
            display: flex;
            align-items: center;
            gap: 14px;
        }

        select.camera-dropdown {
            background: #161e2e;
            color: var(--text-primary);
            border: 1px solid var(--border-bright);
            padding: 6px 12px;
            border-radius: 4px;
            font-size: 13px;
            font-family: 'Inter', sans-serif;
            outline: none;
            cursor: pointer;
            min-width: 280px;
        }

        .status-pill {
            font-family: 'JetBrains Mono', monospace;
            font-size: 11px;
            font-weight: 600;
            padding: 4px 10px;
            border-radius: 4px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        .status-pill.online { background: rgba(16, 185, 129, 0.15); color: var(--status-online); border: 1px solid rgba(16, 185, 129, 0.3); }
        .status-pill.reconnecting { background: rgba(245, 158, 11, 0.15); color: var(--status-warning); border: 1px solid rgba(245, 158, 11, 0.3); }
        .status-pill.offline { background: rgba(239, 68, 68, 0.15); color: var(--status-critical); border: 1px solid rgba(239, 68, 68, 0.3); }

        /* Telemetry Ribbon */
        .telemetry-bar {
            background: #0f1523;
            border-bottom: 1px solid var(--border);
            padding: 6px 24px;
            display: flex;
            align-items: center;
            justify-content: flex-start;
            gap: 24px;
            font-family: 'JetBrains Mono', monospace;
            font-size: 12px;
            color: var(--text-secondary);
            overflow-x: auto;
        }
        .telem-item span.val {
            color: var(--text-primary);
            font-weight: 600;
        }

        /* Main Workspace Layout */
        main {
            display: grid;
            grid-template-columns: 1fr 380px;
            gap: 16px;
            padding: 16px 24px;
            flex: 1;
        }

        @media (max-width: 1024px) {
            main { grid-template-columns: 1fr; }
        }

        /* Left Column: CCTV Stream & Alert Banner */
        .video-container {
            display: flex;
            flex-direction: column;
            gap: 12px;
        }

        .incident-banner {
            padding: 12px 18px;
            border-radius: 6px;
            font-family: 'JetBrains Mono', monospace;
            font-size: 13px;
            font-weight: 600;
            display: flex;
            align-items: center;
            justify-content: space-between;
            transition: background-color 0.2s, border-color 0.2s;
        }
        .incident-banner.normal {
            background: #111827;
            border: 1px solid #1f2937;
            color: #9ca3af;
        }
        .incident-banner.possible {
            background: rgba(245, 158, 11, 0.15);
            border: 1px solid rgba(245, 158, 11, 0.4);
            color: #fbbf24;
        }
        .incident-banner.confirmed {
            background: rgba(239, 68, 68, 0.2);
            border: 1px solid rgba(239, 68, 68, 0.6);
            color: #f87171;
            box-shadow: 0 0 15px rgba(239, 68, 68, 0.2);
        }

        .video-viewport {
            position: relative;
            background: #000000;
            border: 1px solid var(--border);
            border-radius: 6px;
            overflow: hidden;
            display: flex;
            align-items: center;
            justify-content: center;
            min-height: 400px;
        }

        .video-viewport img {
            width: 100%;
            height: auto;
            max-height: 68vh;
            object-fit: contain;
            display: block;
        }

        .viewport-hud {
            position: absolute;
            top: 10px;
            left: 12px;
            font-family: 'JetBrains Mono', monospace;
            font-size: 11px;
            color: rgba(255, 255, 255, 0.7);
            background: rgba(0, 0, 0, 0.6);
            padding: 3px 8px;
            border-radius: 3px;
            pointer-events: none;
        }

        /* Right Column: Cards */
        .sidebar {
            display: flex;
            flex-direction: column;
            gap: 16px;
        }

        .card {
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: 6px;
            padding: 16px;
        }

        .card-header {
            font-size: 13px;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            color: var(--text-secondary);
            margin-bottom: 12px;
            padding-bottom: 8px;
            border-bottom: 1px solid var(--border);
            display: flex;
            justify-content: space-between;
            align-items: center;
        }

        .details-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 10px;
            font-size: 12px;
        }
        .detail-item {
            display: flex;
            flex-direction: column;
            gap: 2px;
        }
        .detail-item .label {
            font-size: 11px;
            color: var(--text-muted);
            text-transform: uppercase;
        }
        .detail-item .value {
            font-family: 'JetBrains Mono', monospace;
            color: var(--text-primary);
            font-weight: 500;
        }

        /* Incidents Table */
        .incident-list {
            max-height: 360px;
            overflow-y: auto;
            display: flex;
            flex-direction: column;
            gap: 8px;
        }

        .incident-item {
            background: #141a29;
            border: 1px solid var(--border);
            border-radius: 4px;
            padding: 10px;
            display: flex;
            flex-direction: column;
            gap: 6px;
            font-size: 12px;
        }
        .incident-item-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .incident-code {
            font-family: 'JetBrains Mono', monospace;
            font-weight: 600;
            color: var(--status-critical);
        }
        .incident-time {
            color: var(--text-muted);
            font-size: 11px;
        }
        .incident-actions {
            display: flex;
            gap: 6px;
            margin-top: 4px;
        }

        button.btn {
            background: #1e293b;
            color: var(--text-primary);
            border: 1px solid var(--border-bright);
            padding: 4px 8px;
            border-radius: 3px;
            font-size: 11px;
            cursor: pointer;
            font-family: 'Inter', sans-serif;
            transition: background 0.15s;
        }
        button.btn:hover { background: #334155; }
        button.btn-danger { background: rgba(239, 68, 68, 0.2); border-color: rgba(239, 68, 68, 0.4); color: #f87171; }
        button.btn-danger:hover { background: rgba(239, 68, 68, 0.35); }
        button.btn-success { background: rgba(16, 185, 129, 0.2); border-color: rgba(16, 185, 129, 0.4); color: #34d399; }
        button.btn-success:hover { background: rgba(16, 185, 129, 0.35); }
    </style>
</head>
<body>
    <header>
        <div class="brand-section">
            <span class="brand-title">TRAFFICGUARD</span>
            <span class="brand-badge">CCTV AID MVP</span>
        </div>
        <div class="controls-section">
            <select id="cameraSelector" class="camera-dropdown" onchange="switchCamera(this.value)">
                <option value="">Loading camera catalog...</option>
            </select>
            <div id="cameraStatusBadge" class="status-pill online">ONLINE</div>
        </div>
    </header>

    <div class="telemetry-bar">
        <div class="telem-item">INFERENCE: <span id="telemInf" class="val">--</span> ms (P95: <span id="telemP95" class="val">--</span> ms)</div>
        <div class="telem-item">DETECTION: <span id="telemFps" class="val">--</span> FPS</div>
        <div class="telem-item">CAPTURE: <span id="telemCap" class="val">--</span> FPS</div>
        <div class="telem-item">CPU: <span id="telemCpu" class="val">--</span>%</div>
        <div class="telem-item">RAM: <span id="telemMem" class="val">--</span> MB</div>
        <div class="telem-item">OBJECTS: <span id="telemVeh" class="val">0</span></div>
    </div>

    <main>
        <div class="video-container">
            <div id="incidentBanner" class="incident-banner normal">
                <span id="bannerText">STATUS: NORMAL - ALL LANES CLEAR</span>
                <span id="bannerScore">RISK: 0.00</span>
            </div>
            <div class="video-viewport">
                <img id="cctvFeed" src="/video_feed" alt="Live CCTV Video Feed">
                <div id="viewportHud" class="viewport-hud">LIVE CCTV // 640x360</div>
            </div>
        </div>

        <div class="sidebar">
            <div class="card">
                <div class="card-header">
                    <span>Incident Inspector</span>
                    <span id="inspectBadge" class="status-pill" style="font-size:10px;">MONITORING</span>
                </div>
                <div class="details-grid">
                    <div class="detail-item">
                        <span class="label">Incident Code</span>
                        <span id="detailCode" class="value">--</span>
                    </div>
                    <div class="detail-item">
                        <span class="label">Collision Score</span>
                        <span id="detailScore" class="value">0.00</span>
                    </div>
                    <div class="detail-item" style="grid-column: span 2;">
                        <span class="label">Vehicles Involved</span>
                        <span id="detailVehicles" class="value">None</span>
                    </div>
                    <div class="detail-item" style="grid-column: span 2;">
                        <span class="label">Camera Location</span>
                        <span id="detailLocation" class="value">CCTV Feed</span>
                    </div>
                </div>
            </div>

            <div class="card" style="flex: 1; display: flex; flex-direction: column;">
                <div class="card-header">
                    <span>Audit Incident Log</span>
                    <button class="btn" onclick="clearIncidents()">Clear All</button>
                </div>
                <div id="incidentList" class="incident-list">
                    <div style="color: var(--text-muted); font-size: 12px; text-align: center; padding: 20px 0;">No incidents recorded.</div>
                </div>
            </div>
        </div>
    </main>

    <script>
        let activeCamId = null;

        async function fetchCameras() {
            try {
                const res = await fetch('/api/cameras');
                const cameras = await res.json();
                const selector = document.getElementById('cameraSelector');
                selector.innerHTML = '';
                cameras.forEach(cam => {
                    const opt = document.createElement('option');
                    opt.value = cam.id;
                    opt.textContent = cam.name + (cam.type === 'hls' ? ' (Live Stream)' : '');
                    if (cam.is_active) {
                        opt.selected = true;
                        activeCamId = cam.id;
                    }
                    selector.appendChild(opt);
                });
            } catch (err) {
                console.error("Failed to load cameras:", err);
            }
        }

        async function switchCamera(camId) {
            if (!camId || camId === activeCamId) return;
            try {
                const res = await fetch(`/api/cameras/${encodeURIComponent(camId)}/activate`, { method: 'POST' });
                if (res.ok) {
                    activeCamId = camId;
                    const img = document.getElementById('cctvFeed');
                    img.src = '/video_feed?t=' + Date.now();
                }
            } catch (err) {
                console.error("Failed to switch camera:", err);
            }
        }

        async function updateTelemetry() {
            try {
                const res = await fetch('/api/telemetry');
                if (!res.ok) return;
                const data = await res.json();

                // Telemetry bar
                document.getElementById('telemInf').textContent = data.inference_avg_ms;
                document.getElementById('telemP95').textContent = data.inference_p95_ms;
                document.getElementById('telemFps').textContent = data.detection_fps;
                document.getElementById('telemCap').textContent = data.capture_fps;
                document.getElementById('telemCpu').textContent = data.cpu_percent;
                document.getElementById('telemMem').textContent = data.memory_mb;
                document.getElementById('telemVeh').textContent = data.vehicle_count;

                // Status pill
                const statusBadge = document.getElementById('cameraStatusBadge');
                statusBadge.textContent = data.camera_status;
                statusBadge.className = 'status-pill ' + data.camera_status.toLowerCase();

                // Incident Alert Banner
                const banner = document.getElementById('incidentBanner');
                const bannerText = document.getElementById('bannerText');
                const bannerScore = document.getElementById('bannerScore');

                if (data.status === 'CONFIRMED COLLISION') {
                    banner.className = 'incident-banner confirmed';
                    bannerText.textContent = 'ALERT: CONFIRMED COLLISION DETECTED [' + (data.incident_code || 'TID-01') + ']';
                    bannerScore.textContent = 'SCORE: ' + data.risk_score.toFixed(2);
                } else if (data.status === 'POSSIBLE COLLISION') {
                    banner.className = 'incident-banner possible';
                    bannerText.textContent = 'STATUS: POSSIBLE COLLISION - ASSESSING VELOCITY DELTA';
                    bannerScore.textContent = 'SCORE: ' + data.risk_score.toFixed(2);
                } else if (data.status === 'OFFLINE') {
                    banner.className = 'incident-banner normal';
                    bannerText.textContent = 'CAMERA FEED OFFLINE - ATTEMPTING RECONNECT';
                    bannerScore.textContent = 'SOURCE: ' + data.camera_status;
                } else {
                    banner.className = 'incident-banner normal';
                    bannerText.textContent = 'STATUS: NORMAL - ALL LANES CLEAR';
                    bannerScore.textContent = 'RISK: ' + data.risk_score.toFixed(2);
                }

                // Incident Inspector
                document.getElementById('detailCode').textContent = data.incident_code || 'NONE';
                document.getElementById('detailScore').textContent = data.risk_score.toFixed(2);
                document.getElementById('detailVehicles').textContent = data.vehicles_involved || 'None';
                document.getElementById('detailLocation').textContent = data.camera_location || 'CCTV Feed';
                document.getElementById('inspectBadge').textContent = data.status;

                document.getElementById('viewportHud').textContent =
                    (data.active_camera_name || 'CCTV') + ' // ' + (data.camera_type || 'FILE').toUpperCase();

            } catch (err) {
                console.error("Telemetry update error:", err);
            }
        }

        async function updateIncidentList() {
            try {
                const res = await fetch('/api/incidents?limit=20');
                if (!res.ok) return;
                const incidents = await res.json();
                const container = document.getElementById('incidentList');

                if (!incidents || incidents.length === 0) {
                    container.innerHTML = '<div style="color: var(--text-muted); font-size: 12px; text-align: center; padding: 20px 0;">No incidents recorded.</div>';
                    return;
                }

                container.innerHTML = '';
                incidents.forEach(inc => {
                    const item = document.createElement('div');
                    item.className = 'incident-item';
                    item.innerHTML = `
                        <div class="incident-item-header">
                            <span class="incident-code">${inc.incident_code || 'TID-01'}</span>
                            <span class="incident-time">${inc.timestamp || ''}</span>
                        </div>
                        <div style="font-size: 11px; color: var(--text-secondary);">
                            Score: <strong>${inc.risk_score ? inc.risk_score.toFixed(2) : '0.00'}</strong> | ${inc.vehicles_involved || 'Vehicles'}
                        </div>
                        <div style="font-size: 11px; color: var(--text-muted);">
                            Status: <strong style="color: ${inc.status === 'VERIFIED' ? '#10b981' : (inc.status === 'FALSE_ALARM' ? '#94a3b8' : '#f59e0b')}">${inc.status}</strong>
                        </div>
                        <div class="incident-actions">
                            <button class="btn btn-success" onclick="verifyIncident(${inc.id})">Verify</button>
                            <button class="btn btn-danger" onclick="markFalseAlarm(${inc.id})">False Alarm</button>
                        </div>
                    `;
                    container.appendChild(item);
                });
            } catch (err) {
                console.error("Incident log update error:", err);
            }
        }

        async function verifyIncident(id) {
            await fetch(`/api/incidents/${id}/verify`, { method: 'POST' });
            updateIncidentList();
        }

        async function markFalseAlarm(id) {
            await fetch(`/api/incidents/${id}/false_alarm`, { method: 'POST' });
            updateIncidentList();
        }

        async function clearIncidents() {
            if (confirm("Clear all recorded incident logs?")) {
                await fetch('/api/incidents/clear', { method: 'POST' });
                updateIncidentList();
                updateTelemetry();
            }
        }

        // Initialize polling cycles
        fetchCameras();
        setInterval(updateTelemetry, 500);
        setInterval(updateIncidentList, 2500);
        updateTelemetry();
        updateIncidentList();
    </script>
</body>
</html>
"""


def create_app(engine: TrafficGuardEngine) -> FastAPI:
    app = FastAPI(title="TrafficGuard MVP", description="Lightweight CCTV Traffic Incident Detection")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/", response_class=HTMLResponse)
    async def dashboard_index():
        return HTMLResponse(content=OPERATOR_DASHBOARD_HTML)

    @app.get("/video_feed")
    def video_feed():
        def frame_generator():
            while True:
                jpeg = engine.get_jpeg_frame()
                if jpeg is not None:
                    yield (b"--frame\r\n"
                           b"Content-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n")
                time.sleep(0.04)

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

    @app.post("/api/cameras/{camera_id}/activate")
    async def activate_camera(camera_id: str):
        success = engine.camera_mgr.set_active_camera(camera_id)
        if not success:
            raise HTTPException(status_code=404, detail="Camera ID not found")
        engine.reset_state()
        return {"success": True, "active_camera": camera_id}

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

    @app.delete("/api/cameras/{camera_id}")
    async def delete_camera(camera_id: str):
        success = engine.camera_mgr.remove_camera(camera_id)
        return {"success": success}

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
            raise HTTPException(status_code=404, detail="Incident not found")
        return {"success": True, "incident_id": incident_id, "status": req.status}

    return app
