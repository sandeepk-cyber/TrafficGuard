# TrafficGuard - Real-Time Accident Detection MVP

**TrafficGuard** is a low-resource computer vision pipeline and operator dashboard designed to ingest traffic camera feeds (HLS, RTSP, MJPEG, or local video), detect vehicles, track kinematics, evaluate collision risk heuristics in real-time, and alert human operators.

---

## 🚀 Quick Start

### 1. Run Simulation Mode (Recommended for testing)
Starts the web dashboard with synthetic road animation and injects an accident event after 5 seconds:
```bash
python run.py --simulate
```

### 2. Run with Local Video File
Runs the CV detection and tracking pipeline against the included test clip:
```bash
python run.py --video data/test/accident.mp4
```

### 3. Run with Live Traffic Stream
Connects to the stream URL configured in `config/cameras.yaml`:
```bash
python run.py --live
```

---

## 🖥️ Operator Dashboard

Open your browser at:
👉 **[http://127.0.0.1:8001](http://127.0.0.1:8001)**

### Key Features:
- **Live Stream + HUD**: Real-time vehicle bounding boxes, speed estimates, and accident score overlay.
- **Telemetry**: Live CPU usage %, memory footprint, FPS, and camera connection state.
- **Incident Review**: High-risk candidates trigger audible and visual alerts. Operators can review snapshots and mark them as `RESOLVED` or `FALSE_POSITIVE`.
- **Incident History**: SQLite persistent audit trail in `data/trafficguard.db`.

---

## ⚙️ Configuration

Edit `config/cameras.yaml` to configure your live camera streams:

```yaml
cameras:
  demo:
    name: "Live Traffic Camera"
    type: "hls"   # 'hls', 'rtsp', 'mjpeg', or 'file'
    url: "https://your-stream-url.m3u8"
    latitude: 40.7128
    longitude: -74.0060
    location: "Metropolitan Traffic Corridor"
```

---

## 🧪 Running Tests

Run the automated test suite:
```bash
python scripts/test_system.py
```
