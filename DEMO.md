# TrafficGuard Enterprise ITS-AID: Operator Demonstration Guide

This document provides a comprehensive operational guide for evaluating the TrafficGuard Video Management System (VMS) and Automatic Incident Detection (ITS-AID) pipeline, benchmarked against authentic surveillance camera footage from the Kaggle dataset [Accident Footages From CCTV](https://www.kaggle.com/datasets/fahaddalwai/cctvfootagevideo).

---

## 1. Quick Start

### Start the VMS Server
Open PowerShell in the `TrafficGuard` workspace and run:

```powershell
python run.py
```

By default, TrafficGuard loads the authentic Kaggle elevated highway high-speed crash benchmark feed (`cctv_kaggle_04_highway_highspeed.mp4`).

### Access the Web Console
Open your web browser and navigate to:
**[http://127.0.0.1:8001](http://127.0.0.1:8001)**

---

## 2. Interactive Operator VMS Features

### A. Live Camera Switching
Use the **Active Channel Selector** in the navigation bar to switch between authentic CCTV feeds in real time. Switching cameras instantly clears prior kinematic tracking state, resets collision latching, and flushes frame buffers.

Available CCTV Channels:
1. **Kaggle CCTV - Elevated Highway High-Speed Crash** (`cctv_kaggle_04`): Elevated pole camera capturing high-speed kinetic highway collision and severe deceleration shock.
2. **Kaggle CCTV - Arterial Junction T-Bone Collision** (`cctv_kaggle_02`): 4th & Commercial arterial crossing severe right-angle vehicle-to-vehicle collision.
3. **Kaggle CCTV - Urban Motorcycle Conflict** (`cctv_kaggle_01`): Downtown Market Street corridor high-risk motorcycle-car interaction.
4. **Kaggle CCTV - Dense Urban Commercial Crossroads** (`cctv_kaggle_03`): Multi-vehicle retail corridor with parking zone (used to demonstrate dense parking false-positive immunity).
5. **Kaggle CCTV - Commercial Truck Heavy Impact** (`cctv_kaggle_05`): Logistics terminal high-momentum truck collision.
6. **Kaggle CCTV - Arterial Bus Corridor Collision** (`cctv_kaggle_06`): Transit corridor multi-lane commercial bus collision.
7. **Kaggle CCTV - Full Incident Compilation** (`cctv_kaggle_master`): 76.6-second continuous multi-sector surveillance stream.
8. **Jane Byrne Interstate Interchange** (`cctv_highway_interchange`): High-density highway interchange CCTV feed.
9. **Urban Intersection Collision** (`cctv_junction_accident`): Street-level traffic light junction incident.
10. **Metropolitan Expressway Collision** (`cctv_expressway_accident`): 1080p surveillance crash benchmark.
11. **Caltrans Live Traffic Feed** (`caltrans_live_cctv`): Live HLS network video feed.

### B. Layer Overlays Control
Use the tactical overlay checkboxes in the console header to toggle real-time CV layers:
- **Boxes**: High-contrast bounding boxes with class tags, confidence scores, and IDs.
- **Vectors**: Real-time directional velocity arrows scaled by speed.
- **Trajectories**: Multi-frame smoothed motion breadcrumb trails.
- **Collision Rings**: Vivid solid red (`(0, 0, 255)`) reinforced corner brackets and impact reticles latched for 7 seconds upon impact.
- **Tactical HUD**: Telemetry status stamp, timestamp, and active camera label.

### C. Incident Triage Desk & CAD Dispatch
When an incident is detected:
- The threat status turns **CRITICAL** with an audible audio alert.
- The incident is latched for **7 seconds** in solid red on the video feed.
- An incident card appears in the **Incident Triage Queue** with timestamp, severity, vehicles involved, and peak risk score.
- Operators can click **Verify Incident** to confirm or **False Alarm** to log triage feedback.
- Click **Dispatch CAD** to generate an official Computer-Aided Dispatch emergency response ticket.

### D. Real-Time Corridor Telemetry
The telemetry panel continuously reports ITS corridor metrics:
- **Corridor Flow Rate**: Vehicles Per Minute (VPM).
- **Level of Service (LOS)**: Density classification from LOS A (free flow) to LOS F (breakdown).
- **Average Speed**: Real-time corridor velocity in km/h.
- **Threat Index**: Aggregated collision risk score (0 to 100%).
- **Hardware Telemetry**: CPU utilization, memory footprint, and inference latency.

---

## 3. Automated Verification Suite

To run the complete 8-part automated test suite validating detection, tracking, false-positive immunity, and CAD dispatch:

```powershell
python scripts/test_system.py
```

Expected Output:
- [TEST 1/8] YOLO Deep Learning Detector & Containment Suppression: PASS
- [TEST 2/8] Kaggle & Enterprise CCTV Feeds Integrity (10 feeds): PASS
- [TEST 3/8] Kinematic Tracker Zero Ghosting & Velocity Stability: PASS
- [TEST 4/8] Persistent 7-Second Red Collision Latching: PASS
- [TEST 5/8] Dense Urban Parking False-Positive Immunity (0 false alarms): PASS
- [TEST 6/8] Positive Crash Detection on Real CCTV (Scene 4 Highway): PASS
- [TEST 7/8] Commercial ITS Analytics & CAD Dispatch: PASS
- [TEST 8/8] Strict Zero Emojis Compliance: PASS

---

## 4. Live Stream Injection

To inject an external live traffic camera (HLS, RTSP, MJPEG, or local video file):

### Method A: Web Console (Dynamic Ingest)
1. Click **+ Add Camera** in the top navigation bar.
2. Enter the camera ID, display name, stream URL (e.g. `http://.../live.m3u8` or `rtsp://...`), and geographic coordinates.
3. Click **Register Camera**. The stream immediately appears in the channel selector.

### Method B: Configuration File
Edit `config/cameras.yaml` to register permanent camera endpoints:

```yaml
cameras:
  my_city_cctv:
    name: Downtown 5th Avenue Traffic Cam
    type: hls
    url: https://your-stream-server.com/live/cctv01.m3u8
    location: 5th Avenue & Market Street
    latitude: 37.7749
    longitude: -122.4194
```
