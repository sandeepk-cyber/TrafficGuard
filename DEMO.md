# TrafficGuard CCTV Incident Monitor - Operator Demonstration Guide

This document provides step-by-step instructions for demonstrating and evaluating the TrafficGuard CCTV accident-detection MVP.

Zero emojis, authentic CCTV surveillance focus.

---

## 1. Quick Start

### Start the Application
Open a terminal in the repository root and execute:

```bash
python run.py
```

The system will start by loading the primary elevated highway crash benchmark clip (`cctv_kaggle_04_highway_highspeed.mp4`).

Output will display the authentic terminal banner:
```text
======================================================================
  TRAFFICGUARD - REAL-TIME CCTV ACCIDENT DETECTION MVP
  Lightweight Local Computer Vision Pipeline (CPU)
======================================================================
  ACTIVE CAMERA : Kaggle CCTV - Elevated Highway High-Speed Crash
  STREAM SOURCE : data/test/cctv_kaggle_04_highway_highspeed.mp4
  PROCESSING RES: 640x360
  DETECTION FPS : 10.0 FPS (Target)
  OPERATOR UI   : http://127.0.0.1:8001
======================================================================
  Press Ctrl+C to stop.
```

### Access the Operator Console
Open a browser and navigate to:
**[http://127.0.0.1:8001](http://127.0.0.1:8001)**

---

## 2. Key Demonstration Scenarios

### Scenario A: High-Speed Kinetic Collision (TID-01)
1. Select **Kaggle CCTV - Elevated Highway High-Speed Crash** from the camera dropdown.
2. Observe normal traffic flow:
   - Green/white bounding boxes with vehicle IDs (e.g. `car #1`, `car #2`).
   - Incident Banner reads: `STATUS: NORMAL - ALL LANES CLEAR`.
3. As the overtaking vehicle approaches at high relative velocity, observe the trajectory convergence.
4. Upon impact and subsequent kinetic arrest:
   - Incident Banner switches to: `ALERT: CONFIRMED COLLISION DETECTED [TID-01]`.
   - Bounding boxes for involved vehicles turn to restrained red outline.
   - Incident Details card populates with collision score, involved track IDs, and camera location.
   - An entry is automatically logged into the SQLite audit list.

### Scenario B: Dense Traffic & Parked Vehicle Immunity
1. Select **Kaggle CCTV - Dense Urban Commercial Crossroads** (`cctv_kaggle_03`).
2. Observe multiple stationary vehicles parked along the curb and vehicles queueing at the crossroad.
3. Verify:
   - Parked vehicles with slight perspective bounding box overlap remain strictly `NORMAL`.
   - Bumper-to-bumper queueing vehicles do not trigger false alarms.
   - Incident Banner remains `STATUS: NORMAL - ALL LANES CLEAR`.

### Scenario C: Arterial Junction Collision
1. Select **Kaggle CCTV - Arterial Junction T-Bone Collision** (`cctv_kaggle_02`).
2. Observe cross-traffic motion at the intersection.
3. Watch the high-energy impact:
   - High spatial overlap and abrupt kinetic deceleration are detected.
   - System flags `CONFIRMED COLLISION` with empirical score >= 0.65.

### Scenario D: Live Stream Ingestion & Offline Handling
1. Switch to **Public HLS Ingest Test Feed** (`public_hls_test_feed`) from the dropdown, or run:
   ```bash
   python run.py --live
   ```
2. The system connects via FFmpeg pipe and streams frames into the detection pipeline.
3. If an invalid or disconnected stream is selected:
   - The status pill transitions to `RECONNECTING` or `OFFLINE`.
   - The display renders a clean dark placeholder: `CAMERA OFFLINE`.
   - No cartoon synthetic animations or fake vehicles are ever generated.

---

## 3. Telemetry & Resource Verification

In the top telemetry bar of the web console, observe real-time metrics:
- **INFERENCE**: Average and P95 latency (typically ~40-50 ms on CPU).
- **DETECTION**: Actual detection rate (throttled to ~10.0 FPS).
- **CAPTURE**: Video capture frame rate (typically ~25-30 FPS from source).
- **CPU**: Process CPU utilization percentage.
- **RAM**: Resident memory footprint (typically ~240 MB).
- **OBJECTS**: Active tracked vehicle count.

---

## 4. Running Automated Tests & Benchmarks

### Sequence-Level Behavioral Test Suite
Run the 13-part test suite validating normal flow, close passes, parked cars, sudden stops, wrong-way driving, track lifecycle, disconnect handling, and zero emojis:
```bash
python scripts/test_system.py
```

### Hardware Performance Benchmark
Run empirical benchmark over 100 consecutive frames:
```bash
python scripts/benchmark_system.py
```
