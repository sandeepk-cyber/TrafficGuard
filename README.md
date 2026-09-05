# TrafficGuard - Real-Time CCTV Traffic Accident Detection MVP

TrafficGuard is a lightweight, low-resource computer vision pipeline and operator console designed for automated traffic incident detection on closed-circuit television (CCTV) cameras. It runs entirely on local CPU resources with low latency, stable multi-object tracking, and empirical multi-factor collision verification.

Zero emojis, zero tactical gimmicks, and strict surveillance engineering standards.

---

## Architectural Overview

TrafficGuard decouples video stream ingestion, neural network inference, and operator streaming to maintain responsiveness and avoid memory queue buildups:

1. **Stream Ingestion (Thread 1)**:
   - Dedicated capture thread reading from local video files or network streams (HLS, RTSP, MJPEG).
   - Ingests into a lock-protected single-slot `latest_frame` buffer.
   - Explicit connection states: `CONNECTING`, `ONLINE`, `RECONNECTING`, and `OFFLINE`.
   - Stale stream timeout (4.0s) with exponential reconnection backoff.
   - Clean dark placeholder frames when offline (zero synthetic cartoon fallbacks).

2. **Detection & Tracking (Thread 2)**:
   - Target processing rate: 10.0 FPS on CPU (configurable 8-15 FPS).
   - Dynamic model introspection handling FP16 and FP32 ONNX runtimes.
   - Letterbox coordinate inversion with boundary clamping.
   - Class-aware non-maximum suppression (NMS) separating vehicles and pedestrians.
   - State-machine multi-object tracker (`TENTATIVE`, `CONFIRMED`, `LOST`, `REMOVED`) with Hungarian data association (`scipy.optimize.linear_sum_assignment`).
   - Image-plane pixel kinematics: pixel velocity (`speed_px_s`), relative speed index, and motion deadband. No fictional km/h speeds without geometric calibration.
   - Zero ghost boxes: missed tracks survive in memory for re-association but are not rendered.

3. **Multi-Factor Incident Engine**:
   - Temporal multi-factor empirical collision model across a 0.5s to 2.0s observation window:
     `Score = S_overlap * S_approach * S_rel_vel * S_post`
   - Differentiates authentic impacts from normal surveillance scenarios:
     - Normal moving traffic maintains `NORMAL` status.
     - Side-by-side parked vehicles maintain `NORMAL` status (approach delta = 0).
     - Slow dense queues maintain `NORMAL` status.
     - Vehicles passing closely in adjacent lanes maintain `NORMAL` status.
   - Incident Classifications:
     - `TID-01 COLLISION`: Closing trajectory + high overlap + post-impact kinetic arrest or abrupt speed drop.
     - `TID-02 STOPPED_VEHICLE`: Vehicle previously traveling at traffic speed that remains stationary in an active flow corridor.
     - `TID-03 WRONG_WAY`: Vehicle traveling against configured corridor heading.

4. **Operator Web Console**:
   - Minimalist, high-contrast dark surveillance console (16:9 responsive CCTV viewport).
   - Real-time stream delivery over HTTP MJPEG.
   - Honest telemetry: Camera Capture FPS, Detection FPS, Average and P95 Inference Latency (ms), Process CPU %, and RAM Footprint (MB).
   - Incident triage desk with SQLite persistent audit trail and verification actions.

---

## Performance Benchmarks (Local CPU)

Benchmark executed on 100 consecutive frames of authentic 640x360 CCTV footage (`cctv_kaggle_04_highway_highspeed.mp4`) using an Intel CPU:

| Stage | Average Latency | P50 Latency | P95 Latency |
| :--- | :--- | :--- | :--- |
| **YOLO Object Detection (ONNX FP16)** | 46.8 ms | 46.0 ms | 50.9 ms |
| **Hungarian Tracking** | 0.15 ms | 0.11 ms | 0.20 ms |
| **Incident Heuristics & Annotations** | 0.43 ms | 0.19 ms | 0.34 ms |
| **End-to-End Pipeline** | **47.4 ms** | **46.3 ms** | **51.6 ms** |

- **Effective Pipeline Throughput**: ~21.1 FPS on CPU
- **Memory Footprint (RSS)**: ~242 MB
- **Zero Memory Leaks**: Stream switching cleanly terminates background workers and releases video decoders.

---

## Quick Start

### 1. Launch Operator Console
Run with the default authentic CCTV crash benchmark:
```bash
python run.py
```

Or specify an explicit camera channel from `config/cameras.yaml`:
```bash
python run.py --camera cctv_kaggle_02_junction
```

Or ingest a local video file directly:
```bash
python run.py --video data/test/cctv_kaggle_04_highway_highspeed.mp4
```

Or connect to the live HLS stream:
```bash
python run.py --live
```

Access the web interface at **[http://127.0.0.1:8001](http://127.0.0.1:8001)**.

---

## CCTV Benchmark Dataset

TrafficGuard incorporates benchmark video clips from the public Kaggle surveillance dataset [Accident Footages From CCTV](https://www.kaggle.com/datasets/fahaddalwai/cctvfootagevideo):

- `cctv_kaggle_04_highway_highspeed.mp4`: Elevated highway high-speed crash benchmark.
- `cctv_kaggle_02_junction_tbone.mp4`: Arterial junction right-angle collision benchmark.
- `cctv_kaggle_01_urban_motorcycle.mp4`: Downtown corridor motorcycle conflict benchmark.
- `cctv_kaggle_03_dense_crossroads.mp4`: Dense urban crossroads and parking sector (false-positive validation).
- `cctv_kaggle_05_truck_collision.mp4`: Logistics expressway commercial truck collision benchmark.
- `cctv_kaggle_06_arterial_bus.mp4`: Metropolitan transit corridor bus collision benchmark.
- `cctv_kaggle_master_feed.mp4`: Multi-sector surveillance compilation benchmark.

---

## Verification & Testing

Run the automated 13-part sequence behavioral test suite:
```bash
python scripts/test_system.py
```

Tests validate:
1. Normal moving traffic maintains NORMAL status without false alarms.
2. Parked vehicles with visual bounding box overlap remain strictly NORMAL.
3. Dense queued traffic maintains steady flow without false alerts.
4. Vehicles passing in adjacent lanes maintain steady speed and trigger no collision.
5. Genuine collision sequence verified with empirical approach, overlap, and arrest.
6. Previously moving vehicle stopped in active traffic correctly flagged as TID-02.
7. Wrong-way vehicle trajectory correctly detected and flagged with TID-03.
8. Single missed detector frame does not drop track or render ghost boxes.
9. Stale track cleanly pruned after exceeding max_missed window.
10. Disconnected camera renders clean OFFLINE frame without synthetic fallback.
11. Real video feed ingests cleanly and achieves ONLINE state.
12. Stream switching cleanly terminates previous background workers and resources.
13. Strict project-wide zero emojis compliance across all source files, docs, and configs.

To measure hardware performance on your CPU:
```bash
python scripts/benchmark_system.py
```

---

## Operational Limitations

1. **Speed Metric**: Reports image-plane pixel velocity (`speed_px_s`) and relative speed index. Real-world metric velocity (km/h) requires per-camera four-point perspective homography calibration.
2. **Severe Occlusion**: Complete occlusion exceeding the tracking window (`max_missed = 3` frames) will initiate a new track upon re-emergence.
3. **Extreme Weather / Night Conditions**: Extreme glare or zero-lux footage may reduce YOLO detection confidence scores below the 0.25 threshold.
