# TrafficGuard - Enterprise Automatic Incident Detection (ITS-AID)

TrafficGuard is a high-performance, low-resource computer vision pipeline and Video Management System (VMS) console designed to ingest authentic elevated CCTV traffic camera feeds (HLS, RTSP, MJPEG, or local CCTV video), detect vehicles, track multi-object kinematics without ghosting, evaluate standardized ITS incident types (TID-01 through TID-04), and trigger CAD emergency dispatches.

---

## Quick Start

### 1. Run CCTV Incident Benchmark (Default)
Runs the computer vision pipeline and VMS console against the 1080p highway collision benchmark:
```bash
python run.py
```
Or specify explicit CCTV footage:
```bash
python run.py --video data/test/accident_cctv.webm
```

### 2. Run with Live CCTV Feed
Connects to the CCTV stream configured in `config/cameras.yaml` (e.g. Caltrans Live HLS):
```bash
python run.py --live
```

---

## Operator VMS Console

Open your browser at:
**[http://127.0.0.1:8001](http://127.0.0.1:8001)**

### Enterprise Features:
- **Zero Ghosting Kinematic Tracking**: Trajectory smoothing and keyframe extrapolation with active-only rendering and boundary purges.
- **Persistent Latched Red Collision Visuals**: Vehicles involved in collisions remain highlighted in vivid solid red (BGR 0, 0, 255) with 3px reinforced corner brackets, collision reticles, and impact vectors for 7 seconds.
- **Commercial ITS Corridor Analytics**: Real-time traffic flow rate (VPM), Level of Service (LOS A through F) corridor density gauge, average velocity, and collision threat index.
- **Operator Incident Triage Desk**: Rapid review with verification, false alarm triage, and official CAD dispatch ticket generation.
- **Layer Overlays Toggle**: Dynamic toggles for Bounding Boxes, Velocity Vectors, Trajectories, Collision Reticles, and HUD.
- **Strictly CCTV Feeds**: Zero dashcams and zero synthetic road animations.

---

## Authentic Kaggle CCTV Dataset & Incident Channels

TrafficGuard integrates the authentic surveillance benchmark dataset [Accident Footages From CCTV](https://www.kaggle.com/datasets/fahaddalwai/cctvfootagevideo) (`fahaddalwai/cctvfootagevideo`). The dataset has been losslessly segmented into six discrete surveillance feeds under `data/test/`:

- `cctv_kaggle_01_urban_motorcycle.mp4`: Downtown Market Street corridor motorcycle conflict.
- `cctv_kaggle_02_junction_tbone.mp4`: 4th & Commercial arterial junction high-energy T-bone impact.
- `cctv_kaggle_03_dense_crossroads.mp4`: Dense urban retail crossing & parking sector (used for zero-false-alarm validation).
- `cctv_kaggle_04_highway_highspeed.mp4`: Elevated highway pole camera high-speed kinetic crash (primary test channel).
- `cctv_kaggle_05_truck_collision.mp4`: Logistics expressway terminal heavy vehicle impact.
- `cctv_kaggle_06_arterial_bus.mp4`: Metropolitan transit corridor bus collision.
- `cctv_kaggle_master_feed.mp4`: Continuous multi-sector surveillance compilation feed (76.6s).

To re-segment or regenerate test clips from Kaggle:
```bash
python tools/segment_kaggle_dataset.py
```

---

## Configuration

Edit `config/cameras.yaml` to configure CCTV camera feeds:

```yaml
cameras:
  cctv_kaggle_04_highway:
    name: Kaggle CCTV - Elevated Highway High-Speed Crash
    type: file
    url: data/test/cctv_kaggle_04_highway_highspeed.mp4
    location: Elevated Highway Pole Cam KM 24.8
  cctv_kaggle_02_junction:
    name: Kaggle CCTV - Arterial Junction T-Bone Collision
    type: file
    url: data/test/cctv_kaggle_02_junction_tbone.mp4
    location: 4th & Commercial Arterial Junction
  caltrans_live_cctv:
    name: Caltrans Live Traffic Feed (IPTV HLS)
    type: hls
    url: https://cph-p2p-msl.akamaized.net/hls/live/200034/test/master.m3u8
    location: State Route 99 Traffic Corridor
```

---

## Automated Verification

Run the automated 8-part verification test suite:
```bash
python scripts/test_system.py
```

Test suite validates:
1. YOLO Deep Learning Detector with Soft-IoM containment suppression
2. Kaggle & Enterprise CCTV Video Feeds Integrity
3. Kinematic Tracker for Zero Ghosting and Velocity Stability
4. Persistent 7-Second Red Collision Latching
5. Dense Urban Parking False-Positive Immunity (0 false alarms)
6. Positive Crash Detection on Real CCTV (Scene 4 Highway)
7. Commercial ITS Analytics (VPM, LOS, Threat Index) & CAD Dispatch
8. Project-Wide Strict Zero Emojis Compliance
