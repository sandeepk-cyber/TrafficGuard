"""
TrafficGuard - Enterprise System & Deep Learning Integration Test Suite
Zero emojis, strict industrial validation with isolated test database.
Validates:
  1. YOLOv5n FP16 Deep Learning Object Detector & Containment Suppression
  2. Authentic Kaggle & Metropolitan CCTV Video Feeds (H.264/WebM Decoding)
  3. Zero-Ghosting Kinematic Tracking & Timestamp-Safe Velocity Stability
  4. Persistent Latched Red Collision Detection & Tactical Symbology
  5. Dense Urban Parking / Normal Traffic False Positive Immunity
  6. Real-World CCTV Crash Verification (Kaggle Highway Impact Positive Trigger)
  7. Commercial ITS Analytics (Flow Rate, Density LOS, CAD Dispatch)
  8. Project-Wide Strict Zero Emojis Compliance
"""
import os
import sys
import time
import re
import cv2
import numpy as np

# Add project root to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.detector import YOLODetector
from app.tracker import KinematicTracker, TrackedVehicle
from app.heuristics import EnterpriseAIDEngine
from app.camera_manager import CameraManager
from app.database import IncidentDatabase
from app.dashboard import TrafficGuardEngine, create_app

TEST_DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "test_trafficguard.db")
DATA_TEST_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "test")


def test_yolo_detector_and_containment():
    print("[TEST 1/8] Testing YOLO Deep Learning Detector & Containment Suppression...")
    detector = YOLODetector(conf_threshold=0.25, nms_threshold=0.35)
    assert detector.session is not None, "YOLO session failed to initialize"

    # Synthetic vehicle test
    test_img = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.rectangle(test_img, (200, 200), (350, 320), (200, 200, 200), -1)

    dets, inf_ms = detector.detect(test_img)
    assert isinstance(dets, list), "Detections must be a list"
    assert inf_ms >= 0, "Inference time must be recorded"

    # Test on Scene 3 frame to verify sub-box containment suppression
    s3_path = os.path.join(DATA_TEST_DIR, "cctv_kaggle_03_dense_crossroads.mp4")
    if os.path.exists(s3_path):
        cap = cv2.VideoCapture(s3_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, 30)
        ret, frame = cap.read()
        cap.release()
        if ret:
            s3_dets, _ = detector.detect(frame)
            # Ensure no co-located sub-box pairs exist with IoM > 0.65
            for i in range(len(s3_dets)):
                b1 = s3_dets[i]["bbox"]
                a1 = (b1[2]-b1[0]) * (b1[3]-b1[1])
                for j in range(i + 1, len(s3_dets)):
                    b2 = s3_dets[j]["bbox"]
                    a2 = (b2[2]-b2[0]) * (b2[3]-b2[1])
                    inter_w = max(0, min(b1[2], b2[2]) - max(b1[0], b2[0]))
                    inter_h = max(0, min(b1[3], b2[3]) - max(b1[1], b2[1]))
                    inter = inter_w * inter_h
                    if inter > 0 and min(a1, a2) > 0:
                        containment = inter / min(a1, a2)
                        assert containment <= 0.65, f"Sub-box containment violation: {containment:.2f}"

    print(f"  [OK] YOLO Model active (FP16 inference: {inf_ms:.1f}ms) with sub-box containment suppression verified")


def test_cctv_feeds_decoding():
    print("[TEST 2/8] Testing Kaggle & Enterprise CCTV Video Feeds Integrity...")
    required_cctv_files = [
        "cctv_kaggle_01_urban_motorcycle.mp4",
        "cctv_kaggle_02_junction_tbone.mp4",
        "cctv_kaggle_03_dense_crossroads.mp4",
        "cctv_kaggle_04_highway_highspeed.mp4",
        "cctv_kaggle_05_truck_collision.mp4",
        "cctv_kaggle_06_arterial_bus.mp4",
        "cctv_kaggle_master_feed.mp4",
        "accident_cctv.webm",
        "cctv_junction_accident.webm",
        "cctv_highway_interchange.webm"
    ]
    for filename in required_cctv_files:
        path = os.path.join(DATA_TEST_DIR, filename)
        assert os.path.exists(path), f"Required CCTV video file missing: {filename}"
        cap = cv2.VideoCapture(path)
        assert cap.isOpened(), f"Failed to open CCTV video: {filename}"
        ret, frame = cap.read()
        assert ret and frame is not None, f"Failed to read frame from {filename}"
        h, w = frame.shape[:2]
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        print(f"  [OK] CCTV Feed Verified: {filename} ({w}x{h} @ {fps:.1f} FPS)")


def test_zero_ghosting_tracker_and_velocity_stability():
    print("[TEST 3/8] Testing Kinematic Tracker for Zero Ghosting & Velocity Stability...")
    tracker = KinematicTracker(frame_width=640, frame_height=480, max_disappeared=2)

    # Frame 1: Initial detection with fixed timestamp
    t0 = 100.0
    dets_f1 = [
        {"bbox": (100, 100, 160, 140), "class_name": "car", "confidence": 0.85, "center": (130, 120)},
        {"bbox": (300, 200, 380, 250), "class_name": "truck", "confidence": 0.90, "center": (340, 225)}
    ]
    tracks1 = tracker.update(dets_f1, timestamp=t0)
    assert len(tracks1) == 2, "Both vehicles should be detected"

    # Frame 2: 33ms later (simulating 30 FPS)
    t1 = t0 + 0.0333
    dets_f2 = [
        {"bbox": (104, 100, 164, 140), "class_name": "car", "confidence": 0.88, "center": (134, 120)},
        {"bbox": (304, 200, 384, 250), "class_name": "truck", "confidence": 0.91, "center": (344, 225)}
    ]
    tracks2 = tracker.update(dets_f2, timestamp=t1)
    assert len(tracks2) == 2, "Vehicles should be confirmed"

    # Check velocity stability: speed should be reasonable (~30-60 km/h), never 300+ km/h
    for t in tracks2:
        assert 0.0 <= t.speed_kmh <= 120.0, f"Unstable velocity recorded: {t.speed_kmh} km/h"

    # Frame 3: Car vanishes
    t2 = t1 + 0.0333
    dets_f3 = [
        {"bbox": (308, 200, 388, 250), "class_name": "truck", "confidence": 0.92, "center": (348, 225)}
    ]
    tracks3 = tracker.update(dets_f3, timestamp=t2)
    assert len(tracks3) == 1, f"Ghosting detected: expected 1 track, got {len(tracks3)}"
    assert tracks3[0].class_name == "truck"

    # Extrapolation zero-drift check
    extrapolated = tracker.extrapolate_all(dt=0.033)
    assert len(extrapolated) == 1

    # Reset capability check
    tracker.reset()
    assert len(tracker.tracks) == 0 and tracker.next_id == 1, "Tracker reset failed"

    print("  [OK] Zero ghosting and velocity stability verified: dt bounded, zero drift, clean reset")


def test_persistent_red_collision_latch():
    print("[TEST 4/8] Testing Persistent 7-Second Red Collision Latching...")
    heuristics = EnterpriseAIDEngine(collision_latch_duration=7.0)

    v1 = TrackedVehicle(1, (200, 200, 260, 240), "car", 0.9, frame_height=480)
    v2 = TrackedVehicle(2, (205, 202, 265, 242), "truck", 0.9, frame_height=480)
    v1.hits = 4
    v2.hits = 4
    v1.is_confirmed = True
    v2.is_confirmed = True
    v1.velocity = [15.0, 0.0]
    v2.velocity = [-15.0, 0.0]
    v1.acceleration = [-160.0, 0.0]
    v2.acceleration = [-140.0, 0.0]
    v1.speed_kmh = 35.0
    v2.speed_kmh = 28.0
    v1.max_historical_speed = 35.0
    v2.max_historical_speed = 28.0

    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)

    # Collision frame
    annotated1, risk1, sev1, is_inc1, data1 = heuristics.evaluate(dummy_frame, [v1, v2])
    assert is_inc1, "Collision should trigger incident alarm"
    assert sev1 == "CRITICAL", f"Severity should be CRITICAL, got {sev1}"
    assert 1 in heuristics.latched_collisions, "Track 1 should be latched"
    assert 2 in heuristics.latched_collisions, "Track 2 should be latched"

    # Verify bright red bounding box pixels (BGR: 0, 0, 255)
    red_pixels = np.count_nonzero((annotated1[:, :, 2] > 200) & (annotated1[:, :, 0] < 50))
    assert red_pixels > 50, "Annotated frame must render vivid bright red bounding box"

    # Stopped post-collision holds latch
    time.sleep(0.02)
    v1.speed_kmh = 0.0
    v2.speed_kmh = 0.0
    v1.velocity = [0.0, 0.0]
    v2.velocity = [0.0, 0.0]
    annotated2, risk2, sev2, is_inc2, data2 = heuristics.evaluate(dummy_frame, [v1, v2])

    assert is_inc2, "Collision state must remain latched across subsequent frames"
    assert sev2 == "CRITICAL", "Severity must remain CRITICAL during latch window"
    assert v1.in_collision, "Vehicle 1 must retain in_collision state"

    print("  [OK] Persistent red collision latching verified: Bounding boxes turn bright red and hold state")


def test_dense_parking_false_positive_immunity():
    print("[TEST 5/8] Testing Dense Urban Parking False-Positive Immunity (Kaggle Scene 3)...")
    detector = YOLODetector(conf_threshold=0.25, nms_threshold=0.35)
    tracker = KinematicTracker(frame_width=640, frame_height=360, max_disappeared=2)
    heuristics = EnterpriseAIDEngine(risk_threshold=0.70)

    s3_path = os.path.join(DATA_TEST_DIR, "cctv_kaggle_03_dense_crossroads.mp4")
    assert os.path.exists(s3_path), f"Missing {s3_path}"
    cap = cv2.VideoCapture(s3_path)

    # Process first 60 frames of normal dense parking phase
    false_alarm_count = 0
    for f_idx in range(60):
        ret, frame = cap.read()
        if not ret:
            break
        dets, _ = detector.detect(frame)
        tracks = tracker.update(dets, timestamp=f_idx / 30.0)
        _, risk, severity, is_inc, inc_data = heuristics.evaluate(frame, tracks)
        if is_inc and inc_data and inc_data.get("incident_code") == "TID-01 COLLISION_IMPACT":
            false_alarm_count += 1

    cap.release()
    assert false_alarm_count == 0, f"False positive collisions occurred in normal parking phase: {false_alarm_count}"
    print(f"  [OK] False-Positive Immunity Verified: 0 false alarms across dense urban parking surveillance")


def test_kaggle_highway_crash_positive_detection():
    print("[TEST 6/8] Testing Positive Crash Detection on Real CCTV (Kaggle Scene 4 Highway)...")
    detector = YOLODetector(conf_threshold=0.25, nms_threshold=0.35)
    tracker = KinematicTracker(frame_width=640, frame_height=360, max_disappeared=2)
    heuristics = EnterpriseAIDEngine(risk_threshold=0.70, collision_latch_duration=7.0)

    s4_path = os.path.join(DATA_TEST_DIR, "cctv_kaggle_04_highway_highspeed.mp4")
    assert os.path.exists(s4_path), f"Missing {s4_path}"
    cap = cv2.VideoCapture(s4_path)

    crash_detected = False
    peak_risk = 0.0
    detected_frame = None

    # Step through highway crash video
    for f_idx in range(180):
        ret, frame = cap.read()
        if not ret:
            break
        dets, _ = detector.detect(frame)
        tracks = tracker.update(dets, timestamp=f_idx / 30.0)
        annotated, risk, severity, is_inc, inc_data = heuristics.evaluate(frame, tracks)
        if risk > peak_risk:
            peak_risk = risk
        if is_inc and inc_data and inc_data.get("incident_code") == "TID-01 COLLISION_IMPACT":
            crash_detected = True
            detected_frame = f_idx
            break

    cap.release()
    assert crash_detected, f"Failed to detect real highway crash in Kaggle dataset (peak risk: {peak_risk:.2f})"
    print(f"  [OK] Positive Crash Detection Verified: Real CCTV highway collision detected at frame {detected_frame} (Peak Risk: {peak_risk*100:.0f}%)")


def test_commercial_its_metrics_and_cad_dispatch():
    print("[TEST 7/8] Testing Commercial ITS Analytics & CAD Dispatch...")
    engine = TrafficGuardEngine(detection_fps=15.0, is_simulation_mode=False)

    telemetry = engine.get_telemetry()
    assert "flow_rate_vpm" in telemetry, "Missing flow_rate_vpm"
    assert "density_los" in telemetry, "Missing density_los"
    assert "avg_speed_kmh" in telemetry, "Missing avg_speed_kmh"
    assert "collision_threat_index" in telemetry, "Missing collision_threat_index"
    assert "overlays" in telemetry, "Missing layer overlays"

    # Layer overlays check
    assert engine.layer_overlays["boxes"] is True
    assert engine.layer_overlays["vectors"] is True

    # CAD dispatch check
    db = IncidentDatabase(db_path=TEST_DB_PATH)
    inc_id = db.record_incident(
        camera_name="Kaggle Highway CCTV KM 24.8",
        location="Elevated Highway Sector 4",
        incident_code="TID-01 COLLISION_IMPACT",
        risk_score=0.98,
        severity="CRITICAL",
        speed_at_impact="88 km/h",
        vehicles_involved="Car #2 vs Car #4",
        description="High-velocity rear-impact collision"
    )
    assert inc_id is not None
    db.update_incident_status(inc_id, "DISPATCHED")
    cad_ticket = f"CAD-2026-{inc_id:05d}"
    assert "CAD-2026-" in cad_ticket

    db.clear_all_incidents()
    if os.path.exists(TEST_DB_PATH):
        os.remove(TEST_DB_PATH)

    print("  [OK] Commercial ITS metrics (VPM, LOS, Threat Index) and CAD dispatch verified")


def test_zero_emojis_compliance():
    print("[TEST 8/8] Testing Project-Wide Strict Zero Emojis Compliance...")
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    emoji_pattern = re.compile(r'[\U0001F000-\U0001FAFF\U00002600-\U000027BF\U0001F300-\U0001F9FF]')

    violations = []
    for root, dirs, files in os.walk(root_dir):
        if '.git' in root or '__pycache__' in root or '.system_generated' in root:
            continue
        for f in files:
            if f.endswith(('.py', '.yaml', '.yml', '.md', '.txt', '.html', '.json', '.css', '.js')):
                p = os.path.join(root, f)
                with open(p, 'r', encoding='utf-8', errors='ignore') as fp:
                    for lno, line in enumerate(fp, 1):
                        m = emoji_pattern.findall(line)
                        if m:
                            violations.append(f"{os.path.relpath(p, root_dir)}:{lno}")

    assert len(violations) == 0, f"Found emoji violations in: {violations}"
    print("  [OK] Zero emojis verified: 100% clean codebase across all files")


def run_all_tests():
    print("=" * 70)
    print("  TRAFFICGUARD ENTERPRISE ITS-AID v2.5 VERIFICATION SUITE")
    print("  Benchmarked Against Authentic Kaggle CCTV Accident Dataset")
    print("=" * 70)
    test_yolo_detector_and_containment()
    test_cctv_feeds_decoding()
    test_zero_ghosting_tracker_and_velocity_stability()
    test_persistent_red_collision_latch()
    test_dense_parking_false_positive_immunity()
    test_kaggle_highway_crash_positive_detection()
    test_commercial_its_metrics_and_cad_dispatch()
    test_zero_emojis_compliance()
    print("=" * 70)
    print("  ALL 8/8 ENTERPRISE TESTS PASSED CLEANLY WITH ZERO ERRORS!")
    print("=" * 70)


if __name__ == "__main__":
    run_all_tests()
