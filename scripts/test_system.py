"""
TrafficGuard - Sequence-Level Behavioral & Architectural Test Suite
Validates 12 sequence-level behavioral scenarios and strict zero emojis compliance.
"""
import os
import sys
import time
import re
import cv2
import numpy as np

# Ensure project root in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.detector import YOLODetector
from app.tracker import KinematicTracker, TrackedVehicle, TrackState
from app.heuristics import IncidentEngine
from app.camera_manager import CameraManager, CameraStreamSource, FRAME_WIDTH, FRAME_HEIGHT
from app.database import IncidentDatabase


def make_dummy_frame():
    return np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)


def test_01_normal_moving_traffic_no_incident():
    print("[TEST 01/13] Scenario 1: Normal Moving Traffic -> No Incident...")
    tracker = KinematicTracker()
    engine = IncidentEngine(risk_threshold=0.65)
    frame = make_dummy_frame()

    for i in range(15):
        t = i * 0.1
        dets = [
            {"bbox": [50 + i * 12, 100, 110 + i * 12, 140], "conf": 0.90, "class_id": 2, "class_name": "car"},
            {"bbox": [80 + i * 14, 200, 140 + i * 14, 240], "conf": 0.92, "class_id": 2, "class_name": "car"},
            {"bbox": [20 + i * 10, 280, 90 + i * 10, 330], "conf": 0.88, "class_id": 7, "class_name": "truck"},
        ]
        tracks = tracker.update(dets, timestamp=t)
        annotated, score, status, is_inc, _ = engine.evaluate(frame, tracks, timestamp=t)

        assert not is_inc, f"Frame {i}: False incident triggered in normal traffic"
        assert status == "NORMAL", f"Frame {i}: Status was {status}, expected NORMAL"

    confirmed = [tr for tr in tracks if tr.state == TrackState.CONFIRMED]
    assert len(confirmed) == 3, f"Expected 3 confirmed tracks, got {len(confirmed)}"
    print("  [PASS] Normal moving traffic maintains NORMAL status without false alarms.")


def test_02_parked_vehicles_no_incident():
    print("[TEST 02/13] Scenario 2: Parked Vehicles (Visual Overlap) -> No Incident...")
    tracker = KinematicTracker()
    engine = IncidentEngine(risk_threshold=0.65)
    frame = make_dummy_frame()

    # Three parked cars with slight perspective overlap (IoU ~ 0.12)
    dets = [
        {"bbox": [100, 120, 160, 180], "conf": 0.89, "class_id": 2, "class_name": "car"},
        {"bbox": [150, 120, 210, 180], "conf": 0.91, "class_id": 2, "class_name": "car"},
        {"bbox": [200, 120, 260, 180], "conf": 0.87, "class_id": 2, "class_name": "car"},
    ]

    for i in range(20):
        t = i * 0.1
        tracks = tracker.update(dets, timestamp=t)
        annotated, score, status, is_inc, _ = engine.evaluate(frame, tracks, timestamp=t)

        assert not is_inc, f"Frame {i}: Parked cars triggered false incident (score: {score:.2f})"
        assert score < 0.30, f"Frame {i}: Parked cars score {score:.2f} too high"
        assert status == "NORMAL"

    print("  [PASS] Parked vehicles with visual bounding box overlap remain strictly NORMAL.")


def test_03_dense_traffic_no_incident():
    print("[TEST 03/13] Scenario 3: Dense Queued Traffic -> No Incident...")
    tracker = KinematicTracker()
    engine = IncidentEngine(risk_threshold=0.65)
    frame = make_dummy_frame()

    # 5 vehicles in a tight queue moving at steady slow speed (4 px/frame)
    for i in range(20):
        t = i * 0.1
        shift = i * 4
        dets = [
            {"bbox": [50 + shift, 150, 100 + shift, 190], "conf": 0.90, "class_id": 2, "class_name": "car"},
            {"bbox": [115 + shift, 150, 165 + shift, 190], "conf": 0.88, "class_id": 2, "class_name": "car"},
            {"bbox": [180 + shift, 150, 230 + shift, 190], "conf": 0.91, "class_id": 2, "class_name": "car"},
            {"bbox": [245 + shift, 150, 295 + shift, 190], "conf": 0.89, "class_id": 2, "class_name": "car"},
            {"bbox": [310 + shift, 150, 360 + shift, 190], "conf": 0.92, "class_id": 2, "class_name": "car"},
        ]
        tracks = tracker.update(dets, timestamp=t)
        annotated, score, status, is_inc, _ = engine.evaluate(frame, tracks, timestamp=t)

        assert not is_inc, f"Frame {i}: Dense queue triggered incident (score: {score:.2f})"
        assert status == "NORMAL"

    print("  [PASS] Dense queue traffic maintains steady flow without false alerts.")


def test_04_vehicles_passing_close_no_incident():
    print("[TEST 04/13] Scenario 4: Close Vehicle Passing Without Collision -> No Incident...")
    tracker = KinematicTracker()
    engine = IncidentEngine(risk_threshold=0.65)
    frame = make_dummy_frame()

    # Vehicle 1 moving East at 20 px/frame, Vehicle 2 moving West at 20 px/frame in adjacent lanes
    for i in range(20):
        t = i * 0.1
        v1_x = 50 + i * 20
        v2_x = 450 - i * 20
        dets = [
            {"bbox": [v1_x, 100, v1_x + 60, 140], "conf": 0.92, "class_id": 2, "class_name": "car"},
            {"bbox": [v2_x, 135, v2_x + 60, 175], "conf": 0.90, "class_id": 2, "class_name": "car"},
        ]
        tracks = tracker.update(dets, timestamp=t)
        annotated, score, status, is_inc, _ = engine.evaluate(frame, tracks, timestamp=t)

        # Even when passing closest (around frame 10), steady speed means no arrest -> no collision
        assert not is_inc, f"Frame {i}: Passing vehicles triggered collision (score: {score:.2f})"

    print("  [PASS] Vehicles passing in adjacent lanes maintain steady speed and trigger no collision.")


def test_05_genuine_collision_sequence():
    print("[TEST 05/13] Scenario 5: Genuine Collision Sequence -> Confirmed Incident...")
    tracker = KinematicTracker()
    engine = IncidentEngine(risk_threshold=0.65)
    frame = make_dummy_frame()

    # Phase 1: Rapid approach (frames 0 to 7)
    for i in range(8):
        t = i * 0.1
        dets = [
            {"bbox": [100 + i * 15, 150, 160 + i * 15, 200], "conf": 0.95, "class_id": 2, "class_name": "car"},
            {"bbox": [320 - i * 12, 150, 380 - i * 12, 200], "conf": 0.93, "class_id": 2, "class_name": "car"},
        ]
        tracks = tracker.update(dets, timestamp=t)
        engine.evaluate(frame, tracks, timestamp=t)

    # Phase 2: High impact deformation and post-impact kinetic arrest (frames 8 to 15)
    confirmed_incident_observed = False
    for i in range(8, 16):
        t = i * 0.1
        # Overlapping locked boxes (IoU ~ 0.30) with zero movement (arrest)
        dets = [
            {"bbox": [210, 150, 270, 200], "conf": 0.94, "class_id": 2, "class_name": "car"},
            {"bbox": [225, 150, 285, 200], "conf": 0.92, "class_id": 2, "class_name": "car"},
        ]
        tracks = tracker.update(dets, timestamp=t)
        annotated, score, status, is_inc, inc_data = engine.evaluate(frame, tracks, timestamp=t)

        if is_inc and status == "CONFIRMED COLLISION":
            confirmed_incident_observed = True
            assert inc_data is not None
            assert "TID-01" in inc_data["incident_code"]
            assert score >= 0.65

    assert confirmed_incident_observed, "Genuine collision failed to trigger CONFIRMED COLLISION"
    print("  [PASS] Genuine collision sequence verified with empirical approach, overlap, and arrest.")


def test_06_vehicle_suddenly_stops():
    print("[TEST 06/13] Scenario 6: Vehicle Suddenly Stops in Active Lane -> TID-02...")
    tracker = KinematicTracker()
    # Configure stopped_duration_threshold to 0.8s for fast sequence testing
    engine = IncidentEngine(stopped_duration_threshold=0.8)
    frame = make_dummy_frame()

    # Step A: Vehicle moving actively for 8 frames
    for i in range(8):
        t = i * 0.1
        dets = [{"bbox": [50 + i * 25, 150, 110 + i * 25, 190], "conf": 0.92, "class_id": 2, "class_name": "car"}]
        tracks = tracker.update(dets, timestamp=t)
        engine.evaluate(frame, tracks, timestamp=t)

    # Step B: Vehicle halts completely in the traffic lane for 20 frames (2.0 seconds)
    stop_x = 50 + 7 * 25
    triggered_tid02 = False
    for i in range(8, 28):
        t = i * 0.1
        dets = [{"bbox": [stop_x, 150, stop_x + 60, 190], "conf": 0.92, "class_id": 2, "class_name": "car"}]
        tracks = tracker.update(dets, timestamp=t)
        annotated, score, status, is_inc, inc_data = engine.evaluate(frame, tracks, timestamp=t)
        if is_inc and inc_data and "TID-02" in inc_data.get("incident_code", ""):
            triggered_tid02 = True

    assert triggered_tid02, "Previously moving vehicle stopped in traffic failed to trigger TID-02"
    print("  [PASS] Previously moving vehicle stopped in active traffic correctly flagged as TID-02.")


def test_07_wrong_way_sequence():
    print("[TEST 07/13] Scenario 7: Wrong-Way Vehicle -> TID-03...")
    tracker = KinematicTracker()
    # Configure corridor heading = 0 degrees (Eastbound flow, rightwards)
    engine = IncidentEngine(corridor_heading=0.0)
    frame = make_dummy_frame()

    # Vehicle A travels East (correct direction), Vehicle B travels West (wrong way)
    triggered_tid03 = False
    for i in range(10):
        t = i * 0.1
        dets = [
            {"bbox": [50 + i * 20, 100, 110 + i * 20, 140], "conf": 0.90, "class_id": 2, "class_name": "car"},
            {"bbox": [400 - i * 20, 200, 460 - i * 20, 240], "conf": 0.92, "class_id": 2, "class_name": "car"},
        ]
        tracks = tracker.update(dets, timestamp=t)
        annotated, score, status, is_inc, inc_data = engine.evaluate(frame, tracks, timestamp=t)
        if is_inc and inc_data and "TID-03" in inc_data.get("incident_code", ""):
            triggered_tid03 = True

    assert triggered_tid03, "Opposing vehicle failed to trigger TID-03 Wrong-Way Driver"
    print("  [PASS] Wrong-way vehicle trajectory correctly detected and flagged with TID-03.")


def test_08_detector_misses_one_frame_track_survives():
    print("[TEST 08/13] Scenario 8: Detector Misses One Frame -> Track Survives...")
    tracker = KinematicTracker()

    # 4 frames with detections to establish CONFIRMED track
    for i in range(4):
        dets = [{"bbox": [100 + i * 10, 100, 160 + i * 10, 150], "conf": 0.90, "class_id": 2, "class_name": "car"}]
        tracks = tracker.update(dets, timestamp=i * 0.1)

    assert len(tracks) == 1
    assert tracks[0].state == TrackState.CONFIRMED
    tid = tracks[0].track_id

    # Frame 5: Detector misses (empty detections)
    # The track is not rendered (zero ghosting), but survives in memory with missed_frames == 1
    tracks_after_miss = tracker.update([], timestamp=0.5)
    assert len(tracks_after_miss) == 0, "Ghost box was rendered on frame with zero detections"
    assert tid in tracker.tracks, "Track was dropped from internal state after single miss"
    assert tracker.tracks[tid].missed_frames == 1

    # Frame 6: Detector re-detects the vehicle -> re-associates same track ID
    dets = [{"bbox": [140, 100, 200, 150], "conf": 0.90, "class_id": 2, "class_name": "car"}]
    tracks_redetect = tracker.update(dets, timestamp=0.6)
    assert len(tracks_redetect) == 1
    assert tracks_redetect[0].track_id == tid, f"Expected track ID {tid} to be maintained, got {tracks_redetect[0].track_id}"
    print("  [PASS] Track survived single detector frame omission without loss or ghosting.")


def test_09_detector_misses_multiple_frames_track_removed():
    print("[TEST 09/13] Scenario 9: Detector Misses Multiple Frames -> Track Removed...")
    tracker = KinematicTracker(max_missed=3)

    # Establish confirmed track
    for i in range(4):
        dets = [{"bbox": [100 + i * 10, 100, 160 + i * 10, 150], "conf": 0.90, "class_id": 2, "class_name": "car"}]
        tracker.update(dets, timestamp=i * 0.1)

    assert len(tracker.tracks) == 1
    tid = next(iter(tracker.tracks.keys()))

    # Miss 4 frames (exceeding max_missed = 3)
    for i in range(4):
        tracker.update([], timestamp=0.4 + (i + 1) * 0.1)

    assert tid not in tracker.tracks, "Stale track was not pruned after exceeding max_missed"
    assert len(tracker.tracks) == 0
    print("  [PASS] Stale track cleanly pruned after exceeding max_missed window.")


def test_10_camera_disconnect_offline_not_simulation():
    print("[TEST 10/13] Scenario 10: Camera Disconnect -> OFFLINE, Not Simulation...")
    # Initialize camera source with non-existent URL
    invalid_cfg = {
        "id": "test_cam_invalid",
        "name": "Invalid Feed",
        "type": "file",
        "url": "non_existent_file_path_12345.mp4"
    }
    src = CameraStreamSource(invalid_cfg)
    time.sleep(0.3)

    success, frame = src.read_frame()
    src.release()

    assert not success, "Invalid camera source reported success"
    assert src.status in ["OFFLINE", "CONNECTING", "RECONNECTING"]
    assert frame is not None
    assert frame.shape == (FRAME_HEIGHT, FRAME_WIDTH, 3)

    # Verify placeholder is clean dark frame, not synthetic cartoon traffic
    mean_val = np.mean(frame)
    assert mean_val < 60.0, f"Expected dark placeholder frame, got mean intensity {mean_val:.1f}"
    print("  [PASS] Disconnected camera renders clean OFFLINE frame without synthetic fallback.")


def test_11_camera_reconnect_online():
    print("[TEST 11/13] Scenario 11: Valid Stream Ingestion -> ONLINE...")
    valid_video = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data", "test", "cctv_kaggle_04_highway_highspeed.mp4"
    )
    assert os.path.exists(valid_video), f"Benchmark video missing: {valid_video}"

    cfg = {
        "id": "test_online_cam",
        "name": "Highway Benchmark",
        "type": "file",
        "url": valid_video,
        "loop": True
    }
    src = CameraStreamSource(cfg)
    time.sleep(0.5)

    success, frame = src.read_frame()
    status = src.status
    src.release()

    assert success, "Valid video source failed to read frame"
    assert status == "ONLINE", f"Expected ONLINE status, got {status}"
    assert frame is not None
    print("  [PASS] Real video feed ingests cleanly and achieves ONLINE state.")


def test_12_stream_switching_resources_released():
    print("[TEST 12/13] Scenario 12: Stream Switching -> Resources Released Cleanly...")
    mgr = CameraManager()
    cam_keys = list(mgr.cameras.keys())
    assert len(cam_keys) >= 2, "At least 2 cameras required in catalog"

    cam1, cam2 = cam_keys[0], cam_keys[1]

    mgr.set_active_camera(cam1)
    time.sleep(0.3)
    src1 = mgr.active_source
    assert src1 is not None and src1.running

    # Switch to second camera
    mgr.set_active_camera(cam2)
    time.sleep(0.3)
    src2 = mgr.active_source

    assert not src1.running, "Previous camera source thread was not stopped upon switch"
    assert src2 is not None and src2.running
    assert mgr.active_camera_id == cam2

    mgr.release()
    print("  [PASS] Stream switching cleanly terminates previous background workers and resources.")


def test_13_strict_zero_emojis_project_wide():
    print("[TEST 13/13] Project-Wide Zero Emojis Compliance Verification...")
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Emoji Unicode range pattern
    emoji_pattern = re.compile(
        "[\U0001F600-\U0001F64F"  # emoticons
        "\U0001F300-\U0001F5FF"  # symbols & pictographs
        "\U0001F680-\U0001F6FF"  # transport & map
        "\U0001F1E0-\U0001F1FF"  # flags
        "\U00002702-\U000027B0"  # dingbats
        "\U0001F900-\U0001F9FF"  # supplemental symbols
        "\U0001FA70-\U0001FAFF"  # symbols and pictographs extended-a
        "\U00002600-\U000026FF]"  # misc symbols
    )

    checked_extensions = {".py", ".yaml", ".yml", ".md", ".html", ".js", ".json"}
    ignore_dirs = {".git", ".idea", "__pycache__", "venv", ".gemini", "scratch"}

    violations = []
    files_scanned = 0

    for root, dirs, files in os.walk(repo_root):
        dirs[:] = [d for d in dirs if d not in ignore_dirs]
        for f in files:
            ext = os.path.splitext(f)[1].lower()
            if ext in checked_extensions:
                filepath = os.path.join(root, f)
                files_scanned += 1
                try:
                    with open(filepath, "r", encoding="utf-8", errors="ignore") as file:
                        for line_no, line in enumerate(file, start=1):
                            if emoji_pattern.search(line):
                                violations.append((os.path.relpath(filepath, repo_root), line_no, line.strip()))
                except Exception:
                    pass

    if violations:
        print(f"  FAILED: Found {len(violations)} emoji violations across {files_scanned} files:")
        for v in violations[:10]:
            print(f"    {v[0]}:{v[1]} -> {v[2]}")
        assert False, f"Strict zero emojis compliance violated in {len(violations)} lines"

    print(f"  [PASS] Scanned {files_scanned} project files: strictly ZERO emojis found.")


def main():
    print("=" * 70)
    print("  TRAFFICGUARD - SEQUENCE BEHAVIORAL & ARCHITECTURAL TEST SUITE")
    print("  12 Sequence Tests + Strict Zero Emojis Validation")
    print("=" * 70)

    tests = [
        test_01_normal_moving_traffic_no_incident,
        test_02_parked_vehicles_no_incident,
        test_03_dense_traffic_no_incident,
        test_04_vehicles_passing_close_no_incident,
        test_05_genuine_collision_sequence,
        test_06_vehicle_suddenly_stops,
        test_07_wrong_way_sequence,
        test_08_detector_misses_one_frame_track_survives,
        test_09_detector_misses_multiple_frames_track_removed,
        test_10_camera_disconnect_offline_not_simulation,
        test_11_camera_reconnect_online,
        test_12_stream_switching_resources_released,
        test_13_strict_zero_emojis_project_wide
    ]

    passed = 0
    start_t = time.time()

    for t in tests:
        try:
            t()
            passed += 1
        except AssertionError as e:
            print(f"\n  [FAIL] {t.__name__}: {e}\n")
            sys.exit(1)
        except Exception as e:
            print(f"\n  [ERROR] {t.__name__} raised unexpected exception: {e}\n")
            import traceback
            traceback.print_exc()
            sys.exit(1)

    dur = time.time() - start_t
    print("=" * 70)
    print(f"  ALL {passed}/{len(tests)} TESTS PASSED SUCCESSFULLY in {dur:.2f}s")
    print("=" * 70)


if __name__ == "__main__":
    main()
