"""
TrafficGuard - Performance & Resource Benchmarking Script
Measures empirical CPU latency, detection throughput, tracking overhead, and memory usage.
Zero emojis, strict industrial benchmarking.
"""
import os
import sys
import time
import cv2
import psutil
import numpy as np

# Ensure project root in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.detector import YOLODetector
from app.tracker import KinematicTracker
from app.heuristics import IncidentEngine
from app.camera_manager import FRAME_WIDTH, FRAME_HEIGHT

BENCHMARK_VIDEO = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "test", "cctv_kaggle_04_highway_highspeed.mp4"
)
NUM_FRAMES = 100


def run_benchmark():
    print("=" * 70)
    print("  TRAFFICGUARD - SYSTEM PERFORMANCE & RESOURCE BENCHMARK")
    print("  Local Low-Resource Computer Vision Pipeline (CPU)")
    print("=" * 70)

    if not os.path.exists(BENCHMARK_VIDEO):
        print(f"Error: Benchmark video not found at {BENCHMARK_VIDEO}")
        sys.exit(1)

    process = psutil.Process(os.getpid())
    mem_start = process.memory_info().rss / (1024 * 1024)

    print(f"  Benchmark Target: {os.path.basename(BENCHMARK_VIDEO)}")
    print(f"  Processing Resolution: {FRAME_WIDTH}x{FRAME_HEIGHT}")
    print(f"  Test Sample: {NUM_FRAMES} consecutive frames")
    print(f"  CPU Cores Available: {psutil.cpu_count(logical=True)} (Physical: {psutil.cpu_count(logical=False)})")
    print("-" * 70)

    # Initialize subsystems
    init_t0 = time.perf_counter()
    detector = YOLODetector(conf_threshold=0.20, nms_threshold=0.40)
    tracker = KinematicTracker()
    engine = IncidentEngine()
    init_dur = (time.perf_counter() - init_t0) * 1000.0

    print(f"  Pipeline Initialization Time : {init_dur:.1f} ms")
    print(f"  Detector Model Format        : ONNX {'FP16' if detector.is_fp16 else 'FP32'}")

    cap = cv2.VideoCapture(BENCHMARK_VIDEO)
    frames = []
    for _ in range(NUM_FRAMES):
        ret, f = cap.read()
        if not ret or f is None:
            break
        frames.append(cv2.resize(f, (FRAME_WIDTH, FRAME_HEIGHT)))
    cap.release()

    actual_frames = len(frames)
    print(f"  Ingested Frame Buffer        : {actual_frames} frames")

    # Metrics storage
    detector_times = []
    tracker_times = []
    heuristics_times = []
    e2e_times = []
    detected_counts = []

    # Warmup
    if frames:
        _ = detector.detect(frames[0])

    print("\n  Executing benchmark iterations...")
    psutil.cpu_percent(interval=None)  # Reset CPU measurement window
    bench_start = time.perf_counter()

    for idx, frame in enumerate(frames):
        t0 = time.perf_counter()

        # 1. Detection
        t_det0 = time.perf_counter()
        dets, det_ms = detector.detect(frame)
        t_det1 = time.perf_counter()
        detector_times.append((t_det1 - t_det0) * 1000.0)
        detected_counts.append(len(dets))

        # 2. Tracking
        t_trk0 = time.perf_counter()
        tracks = tracker.update(dets, timestamp=time.time())
        t_trk1 = time.perf_counter()
        tracker_times.append((t_trk1 - t_trk0) * 1000.0)

        # 3. Incident Evaluation
        t_heu0 = time.perf_counter()
        annotated, score, status, is_inc, _ = engine.evaluate(frame, tracks)
        t_heu1 = time.perf_counter()
        heuristics_times.append((t_heu1 - t_heu0) * 1000.0)

        t1 = time.perf_counter()
        e2e_times.append((t1 - t0) * 1000.0)

    bench_dur = time.perf_counter() - bench_start
    overall_fps = actual_frames / bench_dur if bench_dur > 0 else 0.0
    cpu_usage = psutil.cpu_percent(interval=None)
    mem_end = process.memory_info().rss / (1024 * 1024)

    # Compute statistics
    def stats(arr):
        s = sorted(arr)
        avg = sum(s) / len(s)
        p50 = s[int(len(s) * 0.50)]
        p95 = s[min(int(len(s) * 0.95), len(s) - 1)]
        return avg, p50, p95

    det_avg, det_p50, det_p95 = stats(detector_times)
    trk_avg, trk_p50, trk_p95 = stats(tracker_times)
    heu_avg, heu_p50, heu_p95 = stats(heuristics_times)
    e2e_avg, e2e_p50, e2e_p95 = stats(e2e_times)

    print("\n" + "=" * 70)
    print("  BENCHMARK RESULTS SUMMARY")
    print("=" * 70)
    print(f"  {'STAGE':<28} | {'AVG (ms)':<10} | {'P50 (ms)':<10} | {'P95 (ms)':<10}")
    print("  " + "-" * 66)
    print(f"  {'1. YOLO Object Detection':<28} | {det_avg:<10.2f} | {det_p50:<10.2f} | {det_p95:<10.2f}")
    print(f"  {'2. Hungarian Tracking':<28} | {trk_avg:<10.2f} | {trk_p50:<10.2f} | {trk_p95:<10.2f}")
    print(f"  {'3. Incident Heuristics':<28} | {heu_avg:<10.2f} | {heu_p50:<10.2f} | {heu_p95:<10.2f}")
    print("  " + "-" * 66)
    print(f"  {'End-to-End Pipeline':<28} | {e2e_avg:<10.2f} | {e2e_p50:<10.2f} | {e2e_p95:<10.2f}")
    print("=" * 70)
    print(f"  Effective Pipeline Throughput : {overall_fps:.1f} FPS")
    print(f"  Average Objects per Frame     : {sum(detected_counts)/len(detected_counts):.1f}")
    print(f"  Process CPU Utilization       : {cpu_usage:.1f}%")
    print(f"  Memory Footprint (RSS)        : {mem_end:.1f} MB (Delta: +{mem_end - mem_start:.1f} MB)")
    print("=" * 70)

    # Verification of performance envelope
    assert det_avg < 120.0, f"Detector average latency {det_avg:.1f}ms exceeds 120ms CPU budget"
    assert trk_avg < 5.0, f"Tracking average latency {trk_avg:.1f}ms exceeds 5ms budget"
    assert heu_avg < 5.0, f"Heuristics average latency {heu_avg:.1f}ms exceeds 5ms budget"
    print("\n  [PASS] All performance latency envelopes satisfied.")


if __name__ == "__main__":
    run_benchmark()
