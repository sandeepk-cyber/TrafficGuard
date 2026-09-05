"""
TrafficGuard - Kaggle CCTV Dataset Segmenter and Ingest Preparer
Extracts and packages discrete CCTV surveillance clips from fahaddalwai/cctvfootagevideo.
Strict zero emojis, high-fidelity H.264/MP4 encoding.
"""
import os
import sys
import shutil
import cv2

KAGGLE_SOURCE_PATH = r"C:\Users\pranj\.cache\kagglehub\datasets\fahaddalwai\cctvfootagevideo\versions\1\videoplayback (online-video-cutter.com).mp4"
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "test")

SCENE_DEFINITIONS = [
    {
        "id": "cctv_kaggle_01_urban_motorcycle",
        "name": "Kaggle CCTV 01 - Urban Motorcycle Conflict",
        "start_frame": 0,
        "end_frame": 258,
        "description": "Urban junction CCTV monitoring motorcycle and pedestrian conflict."
    },
    {
        "id": "cctv_kaggle_02_junction_tbone",
        "name": "Kaggle CCTV 02 - Arterial Junction T-Bone",
        "start_frame": 258,
        "end_frame": 619,
        "description": "Multi-lane arterial intersection CCTV capturing high-angle vehicle collision."
    },
    {
        "id": "cctv_kaggle_03_dense_crossroads",
        "name": "Kaggle CCTV 03 - Dense Urban Crossroads",
        "start_frame": 619,
        "end_frame": 1142,
        "description": "Dense urban commercial crossroads with high-density parked and moving vehicles."
    },
    {
        "id": "cctv_kaggle_04_highway_highspeed",
        "name": "Kaggle CCTV 04 - Elevated Highway Crash",
        "start_frame": 1142,
        "end_frame": 1742,
        "description": "Elevated highway pole CCTV capturing high-speed kinetic vehicle collision."
    },
    {
        "id": "cctv_kaggle_05_truck_collision",
        "name": "Kaggle CCTV 05 - Commercial Truck Collision",
        "start_frame": 1742,
        "end_frame": 1982,
        "description": "Industrial corridor CCTV monitoring commercial truck and passenger vehicle impact."
    },
    {
        "id": "cctv_kaggle_06_arterial_bus",
        "name": "Kaggle CCTV 06 - Arterial Bus Corridor",
        "start_frame": 1982,
        "end_frame": 2298,
        "description": "Transit arterial CCTV monitoring bus and passenger vehicle collision."
    }
]


def segment_dataset():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    if not os.path.exists(KAGGLE_SOURCE_PATH):
        print(f"Error: Kaggle source file not found at {KAGGLE_SOURCE_PATH}")
        sys.exit(1)

    cap = cv2.VideoCapture(KAGGLE_SOURCE_PATH)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Processing Kaggle CCTV dataset: {width}x{height} @ {fps:.1f} FPS, {total_frames} frames total.")

    # 1. Copy complete master compilation feed
    master_dest = os.path.join(OUTPUT_DIR, "cctv_kaggle_master_feed.mp4")
    shutil.copyfile(KAGGLE_SOURCE_PATH, master_dest)
    print(f"Exported master compilation: {master_dest} ({os.path.getsize(master_dest)} bytes)")

    # 2. Segment individual CCTV camera clips
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    for scene in SCENE_DEFINITIONS:
        out_filename = f"{scene['id']}.mp4"
        out_path = os.path.join(OUTPUT_DIR, out_filename)
        start_f = scene["start_frame"]
        end_f = min(scene["end_frame"], total_frames)

        cap.set(cv2.CAP_PROP_POS_FRAMES, start_f)
        writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

        frames_written = 0
        for _ in range(start_f, end_f):
            ret, frame = cap.read()
            if not ret:
                break
            writer.write(frame)
            frames_written += 1

        writer.release()
        size_bytes = os.path.getsize(out_path)
        print(f"Segmented {scene['name']} -> {out_filename} ({frames_written} frames, {size_bytes} bytes)")

    cap.release()
    print("Dataset segmentation completed successfully.")


if __name__ == "__main__":
    segment_dataset()
