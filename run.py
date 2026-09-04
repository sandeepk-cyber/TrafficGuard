#!/usr/bin/env python3
"""
TrafficGuard - Real-Time Accident Detection MVP
Main CLI runner supporting Live Stream, Local MP4 Video, and Simulation modes.
"""
import os
import sys
import argparse
import logging
import uvicorn

from app.main import load_camera_config
from app.dashboard import TrafficGuardEngine, create_app
from app.video import FRAME_WIDTH, FRAME_HEIGHT, DETECTION_FPS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("trafficguard")


def parse_args():
    parser = argparse.ArgumentParser(
        description="TrafficGuard - Real-Time Traffic Accident Detection MVP",
        formatter_class=argparse.RawTextHelpFormatter
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--live",
        action="store_true",
        help="Connect to the public live traffic camera stream configured in config/cameras.yaml"
    )
    group.add_argument(
        "--video",
        type=str,
        metavar="PATH",
        help="Process a local MP4 video file fallback (e.g. --video data/test/accident.mp4)"
    )
    group.add_argument(
        "--simulate",
        action="store_true",
        help="Run in simulation mode (triggers a test incident after 5s for dashboard testing)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8001,
        help="Dashboard web server port (default: 8001)"
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Dashboard web server host (default: 127.0.0.1)"
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=DETECTION_FPS,
        help=f"Detection processing rate in FPS (default: {DETECTION_FPS})"
    )
    return parser.parse_args()


def print_banner(mode: str, cam_name: str, cam_url: str, port: int):
    print("=" * 70)
    print("  TRAFFICGUARD - REAL-TIME ACCIDENT DETECTION MVP")
    print("  Local Low-Resource Computer Vision Pipeline (CPU)")
    print("=" * 70)
    print(f"  MODE         : {mode}")
    print(f"  CAMERA NAME  : {cam_name}")
    print(f"  STREAM URL   : {cam_url if cam_url else '(None - Waiting for feed)'}")
    print(f"  RESOLUTION   : {FRAME_WIDTH}x{FRAME_HEIGHT}")
    print(f"  DETECTION FPS: ~{DETECTION_FPS} FPS (Hardware Throttled)")
    print(f"  DASHBOARD    : http://127.0.0.1:{port}")
    print("=" * 70)
    print("  Press Ctrl+C to stop.\n")


def main():
    args = parse_args()

    # Determine mode & camera source
    is_simulation = False
    mode_label = "DEFAULT"

    if args.simulate:
        is_simulation = True
        mode_label = "SIMULATION (Test Incident in 5s)"
        camera_config = {
            "name": "Live Traffic Camera (Simulated)",
            "type": "simulation",
            "url": "",
            "latitude": 37.7749,
            "longitude": -122.4194,
            "location": "Simulated Metropolitan Corridor"
        }
    elif args.video:
        mode_label = f"LOCAL VIDEO FILE ({args.video})"
        if not os.path.exists(args.video):
            logger.error("Video file not found: %s", args.video)
            sys.exit(1)
        camera_config = {
            "name": "Local Test Video",
            "type": "file",
            "url": args.video,
            "latitude": 34.0522,
            "longitude": -118.2437,
            "location": f"Local File: {os.path.basename(args.video)}"
        }
    elif args.live:
        mode_label = "LIVE PUBLIC TRAFFIC CAMERA"
        camera_config = load_camera_config()
        if not camera_config.get("url"):
            logger.warning(
                "No live URL specified in config/cameras.yaml.\n"
                "Please edit config/cameras.yaml and add a working public HLS (.m3u8), MJPEG, or RTSP stream.\n"
                "Directories: https://opencctv.org/cameras/traffic or https://trafficvision.live/map\n"
            )
    else:
        # Default behavior: run on authentic Kaggle CCTV crash benchmark footage
        default_cctv = "data/test/cctv_kaggle_04_highway_highspeed.mp4"
        if os.path.exists(default_cctv):
            mode_label = f"KAGGLE CCTV CRASH BENCHMARK ({default_cctv})"
            camera_config = {
                "name": "Kaggle CCTV - Elevated Highway High-Speed Crash",
                "type": "file",
                "url": default_cctv,
                "latitude": 37.7749,
                "longitude": -122.4194,
                "location": "Elevated Highway Pole Cam KM 24.8"
            }
        else:
            default_cctv = "data/test/accident_cctv.webm"
            if os.path.exists(default_cctv):
                mode_label = f"CCTV BENCHMARK ({default_cctv})"
                camera_config = {
                    "name": "Metropolitan Expressway Collision (CCTV 1080p)",
                    "type": "file",
                    "url": default_cctv,
                    "latitude": 34.0522,
                    "longitude": -118.2437,
                    "location": "Metropolitan Expressway Junction KM 14.2"
                }
            else:
                camera_config = load_camera_config()
                mode_label = "CONFIGURED CCTV FEED"

    print_banner(
        mode=mode_label,
        cam_name=camera_config.get("name", "Traffic Camera"),
        cam_url=camera_config.get("url", ""),
        port=args.port
    )

    # Initialize processing engine
    engine = TrafficGuardEngine(
        camera_config=camera_config,
        detection_fps=args.fps,
        is_simulation_mode=is_simulation
    )
    engine.start()

    # Create FastAPI app
    app = create_app(engine)

    try:
        uvicorn.run(
            app,
            host=args.host,
            port=args.port,
            log_level="warning",
            access_log=False
        )
    except KeyboardInterrupt:
        logger.info("Shutdown signal received.")
    finally:
        engine.stop()
        logger.info("TrafficGuard stopped cleanly.")


if __name__ == "__main__":
    main()

