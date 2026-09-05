#!/usr/bin/env python3
"""
TrafficGuard - Real-Time CCTV Accident Detection MVP
Local low-resource computer vision pipeline on CPU.
Zero emojis, strict industrial standards.
"""
import os
import sys
import argparse
import logging
import uvicorn

from app.dashboard import TrafficGuardEngine, create_app, DEFAULT_DETECTION_FPS
from app.camera_manager import FRAME_WIDTH, FRAME_HEIGHT, CameraManager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("trafficguard")


def parse_args():
    parser = argparse.ArgumentParser(
        description="TrafficGuard - Real-Time CCTV Accident Detection MVP",
        formatter_class=argparse.RawTextHelpFormatter
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--camera",
        type=str,
        default=None,
        help="Specify camera ID from catalog (e.g. --camera cctv_kaggle_04_highway)"
    )
    group.add_argument(
        "--video",
        type=str,
        metavar="PATH",
        help="Process a local video file (e.g. --video data/test/cctv_kaggle_04_highway_highspeed.mp4)"
    )
    group.add_argument(
        "--live",
        action="store_true",
        help="Ingest the public HLS live stream configured in config/cameras.yaml"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8001,
        help="Web console port (default: 8001)"
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Web console host (default: 127.0.0.1)"
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=DEFAULT_DETECTION_FPS,
        help=f"Target detection rate in FPS (default: {DEFAULT_DETECTION_FPS})"
    )
    return parser.parse_args()


def print_banner(camera_name: str, stream_url: str, detection_fps: float, host: str, port: int):
    print("=" * 70)
    print("  TRAFFICGUARD - REAL-TIME CCTV ACCIDENT DETECTION MVP")
    print("  Lightweight Local Computer Vision Pipeline (CPU)")
    print("=" * 70)
    print(f"  ACTIVE CAMERA : {camera_name}")
    print(f"  STREAM SOURCE : {stream_url}")
    print(f"  PROCESSING RES: {FRAME_WIDTH}x{FRAME_HEIGHT}")
    print(f"  DETECTION FPS : {detection_fps:.1f} FPS (Target)")
    print(f"  OPERATOR UI   : http://{host}:{port}")
    print("=" * 70)
    print("  Press Ctrl+C to stop.\n")


def main():
    args = parse_args()

    camera_config = None

    if args.video:
        if not os.path.exists(args.video):
            logger.error("Video file does not exist: %s", args.video)
            sys.exit(1)
        camera_config = {
            "name": f"Local Video: {os.path.basename(args.video)}",
            "type": "file",
            "url": args.video,
            "location": "Local Benchmark File",
            "latitude": None,
            "longitude": None,
            "loop": True
        }
    elif args.live:
        camera_config = {
            "name": "Public HLS Ingest Stream",
            "type": "hls",
            "url": "https://test-streams.mux.dev/x36xhzz/x36xhzz.m3u8",
            "location": "Public Live Ingest Feed",
            "latitude": None,
            "longitude": None,
            "loop": False
        }
    elif args.camera:
        # Check catalog
        mgr = CameraManager()
        if args.camera in mgr.cameras:
            cinfo = mgr.cameras[args.camera]
            camera_config = dict(cinfo)
        else:
            logger.warning("Camera ID '%s' not found in catalog. Using default benchmark.", args.camera)

    # Initialize processing engine
    engine = TrafficGuardEngine(
        camera_config=camera_config,
        detection_fps=args.fps
    )

    cam_info = engine.camera_mgr.get_active_camera_info()
    print_banner(
        camera_name=cam_info.get("name", "Active Camera"),
        stream_url=cam_info.get("url", "Catalog Feed"),
        detection_fps=args.fps,
        host=args.host,
        port=args.port
    )

    engine.start()
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
