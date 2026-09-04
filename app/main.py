"""
TrafficGuard - Configuration & Core Utilities
Zero emojis.
"""
import os
import yaml
import logging

logger = logging.getLogger("trafficguard.main")

CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config", "cameras.yaml")


def load_camera_config(camera_id: str = "cctv_kaggle_04_highway"):
    """
    Loads camera configuration from config/cameras.yaml.
    Returns a dictionary with camera parameters.
    """
    default_config = {
        "name": "Kaggle CCTV - Elevated Highway High-Speed Crash",
        "type": "file",
        "url": "data/test/cctv_kaggle_04_highway_highspeed.mp4",
        "latitude": None,
        "longitude": None,
        "location": "Dataset benchmark clip (Elevated Highway)",
        "loop": True
    }

    if not os.path.exists(CONFIG_PATH):
        logger.warning("Config file not found at %s. Using default configuration.", CONFIG_PATH)
        return default_config

    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        cameras = data.get("cameras", {})
        if camera_id in cameras:
            cfg = cameras[camera_id]
            return {
                "name": cfg.get("name", "Traffic Camera"),
                "type": cfg.get("type", "file"),
                "url": cfg.get("url", ""),
                "latitude": cfg.get("latitude"),
                "longitude": cfg.get("longitude"),
                "location": cfg.get("location", "CCTV Benchmark"),
                "loop": cfg.get("loop", True)
            }
        elif cameras:
            first_key = next(iter(cameras))
            cfg = cameras[first_key]
            return {
                "name": cfg.get("name", "Traffic Camera"),
                "type": cfg.get("type", "file"),
                "url": cfg.get("url", ""),
                "latitude": cfg.get("latitude"),
                "longitude": cfg.get("longitude"),
                "location": cfg.get("location", "CCTV Benchmark"),
                "loop": cfg.get("loop", True)
            }
    except Exception as e:
        logger.error("Error reading camera config from %s: %s", CONFIG_PATH, e)

    return default_config
