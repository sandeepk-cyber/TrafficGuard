"""
TrafficGuard - Configuration & Core Utilities
"""
import os
import yaml
import logging

logger = logging.getLogger("trafficguard.main")

CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config", "cameras.yaml")


def load_camera_config(camera_id: str = "demo"):
    """
    Loads camera configuration from config/cameras.yaml.
    Returns a dictionary with camera parameters.
    """
    default_config = {
        "name": "Live Traffic Camera (Simulated)",
        "type": "simulation",
        "url": "",
        "latitude": 37.7749,
        "longitude": -122.4194,
        "location": "Metropolitan Traffic Corridor (Simulated)"
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
                "type": cfg.get("type", "hls"),
                "url": cfg.get("url", ""),
                "latitude": cfg.get("latitude", 37.7749),
                "longitude": cfg.get("longitude", -122.4194),
                "location": cfg.get("location", "Traffic Corridor")
            }
        elif cameras:
            first_key = next(iter(cameras))
            cfg = cameras[first_key]
            return {
                "name": cfg.get("name", "Traffic Camera"),
                "type": cfg.get("type", "hls"),
                "url": cfg.get("url", ""),
                "latitude": cfg.get("latitude", 37.7749),
                "longitude": cfg.get("longitude", -122.4194),
                "location": cfg.get("location", "Traffic Corridor")
            }
    except Exception as e:
        logger.error("Error reading camera config from %s: %s", CONFIG_PATH, e)

    return default_config
