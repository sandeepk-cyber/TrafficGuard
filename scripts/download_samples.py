"""
TrafficGuard - Public CCTV Footage Downloader & Dataset Scraper
Scrapes and downloads genuine traffic accident and highway CCTV footage for model evaluation.
"""
import os
import sys
import subprocess
import urllib.request
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("trafficguard.scraper")

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "test")

PUBLIC_SAMPLES = [
    {
        "filename": "accident_cctv.webm",
        "url": "https://upload.wikimedia.org/wikipedia/commons/c/cd/Connect_to_the_opposite_lane_and_bypass_the_traffic_accident_section.webm",
        "description": "Metropolitan highway multi-vehicle collision & bypass (CCTV 1080p)"
    },
    {
        "filename": "cctv_junction_accident.webm",
        "url": "https://upload.wikimedia.org/wikipedia/commons/3/3d/Jones_Bouldevard_and_Coppper_Crest_Dr._Car_Accident.webm",
        "description": "Urban intersection vehicle collision (CCTV 1080p)"
    },
    {
        "filename": "cctv_highway_interchange.webm",
        "url": "https://upload.wikimedia.org/wikipedia/commons/9/90/Jane_M._Byrne_Interchange_Traffic.webm",
        "description": "Jane Byrne Interstate expressway interchange traffic flow (CCTV 1080p)"
    }
]


def download_public_samples():
    os.makedirs(DATA_DIR, exist_ok=True)

    for item in PUBLIC_SAMPLES:
        dest = os.path.join(DATA_DIR, item["filename"])
        if os.path.exists(dest):
            logger.info("Sample already present: %s (%d bytes)", item["filename"], os.path.getsize(dest))
            continue

        logger.info("Downloading %s: %s...", item["filename"], item["description"])
        try:
            req = urllib.request.Request(item["url"], headers={"User-Agent": "TrafficGuard-Scraper/1.0"})
            with urllib.request.urlopen(req, timeout=30) as resp, open(dest, "wb") as f:
                f.write(resp.read())
            logger.info("Successfully downloaded %s (%d bytes)", item["filename"], os.path.getsize(dest))
        except Exception as e:
            logger.error("Error downloading %s: %s", item["filename"], e)


def scrape_youtube_cctv(search_query="traffic accident cctv highway", output_name="accident_youtube.mp4"):
    """Uses yt-dlp to scrape a short 20s clip of CCTV footage from YouTube."""
    out_path = os.path.join(DATA_DIR, output_name)
    if os.path.exists(out_path):
        logger.info("YouTube sample already exists: %s", out_path)
        return out_path

    logger.info("Scraping CCTV footage from YouTube for query: '%s'...", search_query)
    cmd = [
        "yt-dlp",
        f"ytsearch1:{search_query}",
        "--download-sections", "*00:00-00:25",
        "-f", "best[ext=mp4]/best",
        "-o", out_path
    ]
    try:
        subprocess.run(cmd, check=True)
        logger.info("Downloaded CCTV video clip to %s", out_path)
        return out_path
    except Exception as e:
        logger.warning("yt-dlp scraping failed or timed out: %s", e)
        return None


if __name__ == "__main__":
    download_public_samples()
    print("Download complete. Files available in data/test/:")
    for f in os.listdir(DATA_DIR):
        print(f" - {f} ({os.path.getsize(os.path.join(DATA_DIR, f))} bytes)")
