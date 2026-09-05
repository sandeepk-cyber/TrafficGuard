"""
TrafficGuard - Video Processing & Synthetic Traffic Stream Engine
"""
import time
import math
import random
import cv2
import numpy as np
import logging

logger = logging.getLogger("trafficguard.video")

FRAME_WIDTH = 640
FRAME_HEIGHT = 480
DETECTION_FPS = 15.0


class SimulatedVehicle:
    """Represents a vehicle in the synthetic simulation."""
    def __init__(self, vid, x, y, speed_x, speed_y, length=70, width=36, color=(50, 150, 240), label="Car"):
        self.id = vid
        self.x = float(x)
        self.y = float(y)
        self.speed_x = float(speed_x)
        self.speed_y = float(speed_y)
        self.length = length
        self.width = width
        self.color = color
        self.label = label
        self.is_collided = False
        self.decelerating = False

    @property
    def speed_kmh(self):
        # Scale speed for display (pixels/frame -> km/h estimate)
        speed = math.hypot(self.speed_x, self.speed_y)
        return max(0.0, speed * 18.0)

    def update(self):
        if self.decelerating:
            self.speed_x *= 0.85
            self.speed_y *= 0.85
            if abs(self.speed_x) < 0.2:
                self.speed_x = 0
            if abs(self.speed_y) < 0.2:
                self.speed_y = 0

        self.x += self.speed_x
        self.y += self.speed_y


class SyntheticTrafficGenerator:
    """
    Generates a realistic animated multi-lane highway traffic scene.
    Injects an accident event at a scheduled time or on demand.
    """
    def __init__(self):
        self.start_time = time.time()
        self.frame_count = 0
        self.incident_triggered = False
        self.accident_injected_time = None
        self.vehicles = []
        self.particles = []
        self._init_traffic()

    def _init_traffic(self):
        self.vehicles = [
            # Lane 1 (top westbound): y=160, speed_x = -4.5
            SimulatedVehicle(101, 620, 150, -4.5, 0, length=75, width=38, color=(40, 130, 230), label="Sedan"),
            SimulatedVehicle(102, 380, 150, -4.0, 0, length=90, width=42, color=(60, 180, 75), label="SUV"),
            SimulatedVehicle(103, 150, 150, -4.2, 0, length=70, width=36, color=(160, 80, 200), label="Sedan"),

            # Lane 2 (middle westbound): y=225, speed_x = -5.5
            SimulatedVehicle(104, 550, 220, -5.5, 0, length=80, width=38, color=(220, 190, 50), label="Sedan"),
            SimulatedVehicle(105, 260, 220, -5.2, 0, length=120, width=45, color=(120, 120, 120), label="Truck"),

            # Lane 3 (eastbound): y=310, speed_x = 5.0
            SimulatedVehicle(201, 50, 310, 5.0, 0, length=78, width=38, color=(50, 90, 220), label="Sedan"),
            SimulatedVehicle(202, 320, 310, 4.8, 0, length=72, width=36, color=(240, 240, 240), label="Coupe"),

            # Lane 4 (bottom eastbound): y=375, speed_x = 4.0
            SimulatedVehicle(203, 120, 375, 4.0, 0, length=85, width=40, color=(80, 190, 230), label="SUV"),
            SimulatedVehicle(204, 450, 375, 3.8, 0, length=130, width=48, color=(200, 100, 50), label="Bus"),
        ]

    def trigger_incident(self):
        """Manually or automatically trigger collision kinematics between vehicle 104 and 105."""
        if not self.incident_triggered:
            self.incident_triggered = True
            self.accident_injected_time = time.time()
            logger.info("SYNTHETIC INCIDENT INJECTED: Vehicles converging rapidly on Lane 2")
            for v in self.vehicles:
                if v.id == 104:
                    # Vehicle 104 suddenly cuts lane and brakes
                    v.speed_x = -7.5
                    v.speed_y = 0.8
                elif v.id == 105:
                    v.speed_x = -1.5
                    v.decelerating = True

    def get_frame(self):
        self.frame_count += 1
        elapsed = time.time() - self.start_time

        # Trigger simulated accident automatically after ~5 seconds if not already triggered
        if elapsed >= 5.0 and not self.incident_triggered:
            self.trigger_incident()

        # Canvas: Asphalt road background
        frame = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)
        frame[:] = (55, 58, 62)  # Asphalt grey

        # Draw grass/shoulders
        cv2.rectangle(frame, (0, 0), (FRAME_WIDTH, 110), (35, 75, 40), -1)      # North shoulder
        cv2.rectangle(frame, (0, 430), (FRAME_WIDTH, FRAME_HEIGHT), (35, 75, 40), -1)  # South shoulder

        # Guardrails / road boundary lines
        cv2.line(frame, (0, 110), (FRAME_WIDTH, 110), (220, 220, 220), 4)
        cv2.line(frame, (0, 430), (FRAME_WIDTH, 430), (220, 220, 220), 4)

        # Center double yellow dividing line
        cv2.line(frame, (0, 268), (FRAME_WIDTH, 268), (20, 190, 240), 2)
        cv2.line(frame, (0, 272), (FRAME_WIDTH, 272), (20, 190, 240), 2)

        # Dashed lane markings
        dash_offset = int((self.frame_count * 5) % 40)
        for x in range(-dash_offset, FRAME_WIDTH + 40, 40):
            # Lane 1/2 boundary (Westbound)
            cv2.line(frame, (x, 190), (x + 20, 190), (240, 240, 240), 2)
            # Lane 3/4 boundary (Eastbound)
            cv2.line(frame, (x, 350), (x + 20, 350), (240, 240, 240), 2)

        # Check collision physics for simulation
        if self.incident_triggered and self.accident_injected_time:
            time_since_crash = time.time() - self.accident_injected_time
            v1 = next((v for v in self.vehicles if v.id == 104), None)
            v2 = next((v for v in self.vehicles if v.id == 105), None)
            if v1 and v2:
                dist = math.hypot(v1.x - v2.x, v1.y - v2.y)
                if dist < 65:
                    v1.is_collided = True
                    v2.is_collided = True
                    v1.decelerating = True
                    v2.decelerating = True
                    # Spawn impact sparks and smoke
                    for _ in range(4):
                        self.particles.append({
                            "x": (v1.x + v2.x) / 2 + random.uniform(-10, 10),
                            "y": (v1.y + v2.y) / 2 + random.uniform(-10, 10),
                            "vx": random.uniform(-3, 3),
                            "vy": random.uniform(-3, 3),
                            "life": random.randint(15, 30),
                            "color": random.choice([(0, 200, 255), (0, 100, 255), (180, 180, 180)])
                        })

        # Update and draw vehicles
        detected_objects = []
        for v in self.vehicles:
            v.update()

            # Wrap around boundaries
            if v.speed_x < 0 and v.x < -150:
                v.x = FRAME_WIDTH + random.randint(30, 100)
                v.is_collided = False
                v.decelerating = False
                v.speed_x = -random.uniform(4.0, 5.5)
            elif v.speed_x > 0 and v.x > FRAME_WIDTH + 150:
                v.x = -random.randint(30, 100)
                v.is_collided = False
                v.decelerating = False
                v.speed_x = random.uniform(4.0, 5.5)

            # Draw vehicle body (rectangle with headlights/taillights)
            vx, vy = int(v.x), int(v.y)
            hw = v.width // 2
            hl = v.length // 2

            # Shadow
            cv2.ellipse(frame, (vx + hl, vy + 4), (hl + 6, hw + 4), 0, 0, 360, (20, 20, 20), -1)

            # Main chassis
            body_color = (0, 0, 220) if v.is_collided else v.color
            cv2.rectangle(frame, (vx, vy - hw), (vx + v.length, vy + hw), body_color, -1)
            cv2.rectangle(frame, (vx, vy - hw), (vx + v.length, vy + hw), (30, 30, 30), 2)

            # Windshield / Roof
            roof_margin = int(v.length * 0.22)
            cv2.rectangle(frame, (vx + roof_margin, vy - hw + 4), (vx + v.length - roof_margin, vy + hw - 4), (30, 35, 45), -1)

            # Lights
            if v.speed_x < 0: # Driving Left
                cv2.circle(frame, (vx + 2, vy - hw + 4), 3, (180, 240, 255), -1) # Headlight
                cv2.circle(frame, (vx + 2, vy + hw - 4), 3, (180, 240, 255), -1)
                cv2.circle(frame, (vx + v.length - 2, vy - hw + 4), 3, (0, 0, 255), -1) # Brake
                cv2.circle(frame, (vx + v.length - 2, vy + hw - 4), 3, (0, 0, 255), -1)
            else: # Driving Right
                cv2.circle(frame, (vx + v.length - 2, vy - hw + 4), 3, (180, 240, 255), -1)
                cv2.circle(frame, (vx + v.length - 2, vy + hw - 4), 3, (180, 240, 255), -1)
                cv2.circle(frame, (vx + 2, vy - hw + 4), 3, (0, 0, 255), -1)
                cv2.circle(frame, (vx + 2, vy + hw - 4), 3, (0, 0, 255), -1)

            # Register detection bbox for heuristic engine
            bx1, by1 = vx, vy - hw
            bx2, by2 = vx + v.length, vy + hw
            cname = "car" if v.label not in ["Bus", "Truck"] else v.label.lower()
            detected_objects.append({
                "id": v.id,
                "bbox": (bx1, by1, bx2, by2),
                "class_name": cname,
                "confidence": 0.95,
                "center": ((bx1 + bx2) / 2.0, (by1 + by2) / 2.0),
                "label": v.label,
                "speed_kmh": round(v.speed_kmh, 1),
                "is_collided": v.is_collided,
                "velocity": (v.speed_x, v.speed_y)
            })

        # Draw particles (smoke/sparks)
        alive_particles = []
        for p in self.particles:
            p["x"] += p["vx"]
            p["y"] += p["vy"]
            p["life"] -= 1
            if p["life"] > 0:
                cv2.circle(frame, (int(p["x"]), int(p["y"])), max(1, p["life"] // 6), p["color"], -1)
                alive_particles.append(p)
        self.particles = alive_particles

        # Timestamp and HUD stamp on top
        time_str = time.strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(frame, f"CAM-SIM01 | {time_str} | 1080p-HQ", (15, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

        return frame, detected_objects


class VideoStreamHandler:
    """
    Handles live camera streams (HLS/RTSP/MJPEG), local video files, or simulation mode.
    """
    def __init__(self, camera_config, is_simulation=False):
        self.camera_config = camera_config
        self.is_simulation = is_simulation or (camera_config.get("type") == "simulation")
        self.url = camera_config.get("url", "")
        self.cap = None
        self.sim_gen = SyntheticTrafficGenerator() if self.is_simulation else None
        self.connected = False

        if not self.is_simulation and self.url:
            self._connect()
        else:
            self.is_simulation = True
            self.sim_gen = SyntheticTrafficGenerator()
            self.connected = True

    def _connect(self):
        try:
            logger.info("Opening video capture for: %s", self.url)
            self.cap = cv2.VideoCapture(self.url)
            if self.cap.isOpened():
                self.connected = True
                logger.info("Successfully connected to video stream.")
            else:
                logger.warning("Could not open stream: %s. Falling back to simulation.", self.url)
                self.is_simulation = True
                self.sim_gen = SyntheticTrafficGenerator()
                self.connected = True
        except Exception as e:
            logger.error("Error opening video stream: %s. Using simulation.", e)
            self.is_simulation = True
            self.sim_gen = SyntheticTrafficGenerator()
            self.connected = True

    def read_frame(self):
        """Returns (success: bool, frame: np.ndarray, synthetic_detections: list or None)"""
        if self.is_simulation and self.sim_gen:
            frame, detections = self.sim_gen.get_frame()
            return True, frame, detections

        if self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
                return True, frame, None
            else:
                # Loop video if file
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret2, frame2 = self.cap.read()
                if ret2:
                    frame2 = cv2.resize(frame2, (FRAME_WIDTH, FRAME_HEIGHT))
                    return True, frame2, None

        # Fallback
        if not self.sim_gen:
            self.sim_gen = SyntheticTrafficGenerator()
            self.is_simulation = True
        frame, detections = self.sim_gen.get_frame()
        return True, frame, detections

    def trigger_incident(self):
        if self.sim_gen:
            self.sim_gen.trigger_incident()

    def release(self):
        if self.cap:
            self.cap.release()
            self.connected = False
