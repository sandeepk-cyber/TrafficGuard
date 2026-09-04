"""
TrafficGuard - Multi-Object Kinematic Tracker with Perspective Normalization
Enterprise AID Tracking with Velocity Smoothing and Keyframe Extrapolation.
"""
import math
import time
from collections import deque
import numpy as np


class TrackedVehicle:
    def __init__(self, track_id, bbox, class_name, confidence, frame_height=480, max_history=30, timestamp=None):
        self.track_id = track_id
        self.class_name = class_name
        self.confidence = confidence
        self.bbox = [float(c) for c in bbox]
        self.frame_height = frame_height
        self.max_history = max_history

        x1, y1, x2, y2 = self.bbox
        self.center = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
        self.history = deque(maxlen=max_history)
        t0 = timestamp if timestamp is not None else time.time()
        self.history.append((t0, self.center[0], self.center[1]))

        self.velocity = [0.0, 0.0]      # (vx, vy) in px/sec
        self.acceleration = [0.0, 0.0]  # (ax, ay) in px/sec^2
        self.speed_kmh = 0.0
        self.heading_deg = 0.0
        self.disappeared = 0
        self.stationary_duration = 0.0
        self.stopped_start_time = None
        self.hits = 1
        self.is_confirmed = False
        self.in_collision = False
        self.collision_latched_until = 0.0
        self.max_historical_speed = 0.0

    def get_perspective_factor(self, y):
        """
        Calibrated perspective scale:
        Objects near the horizon (top of frame) move fewer pixels for the same real distance.
        """
        norm_y = max(0.1, min(1.0, y / float(self.frame_height)))
        return 0.35 + 0.65 * (norm_y ** 2)

    def update(self, bbox, confidence, timestamp=None):
        now = timestamp if timestamp is not None else time.time()
        self.bbox = [float(c) for c in bbox]
        self.confidence = confidence
        self.disappeared = 0
        self.hits += 1
        if self.hits >= 2:
            self.is_confirmed = True

        x1, y1, x2, y2 = self.bbox
        new_cx, new_cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0

        if len(self.history) > 0:
            last_t, last_cx, last_cy = self.history[-1]
            raw_dt = now - last_t
            if raw_dt <= 0.005 or raw_dt > 0.50:
                dt = 1.0 / 30.0
            else:
                dt = raw_dt

            # Multi-frame window (up to 4 frames / ~133ms) to filter out single-frame jitter
            k = min(len(self.history), 4)
            ref_t, ref_cx, ref_cy = self.history[-k]
            raw_window_dt = now - ref_t
            if raw_window_dt <= 0.005 or raw_window_dt > 2.0:
                window_dt = k * (1.0 / 30.0)
            else:
                window_dt = raw_window_dt

            raw_vx = (new_cx - ref_cx) / window_dt
            raw_vy = (new_cy - ref_cy) / window_dt

            # EMA velocity smoothing
            vx = 0.40 * raw_vx + 0.60 * self.velocity[0]
            vy = 0.40 * raw_vy + 0.60 * self.velocity[1]
            pixel_speed = math.hypot(vx, vy)

            # Cumulative displacement from initial observation
            tot_disp = math.hypot(new_cx - self.history[0][1], new_cy - self.history[0][2])

            # Jitter deadband: requires at least 18 px/s and cumulative translation >= 16px to register highway/roadway speed
            if pixel_speed < 18.0 or tot_disp < 16.0:
                vx, vy = 0.0, 0.0
                ax, ay = 0.0, 0.0
                pixel_speed = 0.0
                self.speed_kmh = 0.0
            else:
                if pixel_speed > 550.0:
                    scale = 550.0 / pixel_speed
                    vx *= scale
                    vy *= scale
                    pixel_speed = 550.0

                raw_ax = (vx - self.velocity[0]) / dt
                raw_ay = (vy - self.velocity[1]) / dt
                ax = 0.40 * raw_ax + 0.60 * self.acceleration[0]
                ay = 0.40 * raw_ay + 0.60 * self.acceleration[1]

                depth_scale = self.get_perspective_factor(new_cy)
                self.speed_kmh = min(175.0, round((pixel_speed / depth_scale) * 0.28, 1))

            self.velocity = [vx, vy]
            self.acceleration = [ax, ay]
            self.max_historical_speed = max(self.max_historical_speed, self.speed_kmh)

            if pixel_speed > 2.0:
                self.heading_deg = math.degrees(math.atan2(vy, vx)) % 360.0

            # Stationary tracker (TID-02)
            if self.speed_kmh < 4.0:
                if self.stopped_start_time is None:
                    self.stopped_start_time = now
                self.stationary_duration = round(now - self.stopped_start_time, 1)
            else:
                self.stopped_start_time = None
                self.stationary_duration = 0.0

        self.center = (new_cx, new_cy)
        self.history.append((now, new_cx, new_cy))

    def extrapolate(self, dt):
        """Strict zero-ghosting extrapolation: only extrapolate active moving vehicles."""
        if self.disappeared > 0 or self.speed_kmh < 4.0:
            # Frozen box: Never drift occluded, disappeared, or stopped vehicles
            return

        dx = self.velocity[0] * dt
        dy = self.velocity[1] * dt

        # Clamp max extrapolation jump to prevent wild jumps
        max_jump = 18.0
        dx = max(-max_jump, min(max_jump, dx))
        dy = max(-max_jump, min(max_jump, dy))

        self.bbox[0] += dx
        self.bbox[1] += dy
        self.bbox[2] += dx
        self.bbox[3] += dy

        new_cx = (self.bbox[0] + self.bbox[2]) / 2.0
        new_cy = (self.bbox[1] + self.bbox[3]) / 2.0
        self.center = (new_cx, new_cy)
        self.history.append((time.time(), new_cx, new_cy))

    def mark_missed(self):
        self.disappeared += 1


def compute_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    iou = interArea / float(boxAArea + boxBArea - interArea + 1e-6)
    return iou


class KinematicTracker:
    def __init__(self, frame_width=1280, frame_height=720, max_disappeared=2, min_iou_match=0.15, max_dist_match=85.0):
        self.next_id = 1
        self.tracks = {}  # track_id -> TrackedVehicle
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.max_disappeared = max_disappeared
        self.min_iou_match = min_iou_match
        self.max_dist_match = max_dist_match

    def update(self, detections, timestamp=None):
        """Updates active tracks with new frame detections. Guarantees zero ghosting."""
        if len(self.tracks) == 0:
            for det in detections:
                cname = det.get("class_name", det.get("label", "vehicle"))
                conf = float(det.get("confidence", 0.90))
                t = TrackedVehicle(self.next_id, det["bbox"], cname, conf, self.frame_height, timestamp=timestamp)
                self.tracks[self.next_id] = t
                self.next_id += 1
            return [t for t in self.tracks.values() if (t.is_confirmed or t.hits >= 1) and t.disappeared == 0]

        track_ids = list(self.tracks.keys())
        num_tracks = len(track_ids)
        num_dets = len(detections)

        if num_dets == 0:
            for tid in track_ids:
                self.tracks[tid].mark_missed()
                if self.tracks[tid].disappeared > self.max_disappeared:
                    del self.tracks[tid]
            return [t for t in self.tracks.values() if (t.is_confirmed or t.hits >= 2) and t.disappeared == 0]

        # Bipartite matching matrix
        cost_matrix = np.zeros((num_tracks, num_dets), dtype=np.float32)
        for i, tid in enumerate(track_ids):
            t = self.tracks[tid]
            t_is_person = (t.class_name == "person")
            for j, det in enumerate(detections):
                d_name = det.get("class_name", det.get("label", "vehicle"))
                d_is_person = (d_name == "person")
                # 1. Class consistency gate: Pedestrians cannot match vehicles and vice-versa
                if t_is_person != d_is_person:
                    cost_matrix[i, j] = -1.0
                    continue

                det_center = det.get("center")
                if det_center is None:
                    bx = det["bbox"]
                    det_center = ((bx[0] + bx[2]) / 2.0, (bx[1] + bx[3]) / 2.0)

                iou = compute_iou(t.bbox, det["bbox"])
                dist = math.hypot(t.center[0] - det_center[0], t.center[1] - det_center[1])

                # 2. Maximum physical displacement gate (for zero-overlap candidates)
                # In 30fps video (33ms), a vehicle or person with zero IoU cannot jump > 32px
                if iou == 0.0 and dist > 32.0:
                    cost_matrix[i, j] = -1.0
                    continue

                match_score = (iou * 2.0) + max(0.0, (1.0 - (dist / self.max_dist_match)))
                cost_matrix[i, j] = match_score

        matched_tracks = set()
        matched_dets = set()

        while True:
            max_idx = np.unravel_index(np.argmax(cost_matrix), cost_matrix.shape)
            max_score = cost_matrix[max_idx]
            if max_score < 0.20:
                break
            i, j = max_idx
            if i in matched_tracks or j in matched_dets:
                cost_matrix[i, j] = -1.0
                continue

            tid = track_ids[i]
            conf = float(detections[j].get("confidence", 0.90))
            self.tracks[tid].update(detections[j]["bbox"], conf, timestamp=timestamp)
            matched_tracks.add(i)
            matched_dets.add(j)
            cost_matrix[i, :] = -1.0
            cost_matrix[:, j] = -1.0

        for i, tid in enumerate(track_ids):
            if i not in matched_tracks:
                self.tracks[tid].mark_missed()
                # Immediate purge if disappeared exceeds threshold or if beyond canvas boundary
                t = self.tracks[tid]
                is_out_of_bounds = (t.center[0] < -30 or t.center[0] > self.frame_width + 30 or
                                    t.center[1] < -30 or t.center[1] > self.frame_height + 30)
                if t.disappeared > self.max_disappeared or is_out_of_bounds:
                    del self.tracks[tid]

        for j, det in enumerate(detections):
            if j not in matched_dets:
                cname = det.get("class_name", det.get("label", "vehicle"))
                conf = float(det.get("confidence", 0.90))
                t = TrackedVehicle(self.next_id, det["bbox"], cname, conf, self.frame_height, timestamp=timestamp)
                self.tracks[self.next_id] = t
                self.next_id += 1

        # Strictly return confirmed tracks that are detected in the current keyframe (disappeared == 0)
        return [t for t in self.tracks.values() if (t.is_confirmed or t.hits >= 2) and t.disappeared == 0]

    def reset(self):
        """Purges all tracks and resets ID counter, used during scene changes or stream reload."""
        self.tracks.clear()
        self.next_id = 1

    def extrapolate_all(self, dt):
        """Extrapolates only confirmed tracks that were seen in the latest keyframe."""
        active = []
        for tid, t in list(self.tracks.items()):
            if t.is_confirmed and t.disappeared == 0:
                t.extrapolate(dt)
                if (-30 <= t.center[0] <= self.frame_width + 30 and
                        -30 <= t.center[1] <= self.frame_height + 30):
                    active.append(t)
                else:
                    t.mark_missed()
                    del self.tracks[tid]
        return active
