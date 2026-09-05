"""
TrafficGuard - Multi-Object Kinematic Tracker
State-machine tracking (TENTATIVE, CONFIRMED, LOST, REMOVED) using Hungarian assignment,
class-consistency gating, and image-plane kinematics.
Zero emojis, strict industrial coding standards.
"""
import math
import time
from collections import deque
from enum import Enum
import numpy as np
from scipy.optimize import linear_sum_assignment


class TrackState(str, Enum):
    TENTATIVE = "TENTATIVE"
    CONFIRMED = "CONFIRMED"
    LOST = "LOST"
    REMOVED = "REMOVED"


def compute_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    inter = max(0.0, xB - xA) * max(0.0, yB - yA)
    areaA = max(1.0, (boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
    areaB = max(1.0, (boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))
    return inter / float(areaA + areaB - inter + 1e-6)


class TrackedVehicle:
    def __init__(self, track_id, bbox, class_name, confidence, frame_height=480, max_history=30, timestamp=None):
        self.track_id = int(track_id)
        self.bbox = [float(c) for c in bbox]
        self.class_name = str(class_name)
        self.confidence = float(confidence)
        self.frame_height = float(frame_height)
        self.max_history = int(max_history)

        x1, y1, x2, y2 = self.bbox
        self.center = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
        self.origin_center = self.center
        self.net_displacement = 0.0
        self.max_historical_displacement = 0.0
        self.history = deque(maxlen=max_history)
        t0 = float(timestamp) if timestamp is not None else time.time()
        self.history.append((t0, self.center[0], self.center[1], list(self.bbox)))

        # Lifecycle tracking
        self.state = TrackState.TENTATIVE
        self.hits = 1
        self.age = 1
        self.time_since_update = 0

        # Kinematic state (image-plane)
        self.velocity = [0.0, 0.0]        # (vx, vy) in px/sec
        self.acceleration = [0.0, 0.0]    # (ax, ay) in px/sec^2
        self.speed_px_s = 0.0             # image plane pixel speed
        self.relative_speed = 0.0         # speed normalized by frame height (units/sec)
        self.heading_deg = 0.0
        self.max_historical_speed = 0.0

        # Incident flags & stationary monitoring
        self.stationary_duration = 0.0
        self.stopped_start_time = None
        self.in_collision = False
        self.collision_latched_until = 0.0

    @property
    def is_confirmed(self):
        return self.state == TrackState.CONFIRMED

    @property
    def missed_frames(self):
        return self.time_since_update

    def update(self, bbox, confidence, timestamp=None):
        """Updates track with a matched detection in the current frame."""
        now = float(timestamp) if timestamp is not None else time.time()
        self.bbox = [float(c) for c in bbox]
        self.confidence = float(confidence)
        self.hits += 1
        self.age += 1
        self.time_since_update = 0

        # Promote to CONFIRMED after 3 consistent detections
        if self.state == TrackState.TENTATIVE and self.hits >= 3:
            self.state = TrackState.CONFIRMED
        elif self.state == TrackState.LOST:
            self.state = TrackState.CONFIRMED

        x1, y1, x2, y2 = self.bbox
        new_cx, new_cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0

        if len(self.history) > 0:
            last_t, last_cx, last_cy, _ = self.history[-1]
            raw_dt = now - last_t
            dt = raw_dt if (0.005 < raw_dt < 0.50) else (1.0 / 30.0)

            # Multi-frame window (up to 4 frames) to eliminate single-frame discretization noise
            k = min(len(self.history), 4)
            ref_t, ref_cx, ref_cy, _ = self.history[-k]
            window_dt = (now - ref_t) if (0.01 < (now - ref_t) < 1.0) else (k * (1.0 / 30.0))

            raw_vx = (new_cx - ref_cx) / max(0.01, window_dt)
            raw_vy = (new_cy - ref_cy) / max(0.01, window_dt)

            # Exponential Moving Average (EMA) velocity smoothing
            vx = 0.40 * raw_vx + 0.60 * self.velocity[0]
            vy = 0.40 * raw_vy + 0.60 * self.velocity[1]
            px_speed = math.hypot(vx, vy)

            # Cumulative spatial displacement from origin observation
            tot_disp = math.hypot(new_cx - self.origin_center[0], new_cy - self.origin_center[1])
            self.net_displacement = tot_disp
            self.max_historical_displacement = max(self.max_historical_displacement, tot_disp)

            # Motion deadband: reject sub-pixel macroblock jitter (< 6.0 px/s)
            if px_speed < 6.0:
                vx, vy = 0.0, 0.0
                ax, ay = 0.0, 0.0
                px_speed = 0.0
            else:
                raw_ax = (vx - self.velocity[0]) / dt
                raw_ay = (vy - self.velocity[1]) / dt
                ax = 0.40 * raw_ax + 0.60 * self.acceleration[0]
                ay = 0.40 * raw_ay + 0.60 * self.acceleration[1]

            self.velocity = [vx, vy]
            self.acceleration = [ax, ay]
            self.speed_px_s = round(px_speed, 1)
            self.relative_speed = round(px_speed / max(100.0, self.frame_height), 2)
            self.area = float(max(1.0, (x2 - x1) * (y2 - y1)))

            # Only record high traffic speeds if the vehicle has genuine spatial displacement or confirmed track age
            if self.max_historical_displacement >= 18.0 or self.hits >= 4:
                self.max_historical_speed = max(self.max_historical_speed, self.speed_px_s)

            if px_speed > 4.0:
                self.heading_deg = math.degrees(math.atan2(vy, vx)) % 360.0

            # Stationary duration tracking (TID-02)
            if self.speed_px_s < 8.0:
                if self.stopped_start_time is None:
                    self.stopped_start_time = now
                self.stationary_duration = round(now - self.stopped_start_time, 1)
            else:
                self.stopped_start_time = None
                self.stationary_duration = 0.0

        self.center = (new_cx, new_cy)
        self.history.append((now, new_cx, new_cy, list(self.bbox)))

    def mark_missed(self):
        """Marks track as missed when no matching detection is found."""
        self.age += 1
        self.time_since_update += 1

        if self.state == TrackState.TENTATIVE:
            # Unconfirmed tracks are pruned immediately on miss
            self.state = TrackState.REMOVED
        elif self.state == TrackState.CONFIRMED:
            self.state = TrackState.LOST

    def predict(self, dt=0.033):
        """Advances bounding box position slightly during short missed detection intervals."""
        if self.speed_px_s < 5.0 or self.state == TrackState.REMOVED:
            return

        dx = max(-15.0, min(15.0, self.velocity[0] * dt))
        dy = max(-15.0, min(15.0, self.velocity[1] * dt))

        self.bbox[0] += dx
        self.bbox[1] += dy
        self.bbox[2] += dx
        self.bbox[3] += dy
        self.center = ((self.bbox[0] + self.bbox[2]) / 2.0, (self.bbox[1] + self.bbox[3]) / 2.0)


class KinematicTracker:
    def __init__(
        self,
        frame_width=640,
        frame_height=360,
        max_missed=3,
        min_hits=3,
        max_cost=0.75
    ):
        self.next_id = 1
        self.tracks = {}  # track_id -> TrackedVehicle
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.max_missed = max_missed
        self.min_hits = min_hits
        self.max_cost = max_cost

    def update(self, detections, timestamp=None):
        """
        Associates detections with existing tracks using Hungarian assignment on
        a combined IoU and normalized centroid distance cost matrix.

        Returns:
            active_tracks: list of confirmed TrackedVehicle objects visible in current frame
        """
        now = timestamp if timestamp is not None else time.time()
        track_ids = list(self.tracks.keys())
        num_tracks = len(track_ids)
        num_dets = len(detections)

        # 1. Empty tracker initialization
        if num_tracks == 0:
            for det in detections:
                cname = det.get("class_name", det.get("label", "vehicle"))
                conf = float(det.get("confidence", 0.90))
                t = TrackedVehicle(
                    self.next_id, det["bbox"], cname, conf,
                    frame_height=self.frame_height, timestamp=now
                )
                self.tracks[self.next_id] = t
                self.next_id += 1
            return []

        # 2. Advance all tracks with missed prediction if no detections
        if num_dets == 0:
            to_delete = []
            for tid in track_ids:
                t = self.tracks[tid]
                t.mark_missed()
                if t.time_since_update > self.max_missed or t.state == TrackState.REMOVED:
                    to_delete.append(tid)
            for tid in to_delete:
                del self.tracks[tid]
            return []

        # 3. Construct Hungarian cost matrix
        # Cost = 0.65 * (1 - IoU) + 0.35 * (normalized centroid distance)
        max_dim = math.hypot(self.frame_width, self.frame_height)
        cost_matrix = np.full((num_tracks, num_dets), 1e5, dtype=np.float32)

        for i, tid in enumerate(track_ids):
            t = self.tracks[tid]
            t_is_ped = (t.class_name == "person")
            for j, det in enumerate(detections):
                d_name = det.get("class_name", det.get("label", "vehicle"))
                d_is_ped = (d_name == "person")

                # Strict class category separation: pedestrians cannot match vehicles
                if t_is_ped != d_is_ped:
                    continue

                det_center = det.get("center")
                if det_center is None:
                    bx = det["bbox"]
                    det_center = ((bx[0] + bx[2]) / 2.0, (bx[1] + bx[3]) / 2.0)

                iou = compute_iou(t.bbox, det["bbox"])
                dist = math.hypot(t.center[0] - det_center[0], t.center[1] - det_center[1])
                norm_dist = min(1.0, dist / (0.15 * max_dim))

                # Physical gating: zero-IoU match cannot jump across frame
                if iou == 0.0 and dist > 40.0:
                    continue

                cost = 0.65 * (1.0 - iou) + 0.35 * norm_dist
                cost_matrix[i, j] = cost

        # 4. Hungarian Assignment (linear_sum_assignment)
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        matched_tracks = set()
        matched_dets = set()

        for r, c in zip(row_ind, col_ind):
            if cost_matrix[r, c] <= self.max_cost:
                tid = track_ids[r]
                conf = float(detections[c].get("confidence", 0.90))
                self.tracks[tid].update(detections[c]["bbox"], conf, timestamp=now)
                matched_tracks.add(r)
                matched_dets.add(c)

        # 5. Handle unmatched tracks
        to_delete = []
        for i, tid in enumerate(track_ids):
            if i not in matched_tracks:
                t = self.tracks[tid]
                t.mark_missed()
                t.predict(dt=1.0 / 30.0)

                # Boundary check: purge out-of-frame tracks immediately
                out_of_frame = (
                    t.center[0] < -20 or t.center[0] > self.frame_width + 20 or
                    t.center[1] < -20 or t.center[1] > self.frame_height + 20
                )
                if t.time_since_update > self.max_missed or t.state == TrackState.REMOVED or out_of_frame:
                    to_delete.append(tid)

        for tid in to_delete:
            del self.tracks[tid]

        # 6. Initialize new tracks for unmatched detections
        for j, det in enumerate(detections):
            if j not in matched_dets:
                cname = det.get("class_name", det.get("label", "vehicle"))
                conf = float(det.get("confidence", 0.90))
                t = TrackedVehicle(
                    self.next_id, det["bbox"], cname, conf,
                    frame_height=self.frame_height, timestamp=now
                )
                self.tracks[self.next_id] = t
                self.next_id += 1

        # 7. Return only CONFIRMED tracks that were detected in the current frame
        return [
            t for t in self.tracks.values()
            if t.state == TrackState.CONFIRMED and t.time_since_update == 0
        ]

    def reset(self):
        """Purges all active tracks and resets ID counter."""
        self.tracks.clear()
        self.next_id = 1
