"""
TrafficGuard - Incident Detection Engine (ITS-AID)
Temporal multi-factor collision detection, stopped vehicle detection, and wrong-way monitoring.
Restrained CCTV visualization with zero gimmicks or tactical embellishments.
Zero emojis, strict industrial coding standards.
"""
import math
import time
import cv2
import numpy as np


def compute_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    inter = max(0.0, xB - xA) * max(0.0, yB - yA)
    areaA = max(1.0, (boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
    areaB = max(1.0, (boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))
    return inter / float(areaA + areaB - inter + 1e-6)


class IncidentEngine:
    def __init__(
        self,
        risk_threshold=0.65,
        stopped_duration_threshold=5.0,
        corridor_heading=None,
        latch_duration=5.0
    ):
        self.risk_threshold = float(risk_threshold)
        self.stopped_duration_threshold = float(stopped_duration_threshold)
        self.corridor_heading = corridor_heading  # degrees, or None if bidirectional
        self.latch_duration = float(latch_duration)

        # Collision candidate tracker: (tid1, tid2) -> dict of temporal evidence
        self.pair_evidence = {}

        # Active latched incident state: track_id -> dict(expiry, code, partner_id, score)
        self.latched_incidents = {}

        # Global incident summary
        self.current_incident = None

    def reset(self):
        """Clears all internal pair evidence and latched incident states."""
        self.pair_evidence.clear()
        self.latched_incidents.clear()
        self.current_incident = None

    def evaluate(self, frame, tracked_vehicles, timestamp=None):
        """
        Evaluates confirmed vehicles for:
          - TID-01: Collision (Temporal Multi-Factor Verification)
          - TID-02: Stopped Vehicle in Active Flow
          - TID-03: Wrong-Way Driver (if corridor_heading is configured)

        Returns:
            annotated_frame: np.ndarray with restrained bounding boxes
            max_score: float (0.0 to 1.0)
            status: str ("NORMAL", "POSSIBLE COLLISION", "CONFIRMED COLLISION")
            is_incident: bool
            incident_data: dict or None
        """
        now = float(timestamp) if timestamp is not None else time.time()
        annotated = frame.copy() if frame is not None else None

        # 1. Purge expired latches and stale pair evidence
        self.latched_incidents = {
            tid: info for tid, info in self.latched_incidents.items()
            if info["expiry"] > now
        }

        # Filter pair evidence for tracks no longer active
        active_ids = {v.track_id for v in tracked_vehicles}
        self.pair_evidence = {
            pair: ev for pair, ev in self.pair_evidence.items()
            if pair[0] in active_ids and pair[1] in active_ids and (now - ev["last_seen"] < 1.0)
        }

        max_score = 0.0
        primary_code = None
        frame_h, frame_w = frame.shape[:2] if frame is not None else (360, 640)
        involved_ids = set()
        incident_details = None

        n = len(tracked_vehicles)

        # 2. Pairwise Collision Evaluation (TID-01)
        for i in range(n):
            v1 = tracked_vehicles[i]
            for j in range(i + 1, n):
                v2 = tracked_vehicles[j]

                # TID-01 evaluates vehicle-to-vehicle collision; pedestrians are excluded
                if v1.class_name == "person" or v2.class_name == "person":
                    continue

                # Filter horizon micro-specks at vanishing points
                w1 = max(1.0, v1.bbox[2] - v1.bbox[0])
                h1 = max(1.0, v1.bbox[3] - v1.bbox[1])
                w2 = max(1.0, v2.bbox[2] - v2.bbox[0])
                h2 = max(1.0, v2.bbox[3] - v2.bbox[1])
                if min(w1, w2) < 20.0 or min(h1, h2) < 13.0 or min(v1.area, v2.area) < 220.0:
                    continue

                # Filter foreground camera occlusion mega-boxes
                if v1.area > 0.25 * frame_w * frame_h or v2.area > 0.25 * frame_w * frame_h:
                    continue

                pair_key = (min(v1.track_id, v2.track_id), max(v1.track_id, v2.track_id))
                score, is_confirmed_crash = self._evaluate_pairwise_collision(v1, v2, now)

                if score > 0.30:
                    ev = self.pair_evidence.setdefault(pair_key, {
                        "first_seen": now,
                        "last_seen": now,
                        "frame_count": 0,
                        "peak_score": 0.0
                    })
                    ev["last_seen"] = now
                    ev["frame_count"] += 1
                    ev["peak_score"] = max(ev["peak_score"], score)

                    # Temporal consistency boost after multi-frame observation
                    temporal_factor = min(1.0, 0.60 + 0.10 * ev["frame_count"])
                    score = min(1.0, score * temporal_factor)

                if score >= self.risk_threshold or is_confirmed_crash:
                    # Latch collision
                    expiry = now + self.latch_duration
                    cur_details = {
                        "incident_code": "TID-01 COLLISION",
                        "severity": "CRITICAL" if score >= 0.75 else "WARNING",
                        "risk_score": round(score, 2),
                        "vehicles_involved": f"{v1.class_name} #{v1.track_id}, {v2.class_name} #{v2.track_id}",
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(now)),
                        "description": f"Vehicle collision detected between #{v1.track_id} and #{v2.track_id}"
                    }
                    self.latched_incidents[v1.track_id] = {
                        "expiry": expiry, "code": "TID-01 COLLISION",
                        "partner_id": v2.track_id, "score": score,
                        "details": cur_details
                    }
                    self.latched_incidents[v2.track_id] = {
                        "expiry": expiry, "code": "TID-01 COLLISION",
                        "partner_id": v1.track_id, "score": score,
                        "details": cur_details
                    }
                    v1.in_collision = True
                    v2.in_collision = True
                    involved_ids.update([v1.track_id, v2.track_id])

                    if score > max_score:
                        max_score = score
                        primary_code = "TID-01 COLLISION"
                        incident_details = cur_details

        # 3. Single Vehicle Incidents (TID-02 Stopped Vehicle, TID-03 Wrong-Way)
        for v in tracked_vehicles:
            if v.class_name == "person":
                continue

            # TID-02: Stopped Vehicle
            # Requires vehicle that was previously moving in active traffic and is now stopped for >= threshold
            was_actively_moving = (v.max_historical_speed >= 25.0)
            if was_actively_moving and v.stationary_duration >= self.stopped_duration_threshold:
                s_stop = min(1.0, 0.70 + (v.stationary_duration - self.stopped_duration_threshold) * 0.08)
                if s_stop > max_score and v.track_id not in self.latched_incidents:
                    max_score = s_stop
                    primary_code = "TID-02 STOPPED_VEHICLE"
                    involved_ids.add(v.track_id)
                    incident_details = {
                        "incident_code": "TID-02 STOPPED_VEHICLE",
                        "severity": "WARNING",
                        "risk_score": round(s_stop, 2),
                        "vehicles_involved": f"{v.class_name} #{v.track_id}",
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(now)),
                        "description": f"Vehicle #{v.track_id} stationary in flow for {v.stationary_duration:.1f}s"
                    }

            # TID-03: Wrong-Way Driver
            if self.corridor_heading is not None and v.speed_px_s >= 20.0:
                heading_diff = abs(v.heading_deg - self.corridor_heading) % 360.0
                if heading_diff > 180.0:
                    heading_diff = 360.0 - heading_diff
                if heading_diff >= 120.0 and v.age >= 10:
                    s_wwd = 0.85
                    if s_wwd > max_score:
                        max_score = s_wwd
                        primary_code = "TID-03 WRONG_WAY"
                        involved_ids.add(v.track_id)
                        incident_details = {
                            "incident_code": "TID-03 WRONG_WAY",
                            "severity": "CRITICAL",
                            "risk_score": 0.85,
                            "vehicles_involved": f"{v.class_name} #{v.track_id}",
                            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(now)),
                            "description": f"Vehicle #{v.track_id} traveling opposite designated traffic heading"
                        }

        # 4. Integrate latched collisions into frame status
        for v in tracked_vehicles:
            if v.track_id in self.latched_incidents:
                latch = self.latched_incidents[v.track_id]
                v.in_collision = True
                involved_ids.add(v.track_id)
                if latch["score"] > max_score:
                    max_score = latch["score"]
                    primary_code = latch["code"]
                    if latch.get("details"):
                        incident_details = latch["details"]

        # 5. Determine high-level status
        if max_score >= self.risk_threshold:
            status = "CONFIRMED COLLISION" if primary_code == "TID-01 COLLISION" else "CONFIRMED INCIDENT"
            is_incident = True
        elif max_score >= 0.40:
            status = "POSSIBLE COLLISION"
            is_incident = False
        else:
            status = "NORMAL"
            is_incident = False

        # 6. Restrained CCTV Visual Rendering
        if annotated is not None:
            self._render_restrained_annotations(annotated, tracked_vehicles, involved_ids, is_incident)

        return annotated, round(max_score, 2), status, is_incident, incident_details

    def _evaluate_pairwise_collision(self, v1, v2, now):
        """
        Empirical multi-factor temporal collision evaluation:
          score = S_overlap * S_approach * S_rel_vel * S_post
        """
        # A. Spatial Overlap Score
        iou = compute_iou(v1.bbox, v2.bbox)
        dx = v2.center[0] - v1.center[0]
        dy = v2.center[1] - v1.center[1]
        dist = max(1.0, math.hypot(dx, dy))

        w1 = max(10.0, v1.bbox[2] - v1.bbox[0])
        w2 = max(10.0, v2.bbox[2] - v2.bbox[0])
        center_thresh = (w1 + w2) * 0.55

        if iou > 0.05:
            s_overlap = min(1.0, 0.40 + iou * 2.0)
        elif dist < center_thresh:
            s_overlap = max(0.0, 1.0 - (dist / center_thresh)) * 0.50
        else:
            return 0.0, False

        # B. Approach History Score (Were they closing in on each other?)
        # Look back up to 15 frames (~0.5s) in trajectory history
        k = min(len(v1.history), len(v2.history), 15)
        if k < 2:
            # Insufficient history (synthetic or provisional tracks)
            s_approach = 0.50
            delta_dist = 0.0
        else:
            prior_c1 = (v1.history[-k][1], v1.history[-k][2])
            prior_c2 = (v2.history[-k][1], v2.history[-k][2])
            dist_prior = math.hypot(prior_c2[0] - prior_c1[0], prior_c2[1] - prior_c1[1])
            delta_dist = dist_prior - dist

            # If distance did not decrease over the window, they are not approaching
            if delta_dist <= 2.0:
                s_approach = 0.0
            else:
                s_approach = min(1.0, delta_dist / 25.0)

        # Instantaneous closing velocity vector
        dvx = v2.velocity[0] - v1.velocity[0]
        dvy = v2.velocity[1] - v1.velocity[1]
        closing_speed = -(dx * dvx + dy * dvy) / dist
        if closing_speed > 10.0:
            s_approach = max(s_approach, min(1.0, closing_speed / 40.0))

        # Two stationary or co-moving vehicles that never approached have score 0
        if s_approach <= 0.05:
            return 0.0, False

        # C. Relative Velocity / Pre-Impact Motion Score
        max_prior_speed = max(v1.max_historical_speed, v2.max_historical_speed, v1.speed_px_s, v2.speed_px_s)
        if max_prior_speed < 12.0:
            # Neither vehicle ever moved at traffic speed (e.g. parked cars)
            return 0.0, False

        s_rel_vel = min(1.0, max_prior_speed / 45.0)

        # D. Post-Impact Response Score (Velocity drop, abrupt deceleration, or kinetic arrest)
        current_max_spd = max(v1.speed_px_s, v2.speed_px_s)
        speed_drop = max_prior_speed - current_max_spd

        # 1. Abrupt deceleration shock
        if speed_drop >= 20.0:
            s_post = min(1.0, 0.60 + (speed_drop / 50.0))
        # 2. Kinetic arrest (stopped and locked together after rapid approach)
        elif current_max_spd < 10.0 and max_prior_speed >= 25.0:
            s_post = 0.85
        # 3. Physical impact deformation with high overlap
        elif iou >= 0.15 and max_prior_speed >= 25.0:
            s_post = 0.80
        # 4. Passing vehicles in adjacent lanes maintain steady speed (speed_drop ~ 0)
        else:
            s_post = 0.10

        total_score = s_overlap * s_approach * s_rel_vel * s_post
        is_confirmed = (total_score >= self.risk_threshold) or (iou >= 0.18 and s_approach >= 0.50 and s_post >= 0.70)
        final_score = min(1.0, total_score * 1.5) if is_confirmed else total_score

        return final_score, is_confirmed

    def _render_restrained_annotations(self, frame, tracked_vehicles, involved_ids, is_incident):
        """Renders clean, restrained 1.5px bounding boxes without tactical gimmicks."""
        for v in tracked_vehicles:
            x1, y1, x2, y2 = [int(c) for c in v.bbox]
            in_collision = v.in_collision or (v.track_id in involved_ids)

            if in_collision:
                box_color = (0, 0, 255)  # Restrained red
                thickness = 2
            else:
                box_color = (200, 180, 50) if v.class_name == "person" else (220, 150, 40)
                thickness = 1

            # Simple clean rectangle
            cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, thickness)

            # Minimal label: class + ID
            label = f"{v.class_name} #{v.track_id}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.40, 1)
            y_label = max(y1 - 4, th + 2)

            # Restrained dark background for text readability
            cv2.rectangle(frame, (x1, y_label - th - 2), (x1 + tw + 4, y_label + 2), (20, 20, 20), -1)
            cv2.putText(frame, label, (x1 + 2, y_label), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255, 255, 255), 1, cv2.LINE_AA)


# Compatibility alias
EnterpriseAIDEngine = IncidentEngine

