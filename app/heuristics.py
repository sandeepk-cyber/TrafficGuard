"""
TrafficGuard - Enterprise Automatic Incident Detection (AID) Heuristics Engine
Implements standardized ITS Incident Types:
  - TID-01: Collision & Kinetic Deceleration Shock
  - TID-02: Stationary Vehicle in Active Corridor
  - TID-03: Wrong-Way Driver Detection
  - TID-04: Queue / Sudden Congestion Shockwave
Zero emojis, commercial VMS visual symbology with 7-second persistent red collision latching.
"""
import math
import time
import cv2
import numpy as np


class EnterpriseAIDEngine:
    def __init__(self, risk_threshold=0.70, corridor_azimuth=0.0, collision_latch_duration=7.0):
        self.risk_threshold = risk_threshold
        self.corridor_azimuth = corridor_azimuth
        self.collision_latch_duration = collision_latch_duration
        self.last_incident_time = 0.0
        # Latched collision registry: track_id -> dict(expiry, incident_code, partner_id, risk, impact_pos)
        self.latched_collisions = {}

    def evaluate(self, frame, tracked_vehicles, overlays=None):
        """
        Evaluates ITS incident criteria and renders tactical HUD.
        Returns:
            annotated_frame: np.ndarray
            max_risk_score: float (0.0 to 1.0)
            severity: str ('NORMAL', 'WATCH', 'WARNING', 'CRITICAL')
            is_incident: bool
            incident_data: dict or None
        """
        if overlays is None:
            overlays = {"boxes": True, "vectors": True, "trails": True, "rings": True, "hud": True}

        annotated = frame.copy()
        h, w = frame.shape[:2]
        now = time.time()

        # 1. Clean up expired latched collisions
        self.latched_collisions = {
            tid: info for tid, info in self.latched_collisions.items()
            if info["expiry"] > now
        }

        max_risk = 0.0
        primary_incident_code = None
        critical_pair = None
        flagged_vehicle = None

        n = len(tracked_vehicles)

        # 2. Pairwise Collision / Convergence Analysis (Highest Priority)
        for i in range(n):
            v1 = tracked_vehicles[i]
            for j in range(i + 1, n):
                v2 = tracked_vehicles[j]

                dx = v2.center[0] - v1.center[0]
                dy = v2.center[1] - v1.center[1]
                dist = max(1.0, math.hypot(dx, dy))

                w1 = max(10.0, v1.bbox[2] - v1.bbox[0])
                h1 = max(10.0, v1.bbox[3] - v1.bbox[1])
                w2 = max(10.0, v2.bbox[2] - v2.bbox[0])
                h2 = max(10.0, v2.bbox[3] - v2.bbox[1])
                avg_size = ((math.hypot(w1, h1) + math.hypot(w2, h2)) / 2.0)
                norm_dist = dist / max(15.0, avg_size)

                # Bounding Box Overlap (IoU)
                xA = max(v1.bbox[0], v2.bbox[0])
                yA = max(v1.bbox[1], v2.bbox[1])
                xB = min(v1.bbox[2], v2.bbox[2])
                yB = min(v1.bbox[3], v2.bbox[3])
                inter = max(0.0, xB - xA) * max(0.0, yB - yA)
                a1 = w1 * h1
                a2 = w2 * h2
                iou = inter / float(a1 + a2 - inter + 1e-6)

                dvx = v2.velocity[0] - v1.velocity[0]
                dvy = v2.velocity[1] - v1.velocity[1]
                closing_speed = -(dx * dvx + dy * dvy) / dist

                a1_mag = math.hypot(v1.acceleration[0], v1.acceleration[1])
                a2_mag = math.hypot(v2.acceleration[0], v2.acceleration[1])

                is_colliding = False
                pair_risk = 0.0

                # Require confirmed tracks with history for kinematic collision evaluation
                if v1.hits < 3 or v2.hits < 3:
                    continue

                # 1. Person-to-person proximity is excluded from vehicle collision checks
                if v1.class_name == "person" and v2.class_name == "person":
                    continue

                # 2. Vehicle-to-Pedestrian Conflict
                if v1.class_name == "person" or v2.class_name == "person":
                    veh = v1 if v2.class_name == "person" else v2
                    ped = v2 if v2.class_name == "person" else v1
                    veh_hist_spd = getattr(veh, "max_historical_speed", veh.speed_kmh)
                    # Requires moving vehicle entering pedestrian space with physical overlap or closing vector
                    if (veh.speed_kmh >= 18.0 or veh_hist_spd >= 25.0):
                        if iou >= 0.12 or (dist < 25.0 and closing_speed > 15.0):
                            is_colliding = True
                            pair_risk = max(0.90, min(1.0, 0.82 + iou * 1.5))
                else:
                    # 3. Vehicle-to-Vehicle Collision
                    # Spatial displacement gate: At least one vehicle must have translated across the roadway (>= 25px)
                    # to prevent jittering parked cars sitting side-by-side from triggering false alarms.
                    if len(v1.history) >= 3 or len(v2.history) >= 3:
                        disp1 = math.hypot(v1.center[0] - v1.history[0][1], v1.center[1] - v1.history[0][2]) if len(v1.history) > 0 else 0.0
                        disp2 = math.hypot(v2.center[0] - v2.history[0][1], v2.center[1] - v2.history[0][2]) if len(v2.history) > 0 else 0.0
                        if max(disp1, disp2) < 25.0:
                            continue

                    max_spd = max(v1.speed_kmh, v2.speed_kmh)
                    max_hist_spd = max(
                        getattr(v1, "max_historical_speed", v1.speed_kmh),
                        getattr(v2, "max_historical_speed", v2.speed_kmh)
                    )
                    max_acc = max(a1_mag, a2_mag)

                    # Dynamic overlap collision: moving vehicles impacting with deformation/overlap
                    if iou >= 0.12 and (max_spd >= 18.0 or (max_hist_spd >= 25.0 and max_acc >= 110.0)):
                        is_colliding = True
                        pair_risk = max(0.92, min(1.0, 0.80 + iou * 1.6))
                    # High-speed kinetic convergence
                    elif dist < 45.0 and closing_speed > 25.0 and (max_spd >= 25.0 or max_acc >= 120.0):
                        is_colliding = True
                        pair_risk = max(0.88, min(1.0, 0.75 + (closing_speed / 100.0)))
                    # Post-collision stationary cluster (ONLY if prior collision latch or abrupt kinetic arrest)
                    elif v1.speed_kmh < 4.0 and v2.speed_kmh < 4.0 and norm_dist < 0.75:
                        has_prior_latch = (v1.track_id in self.latched_collisions or v2.track_id in self.latched_collisions)
                        had_kinetic_arrest = max_hist_spd >= 28.0 and max_acc >= 80.0
                        if has_prior_latch or had_kinetic_arrest:
                            is_colliding = True
                            pair_risk = 0.85

                if is_colliding and pair_risk >= 0.75:
                    expiry = now + self.collision_latch_duration
                    impact_pos = ((v1.center[0] + v2.center[0]) / 2.0, (v1.center[1] + v2.center[1]) / 2.0)
                    self.latched_collisions[v1.track_id] = {
                        "expiry": expiry,
                        "incident_code": "TID-01 COLLISION_IMPACT",
                        "partner_id": v2.track_id,
                        "risk": pair_risk,
                        "impact_pos": impact_pos
                    }
                    self.latched_collisions[v2.track_id] = {
                        "expiry": expiry,
                        "incident_code": "TID-01 COLLISION_IMPACT",
                        "partner_id": v1.track_id,
                        "risk": pair_risk,
                        "impact_pos": impact_pos
                    }
                    if pair_risk > max_risk:
                        max_risk = pair_risk
                        primary_incident_code = "TID-01 COLLISION_IMPACT"
                        critical_pair = (v1, v2, dist)

        # 3. Single-Vehicle Anomalies: TID-01 (Decel), TID-02 (Stopped), TID-03 (Wrong-Way)
        for v in tracked_vehicles:
            if v.class_name == "person" or v.hits < 4:
                continue

            # TID-01: Severe Deceleration Shock (Requires high corridor velocity, negative kinetic vector, and rapid arrest)
            accel_mag = math.hypot(v.acceleration[0], v.acceleration[1])
            is_decelerating = (v.velocity[0] * v.acceleration[0] + v.velocity[1] * v.acceleration[1]) < 0
            speed_drop = getattr(v, "max_historical_speed", 0.0) - v.speed_kmh
            tot_disp = math.hypot(v.center[0] - v.history[0][1], v.center[1] - v.history[0][2]) if len(v.history) > 0 else 0.0

            if tot_disp >= 18.0 and is_decelerating and speed_drop >= 25.0 and accel_mag > 180.0:
                s_decel = min(1.0, accel_mag / 300.0)
                if s_decel * 0.90 > max_risk:
                    max_risk = s_decel * 0.90
                    primary_incident_code = "TID-01 COLLISION_IMPACT"
                    flagged_vehicle = v
                self.latched_collisions[v.track_id] = {
                    "expiry": now + 5.0,
                    "incident_code": "TID-01 COLLISION_IMPACT",
                    "partner_id": None,
                    "risk": max(0.82, s_decel),
                    "impact_pos": v.center
                }

            # TID-02: Stationary Vehicle in Active Flow (Guards parked vehicles with zero corridor motion)
            if v.stationary_duration > 4.0 and getattr(v, "max_historical_speed", 0.0) >= 18.0:
                s_stop = min(1.0, 0.65 + (v.stationary_duration - 4.0) * 0.08)
                if s_stop > max_risk:
                    max_risk = s_stop
                    primary_incident_code = "TID-02 STOPPED_VEHICLE"
                    flagged_vehicle = v

            # TID-03: Wrong-Way Driver
            if self.corridor_azimuth != 0.0 and v.speed_kmh > 15.0:
                diff_angle = abs(v.heading_deg - self.corridor_azimuth) % 360.0
                if diff_angle > 180.0:
                    diff_angle = 360.0 - diff_angle
                if diff_angle > 125.0:
                    s_wwd = 0.92
                    if s_wwd > max_risk:
                        max_risk = s_wwd
                        primary_incident_code = "TID-03 WRONG_WAY_DRIVER"
                        flagged_vehicle = v

        # 4. Integrate Latched Collisions into Current Frame Risk
        for v in tracked_vehicles:
            if v.track_id in self.latched_collisions:
                latch_data = self.latched_collisions[v.track_id]
                v.in_collision = True
                if latch_data["risk"] > max_risk:
                    max_risk = latch_data["risk"]
                    primary_incident_code = latch_data["incident_code"]

        # 5. Classify Severity Level
        if max_risk >= 0.80:
            severity = "CRITICAL"
        elif max_risk >= 0.60:
            severity = "WARNING"
        elif max_risk >= 0.35:
            severity = "WATCH"
        else:
            severity = "NORMAL"

        is_incident = max_risk >= self.risk_threshold

        # 6. Tactical HUD Rendering (Zero Emojis, Vivid Solid Red for Collisions)
        drawn_pairs = set()

        for v in tracked_vehicles:
            in_latched_collision = (v.track_id in self.latched_collisions)
            is_critical = (in_latched_collision or
                           (critical_pair and (v.track_id == critical_pair[0].track_id or v.track_id == critical_pair[1].track_id)) or
                           (flagged_vehicle and v.track_id == flagged_vehicle.track_id and severity == "CRITICAL"))
            is_warning = (severity == "WARNING" and flagged_vehicle and v.track_id == flagged_vehicle.track_id)

            if is_critical:
                color = (0, 0, 255)       # Bright Solid Pure Red (BGR: 0, 0, 255)
                box_thickness = 2
                bracket_thickness = 3
            elif is_warning:
                color = (0, 165, 255)     # Warning Amber
                box_thickness = 1
                bracket_thickness = 2
            else:
                color = (0, 220, 100)     # Normal Neon Green
                box_thickness = 1
                bracket_thickness = 2

            # A. Trajectory History Polyline
            if overlays.get("trails", True) and len(v.history) > 1:
                pts = np.array([[int(pt[1]), int(pt[2])] for pt in v.history], dtype=np.int32)
                cv2.polylines(annotated, [pts], False, color, 1, cv2.LINE_AA)

            # B. Precision Bounding Box & Reinforced Corner Brackets
            x1, y1, x2, y2 = [int(coord) for coord in v.bbox]

            if overlays.get("boxes", True):
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, box_thickness)

                # Reinforced Corner Brackets
                b_len = min(16, int((x2 - x1) * 0.28))
                cv2.line(annotated, (x1, y1), (x1 + b_len, y1), color, bracket_thickness)
                cv2.line(annotated, (x1, y1), (x1, y1 + b_len), color, bracket_thickness)
                cv2.line(annotated, (x2, y1), (x2 - b_len, y1), color, bracket_thickness)
                cv2.line(annotated, (x2, y1), (x2, y1 + b_len), color, bracket_thickness)
                cv2.line(annotated, (x1, y2), (x1 + b_len, y2), color, bracket_thickness)
                cv2.line(annotated, (x1, y2), (x1, y2 - b_len), color, bracket_thickness)
                cv2.line(annotated, (x2, y2), (x2 - b_len, y2), color, bracket_thickness)
                cv2.line(annotated, (x2, y2), (x2, y2 - b_len), color, bracket_thickness)

            # C. Velocity Vector Arrow
            if overlays.get("vectors", True):
                vx, vy = v.velocity
                v_len = math.hypot(vx, vy)
                if v_len > 3.0:
                    end_x = int(v.center[0] + (vx / v_len) * min(26, v_len * 0.35))
                    end_y = int(v.center[1] + (vy / v_len) * min(26, v_len * 0.35))
                    vec_color = (0, 0, 255) if is_critical else (0, 240, 240)
                    cv2.arrowedLine(annotated, (int(v.center[0]), int(v.center[1])), (end_x, end_y),
                                    vec_color, 1, tipLength=0.35)

            # D. Dynamic Collision Target Reticle (for Critical Vehicles)
            if is_critical and overlays.get("rings", True):
                pulse_rad = int(22 + 8 * math.sin(now * 9.0))
                cx, cy = int(v.center[0]), int(v.center[1])
                cv2.circle(annotated, (cx, cy), pulse_rad, (0, 0, 255), 2, cv2.LINE_AA)
                cv2.drawMarker(annotated, (cx, cy), (0, 0, 255), cv2.MARKER_CROSS, 18, 2)

            # E. Precision Data Label Badge
            if overlays.get("boxes", True):
                if is_critical:
                    tag = f"[COLLISION IMPACT] {v.class_name.upper()} #{v.track_id} | {int(v.speed_kmh)} KM/H"
                    bg_color = (15, 15, 180)   # Dark Crimson Red Badge
                    text_color = (255, 255, 255)
                elif is_warning:
                    tag = f"[WARNING] {v.class_name.upper()} #{v.track_id} | {int(v.speed_kmh)} KM/H"
                    bg_color = (15, 110, 190)  # Amber Badge
                    text_color = (255, 255, 255)
                else:
                    tag = f"{v.class_name.upper()} #{v.track_id} | {int(v.speed_kmh)} KM/H"
                    if v.stationary_duration > 2.0:
                        tag += f" [STOP {int(v.stationary_duration)}S]"
                    bg_color = (12, 16, 24)
                    text_color = (230, 240, 255)

                (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)
                badge_y1 = max(0, y1 - 20)
                badge_y2 = max(20, y1)
                cv2.rectangle(annotated, (x1, badge_y1), (x1 + tw + 8, badge_y2), bg_color, -1)
                cv2.rectangle(annotated, (x1, badge_y1), (x1 + tw + 8, badge_y2), color, 1)
                cv2.putText(annotated, tag, (x1 + 4, max(14, y1 - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.38, text_color, 1, cv2.LINE_AA)

            # F. Connecting Vector Line to Colliding Partner
            if in_latched_collision:
                partner_id = self.latched_collisions[v.track_id].get("partner_id")
                if partner_id:
                    pair_key = tuple(sorted([v.track_id, partner_id]))
                    if pair_key not in drawn_pairs:
                        drawn_pairs.add(pair_key)
                        partner = next((p for p in tracked_vehicles if p.track_id == partner_id), None)
                        if partner:
                            pt1 = (int(v.center[0]), int(v.center[1]))
                            pt2 = (int(partner.center[0]), int(partner.center[1]))
                            cv2.line(annotated, pt1, pt2, (0, 0, 255), 2, cv2.LINE_AA)
                            mid = ((pt1[0] + pt2[0]) // 2, (pt1[1] + pt2[1]) // 2)
                            cv2.drawMarker(annotated, mid, (0, 0, 255), cv2.MARKER_TILTED_CROSS, 20, 2)
                            cv2.circle(annotated, mid, int(26 + 6 * math.sin(now * 11.0)), (0, 0, 255), 2, cv2.LINE_AA)

        # 7. Header Alarm Banner (When Incident Active)
        incident_data = None
        if is_incident:
            code_str = primary_incident_code or "TID-01 COLLISION_IMPACT"
            cv2.rectangle(annotated, (0, 0), (w, 36), (15, 15, 220), -1)
            cv2.line(annotated, (0, 36), (w, 36), (255, 255, 255), 1)
            cv2.putText(annotated, f"INCIDENT ALARM [{code_str}] - OPERATOR VERIFICATION REQUIRED",
                        (20, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 1, cv2.LINE_AA)

            involved_list = []
            for tid in self.latched_collisions:
                veh = next((v for v in tracked_vehicles if v.track_id == tid), None)
                if veh:
                    involved_list.append(f"{veh.class_name} #{veh.track_id}")
            if not involved_list and critical_pair:
                involved_list = [f"{critical_pair[0].class_name} #{critical_pair[0].track_id}",
                                 f"{critical_pair[1].class_name} #{critical_pair[1].track_id}"]
            elif not involved_list and flagged_vehicle:
                involved_list = [f"{flagged_vehicle.class_name} #{flagged_vehicle.track_id}"]

            incident_data = {
                "incident_code": code_str,
                "risk_score": round(max_risk, 2),
                "severity": severity,
                "description": f"{code_str}: {', '.join(involved_list) if involved_list else 'Traffic Corridor'} involved with peak risk {int(max_risk * 100)}%",
                "vehicles_involved": ", ".join(involved_list) if involved_list else "Corridor Traffic",
                "speed_at_impact": f"{int(tracked_vehicles[0].speed_kmh if tracked_vehicles else 0)} km/h"
            }

        # 8. Telemetry HUD Card at bottom-left
        if overlays.get("hud", True):
            card_w, card_h = 240, 42
            cv2.rectangle(annotated, (12, h - card_h - 12), (12 + card_w, h - 12), (8, 12, 20), -1)
            cv2.rectangle(annotated, (12, h - card_h - 12), (12 + card_w, h - 12), (30, 45, 68), 1)

            fill_w = int(100 * max_risk)
            bar_color = (0, 220, 100) if severity == "NORMAL" else (
                (0, 190, 255) if severity == "WATCH" else (
                    (0, 120, 255) if severity == "WARNING" else (0, 0, 255)
                )
            )
            cv2.rectangle(annotated, (85, h - 40), (85 + fill_w, h - 28), bar_color, -1)
            cv2.rectangle(annotated, (85, h - 40), (185, h - 28), (60, 75, 95), 1)

            cv2.putText(annotated, f"RISK {int(max_risk * 100)}%", (20, h - 29),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.40, (220, 230, 245), 1, cv2.LINE_AA)
            cv2.putText(annotated, severity, (194, h - 29),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, bar_color, 1, cv2.LINE_AA)

        return annotated, max_risk, severity, is_incident, incident_data

