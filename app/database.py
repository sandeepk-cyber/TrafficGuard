"""
TrafficGuard - Enterprise Incident & CAD Dispatch Database
SQLite persistent audit trail with zero emojis and standardized ITS classifications.
"""
import os
import sqlite3
import time
import logging

logger = logging.getLogger("trafficguard.db")

DB_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
DB_PATH = os.path.join(DB_DIR, "trafficguard.db")


def init_db(db_path=DB_PATH):
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS incidents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            time_epoch REAL,
            camera_name TEXT,
            location TEXT,
            latitude REAL DEFAULT 0.0,
            longitude REAL DEFAULT 0.0,
            incident_code TEXT DEFAULT 'TID-01 COLLISION_IMPACT',
            risk_score REAL,
            severity TEXT,
            status TEXT DEFAULT 'PENDING',
            speed_at_impact TEXT DEFAULT 'Unknown',
            vehicles_involved TEXT DEFAULT 'Vehicles',
            snapshot_base64 TEXT,
            description TEXT
        )
    """)
    # Automatic column additions for existing databases
    cursor.execute("PRAGMA table_info(incidents)")
    cols = [row[1] for row in cursor.fetchall()]
    for col, col_type in [
        ("latitude", "REAL DEFAULT 0.0"),
        ("longitude", "REAL DEFAULT 0.0"),
        ("incident_code", "TEXT DEFAULT 'TID-01 COLLISION_IMPACT'"),
        ("speed_at_impact", "TEXT DEFAULT 'Unknown'"),
        ("vehicles_involved", "TEXT DEFAULT 'Vehicles'")
    ]:
        if col not in cols:
            cursor.execute(f"ALTER TABLE incidents ADD COLUMN {col} {col_type}")
    conn.commit()
    conn.close()


class IncidentDatabase:
    def __init__(self, db_path=DB_PATH):
        self.db_path = db_path
        init_db(self.db_path)

    def _get_connection(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def record_incident(self, camera_name, location, risk_score, severity,
                        incident_code="TID-01 COLLISION_IMPACT",
                        speed_at_impact="Unknown", vehicles_involved="Vehicles",
                        latitude=37.7749, longitude=-122.4194,
                        snapshot_base64="", description=""):
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            now = time.time()
            cursor.execute("""
                INSERT INTO incidents (
                    time_epoch, camera_name, location, latitude, longitude,
                    incident_code, risk_score, severity, status, speed_at_impact,
                    vehicles_involved, snapshot_base64, description
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'PENDING', ?, ?, ?, ?)
            """, (now, camera_name, location, latitude, longitude,
                  incident_code, risk_score, severity, speed_at_impact,
                  vehicles_involved, snapshot_base64, description))
            incident_id = cursor.lastrowid
            conn.commit()
            conn.close()
            logger.info("Incident #%d recorded in SQLite DB (Code: %s, Score: %.2f)", incident_id, incident_code, risk_score)
            return incident_id
        except Exception as e:
            logger.error("Error recording incident to DB: %s", e)
            return None

    def get_incidents(self, limit=50):
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute("""
                SELECT id, timestamp, time_epoch, camera_name, location, latitude, longitude,
                       incident_code, risk_score, severity, status, speed_at_impact,
                       vehicles_involved, snapshot_base64, description
                FROM incidents
                ORDER BY id DESC
                LIMIT ?
            """, (limit,))
            rows = [dict(row) for row in cursor.fetchall()]
            conn.close()
            return rows
        except Exception as e:
            logger.error("Error fetching incidents: %s", e)
            return []

    def update_incident_status(self, incident_id: int, new_status: str):
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE incidents
                SET status = ?
                WHERE id = ?
            """, (new_status.upper(), incident_id))
            conn.commit()
            conn.close()
            logger.info("Incident #%d status updated to %s", incident_id, new_status.upper())
            return True
        except Exception as e:
            logger.error("Error updating incident #%d: %s", incident_id, e)
            return False

    def clear_all_incidents(self):
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute("DELETE FROM incidents")
            conn.commit()
            conn.close()
            logger.info("All incident records cleared from SQLite DB.")
            return True
        except Exception as e:
            logger.error("Error clearing incident records: %s", e)
            return False

    def get_stats(self):
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM incidents")
            total = cursor.fetchone()[0]
            cursor.execute("SELECT COUNT(*) FROM incidents WHERE status = 'PENDING'")
            pending = cursor.fetchone()[0]
            cursor.execute("SELECT COUNT(*) FROM incidents WHERE status = 'RESOLVED'")
            resolved = cursor.fetchone()[0]
            cursor.execute("SELECT COUNT(*) FROM incidents WHERE status = 'FALSE_POSITIVE'")
            false_positives = cursor.fetchone()[0]
            conn.close()
            return {
                "total_incidents": total,
                "pending": pending,
                "resolved": resolved,
                "false_positives": false_positives
            }
        except Exception as e:
            logger.error("Error fetching DB stats: %s", e)
            return {"total_incidents": 0, "pending": 0, "resolved": 0, "false_positives": 0}
