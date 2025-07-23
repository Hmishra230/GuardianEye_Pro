import sqlite3
import os
import csv
import json
from datetime import datetime, timedelta
from contextlib import closing

DB_PATH = 'events.db'

class EventDB:
    def __init__(self, db_path=DB_PATH):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        with closing(sqlite3.connect(self.db_path)) as conn:
            c = conn.cursor()
            c.execute('''
                CREATE TABLE IF NOT EXISTS events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT,
                    object_id INTEGER,
                    speed REAL,
                    direction REAL,
                    classification TEXT,
                    bbox TEXT,
                    trajectory TEXT,
                    snapshot_path TEXT
                )
            ''')
            c.execute('''
                CREATE TABLE IF NOT EXISTS schema_migrations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    migration TEXT,
                    applied_at TEXT
                )
            ''')
            conn.commit()

    def insert_event(self, event):
        try:
            with closing(sqlite3.connect(self.db_path)) as conn:
                c = conn.cursor()
                
                # Prepare event data with proper defaults
                timestamp = event.get('timestamp', datetime.now().isoformat())
                object_id = event.get('object_id', 0)
                speed = event.get('speed', 0.0)
                direction = event.get('direction', 0.0)
                classification = event.get('classification', 'unknown')
                bbox = json.dumps(event.get('bbox', []))
                trajectory = json.dumps(event.get('trajectory', []))
                snapshot_path = event.get('snapshot_path', '')
                
                print(f"[DB] Inserting event: timestamp={timestamp}, object_id={object_id}, speed={speed}, classification={classification}")
                
                c.execute('''
                    INSERT INTO events (timestamp, object_id, speed, direction, classification, bbox, trajectory, snapshot_path)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    timestamp,
                    object_id,
                    speed,
                    direction,
                    classification,
                    bbox,
                    trajectory,
                    snapshot_path
                ))
                
                conn.commit()
                event_id = c.lastrowid
                print(f"[DB] Event successfully inserted with ID: {event_id}")
                return event_id
                
        except sqlite3.Error as e:
            print(f"[DB ERROR] SQLite error inserting event: {e}")
            raise
        except Exception as e:
            print(f"[DB ERROR] Unexpected error inserting event: {e}")
            raise

    def search_events(self, start_time=None, end_time=None, classification=None, min_speed=None, max_speed=None):
        query = 'SELECT * FROM events WHERE 1=1'
        params = []
        if start_time:
            query += ' AND timestamp >= ?'
            params.append(start_time)
        if end_time:
            query += ' AND timestamp <= ?'
            params.append(end_time)
        if classification:
            query += ' AND classification = ?'
            params.append(classification)
        if min_speed:
            query += ' AND speed >= ?'
            params.append(min_speed)
        if max_speed:
            query += ' AND speed <= ?'
            params.append(max_speed)
        with closing(sqlite3.connect(self.db_path)) as conn:
            c = conn.cursor()
            c.execute(query, params)
            rows = c.fetchall()
            return [self._row_to_event(row) for row in rows]

    def get_stats(self):
        with closing(sqlite3.connect(self.db_path)) as conn:
            c = conn.cursor()
            c.execute('SELECT COUNT(*), AVG(speed), MAX(speed), MIN(speed) FROM events')
            count, avg_speed, max_speed, min_speed = c.fetchone()
            return {
                'count': count,
                'avg_speed': avg_speed,
                'max_speed': max_speed,
                'min_speed': min_speed
            }

    def export_events(self, fmt='csv', out_path='events_export'):
        events = self.search_events()
        if fmt == 'csv':
            out_file = out_path + '.csv'
            with open(out_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['id', 'timestamp', 'object_id', 'speed', 'direction', 'classification', 'bbox', 'trajectory'])
                for ev in events:
                    writer.writerow([
                        ev['id'], ev['timestamp'], ev['object_id'], ev['speed'], ev['direction'],
                        ev['classification'], json.dumps(ev['bbox']), json.dumps(ev['trajectory'])
                    ])
            return out_file
        elif fmt == 'json':
            out_file = out_path + '.json'
            with open(out_file, 'w') as f:
                json.dump(events, f, indent=2)
            return out_file
        else:
            raise ValueError('Unsupported format')

    def cleanup_old(self, days=30):
        cutoff = (datetime.now() - timedelta(days=days)).isoformat()
        with closing(sqlite3.connect(self.db_path)) as conn:
            c = conn.cursor()
            c.execute('DELETE FROM events WHERE timestamp < ?', (cutoff,))
            conn.commit()

    def migrate(self, migration_sql, migration_name):
        with closing(sqlite3.connect(self.db_path)) as conn:
            c = conn.cursor()
            c.execute('SELECT COUNT(*) FROM schema_migrations WHERE migration = ?', (migration_name,))
            if c.fetchone()[0] == 0:
                c.executescript(migration_sql)
                c.execute('INSERT INTO schema_migrations (migration, applied_at) VALUES (?, ?)', (migration_name, datetime.now().isoformat()))
                conn.commit()

    def _row_to_event(self, row):
        return {
            'id': row[0],
            'timestamp': row[1],
            'object_id': row[2],
            'speed': row[3],
            'direction': row[4],
            'classification': row[5],
            'bbox': json.loads(row[6]) if row[6] else [],
            'trajectory': json.loads(row[7]) if row[7] else [],
            'snapshot_path': row[8]
        } 