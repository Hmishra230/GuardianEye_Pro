from flask import Flask, render_template, jsonify, request
from datetime import datetime, timedelta
import sqlite3
import json
from collections import defaultdict, Counter
import os

class VideoAnalyticsDashboard:
    def __init__(self, db_path='events.db'):
        self.db_path = db_path
    
    def get_dashboard_stats(self, start_date=None, end_date=None, camera_filter=None, category_filter=None):
        """Get comprehensive dashboard statistics"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Base query
            base_query = """
                SELECT timestamp, classification, snapshot_path, object_id, speed, direction, bbox
                FROM events 
                WHERE 1=1
            """
            params = []
            
            # Add filters
            if start_date:
                base_query += " AND date(timestamp) >= ?"
                params.append(start_date)
            if end_date:
                base_query += " AND date(timestamp) <= ?"
                params.append(end_date)
            if camera_filter and camera_filter != 'all':
                base_query += " AND snapshot_path LIKE ?"
                params.append(f'%{camera_filter}%')
            if category_filter and category_filter != 'all':
                base_query += " AND classification = ?"
                params.append(category_filter)
            
            base_query += " ORDER BY timestamp DESC"
            
            cursor.execute(base_query, params)
            events = cursor.fetchall()
            
            # Process data for dashboard
            stats = self._process_events_for_dashboard(events)
            conn.close()
            
            return stats
            
        except Exception as e:
            print(f"Error getting dashboard stats: {e}")
            return self._get_empty_stats()
    
    def _process_events_for_dashboard(self, events):
        """Process events data for dashboard widgets"""
        if not events:
            return self._get_empty_stats()
        
        # Initialize counters
        daily_counts = defaultdict(int)
        camera_counts = defaultdict(int)
        category_counts = defaultdict(int)
        hourly_trends = defaultdict(int)
        recent_events = []
        
        # Process each event
        for event in events:
            timestamp_str, classification, snapshot_path, object_id, speed, direction, bbox = event
            
            try:
                # Parse timestamp
                timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                date_str = timestamp.strftime('%Y-%m-%d')
                hour = timestamp.hour
                
                # Count by date
                daily_counts[date_str] += 1
                
                # Count by camera (extract from snapshot path)
                camera = self._extract_camera_from_path(snapshot_path)
                camera_counts[camera] += 1
                
                # Count by category
                category_counts[classification or 'Unknown'] += 1
                
                # Hourly trends
                hourly_trends[hour] += 1
                
                # Recent events (last 10)
                if len(recent_events) < 10:
                    recent_events.append({
                        'timestamp': timestamp_str,
                        'classification': classification or 'Unknown',
                        'camera': camera,
                        'snapshot_path': snapshot_path,
                        'speed': speed or 0,
                        'direction': direction or 0
                    })
                    
            except Exception as e:
                print(f"Error processing event: {e}")
                continue
        
        # Prepare trend data (last 7 days)
        trend_data = self._prepare_trend_data(daily_counts)
        
        return {
            'total_detections': len(events),
            'daily_counts': dict(daily_counts),
            'camera_counts': dict(camera_counts),
            'category_counts': dict(category_counts),
            'hourly_trends': dict(hourly_trends),
            'trend_data': trend_data,
            'recent_events': recent_events,
            'top_cameras': self._get_top_items(camera_counts, 5),
            'top_categories': self._get_top_items(category_counts, 5)
        }
    
    def _extract_camera_from_path(self, snapshot_path):
        """Extract camera identifier from snapshot path"""
        if not snapshot_path:
            return 'Camera-Unknown'
        
        # Try to extract camera info from path
        path_parts = snapshot_path.split('/')
        for part in path_parts:
            if 'cam' in part.lower() or 'camera' in part.lower():
                return part
        
        # Default naming
        return f'Camera-{hash(snapshot_path) % 10 + 1}'
    
    def _prepare_trend_data(self, daily_counts):
        """Prepare trend data for the last 7 days"""
        today = datetime.now().date()
        trend_data = []
        
        for i in range(6, -1, -1):
            date = today - timedelta(days=i)
            date_str = date.strftime('%Y-%m-%d')
            count = daily_counts.get(date_str, 0)
            trend_data.append({
                'date': date_str,
                'count': count,
                'label': date.strftime('%m/%d')
            })
        
        return trend_data
    
    def _get_top_items(self, counter_dict, limit):
        """Get top items from counter dictionary"""
        return [{'name': k, 'count': v} for k, v in 
                sorted(counter_dict.items(), key=lambda x: x[1], reverse=True)[:limit]]
    
    def _get_empty_stats(self):
        """Return empty stats structure"""
        return {
            'total_detections': 0,
            'daily_counts': {},
            'camera_counts': {},
            'category_counts': {},
            'hourly_trends': {},
            'trend_data': [],
            'recent_events': [],
            'top_cameras': [],
            'top_categories': []
        }
    
    def get_filtered_events(self, start_date=None, end_date=None, camera_filter=None, category_filter=None, limit=100):
        """Get filtered events for logs view"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            query = """
                SELECT id, timestamp, classification, snapshot_path, object_id, speed, direction, bbox
                FROM events 
                WHERE 1=1
            """
            params = []
            
            # Add filters
            if start_date:
                query += " AND date(timestamp) >= ?"
                params.append(start_date)
            if end_date:
                query += " AND date(timestamp) <= ?"
                params.append(end_date)
            if camera_filter and camera_filter != 'all':
                query += " AND snapshot_path LIKE ?"
                params.append(f'%{camera_filter}%')
            if category_filter and category_filter != 'all':
                query += " AND classification = ?"
                params.append(category_filter)
            
            query += " ORDER BY timestamp DESC LIMIT ?"
            params.append(limit)
            
            cursor.execute(query, params)
            events = cursor.fetchall()
            conn.close()
            
            # Format events
            formatted_events = []
            for event in events:
                event_id, timestamp_str, classification, snapshot_path, object_id, speed, direction, bbox = event
                
                formatted_events.append({
                    'id': event_id,
                    'timestamp': timestamp_str,
                    'classification': classification or 'Unknown',
                    'camera': self._extract_camera_from_path(snapshot_path),
                    'snapshot_path': snapshot_path,
                    'object_id': object_id,
                    'speed': speed or 0,
                    'direction': direction or 0,
                    'bbox': bbox
                })
            
            return formatted_events
            
        except Exception as e:
            print(f"Error getting filtered events: {e}")
            return []
    
    def get_available_cameras(self):
        """Get list of available cameras from events"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("SELECT DISTINCT snapshot_path FROM events WHERE snapshot_path IS NOT NULL")
            paths = cursor.fetchall()
            conn.close()
            
            cameras = set()
            for path_tuple in paths:
                camera = self._extract_camera_from_path(path_tuple[0])
                cameras.add(camera)
            
            return sorted(list(cameras))
            
        except Exception as e:
            print(f"Error getting cameras: {e}")
            return []
    
    def get_available_categories(self):
        """Get list of available object categories"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("SELECT DISTINCT classification FROM events WHERE classification IS NOT NULL")
            categories = cursor.fetchall()
            conn.close()
            
            return sorted([cat[0] for cat in categories if cat[0]])
            
        except Exception as e:
            print(f"Error getting categories: {e}")
            return []
