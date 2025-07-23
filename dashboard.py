import cv2
import numpy as np
import os
import time
import threading
import sqlite3
from datetime import datetime, timedelta
from flask import Flask, render_template, Response, request, jsonify, send_from_directory, redirect, url_for
from werkzeug.utils import secure_filename
import json

# Import from existing modules
from motion_detector import MotionDetector
from alert_system import AlertSystem
from roi_manager import ROIManager
from event_db import EventDB
import config

# Create Flask application
app = Flask(__name__)
app.config['SECRET_KEY'] = 'intrusion_detection_dashboard'
app.config['UPLOAD_FOLDER'] = 'uploaded_videos'
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max upload size

# Ensure directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs('snapshots', exist_ok=True)
os.makedirs('recorded_streams', exist_ok=True)

# Global variables
class DashboardState:
    def __init__(self):
        self.frame = None
        self.processed_frame = None
        self.detection_active = False
        self.roi_points = []
        self.video_source = None
        self.video_path = None
        self.rtsp_url = None
        self.source_type = None  # 'webcam', 'rtsp', 'file'
        self.event_count = 0
        self.events = []
        self.lock = threading.Lock()
        self.recording = False
        self.recording_path = None
        self.video_writer = None
        self.fps = 20
        self.frame_width = 640
        self.frame_height = 480
        self.settings = {
            'motion_threshold': config.MOTION_DETECTION_THRESHOLD,
            'min_area': config.MIN_OBJECT_AREA,
            'max_area': config.MAX_OBJECT_AREA,
            'min_speed': 5,  # Lowered from config.MIN_SPEED_THRESHOLD (50) for testing
            'morph_kernel': config.MORPH_KERNEL_SIZE[0],
        }

state = DashboardState()
motion_detector = MotionDetector()
alert_system = AlertSystem()
event_db = EventDB()

# Video processing thread
class VideoThread(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)
        self.daemon = True
        self.stop_event = threading.Event()
    
    def stop(self):
        self.stop_event.set()
    
    def run(self):
        global state, motion_detector, alert_system
        
        cap = None
        last_frame_time = time.time()
        
        while not self.stop_event.is_set():
            try:
                # Initialize or reinitialize video source if needed
                if cap is None or not cap.isOpened():
                    if state.source_type == 'webcam' and state.video_source is not None:
                        cap = cv2.VideoCapture(state.video_source)
                    elif state.source_type == 'rtsp' and state.rtsp_url:
                        cap = cv2.VideoCapture(state.rtsp_url)
                    elif state.source_type == 'file' and state.video_path:
                        cap = cv2.VideoCapture(state.video_path)
                    else:
                        time.sleep(0.1)
                        continue
                        
                if not state.detection_active:
                    print(f"[DEBUG] Detection not active")
                elif len(state.roi_points) < 3:
                    print(f"[DEBUG] Not enough ROI points: {len(state.roi_points)}")
                    
                    # Set frame dimensions
                    if cap.isOpened():
                        state.frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        state.frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        state.fps = cap.get(cv2.CAP_PROP_FPS)
                        if state.fps <= 0:
                            state.fps = 20  # Default FPS if not available
                
                # Read frame
                ret, frame = cap.read()
                
                if not ret:
                    # End of video file or stream error
                    if state.source_type == 'file':
                        # Loop video file if it's ended
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        continue
                    else:
                        # Reconnect for stream errors
                        cap.release()
                        cap = None
                        time.sleep(1)
                        continue
                
                # Store original frame
                with state.lock:
                    state.frame = frame.copy()
                
                # Process frame if detection is active
                if state.detection_active and len(state.roi_points) >= 3:
                    print(f"[DEBUG] Detection active with {len(state.roi_points)} ROI points")
                    # Create ROI mask
                    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
                    pts = np.array(state.roi_points, np.int32)
                    cv2.fillPoly(mask, [pts], 255)
                    
                    # Save ROI mask and frame periodically for debugging
                    frame_count = getattr(self, 'frame_count', 0) + 1
                    setattr(self, 'frame_count', frame_count)
                    
                    if frame_count % 30 == 0:  # Save every 30 frames
                        cv2.imwrite(f"debug_roi_mask_{frame_count}.jpg", mask)
                        cv2.imwrite(f"debug_frame_{frame_count}.jpg", frame)
                        print(f"[DEBUG] Saved ROI mask and frame at frame {frame_count}")
                        print(f"[DEBUG] ROI mask non-zero pixels: {np.count_nonzero(mask)}")
                        print(f"[DEBUG] ROI points used: {state.roi_points}")
                    
                    # Process frame with motion detector
                    try:
                        # Apply settings to motion detector
                        motion_detector.bg_subtractor.setVarThreshold(state.settings['motion_threshold'])
                        
                        # Process frame with motion detector, passing the current settings
                        results = motion_detector.process(
                            frame, 
                            mask, 
                            min_area=state.settings['min_area'], 
                            max_area=state.settings['max_area'],
                            morph_kernel=state.settings['morph_kernel']
                        )
                        print(f"[DEBUG] Motion detector processed frame, settings: threshold={state.settings['motion_threshold']}, min_speed={state.settings['min_speed']}, min_area={state.settings['min_area']}, max_area={state.settings['max_area']}")
                        print(f"[DEBUG] Mask shape: {mask.shape}, non-zero pixels: {np.count_nonzero(mask)}")
                        print(f"[DEBUG] Frame shape: {frame.shape}")
                    except Exception as e:
                        print(f"[ERROR] Exception in motion detection: {e}")
                        results = []
                    processed_frame = frame.copy()
                    
                    # Draw ROI on frame
                    cv2.polylines(processed_frame, [pts], True, (0, 255, 0), 2)
                    
                    # Debug: Print detection results
                    print(f"[DEBUG] Detection active, found {len(results)} potential objects")
                    
                    # Process detected objects
                    for obj in results:
                        centroid = obj['centroid']
                        x, y, w, h = obj['bbox']
                        speed = obj['speed']
                        area = w * h
                        
                        print(f"[DEBUG] Object: speed={speed:.1f}, area={area:.1f}, min_speed={state.settings['min_speed']}, min_area={state.settings['min_area']}, max_area={state.settings['max_area']}")
                        
                        # Apply filtering based on settings
                        if speed < state.settings['min_speed']:
                            print(f"[DEBUG] Object rejected: speed {speed:.1f} < min_speed {state.settings['min_speed']}")
                            continue
                        if area < state.settings['min_area'] or area > state.settings['max_area']:
                            print(f"[DEBUG] Object rejected: area {area:.1f} outside range [{state.settings['min_area']}-{state.settings['max_area']}]")
                            continue
                        
                        print(f"[DEBUG] Object ACCEPTED: speed={speed:.1f}, area={area:.1f}")
                        
                        # Generate alert
                        cv2.rectangle(processed_frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                        cv2.putText(processed_frame, f'Speed: {speed:.1f}', (x, y - 10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                        cv2.putText(processed_frame, f'ALERT!', (x, y + h + 20), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        
                        # Save event
                        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                        # Save snapshot first
                        snapshot_filename = f'event_{state.event_count + 1}_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.jpg'
                        snapshot_path = os.path.join('snapshots', snapshot_filename)
                        cv2.imwrite(snapshot_path, processed_frame)
                        
                        event = {
                            'timestamp': timestamp,
                            'object_id': obj.get('id', 0),
                            'speed': speed,
                            'direction': obj.get('direction', 0),
                            'classification': obj.get('classification', 'unknown'),
                            'bbox': [x, y, w, h],
                            'trajectory': obj.get('trajectory', []),
                            'snapshot_path': snapshot_path
                        }
                        
                        # Save to database with error handling
                        try:
                            event_id = event_db.insert_event(event)
                            print(f"[SUCCESS] Event saved to database: ID={event_id}, timestamp={timestamp}")
                            
                            # Add the database ID to the event for consistency
                            event['id'] = event_id
                            
                        except Exception as db_error:
                            print(f"[ERROR] Failed to save event to database: {db_error}")
                            # Continue processing even if database save fails
                        
                        # Add to recent events list
                        with state.lock:
                            state.events.append(event)
                            state.event_count += 1
                            print(f"[DEBUG] Event added to memory. Total events in memory: {len(state.events)}")
                            # Keep only recent events in memory
                            if len(state.events) > 100:
                                state.events = state.events[-100:]
                                print(f"[DEBUG] Trimmed events list to 100 most recent events")
                        
                        # Log successful detection for verification
                        print(f"[ALERT] INTRUSION DETECTED! Speed: {speed:.1f}, Area: {area:.1f}, Snapshot: {snapshot_path}")
                        

                    
                    with state.lock:
                        state.processed_frame = processed_frame
                else:
                    # Just draw ROI on frame if detection is not active
                    if len(state.roi_points) >= 3:
                        processed_frame = frame.copy()
                        pts = np.array(state.roi_points, np.int32)
                        cv2.polylines(processed_frame, [pts], True, (0, 255, 0), 2)
                        with state.lock:
                            state.processed_frame = processed_frame
                    else:
                        with state.lock:
                            state.processed_frame = frame.copy()
                
                # Handle recording if active
                if state.recording and state.video_writer is not None:
                    state.video_writer.write(frame)
                
                # Control frame rate
                elapsed = time.time() - last_frame_time
                sleep_time = max(0, 1.0/state.fps - elapsed)
                if sleep_time > 0:
                    time.sleep(sleep_time)
                last_frame_time = time.time()
                
            except Exception as e:
                print(f"Error in video processing thread: {e}")
                time.sleep(0.5)

# Start video thread
video_thread = None

# Routes
@app.route('/')
def index():
    return render_template('dashboard.html')

@app.route('/logs')
def logs():
    return render_template('logs.html')

@app.route('/video_feed')
def video_feed():
    def generate_frames():
        while True:
            with state.lock:
                if state.processed_frame is not None:
                    frame = state.processed_frame.copy()
                elif state.frame is not None:
                    frame = state.frame.copy()
                else:
                    # Generate blank frame if no video
                    frame = np.zeros((480, 640, 3), dtype=np.uint8)
                    cv2.putText(frame, "No video source connected", (50, 240), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                
                # Draw ROI on the frame if points exist
                if state.roi_points and len(state.roi_points) >= 3:
                    # Draw filled polygon with transparency
                    overlay = frame.copy()
                    points = np.array(state.roi_points, np.int32)
                    points = points.reshape((-1, 1, 2))
                    cv2.fillPoly(overlay, [points], (0, 255, 0, 64))
                    cv2.addWeighted(overlay, 0.2, frame, 0.8, 0, frame)
                    
                    # Draw ROI outline
                    cv2.polylines(frame, [points], True, (0, 255, 0), 2)
                    
                    # Draw points with numbers
                    for i, point in enumerate(state.roi_points):
                        cv2.circle(frame, tuple(point), 5, (0, 255, 0), -1)
                        cv2.putText(frame, str(i+1), (point[0]+5, point[1]+5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Add timestamp
            cv2.putText(frame, datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 
                        (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                        0.6, (255, 255, 255), 1)
            
            # Convert to JPEG
            ret, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            
            time.sleep(0.033)  # ~30 FPS
    
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/set_source', methods=['POST'])
def set_source():
    global video_thread, state
    
    data = request.json
    source_type = data.get('source_type')
    
    # Stop existing video thread
    if video_thread is not None:
        video_thread.stop()
        video_thread.join(timeout=2)
    
    # Reset state
    with state.lock:
        state.frame = None
        state.processed_frame = None
        state.source_type = source_type
    
    if source_type == 'webcam':
        camera_id = int(data.get('camera_id', 0))
        with state.lock:
            state.video_source = camera_id
    
    elif source_type == 'rtsp':
        rtsp_url = data.get('rtsp_url')
        with state.lock:
            state.rtsp_url = rtsp_url
    
    elif source_type == 'file':
        # File will be uploaded separately
        pass
    
    # Start new video thread
    video_thread = VideoThread()
    video_thread.start()
    
    return jsonify({'status': 'success'})

@app.route('/api/upload_video', methods=['POST'])
def upload_video():
    global video_thread, state
    
    if 'video' not in request.files:
        return jsonify({'status': 'error', 'message': 'No video file uploaded'})
    
    file = request.files['video']
    if file.filename == '':
        return jsonify({'status': 'error', 'message': 'No file selected'})
    
    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Stop existing video thread
        if video_thread is not None:
            video_thread.stop()
            video_thread.join(timeout=2)
        
        # Set new video source
        with state.lock:
            state.source_type = 'file'
            state.video_path = filepath
        
        # Start new video thread
        video_thread = VideoThread()
        video_thread.start()
        
        return jsonify({'status': 'success', 'filename': filename})
    
    return jsonify({'status': 'error', 'message': 'Failed to upload file'})

@app.route('/api/set_roi', methods=['GET', 'POST'])
def set_roi():
    if request.method == 'POST':
        data = request.json
        points = data.get('points', [])
        
        print(f"[DEBUG] Setting ROI points: {points}")
        
        with state.lock:
            state.roi_points = points
        
        # Save ROI points to file for persistence
        try:
            np.save(config.ROI_SAVE_PATH, np.array(state.roi_points))
            print(f"[DEBUG] Saved ROI points to {config.ROI_SAVE_PATH}")
        except Exception as e:
            print(f"[ERROR] Failed to save ROI points: {e}")
        
        return jsonify({'status': 'success'})
    else:  # GET method
        # Load ROI points from file if available
        if os.path.exists(config.ROI_SAVE_PATH):
            try:
                loaded_points = np.load(config.ROI_SAVE_PATH).tolist()
                with state.lock:
                    if not state.roi_points:  # Only update if not already set
                        state.roi_points = loaded_points
            except Exception as e:
                print(f"Error loading ROI points: {e}")
        
        with state.lock:
            return jsonify({'points': state.roi_points})

@app.route('/api/toggle_detection', methods=['POST'])
def toggle_detection():
    data = request.json
    active = data.get('active', False)
    
    with state.lock:
        state.detection_active = active
    
    print(f"[DEBUG] Detection toggled: {active}, ROI points: {len(state.roi_points)}")
    print(f"[DEBUG] Current settings: {state.settings}")
    
    # Force reload ROI points from file
    if os.path.exists(config.ROI_SAVE_PATH):
        try:
            loaded_points = np.load(config.ROI_SAVE_PATH).tolist()
            with state.lock:
                state.roi_points = loaded_points
                print(f"[DEBUG] Loaded ROI points from file: {state.roi_points}")
        except Exception as e:
            print(f"[ERROR] Failed to load ROI points: {e}")
    
    return jsonify({'status': 'success', 'active': state.detection_active})

@app.route('/api/toggle_recording', methods=['POST'])
def toggle_recording():
    global state
    
    data = request.json
    recording = data.get('recording', False)
    
    with state.lock:
        if recording and not state.recording:
            # Start recording
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            state.recording_path = os.path.join('recorded_streams', f'recording_{timestamp}.avi')
            
            # Create video writer
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            state.video_writer = cv2.VideoWriter(
                state.recording_path, fourcc, state.fps, 
                (state.frame_width, state.frame_height)
            )
            state.recording = True
            
        elif not recording and state.recording:
            # Stop recording
            if state.video_writer is not None:
                state.video_writer.release()
                state.video_writer = None
            state.recording = False
    
    return jsonify({'status': 'success', 'recording': state.recording})

@app.route('/api/get_events')
def get_events():
    # Get query parameters for filtering
    start_date = request.args.get('start_date')
    end_date = request.args.get('end_date')
    limit = int(request.args.get('limit', 100))  # Increased default limit
    offset = int(request.args.get('offset', 0))
    
    try:
        # Fetch events from database with filtering
        events = event_db.search_events(
            start_time=start_date,
            end_time=end_date
        )
        
        # Sort events by timestamp (newest first)
        events.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
        
        # Apply pagination
        total_count = len(events)
        paginated_events = events[offset:offset + limit]
        
        # Add debug logging
        print(f"[DEBUG] Fetched {total_count} events, returning {len(paginated_events)} (offset: {offset}, limit: {limit})")
        
        return jsonify({
            'events': paginated_events,
            'count': total_count,
            'offset': offset,
            'limit': limit,
            'timestamp': datetime.now().isoformat()  # Add server timestamp
        })
    except Exception as e:
        print(f"[ERROR] Error fetching events: {e}")
        return jsonify({
            'events': [],
            'count': 0,
            'offset': offset,
            'limit': limit,
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        })

@app.route('/api/get_settings')
def get_settings():
    with state.lock:
        return jsonify(state.settings)

@app.route('/api/update_settings', methods=['POST'])
def update_settings():
    data = request.json
    
    with state.lock:
        for key, value in data.items():
            if key in state.settings:
                state.settings[key] = value
    
    return jsonify({'status': 'success', 'settings': state.settings})

@app.route('/api/get_snapshots')
def get_snapshots():
    snapshots = []
    for filename in os.listdir('snapshots'):
        if filename.endswith('.jpg'):
            snapshots.append({
                'filename': filename,
                'url': url_for('serve_snapshot', filename=filename),
                'timestamp': filename.split('_', 1)[1].rsplit('.', 1)[0].replace('_', ' ')
            })
    
    # Sort by timestamp (newest first)
    snapshots.sort(key=lambda x: x['timestamp'], reverse=True)
    
    return jsonify({'snapshots': snapshots})

@app.route('/snapshots/<filename>')
def serve_snapshot(filename):
    return send_from_directory('snapshots', filename)

@app.route('/api/get_recordings')
def get_recordings():
    recordings = []
    for filename in os.listdir('recorded_streams'):
        if filename.endswith(('.avi', '.mp4')):
            file_path = os.path.join('recorded_streams', filename)
            recordings.append({
                'filename': filename,
                'url': url_for('serve_recording', filename=filename),
                'size': os.path.getsize(file_path) / (1024 * 1024),  # Size in MB
                'timestamp': filename.split('_', 1)[1].rsplit('.', 1)[0].replace('_', ' ')
            })
    
    # Sort by timestamp (newest first)
    recordings.sort(key=lambda x: x['timestamp'], reverse=True)
    
    return jsonify({'recordings': recordings})

@app.route('/recorded_streams/<filename>')
def serve_recording(filename):
    return send_from_directory('recorded_streams', filename)

@app.route('/api/get_available_cameras')
def get_available_cameras():
    cameras = []
    for i in range(5):  # Check first 5 camera indices
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            cameras.append(i)
            cap.release()
    
    return jsonify({'cameras': cameras})

# Video Analytics Dashboard Functions
def get_video_analytics_stats(start_date=None, end_date=None, camera_filter=None, category_filter=None):
    """Get comprehensive dashboard statistics"""
    try:
        conn = sqlite3.connect('events.db')
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
        conn.close()
        
        # Process data for dashboard
        return process_events_for_dashboard(events)
        
    except Exception as e:
        print(f"Error getting dashboard stats: {e}")
        return get_empty_stats()

def process_events_for_dashboard(events):
    """Process events data for dashboard widgets"""
    if not events:
        return get_empty_stats()
    
    # Initialize counters
    daily_counts = {}
    camera_counts = {}
    category_counts = {}
    hourly_trends = {}
    recent_events = []
    
    # Process each event
    for event in events:
        timestamp_str, classification, snapshot_path, object_id, speed, direction, bbox = event
        
        try:
            # Parse timestamp - handle different formats
            if 'T' in timestamp_str:
                timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
            else:
                timestamp = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S')
            
            date_str = timestamp.strftime('%Y-%m-%d')
            hour = timestamp.hour
            
            # Count by date
            daily_counts[date_str] = daily_counts.get(date_str, 0) + 1
            
            # Count by camera (extract from snapshot path or use default)
            camera = extract_camera_from_path(snapshot_path)
            camera_counts[camera] = camera_counts.get(camera, 0) + 1
            
            # Count by category
            category = classification or 'Unknown'
            category_counts[category] = category_counts.get(category, 0) + 1
            
            # Hourly trends
            hourly_trends[hour] = hourly_trends.get(hour, 0) + 1
            
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
            print(f"Error processing event {timestamp_str}: {e}")
            # Still count the event even if there's a parsing error
            daily_counts['Unknown-Date'] = daily_counts.get('Unknown-Date', 0) + 1
            category_counts[classification or 'Unknown'] = category_counts.get(classification or 'Unknown', 0) + 1
            continue
    
    # Prepare trend data (last 7 days)
    trend_data = prepare_trend_data(daily_counts)
    
    return {
        'total_detections': len(events),
        'daily_counts': daily_counts,
        'camera_counts': camera_counts,
        'category_counts': category_counts,
        'hourly_trends': hourly_trends,
        'trend_data': trend_data,
        'recent_events': recent_events,
        'top_cameras': get_top_items(camera_counts, 5),
        'top_categories': get_top_items(category_counts, 5)
    }

def extract_camera_from_path(snapshot_path):
    """Extract camera identifier from snapshot path"""
    if not snapshot_path:
        return 'Camera-1'
    
    # Try to extract camera info from path
    path_parts = snapshot_path.split('/')
    for part in path_parts:
        if 'cam' in part.lower() or 'camera' in part.lower():
            return part
    
    # Check for common patterns in filename
    filename = path_parts[-1] if path_parts else snapshot_path
    if 'cam' in filename.lower():
        return filename.split('.')[0]  # Remove extension
    
    # Default naming based on hash for consistency
    camera_id = abs(hash(snapshot_path)) % 5 + 1
    return f'Camera-{camera_id}'

def prepare_trend_data(daily_counts):
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

def get_top_items(counter_dict, limit):
    """Get top items from counter dictionary"""
    return [{'name': k, 'count': v} for k, v in 
            sorted(counter_dict.items(), key=lambda x: x[1], reverse=True)[:limit]]

def get_empty_stats():
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

def get_filtered_analytics_events(start_date=None, end_date=None, camera_filter=None, category_filter=None, limit=100):
    """Get filtered events for logs view"""
    try:
        conn = sqlite3.connect('events.db')
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
                'camera': extract_camera_from_path(snapshot_path),
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

def get_available_analytics_cameras():
    """Get list of available cameras from events"""
    try:
        conn = sqlite3.connect('events.db')
        cursor = conn.cursor()
        
        cursor.execute("SELECT DISTINCT snapshot_path FROM events WHERE snapshot_path IS NOT NULL")
        paths = cursor.fetchall()
        conn.close()
        
        cameras = set()
        for path_tuple in paths:
            camera = extract_camera_from_path(path_tuple[0])
            cameras.add(camera)
        
        return sorted(list(cameras))
        
    except Exception as e:
        print(f"Error getting cameras: {e}")
        return []

def get_available_analytics_categories():
    """Get list of available object categories"""
    try:
        conn = sqlite3.connect('events.db')
        cursor = conn.cursor()
        
        cursor.execute("SELECT DISTINCT classification FROM events WHERE classification IS NOT NULL")
        categories = cursor.fetchall()
        conn.close()
        
        return sorted([cat[0] for cat in categories if cat[0]])
        
    except Exception as e:
        print(f"Error getting categories: {e}")
        return []

# Video Analytics Dashboard Routes
@app.route('/video-analytics')
def video_analytics_dashboard():
    """Main Video Analytics Dashboard page"""
    return render_template('video_analytics.html')

@app.route('/api/analytics/dashboard-stats')
def get_analytics_dashboard_stats():
    """Get dashboard statistics with optional filters"""
    start_date = request.args.get('start_date')
    end_date = request.args.get('end_date')
    camera_filter = request.args.get('camera')
    category_filter = request.args.get('category')
    
    stats = get_video_analytics_stats(
        start_date=start_date,
        end_date=end_date,
        camera_filter=camera_filter,
        category_filter=category_filter
    )
    
    return jsonify(stats)

@app.route('/api/analytics/filtered-events')
def get_analytics_filtered_events():
    """Get filtered events for logs view"""
    start_date = request.args.get('start_date')
    end_date = request.args.get('end_date')
    camera_filter = request.args.get('camera')
    category_filter = request.args.get('category')
    limit = int(request.args.get('limit', 100))
    
    events = get_filtered_analytics_events(
        start_date=start_date,
        end_date=end_date,
        camera_filter=camera_filter,
        category_filter=category_filter,
        limit=limit
    )
    
    return jsonify({'events': events})

@app.route('/api/analytics/available-cameras')
def get_analytics_available_cameras():
    """Get available cameras from events"""
    cameras = get_available_analytics_cameras()
    return jsonify({'cameras': cameras})

@app.route('/api/analytics/available-categories')
def get_analytics_available_categories():
    """Get available object categories"""
    categories = get_available_analytics_categories()
    return jsonify({'categories': categories})

if __name__ == '__main__':
    # Start video thread
    video_thread = VideoThread()
    video_thread.start()
    
    # Run Flask app
    app.run(host='0.0.0.0', port=4000, debug=False)