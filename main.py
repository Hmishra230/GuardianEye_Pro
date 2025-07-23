import cv2
import numpy as np
import sys
import traceback
import config
from roi_manager import ROIManager
from motion_detector import MotionDetector
from alert_system import AlertSystem
import threading
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import queue
import time

# Shared state for GUI and OpenCV loop
class SharedState:
    def __init__(self):
        self.motion_threshold = config.MOTION_DETECTION_THRESHOLD
        self.min_area = config.MIN_OBJECT_AREA
        self.max_area = config.MAX_OBJECT_AREA
        self.min_speed = config.MIN_SPEED_THRESHOLD
        self.morph_kernel = config.MORPH_KERNEL_SIZE[0]
        self.event_log = []
        self.stats = {'events': 0, 'frames': 0}
        self.roi_coords = []
        self.camera_source = 0
        self.settings_changed = False
        self.quit = False
        self.save_settings = False
        self.load_settings = False
        self.selected_camera = 0
        self.available_cameras = [0]
        self.update_roi = False
        self.detection_status = 'Waiting for ROI...'
        # New fields for multi-source
        self.source_type = config.DEFAULT_SOURCE_TYPE  # 'webcam', 'rtsp', 'file'
        self.rtsp_url = ''
        self.video_file_path = ''
        self.connection_status = 'Disconnected'  # 'Connected', 'Disconnected', 'Error'
        self.file_loop = config.VIDEO_FILE_LOOP
        self.last_used_source = None
        self.last_used_path = None

shared_state = SharedState()

# Tkinter GUI (must run in main thread)
class ControlPanel:
    def __init__(self, shared_state):
        self.shared_state = shared_state
        self.root = tk.Tk()
        self.root.title('Detection Control Panel')
        self.build_gui()

    def build_gui(self):
        self.roi_label = tk.Label(self.root, text='ROI Coordinates: []')
        self.roi_label.pack()
        self.motion_slider = tk.Scale(self.root, from_=1, to=100, orient='horizontal', label='Motion Threshold', command=self.update_motion)
        self.motion_slider.set(self.shared_state.motion_threshold)
        self.motion_slider.pack()
        self.min_area_slider = tk.Scale(self.root, from_=10, to=1000, orient='horizontal', label='Min Object Area', command=self.update_min_area)
        self.min_area_slider.set(self.shared_state.min_area)
        self.min_area_slider.pack()
        self.max_area_slider = tk.Scale(self.root, from_=500, to=5000, orient='horizontal', label='Max Object Area', command=self.update_max_area)
        self.max_area_slider.set(self.shared_state.max_area)
        self.max_area_slider.pack()
        self.speed_slider = tk.Scale(self.root, from_=1, to=200, orient='horizontal', label='Min Speed', command=self.update_speed)
        self.speed_slider.set(self.shared_state.min_speed)
        self.speed_slider.pack()
        self.morph_slider = tk.Scale(self.root, from_=1, to=15, orient='horizontal', label='Morph Kernel Size', command=self.update_morph)
        self.morph_slider.set(self.shared_state.morph_kernel)
        self.morph_slider.pack()
        self.camera_var = tk.StringVar()
        self.camera_dropdown = ttk.Combobox(self.root, textvariable=self.camera_var, values=[str(i) for i in self.shared_state.available_cameras])
        self.camera_dropdown.current(0)
        self.camera_dropdown.pack()
        self.camera_dropdown.bind('<<ComboboxSelected>>', self.change_camera)
        self.log_text = tk.Text(self.root, height=8, width=50)
        self.log_text.pack()
        self.stats_label = tk.Label(self.root, text='Events: 0 | Frames: 0')
        self.stats_label.pack()
        self.save_btn = tk.Button(self.root, text='Save Settings', command=self.save_settings)
        self.save_btn.pack(side='left')
        self.load_btn = tk.Button(self.root, text='Load Settings', command=self.load_settings)
        self.load_btn.pack(side='left')
        # Source selection
        self.source_var = tk.StringVar(value=self.shared_state.source_type)
        self.webcam_radio = tk.Radiobutton(self.root, text='Webcam', variable=self.source_var, value='webcam', command=self.update_source_type)
        self.rtsp_radio = tk.Radiobutton(self.root, text='RTSP Stream', variable=self.source_var, value='rtsp', command=self.update_source_type)
        self.file_radio = tk.Radiobutton(self.root, text='Video File', variable=self.source_var, value='file', command=self.update_source_type)
        self.webcam_radio.pack(anchor='w')
        self.rtsp_radio.pack(anchor='w')
        self.file_radio.pack(anchor='w')
        # RTSP URL entry
        self.rtsp_url_label = tk.Label(self.root, text='RTSP URL:')
        self.rtsp_url_entry = tk.Entry(self.root, width=40)
        self.rtsp_url_entry.insert(0, self.shared_state.rtsp_url)
        self.rtsp_url_label.pack(anchor='w')
        self.rtsp_url_entry.pack(anchor='w')
        # File browser
        self.file_label = tk.Label(self.root, text='Video File:')
        self.file_path_var = tk.StringVar()
        self.file_entry = tk.Entry(self.root, textvariable=self.file_path_var, width=40)
        self.browse_btn = tk.Button(self.root, text='Browse...', command=self.browse_file)
        self.file_label.pack(anchor='w')
        self.file_entry.pack(anchor='w', side='left')
        self.browse_btn.pack(anchor='w', side='left')
        # Connect/Start button
        self.connect_btn = tk.Button(self.root, text='Connect/Start', command=self.connect_source)
        self.connect_btn.pack(anchor='w')
        # Connection status
        self.status_label = tk.Label(self.root, text='Connection Status: Disconnected', fg='red')
        self.status_label.pack(anchor='w')
        self.root.protocol('WM_DELETE_WINDOW', self.on_close)
        self.update_gui()

    def update_gui(self):
        self.roi_label.config(text=f'ROI Coordinates: {self.shared_state.roi_coords}')
        self.log_text.delete(1.0, tk.END)
        for entry in self.shared_state.event_log[-10:]:
            self.log_text.insert(tk.END, entry + '\n')
        self.stats_label.config(text=f"Events: {self.shared_state.stats['events']} | Frames: {self.shared_state.stats['frames']}")
        # Update connection status
        status = self.shared_state.connection_status
        color = 'green' if status == 'Connected' else ('red' if status == 'Error' else 'orange')
        self.status_label.config(text=f'Connection Status: {status}', fg=color)
        self.root.after(500, self.update_gui)

    def update_motion(self, val):
        self.shared_state.motion_threshold = int(val)
        self.shared_state.settings_changed = True
    def update_min_area(self, val):
        self.shared_state.min_area = int(val)
        self.shared_state.settings_changed = True
    def update_max_area(self, val):
        self.shared_state.max_area = int(val)
        self.shared_state.settings_changed = True
    def update_speed(self, val):
        self.shared_state.min_speed = int(val)
        self.shared_state.settings_changed = True
    def update_morph(self, val):
        self.shared_state.morph_kernel = int(val)
        self.shared_state.settings_changed = True
    def change_camera(self, event):
        try:
            self.shared_state.selected_camera = int(self.camera_var.get())
            self.shared_state.camera_source = self.shared_state.selected_camera
            self.shared_state.settings_changed = True
        except Exception:
            pass
    def save_settings(self):
        self.shared_state.save_settings = True
    def load_settings(self):
        self.shared_state.load_settings = True
    def on_close(self):
        self.shared_state.quit = True
        self.root.quit()
    def update_source_type(self):
        stype = self.source_var.get()
        self.shared_state.source_type = stype
        # Show/hide RTSP and file widgets
        if stype == 'webcam':
            self.rtsp_url_label.pack_forget()
            self.rtsp_url_entry.pack_forget()
            self.file_label.pack_forget()
            self.file_entry.pack_forget()
            self.browse_btn.pack_forget()
        elif stype == 'rtsp':
            self.rtsp_url_label.pack(anchor='w')
            self.rtsp_url_entry.pack(anchor='w')
            self.file_label.pack_forget()
            self.file_entry.pack_forget()
            self.browse_btn.pack_forget()
        elif stype == 'file':
            self.rtsp_url_label.pack_forget()
            self.rtsp_url_entry.pack_forget()
            self.file_label.pack(anchor='w')
            self.file_entry.pack(anchor='w', side='left')
            self.browse_btn.pack(anchor='w', side='left')
    def browse_file(self):
        filetypes = [("Video files", "*.mp4 *.avi *.mov *.mkv *.flv")]
        path = filedialog.askopenfilename(title="Select Video File", filetypes=filetypes)
        if path:
            self.file_path_var.set(path)
            self.shared_state.video_file_path = path
    def connect_source(self):
        stype = self.source_var.get()
        self.shared_state.source_type = stype
        if stype == 'rtsp':
            self.shared_state.rtsp_url = self.rtsp_url_entry.get()
        elif stype == 'file':
            self.shared_state.video_file_path = self.file_path_var.get()
        self.shared_state.settings_changed = True
        self.status_label.config(text='Connection Status: Connecting...', fg='orange')

# Enumerate available cameras
def enumerate_cameras(max_test=5):
    available = []
    for i in range(max_test):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            available.append(i)
            cap.release()
    return available

# OpenCV video processing in a background thread
class VideoThread(threading.Thread):
    def __init__(self, shared_state):
        super().__init__(daemon=True)
        self.shared_state = shared_state
        self.cap = None
        self.roi_manager = None
        self.motion_detector = None
        self.alert_system = None
        self.last_frame = None

    def run(self):
        # Initialize video source
        self.cap = initialize_video_source(self.shared_state.source_type, self.shared_state)
        if not self.cap or not self.cap.isOpened():
            self.shared_state.connection_status = 'Error'
            print(f"[ERROR] Could not open video source: {self.shared_state.source_type}")
            return
        self.shared_state.connection_status = 'Connected'
        window_name = 'Critical Region Throw Detection'
        cv2.namedWindow(window_name)
        self.roi_manager = ROIManager(window_name)
        self.motion_detector = MotionDetector()
        self.alert_system = AlertSystem()
        detection_status = 'Waiting for ROI...'
        last_frame = None
        cv2.setMouseCallback(window_name, self.roi_manager.mouse_callback)
        while not self.shared_state.quit:
            if self.shared_state.settings_changed:
                self.cap.release()
                self.cap = initialize_video_source(self.shared_state.source_type, self.shared_state)
                if not self.cap or not self.cap.isOpened():
                    self.shared_state.connection_status = 'Error'
                    print(f"[ERROR] Could not open video source: {self.shared_state.source_type}")
                    break
                self.shared_state.settings_changed = False
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.1)
                continue
            display_frame = frame.copy()
            key = cv2.waitKey(1) & 0xFF
            display_frame = self.roi_manager.draw_roi(display_frame)
            self.shared_state.roi_coords = self.roi_manager.roi_points.copy()
            if self.roi_manager.finished and len(self.roi_manager.roi_points) >= 3:
                mask = np.zeros(frame.shape[:2], dtype=np.uint8)
                pts = np.array(self.roi_manager.roi_points, np.int32)
                cv2.fillPoly(mask, [pts], 255)
                fg_mask = self.motion_detector.background_subtraction(frame, mask)
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.shared_state.morph_kernel, self.shared_state.morph_kernel))
                filtered_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
                contours = self.motion_detector.detect_contours(filtered_mask)
                detected = False
                for cnt in contours:
                    x, y, w, h = cv2.boundingRect(cnt)
                    centroid = (int(x + w / 2), int(y + h / 2))
                    if not self.roi_manager.point_in_roi(centroid):
                        continue
                    area = cv2.contourArea(cnt)
                    hull = cv2.convexHull(cnt)
                    hull_area = cv2.contourArea(hull)
                    solidity = float(area) / hull_area if hull_area > 0 else 0
                    if area < 20 or area > self.shared_state.max_area:  # Lowered min_area for testing
                        continue
                    if solidity < 0.7:
                        continue
                    speed = 0
                    direction = None
                    # Use new multi-object tracking if available
                    results = self.motion_detector.process(frame, mask)
                    for obj in results:
                        # Match by centroid proximity instead of exact bbox
                        obj_cx, obj_cy = obj['centroid']
                        cx, cy = centroid
                        if np.linalg.norm(np.array([obj_cx, obj_cy]) - np.array([cx, cy])) < 10:
                            speed = obj['speed']
                            direction = obj['direction']
                            print(f"[DEBUG] Detected object: speed={speed:.2f}, area={area:.2f}, centroid={centroid}")
                            break
                    if speed < 5:  # Lowered min_speed for testing
                        continue
                    if True:  # temporal consistency handled in process()
                        self.alert_system.alert(display_frame, (x, y, w, h), speed)
                        detection_status = 'THROWN OBJECT DETECTED!'
                        detected = True
                        self.shared_state.stats['events'] += 1
                        self.shared_state.event_log.append(f'Event {self.shared_state.stats["events"]}: Speed={speed:.1f}, Area={area:.1f}')
                        break
                if not detected:
                    detection_status = 'Monitoring...'
            else:
                detection_status = 'Waiting for ROI...'
            self.shared_state.stats['frames'] += 1
            cv2.putText(display_frame, f'Status: {detection_status}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            self.alert_system.overlay_event_log(display_frame)
            cv2.imshow(window_name, display_frame)
            last_frame = display_frame.copy()
            if key == ord('q') or key == ord('Q'):
                self.shared_state.quit = True
                break
            elif key == ord('r') or key == ord('R'):
                self.roi_manager.reset()
                detection_status = 'ROI reset.'
            elif key == ord('s') or key == ord('S'):
                if last_frame is not None:
                    try:
                        cv2.imwrite('saved_frame.jpg', last_frame)
                        print('Frame saved as saved_frame.jpg')
                    except Exception as e:
                        print(f'Error saving frame: {e}')
            if self.shared_state.save_settings:
                try:
                    np.save('gui_settings.npy', [self.shared_state.motion_threshold, self.shared_state.min_area, self.shared_state.max_area, self.shared_state.min_speed, self.shared_state.morph_kernel])
                    print('Settings saved.')
                except Exception as e:
                    print(f'Error saving settings: {e}')
                self.shared_state.save_settings = False
            if self.shared_state.load_settings:
                try:
                    vals = np.load('gui_settings.npy', allow_pickle=True)
                    self.shared_state.motion_threshold, self.shared_state.min_area, self.shared_state.max_area, self.shared_state.min_speed, self.shared_state.morph_kernel = vals
                    print('Settings loaded.')
                except Exception as e:
                    print(f'Error loading settings: {e}')
                self.shared_state.load_settings = False
        try:
            self.roi_manager.save_roi()
        except Exception as e:
            print(f'Error saving ROI: {e}')
        self.cap.release()
        cv2.destroyAllWindows()

def initialize_video_source(source_type, shared_state):
    if source_type == 'webcam':
        return cv2.VideoCapture(shared_state.camera_source)
    elif source_type == 'rtsp':
        return cv2.VideoCapture(shared_state.rtsp_url)
    elif source_type == 'file':
        return cv2.VideoCapture(shared_state.video_file_path)
    return None

def main():
    shared_state.available_cameras = enumerate_cameras()
    shared_state.camera_source = 0  # Force use of camera index 0
    # Start OpenCV video processing in a background thread
    video_thread = VideoThread(shared_state)
    video_thread.start()
    # Start Tkinter GUI in main thread
    gui = ControlPanel(shared_state)
    gui.root.mainloop()
    # When GUI closes, signal video thread to quit
    shared_state.quit = True
    video_thread.join()

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print('Unhandled exception:', e)
        traceback.print_exc()
        sys.exit(1) 