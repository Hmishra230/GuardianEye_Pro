import cv2
import numpy as np
import config
from collections import deque

class TrackedObject:
    def __init__(self, object_id, centroid, bbox):
        self.id = object_id
        self.centroids = deque([centroid], maxlen=30)
        self.bboxes = deque([bbox], maxlen=30)
        self.kalman = cv2.KalmanFilter(4, 2)
        self.kalman.measurementMatrix = np.eye(2, 4, dtype=np.float32)
        self.kalman.transitionMatrix = np.array([[1,0,1,0],[0,1,0,1],[0,0,1,0],[0,0,0,1]], np.float32)
        self.kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
        self.kalman.statePre = np.array([[centroid[0]], [centroid[1]], [0], [0]], dtype=np.float32)
        self.kalman.statePost = np.array([[centroid[0]], [centroid[1]], [0], [0]], dtype=np.float32)
        self.last_seen = 0
        self.direction = None
        self.trajectory = []
        self.classification = None

    def update(self, centroid, bbox):
        self.centroids.append(centroid)
        self.bboxes.append(bbox)
        self.kalman.correct(np.array([[np.float32(centroid[0])], [np.float32(centroid[1])]]))
        self.last_seen = 0
        self.trajectory.append(centroid)
        if len(self.centroids) > 1:
            dx = self.centroids[-1][0] - self.centroids[0][0]
            dy = self.centroids[-1][1] - self.centroids[0][1]
            self.direction = np.arctan2(dy, dx)

    def predict(self):
        pred = self.kalman.predict()
        return int(pred[0]), int(pred[1])

    def get_speed(self):
        if len(self.centroids) < 2:
            return 0
        return np.linalg.norm(np.array(self.centroids[-1]) - np.array(self.centroids[-2]))

    def get_direction(self):
        return self.direction

    def get_trajectory(self):
        return list(self.trajectory)

    def classify_motion(self):
        # Placeholder for ML-based motion classification
        # Example: return 'thrown' or 'natural'
        return 'thrown' if self.get_speed() > config.MIN_SPEED_THRESHOLD else 'natural'

class MotionDetector:
    def __init__(self):
        # Create background subtractor with more sensitive parameters
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=16)
        self.objects = {}
        self.next_id = 1
        self.max_lost = 10
        self.frame_count = 0
        self.adaptive_lr = 0.01  # Increased learning rate for faster adaptation

    def background_subtraction(self, frame, roi_mask=None):
        # Adaptive background learning with more sensitivity
        fg_mask = self.bg_subtractor.apply(frame, learningRate=self.adaptive_lr)
        
        # Apply ROI mask if provided
        if roi_mask is not None:
            # Make sure we're only detecting within the ROI
            fg_mask = cv2.bitwise_and(fg_mask, fg_mask, mask=roi_mask)
            
            # Debug visualization - save a snapshot of the masked frame
            if self.frame_count % 30 == 0:  # Every 30 frames
                try:
                    masked_frame = cv2.bitwise_and(frame, frame, mask=roi_mask)
                    cv2.imwrite(f'debug_masked_frame_{self.frame_count}.jpg', masked_frame)
                    cv2.imwrite(f'debug_fg_mask_{self.frame_count}.jpg', fg_mask)
                except Exception as e:
                    print(f"[DEBUG] Error saving debug frame: {e}")
        
        self.frame_count += 1
        return fg_mask

    def filter_noise(self, fg_mask, morph_kernel_size=None):
        # Use provided morph_kernel_size if available, otherwise use config value
        if morph_kernel_size is None:
            kernel_size = config.MORPH_KERNEL_SIZE
        else:
            kernel_size = (morph_kernel_size, morph_kernel_size)
        
        print(f"[DEBUG-MD] Using morph kernel size: {kernel_size}")
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel_size)
        opened = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        return opened

    def detect_contours(self, mask):
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return contours

    def update_tracks(self, detections):
        # Hungarian algorithm or greedy matching for assignment
        print(f"[DEBUG-MD] Updating tracks with {len(detections)} detections, current objects: {len(self.objects)}")
        assigned = set()
        for det in detections:
            centroid, bbox = det
            min_dist = float('inf')
            min_id = None
            for obj_id, obj in self.objects.items():
                dist = np.linalg.norm(np.array(obj.centroids[-1]) - np.array(centroid))
                if dist < 50 and obj_id not in assigned:
                    if dist < min_dist:
                        min_dist = dist
                        min_id = obj_id
            if min_id is not None:
                self.objects[min_id].update(centroid, bbox)
                assigned.add(min_id)
                print(f"[DEBUG-MD] Updated existing object ID {min_id}, distance={min_dist:.2f}")
            else:
                self.objects[self.next_id] = TrackedObject(self.next_id, centroid, bbox)
                print(f"[DEBUG-MD] Created new object ID {self.next_id}")
                self.next_id += 1
        # Mark lost objects
        lost_ids = []
        for obj_id, obj in self.objects.items():
            if obj_id not in assigned:
                obj.last_seen += 1
                obj.kalman.predict()
                if obj.last_seen > self.max_lost:
                    lost_ids.append(obj_id)
        for obj_id in lost_ids:
            del self.objects[obj_id]

    def process(self, frame, roi_mask=None, min_area=None, max_area=None, morph_kernel=None):
        print(f"[DEBUG-MD] Processing frame with ROI mask: {roi_mask is not None}")
        fg_mask = self.background_subtraction(frame, roi_mask)
        print(f"[DEBUG-MD] Background subtraction complete, non-zero pixels: {np.count_nonzero(fg_mask)}")
        filtered_mask = self.filter_noise(fg_mask, morph_kernel_size=morph_kernel)
        print(f"[DEBUG-MD] Filtered mask, non-zero pixels: {np.count_nonzero(filtered_mask)}")
        contours = self.detect_contours(filtered_mask)
        print(f"[DEBUG-MD] Detected {len(contours)} contours")
        
        # Use provided min_area and max_area if available, otherwise use config values
        min_area = min_area if min_area is not None else config.MIN_OBJECT_AREA
        max_area = max_area if max_area is not None else config.MAX_OBJECT_AREA
        print(f"[DEBUG-MD] Using area limits: min_area={min_area}, max_area={max_area}")
        
        detections = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            centroid = (int(x + w / 2), int(y + h / 2))
            area = cv2.contourArea(cnt)
            hull = cv2.convexHull(cnt)
            hull_area = cv2.contourArea(hull)
            solidity = float(area) / hull_area if hull_area > 0 else 0
            
            print(f"[DEBUG-MD] Contour: area={area}, solidity={solidity:.2f}, centroid={centroid}")
            
            if area < min_area or area > max_area:
                print(f"[DEBUG-MD] Rejected: area {area} outside range [{min_area}-{max_area}]")
                continue
            if solidity < 0.7:
                print(f"[DEBUG-MD] Rejected: solidity {solidity:.2f} < 0.7")
                continue
            
            print(f"[DEBUG-MD] Accepted contour: area={area}, centroid={centroid}")
            detections.append((centroid, (x, y, w, h)))
        self.update_tracks(detections)
        # Trajectory and direction analysis, ML classification
        results = []
        for obj in self.objects.values():
            speed = obj.get_speed()
            direction = obj.get_direction()
            traj = obj.get_trajectory()
            classification = obj.classify_motion()
            results.append({
                'id': obj.id,
                'centroid': obj.centroids[-1],
                'bbox': obj.bboxes[-1],
                'speed': speed,
                'direction': direction,
                'trajectory': traj,
                'classification': classification
            })
        return results