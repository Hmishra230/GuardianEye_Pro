import cv2
import numpy as np
import os
import config

class ROIManager:
    def __init__(self, window_name):
        self.window_name = window_name
        self.roi_points = []
        self.drawing = False
        self.finished = False
        self.load_roi()

    def mouse_callback(self, event, x, y, flags, param):
        if self.finished:
            return
        if event == cv2.EVENT_LBUTTONDOWN:
            self.roi_points.append((x, y))
            self.drawing = True
        elif event == cv2.EVENT_RBUTTONDOWN and len(self.roi_points) > 2:
            self.finished = True

    def draw_roi(self, frame):
        if len(self.roi_points) > 0:
            pts = np.array(self.roi_points, np.int32)
            pts = pts.reshape((-1, 1, 2))
            cv2.polylines(frame, [pts], isClosed=self.finished, color=(0, 255, 0), thickness=2)
            for pt in self.roi_points:
                cv2.circle(frame, pt, 4, (0, 255, 0), -1)
        return frame

    def point_in_roi(self, point):
        if len(self.roi_points) < 3:
            return False
        pts = np.array(self.roi_points, np.int32)
        return cv2.pointPolygonTest(pts, point, False) >= 0

    def save_roi(self):
        if len(self.roi_points) >= 3:
            np.save(config.ROI_SAVE_PATH, np.array(self.roi_points))

    def load_roi(self):
        if os.path.exists(config.ROI_SAVE_PATH):
            try:
                self.roi_points = [tuple(pt) for pt in np.load(config.ROI_SAVE_PATH)]
                self.finished = True
            except Exception:
                self.roi_points = []
                self.finished = False

    def reset(self):
        self.roi_points = []
        self.finished = False 