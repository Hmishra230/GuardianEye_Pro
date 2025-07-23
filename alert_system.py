import cv2
import os
from datetime import datetime
import config

class AlertSystem:
    def __init__(self):
        self.event_count = 0
        if config.ALERT_SNAPSHOT_SAVE and not os.path.exists(config.ALERT_SNAPSHOT_DIR):
            os.makedirs(config.ALERT_SNAPSHOT_DIR)

    def alert(self, frame, bbox, speed):
        self.event_count += 1
        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        # Draw bounding box and speed
        x, y, w, h = bbox
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.putText(frame, f'Speed: {speed:.1f}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.putText(frame, f'ALERT!', (x, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        # Console log
        print(f'[{timestamp}] ALERT: Thrown object detected at {bbox} with speed {speed:.1f} px/frame')
        # Save snapshot
        if config.ALERT_SNAPSHOT_SAVE:
            filename = os.path.join(config.ALERT_SNAPSHOT_DIR, f'event_{self.event_count}_{timestamp}.jpg')
            try:
                cv2.imwrite(filename, frame)
            except Exception as e:
                print(f'Error saving snapshot: {e}')

    def overlay_event_log(self, frame):
        cv2.putText(frame, f'Events: {self.event_count}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2) 