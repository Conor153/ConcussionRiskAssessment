import cv2 as cv
import numpy as np
from ultralytics import YOLO
from time import time
from collections import deque

class RealTimeSpeedTracker:
    
    def __init__(self, model_path, fps):
        self.model = YOLO(model_path)
        self.fps = fps
        
        # Tracking data (keep only last 10 positions for speed calculation)
        self.positions = {}  # track_id: deque of (x, y, time)
        self.current_speeds = {}  # track_id: current_speed_kmh
        
        # Court calibration
        self.pixel_to_meter_ratio = None
        self.perspective_matrix = None
        
        print(f"Speed tracker initialized with model: {model_path}")
    
    def set_field_calibration(self, corners):
        """Set 4 court corners for perspective transformation"""
        corners = np.array(corners, dtype=np.float32)
        
        # Calculate pixel-to-meter ratio
        top_width = np.linalg.norm(corners[1] - corners[0])
        bottom_width = np.linalg.norm(corners[2] - corners[3])
        left_height = np.linalg.norm(corners[3] - corners[0])
        right_height = np.linalg.norm(corners[2] - corners[1])
        
        avg_width_pixels = (top_width + bottom_width) / 2
        avg_height_pixels = (left_height + right_height) / 2
        #NFL Pitch
        pitch_width = 109.7 #meters of NFL Field endzone to endzone
        pitch_height = 48.8 #meters of NFL Field sideline to sideline

        width_ratio = pitch_width / avg_width_pixels
        height_ratio = pitch_height / avg_height_pixels
        self.pixel_to_meter_ratio = (width_ratio + height_ratio) / 2
        target = np.array([[0, 0], [pitch_width-1, 0], [pitch_width-1, pitch_height-1], [0, pitch_height-1]], dtype=np.float32)

        # Perspective transformation
        #dst_corners = np.array([[0, 0], [600, 0], [600, 150], [0, 150]], dtype=np.float32)
        self.perspective_matrix = cv.getPerspectiveTransform(corners, target)
        
        print(f"Court calibrated - Ratio: {self.pixel_to_meter_ratio:.6f} m/pixel")
    
    def apply_perspective_correction(self, point):
        """Apply perspective correction to point"""
        if self.perspective_matrix is not None:
            point_array = np.array([[[point[0], point[1]]]], dtype=np.float32)
            corrected = cv.perspectiveTransform(point_array, self.perspective_matrix)
            return tuple(corrected[0][0])
        return point
    
    def calculate_speed(self, track_id, position, current_time):
        """Calculate current speed for track_id"""
        if track_id not in self.positions:
            self.positions[track_id] = deque(maxlen=10)
        
        # Store position with timestamp (NO perspective correction)
        self.positions[track_id].append((position[0], position[1], current_time))
        
        # Need at least 2 points for speed
        if len(self.positions[track_id]) < 2:
            self.current_speeds[track_id] = 0.0
            return 0.0
        
        positions = list(self.positions[track_id])
        
        if len(positions) >= 5:
            old_pos = positions[-2]
            new_pos = positions[-1]
        else:
            old_pos = positions[0]
            new_pos = positions[-1]
        
        # Calculate distance in PIXELS
        dx = new_pos[0] - old_pos[0]
        dy = new_pos[1] - old_pos[1]
        distance_pixels = np.sqrt(dx*dx + dy*dy)
        time_diff = new_pos[2] - old_pos[2]
        
        if time_diff > 0 and self.pixel_to_meter_ratio:
            # Convert pixels to meters using ratio
            distance_meters = distance_pixels * self.pixel_to_meter_ratio
            speed_ms = distance_meters / time_diff
            speed_kmh = speed_ms * 3.6
            
            # Filter unrealistic speeds
            if speed_kmh > 50 or speed_kmh < 0:
                speed_kmh = 0.0
                
            self.current_speeds[track_id] = speed_kmh
            return speed_kmh
        
        self.current_speeds[track_id] = 0.0
        return 0.0