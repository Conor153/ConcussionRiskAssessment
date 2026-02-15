import cv2
import numpy as np
import math 
from time import time
from collections import deque

class BirdsEyeView:
    def __init__(self, source, target):
        self.source = source
        self.target = target
        source = self.source.astype(np.float32)
        target = self.target.astype(np.float32)
        self.matrix = cv2.getPerspectiveTransform(source, target)
        self.x = None
        self.y = None
        self.positions = {} # track_id: deque of (x, y, time)
        self.current_speeds = {}  # track_id: current_speed_kmh
        self.acceleration = {} # track_id: 
        self.g_force = {} # track_id: 
    
    #Transform the BBox to get the centre bottom co-ordinates
    def transform_points(self, points: np.ndarray) -> np.ndarray:
        
        bottom_centre_points = []
        for p in points:
            x1 = p[0]
            x2 = p[2]
            y = p[3]
            centre_x = (x1+x2)/2
            #Append the co-ordinates to the bottom_centre_points
            bottom_centre_points.append([centre_x, y])
        #Convert the array to a numpy array
        bottom_centre_points = np.array(bottom_centre_points, dtype=np.float32)
        #reshaoe the numpy array and transform the points for homography
        reshaped_points = bottom_centre_points.reshape(-1, 1, 2).astype(np.float32)
        transformed_points = cv2.perspectiveTransform(reshaped_points, self.matrix)
        return transformed_points.reshape(-1, 2)
    
    def calculate_speed(self, track_id, position, current_time):
        """Calculate current speed for track_id"""
        #Store a queue of the tracked players position
        #Declare positions and current_speeds queues
        if track_id not in self.positions:
            self.positions[track_id] = deque(maxlen=10)
        
        if track_id not in self.current_speeds:
            self.current_speeds[track_id] = deque(maxlen=10)
        #Append x,y and the time it has taken to reach the co-ordinates
        self.positions[track_id].append((position[0], position[1], current_time))
        positions = list(self.positions[track_id])
        
        if len(positions) < 2:
            self.current_speeds[track_id].append((0.0, current_time))
            return 0.0
        
        if len(positions) >= 5:
            old_pos = positions[-5]
            new_pos = positions[-1]
        else:
            old_pos = positions[0]
            new_pos = positions[-1]
        
        dx = new_pos[0] - old_pos[0]
        dy = new_pos[1] - old_pos[1]
        distance_meters = np.sqrt(dx*dx + dy*dy)
        time_diff = new_pos[2] - old_pos[2]
        
        if time_diff > 0:
            speed_ms = distance_meters / time_diff
            speed_kmh = speed_ms * 3.6
            
            # Filter unrealistic speeds
            if speed_kmh > 80 or speed_kmh < 0:
                speed_kmh = 0.0
            self.current_speeds[track_id].append((speed_kmh, current_time))
            return speed_kmh
        else: 
            self.current_speeds[track_id].append((0.0, current_time))
            return 0.0
    
    def calculate_acceleration(self, track_id):
        if track_id not in self.acceleration:
            self.acceleration[track_id] = deque(maxlen=10)
        speed = list(self.current_speeds[track_id])

        if len(speed) < 2:
            self.acceleration[track_id].append((0.0))
            return 0.0

        if len(speed) >= 5:
            old_speed = speed[-5]
            new_speed = speed[-1]
        else:
            old_speed = speed[0]
            new_speed = speed[-1]
        time_diff = new_speed[1] - old_speed[1] 
        if time_diff > 0:
            acceleration = (new_speed[0] - old_speed[0]) / time_diff
            acceleration = acceleration/3.6
            self.acceleration[track_id].append(acceleration)
            return acceleration
        else:
            self.acceleration[track_id].append((0.0))
            return 0.0
    
    def calculate_GForce(self, track_id):
        if track_id not in self.acceleration:
            return 0.0
        accel = self.acceleration[track_id][-1]
        g_Force = accel / 9.81
        return abs(g_Force)