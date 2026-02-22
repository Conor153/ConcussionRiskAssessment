import cv2
import numpy as np
import math 
from time import time
from collections import deque
import matplotlib.pyplot as plt

class BirdsEyeView:
    def __init__(self, source, target):
        self.source = source.astype(np.float32)
        self.target = target.astype(np.float32)
        self.previous_grey_frame = None
        self.previous_points = None
        # # params for ShiTomasi corner detection
        self.feature_params = dict( maxCorners = 300,
                            qualityLevel = 0.3,
                            minDistance = 7,
                            blockSize = 7 )
        # Parameters for lucas kanade optical flow
        self.lk_params = dict( winSize  = (15, 15),
                        maxLevel = 2,
                        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
        
        # H, mask = cv2.find_homography(src_points, dst_points)
        # warped = cv2.warpPerspective(image, H, (width, height))

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

    #Update the homography
    def update_matrix(self, frame):
        #convert the current frame to grey
        current_grey_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        current_points = cv2.goodFeaturesToTrack(
            current_grey_frame,
            mask=None,
            **self.feature_params
        )
        if self.previous_grey_frame is not None and self.previous_points is not None:
            #Create the optical flow
            new_points, status, _ = cv2.calcOpticalFlowPyrLK(self.previous_grey_frame,current_grey_frame, self.previous_points.reshape(-1, 1, 2), None, **self.lk_params)
            good_new_points = new_points[status==1]
            good_previous_points = self.previous_points[status==1]
            if len(good_new_points) >= 4:
                # Camera motion H from corners
                camera_H, _ = cv2.findHomography(good_previous_points, good_new_points, cv2.RANSAC, 5.0)
                backup_source = self.source.copy()
                if camera_H is not None:
                # Apply camera motion to update your 4 field points
                    self.source = cv2.perspectiveTransform(
                    self.source.reshape(-1, 1, 2), camera_H).reshape(-1, 2)
                    h, w = frame.shape[:2]
                    if np.any(self.source < 0) or np.any(self.source[:, 0] > w ) or np.any(self.source[:, 1] > h):
                        self.source = backup_source  # reset to last good
                        return
                #H is the new matrix of the original source points 
                H, mask = cv2.findHomography(self.source, self.target)
                if H is not None:
                    self.matrix = H
                print(f"Corners tracked: {len(good_new_points)}")
                print(f"Source points: {self.source}")
                print(f"Camera H: {camera_H}")
        #Set the current frame as the previoius for the next frame
        self.previous_grey_frame = current_grey_frame
        self.previous_points = current_points
        
    
    def calculate_speed(self, track_id, position, current_time):
        """Calculate current speed for track_id"""
        #Store a queue of the tracked players position
        #Declare positions and current_speeds queues
        if track_id not in self.positions:
            self.positions[track_id] = deque(maxlen=15)
        
        if track_id not in self.current_speeds:
            self.current_speeds[track_id] = deque(maxlen=15)
        #Append x,y and the time it has taken to reach the co-ordinates
        self.positions[track_id].append((position[0], position[1], current_time))
        positions = list(self.positions[track_id])
        
        if len(positions) < 2:
            self.current_speeds[track_id].append((0.0, current_time))
            return 0.0
        
        if len(positions) >= 7:
            old_pos = positions[-7]
            new_pos = positions[-1]
        elif len(positions) >= 4:
            old_pos = positions[-4]
            new_pos = positions[-1]
        else:
            old_pos = positions[0]
            new_pos = positions[-1]
        
        dx = new_pos[0] - old_pos[0]
        dy = new_pos[1] - old_pos[1]

        #distance_meters = np.sqrt(dx*dx + dy*dy)
        distance_meters = np.sqrt(dx*dx + dy*dy)

        time_diff = new_pos[2] - old_pos[2]
        
        if time_diff > 0:
            speed_ms = distance_meters / time_diff
            #speed_kmh = speed_ms * 3.6
            
            # Filter unrealistic speeds
            if speed_ms > 200 or speed_ms < 0:
                speed_ms = 0
            self.current_speeds[track_id].append((speed_ms, current_time))
            return speed_ms
        else: 
            self.current_speeds[track_id].append((0.0, current_time))
            return 0.0
    
    def calculate_acceleration(self, track_id):
        if track_id not in self.acceleration:
            self.acceleration[track_id] = deque(maxlen=15)
        speed = list(self.current_speeds[track_id])

        if len(speed) < 2:
            self.acceleration[track_id].append((0.0))
            return 0.0

        if len(speed) >= 7:
            old_speed = speed[-7]
            new_speed = speed[-1]
        elif len(speed) >= 4:
            old_speed = speed[-4]
            new_speed = speed[-1]
        else:
            old_speed = speed[0]
            new_speed = speed[-1]
        time_diff = new_speed[1] - old_speed[1] 
        if time_diff > 0:
            acceleration = (new_speed[0] - old_speed[0]) / time_diff
            #acceleration = acceleration/3.6
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