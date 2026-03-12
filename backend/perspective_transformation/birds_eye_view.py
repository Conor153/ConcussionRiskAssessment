import cv2
import numpy as np
import math 
from time import time
from collections import deque
import matplotlib.pyplot as plt

class BirdsEyeView:
    def __init__(self, source, target):
        #Source and Traget are used to calculate homography which is the matrix
        self.source = source.astype(np.float32)
        self.target = target.astype(np.float32)
        self.matrix = cv2.getPerspectiveTransform(source, target)
        #Previous grey frame and points are required to update the homography between frames
        self.previous_grey_frame = None
        self.previous_points = None

        #Parameters for ShiTomasi corner detection
        self.feature_params = dict( maxCorners = 300,
                            qualityLevel = 0.3,
                            minDistance = 7,
                            blockSize = 7 )
        # Parameters for lucas kanade optical flow
        self.lk_params = dict( winSize  = (15, 15),
                        maxLevel = 2,
                        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
        
        #Queues used to store the details of thee tracked player
        self.positions = {} # track_id: deque of (x, y, time)
        self.current_speeds = {}  # track_id: current_speed_kmh
        self.acceleration = {} # track_id: 
        self.g_force = {} # track_id: 
        self.angle = {}
        self.angular_velocity = {} 
        self.angular_acceleration = {} 

        self.RED_GFORCE = 80
        self.YELLOW_GFORCE = 49
        self.RED_ANGULAR_ACCELERATION = 5875
        self.YELLOW_ANGULAR_ACCELERATION= 3512

    #Transform the BBox to get the centre bottom co-ordinates
    def transform_points(self, points: np.ndarray) -> np.ndarray:
        """Function to transform the players co-ordinates with the matrix"""
        #Array to store the co-ordinates
        bottom_centre_points = []
        for p in points:
            x1 = p[0] #Left X
            x2 = p[2] #Right X
            y = p[3]  #Bottom
            #Get the centre of X
            centre_x = (x1+x2)/2
            #Append the co-ordinates to the bottom_centre_points
            bottom_centre_points.append([centre_x, y])
        #Convert the array to a numpy array
        bottom_centre_points = np.array(bottom_centre_points, dtype=np.float32)
        #reshape the numpy array and transform the points for homography
        reshaped_points = bottom_centre_points.reshape(-1, 1, 2).astype(np.float32)
        transformed_points = cv2.perspectiveTransform(reshaped_points, self.matrix)
        return transformed_points.reshape(-1, 2)

    #Update the homography
    def update_matrix(self, frame):
        """Function to update the homography matrix between frames"""
        #convert the current frame to grey and apply corner detection
        current_grey_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        current_points = cv2.goodFeaturesToTrack(
            current_grey_frame,
            mask=None,
            **self.feature_params
        )

        if self.previous_grey_frame is not None and self.previous_points is not None:
            #Apply a Lucas Kanade optical flow to update the homography based on ther camera movement
            new_points, status, _ = cv2.calcOpticalFlowPyrLK(self.previous_grey_frame,current_grey_frame, self.previous_points.reshape(-1, 1, 2), None, **self.lk_params)
            good_new_points = new_points[status==1]
            good_previous_points = self.previous_points[status==1]
            if len(good_new_points) >= 4:
                #Find the camera movement based on the movement of the keypoints
                camera_H, _ = cv2.findHomography(good_previous_points, good_new_points, cv2.RANSAC, 5.0)
                backup_source = self.source.copy()
                if camera_H is not None:
                    #Update source points based on the camera movement
                    self.source = cv2.perspectiveTransform(self.source.reshape(-1, 1, 2), camera_H).reshape(-1, 2)
                    h, w = frame.shape[:2]
                    #If the new source points are outside the frame take re use the old ones
                    if np.any(self.source < 0) or np.any(self.source[:, 0] > w ) or np.any(self.source[:, 1] > h):
                        self.source = backup_source
                        return
                #H is the new homography matrix of the original source points based on the camera movement
                H, mask = cv2.findHomography(self.source, self.target)
                if H is not None:
                    self.matrix = H
        #Set the current frame as the previoius frame so that the homography can be updates in the next frame
        self.previous_grey_frame = current_grey_frame
        self.previous_points = current_points
        
    
    def calculate_speed(self, track_id, position, current_time):
        """Calculate current speed for track_id"""
        #Store a queue of the tracked players positions and speed
        if track_id not in self.positions:
            self.positions[track_id] = deque(maxlen=15)
        
        if track_id not in self.current_speeds:
            self.current_speeds[track_id] = deque(maxlen=15)

        #Append x,y and the time it has taken to reach the co-ordinates
        self.positions[track_id].append((position[0], position[1], current_time))
        positions = list(self.positions[track_id])
        #If we are at frame 1 then there is only 1 position stored therfore set speed to 0
        if len(positions) < 2:
            self.current_speeds[track_id].append((0.0, current_time))
            return 0.0
        #If 7 frames are available take the posions of the most recent frame and the 7th last frame
        #If there are only 4 take the posions of the most recent frame and the 4th last frame
        #E;se take last 2 positions
        if len(positions) >= 7:
            old_pos = positions[-7]
            new_pos = positions[-1]
        elif len(positions) >= 4:
            old_pos = positions[-4]
            new_pos = positions[-1]
        else:
            old_pos = positions[0]
            new_pos = positions[-1]
        
        #Subtarct the X and Y positions and calculate the euclidean distance to get the displacement of the player
        dx = new_pos[0] - old_pos[0]
        dy = new_pos[1] - old_pos[1]
        distance_meters = np.sqrt(dx*dx + dy*dy)
        #Get the time at both frames and subtract to get time difference
        time_diff = new_pos[2] - old_pos[2]
        if time_diff > 0:
            #Calculate speed in Metres per second
            speed_ms = distance_meters / time_diff
            # Filter unrealistic speeds
            if speed_ms > 200 or speed_ms < 0:
                speed_ms = 0
            #Append the speed to speeds queue
            self.current_speeds[track_id].append((speed_ms, current_time))
            return speed_ms
        else: 
            self.current_speeds[track_id].append((0.0, current_time))
            return 0.0
    
    def calculate_acceleration(self, track_id):
        """Calculate current acceleration for track_id"""
        #Store a queue of the tracked players acceleration
        if track_id not in self.acceleration:
            self.acceleration[track_id] = deque(maxlen=15)
        #Convert the speed queue to a list
        speed = list(self.current_speeds[track_id])
        #If we are at frame 1 then there is only 1 speed is stored therfore set speed to 0
        if len(speed) < 3:
            self.acceleration[track_id].append((0.0))
            return 0.0
        #If 7 frames are available take the speed of the most recent frame and the 7th last frame
        #If there are only 3 take the speed of the most recent frame and the 4th last frame
        #E;se take last 2 speed results
        # if len(speed) >= 7:
        #     old_speed = speed[-7]
        #     new_speed = speed[-1]
        # if len(speed) >= 3:
        #     old_speed = speed[-3]
        #     new_speed = speed[-1]
        else:
            old_speed = speed[-2]
            new_speed = speed[-1]
        #Get the time at both frames and subtract to get time difference
        time_diff = new_speed[1] - old_speed[1] 
        if time_diff > 0:
            #Subtarct both speeds from one another and divide by the time to calculate the acceraltion fo the player
            #This result can be neagtive as the player may slow down or stop
            acceleration = (new_speed[0] - old_speed[0]) / time_diff
            #Append the acceleration to acceleration queue
            self.acceleration[track_id].append(acceleration)
            return acceleration
        else:
            self.acceleration[track_id].append((0.0))
            return 0.0
    
    def calculate_GForce(self, track_id):
        #IF the track_if is not in acceleration the no acelration has been stored
        if track_id not in self.acceleration:
            return 0.0
        #Retreive the most recent acceleration
        accel = self.acceleration[track_id][-1]
        #Divide the acceraltion by the force of gravity which is 9.81
        g_Force = accel / 9.81
        return abs(g_Force)


    def calculate_angle(self, track_id, nose, left_ear, right_ear, current_time):
        """Function to calculate angle displacent"""
        #If the track id is not in angle then add it to angle
        if track_id not in self.angle:
            self.angle[track_id] = deque(maxlen=15)
        #print(f"ID: {track_id} / Nose: {nose} / LeftEar {left_ear} / RightEar {right_ear} / CurrentTime {current_time}")
        #Subtract the X and Y of the left and right ear
        dx = right_ear[0] - left_ear[0]
        dy = right_ear[1] - left_ear[1]
        #Calculate the Angular Displacemnt 
        angle_rad = np.arctan2(dy, dx)
        #ppend the displacement value
        self.angle[track_id].append((angle_rad, current_time))
        return angle_rad

    def calculate_anglular_velocity(self, track_id, current_time):
        """Function to calculate anglular velocity"""
        #If angular velocity queue does not exist for the tracking ID append it
        if track_id not in self.angular_velocity:
            self.angular_velocity[track_id] = deque(maxlen=15)
        #Convert the angle to a list
        angle = list(self.angle[track_id])
        #If 7 frames are available take the anglular displacement of the most recent frame and the 7th last frame
        #If there are only 4 take the anglular displacement of the most recent frame and the 4th last frame
        #Else take last 2 anglular displacement results
        if len(angle) >= 7:
            old_angle = angle[-7]
            new_angle = angle[-1]
        elif len(angle) >= 4:
            old_angle = angle[-4]
            new_angle = angle[-1]
        else:
            old_angle = angle[0]
            new_angle = angle[-1]
        #Get the time at both frames and subtract to get time difference
        time_diff = new_angle[1] - old_angle[1]
        if time_diff > 0:
            angular_velocity = (new_angle[0] - old_angle[0])/time_diff
            self.angular_velocity[track_id].append((angular_velocity, current_time))
            return angular_velocity
        else: 
            self.angular_velocity[track_id].append((0.0, current_time))
            return 0

    def calculate_anglular_acceleration(self, track_id):
        """Function to calculate anglular acceleration"""
        #If angular velocity queue does not exist for the tracking ID append it
        if track_id not in self.angular_acceleration:
            self.angular_acceleration[track_id] = deque(maxlen=15)
        angular_velocity = list(self.angular_velocity[track_id])

        #If 7 frames are available take the angular velocity of the most recent frame and the 7th last frame
        #If there are only 4 take the angular velocity of the most recent frame and the 4th last frame
        #Else take last 2 angular velocity results
        if len(angular_velocity) >= 4:
            old_angular_velocity = angular_velocity[-4]
            new_angular_velocity = angular_velocity[-1]
        else:
            old_angular_velocity = angular_velocity[0]
            new_angular_velocity = angular_velocity[-1]
        #Get the time at both frames and subtract to get time difference 
        time_diff = new_angular_velocity[1] - old_angular_velocity[1]
        if time_diff > 0:
            #Calculate the angular acceleration by subtracting the new and old angular velocitys to determine the acceleration that the angle moved
            angular_acceleration = (new_angular_velocity[0] - old_angular_velocity[0]) / time_diff
            self.angular_acceleration[track_id].append(angular_acceleration)
            return angular_acceleration
        else:
            self.angular_acceleration[track_id].append((0.0))
            return 0
        
    def calculate_risk(self, g_force, angular_acceleration):
        if g_force >= self.RED_GFORCE or abs(angular_acceleration) >= self.RED_ANGULAR_ACCELERATION:
            return "RED", (0,0,255)
        elif g_force >= self.YELLOW_GFORCE or abs(angular_acceleration) >= self.YELLOW_ANGULAR_ACCELERATION:
            return "YELLOW", (0,255,255)
        else:
            return "GREEN", (0,255,0)
