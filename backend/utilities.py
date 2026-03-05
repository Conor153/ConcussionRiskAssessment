import cv2 as cv
import torch
from ultralytics import YOLO
import numpy as np
from time import time
from perspective_transformation import BirdsEyeView #AngularCalculations
import math 
from sklearn.cluster import KMeans
from collections import defaultdict, deque

#model is a pretrained Yolo Model
#model = YOLO('../models/CoLab_T4/CoLab_T4GPU.pt')
#model = YOLO('../models/AMDCPUv1TrainYOLON/weights/AMDCPUv1TrainYOLOn.pt')
#model = YOLO('../models/AMDGPUv1TrainYOLON/weights/AMDGPUv1TrainYOLOn.pt')

#Models
model = YOLO('../models/AMDGPUv2TrainYOLOS/weights/AMDGPUTrainYOLOs.pt')
pose_model = YOLO('../models/PoseDetection/yolo11n-pose.pt')

#Use GPU if available
if torch.cuda.is_available():
    model.to('cuda:0')
    pose_model.to('cuda:0')
print(f"Model device: {model.device}, Pose device: {pose_model.device}")

#Function to resize the window frame
def rescaleFrame(frame, scale=0.75):
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)
    dimensions = (width, height)
    return cv.resize(frame, dimensions, interpolation=cv.INTER_AREA)

#Function to get the domiante colour of the players jersey
def get_dominant_colour(frame, bbox):
    #Get the co-ordinates of the box
    x1, y1, x2, y2 = map(int, bbox)
    #Ensure coordinates are within bounds
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(frame.shape[1], x2)
    y2 = min(frame.shape[0], y2)
    #Extract the player region
    playerBox = frame[y1:y2, x1:x2]
    #Get the height of the players box
    height = playerBox.shape[0]
    #Get the upperpart of the player body to get jersey colour
    torso = playerBox[int(height*0.30):int(height*0.5), :]
    #Convert the jersey colour to HSV
    hsv_roi = cv.cvtColor(torso, cv.COLOR_BGR2HSV)
    #Create mask to exclude green colours for grass
    lower_green = np.array([35, 40, 40])
    upper_green = np.array([85, 255, 255])
    green_mask = cv.inRange(hsv_roi, lower_green, upper_green)
    #Exclude colour green
    mask = cv.bitwise_not(green_mask)
    #Calculate mean colour
    if cv.countNonZero(mask) > 0:
        avg_colour_bgr = cv.mean(torso, mask=mask)[:3]
    else:
        avg_colour_bgr = cv.mean(torso)[:3]
    #Return the average colour
    return np.array(avg_colour_bgr)

#Extract the team colours
def extract_team_colours_from_frame(frame, model):
    results = model(frame)
    result = results[0]
    boxes = result.boxes
    player_colours = []
    #For all boxes get the dominant colour
    for box in boxes:
        cls = int(box.cls[0])
        if cls == 1:#Jersey
            bbox = box.xyxy[0].cpu().numpy()
            colour = get_dominant_colour(frame, bbox)
            if colour is not None:
                player_colours.append(colour)
    #Use KMeans to find 2 dominant team colours
    player_colours = np.array(player_colours)
    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
    kmeans.fit(player_colours)
    team1_colour = kmeans.cluster_centers_[0]
    team2_colour = kmeans.cluster_centers_[1]
    #Convert to HSV for better colour range definition
    team1_hsv = cv.cvtColor(np.uint8([[team1_colour]]), cv.COLOR_BGR2HSV)[0][0]
    team2_hsv = cv.cvtColor(np.uint8([[team2_colour]]), cv.COLOR_BGR2HSV)[0][0]
    #Create colour ranges
    team1_range = create_colour_range(team1_hsv)
    team2_range = create_colour_range(team2_hsv)
    #return team1 range and colour, team2 range and colour 
    return (team1_colour, team1_range), (team2_colour, team2_range), player_colours

#Create a range for the team colour
def create_colour_range(hsv_colour, h_tolerance=15, s_tolerance=60, v_tolerance=60):
    h, s, v = hsv_colour
    #Convert to int to avoid overflow warning
    h, s, v = int(h), int(s), int(v)
    lower = np.array([
        max(0, h - h_tolerance),
        max(0, s - s_tolerance),
        max(0, v - v_tolerance)
    ], dtype=np.uint8)
    upper = np.array([
        min(179, h + h_tolerance),
        min(255, s + s_tolerance),
        min(255, v + v_tolerance)
    ], dtype=np.uint8)
    return (lower, upper)

#Classify the team player belongs to based on jersey colour
def classify_team_by_colour(colour_bgr, team1_info, team2_info):
    if colour_bgr is None:
        return "Unknown", (128, 128, 128)
    team1_colour, _ = team1_info
    team2_colour, _ = team2_info
    # Calculate Euclidean distance in BGR space
    dist1 = np.linalg.norm(colour_bgr - team1_colour)
    dist2 = np.linalg.norm(colour_bgr - team2_colour)
    if dist1 < dist2:
        return "Team 1", (0, 0, 255)
    else:
        return "Team 2", (255, 0, 0)

def pose_estimation(frame, bbox):
    """Pose Estimation Function
    Used to Identify the keypoints of the body on the players bounding box
    The key points are mapped to the identified position
    Returned is the nose, and ears"""
    
    # Get the corners pof the bounding box
    x1, y1, x2, y2 = map(int, bbox)
    player_box = frame[y1:y2, x1:x2]

    if player_box.size == 0:
        return None, None, None
    
    #Connect the node to the left and right ears
    connections = [(0,3),(0,4),(3,4)]
    #Run the model ove rthe bouning box
    player_pose = pose_model(player_box, verbose=False)
    
    if player_pose is None or len(player_pose) == 0:
        return None, None, None
    if player_pose[0].keypoints is None:
        return None, None, None
    
    keypoints = player_pose[0].keypoints.xy.cpu().numpy()
    
    #If there is 
    if len(keypoints) == 0:
        return None, None, None
    
    #For 
    for person_kpts in keypoints:
        person_kpts = person_kpts[0:5]
        for start, end in connections:
            if start < len(person_kpts) and end < len(person_kpts):
                pt1, pt2 = person_kpts[start], person_kpts[end]
                if pt1[0] > 0 and pt1[1] > 0 and pt2[0] > 0 and pt2[1] > 0:
                    cv.line(frame, (int(pt1[0])+x1, int(pt1[1])+y1),
                            (int(pt2[0])+x1, int(pt2[1])+y1), (245, 66, 230), 3)
        for pt in person_kpts:
            if pt[0] > 0 and pt[1] > 0:
                cv.circle(frame, (int(pt[0])+x1, int(pt[1])+y1), 5, (245, 117, 66), -1)

    #Return the Node, Left ear and Right ear
    return person_kpts[0], person_kpts[3], person_kpts[4]

def create_source(first_frame):
    """A function which allows users to select the source co-ordinates from frame 1"""
    source_coordinates = []
    def get_pixel_coordinates(event, x, y, flags, param):
        if event == cv.EVENT_LBUTTONDOWN:  # left mouse click
            source_coordinates.append((x, y))
            cv.circle(param['frame'], (x, y), 5, (0, 0, 255), -1)

    print("\nClick 4 corners of the field. Press ESC when done (need 4 points)")
    param = {'frame': first_frame.copy()}
    cv.namedWindow("Frame", cv.WINDOW_GUI_EXPANDED)
    cv.setMouseCallback("Frame", get_pixel_coordinates, param)
    
    while True:
        display_frame = param['frame'].copy()
        cv.putText(display_frame, f"Points: {len(source_coordinates)}/4 - Press Q when done", 
                  (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv.imshow("Frame", display_frame)
        key = cv.waitKey(1) & 0xFF
        # Press Q to exit
        if key == ord('q'):
            break
    
    cv.destroyAllWindows()
    print(f"Selected coordinates: {source_coordinates}")
    print(f"Selected coordinates: {source_coordinates[1]}")
    print(f"Selected coordinates: {source_coordinates[1][1]}")
    source_coordinates = sorted(source_coordinates, key=lambda y: (y[1], -y[0]))
    print(f"Selected coordinates: {source_coordinates}")
        

    return source_coordinates

def read_video():
    """ Main video processing thread"""

    #Capture the uploaded video
    capture = cv.VideoCapture("../dataset/videos/video4.mp4")


    #Extract team colours from first frame
    # Get video properties
    frame_count = 0
    fps = capture.get(cv.CAP_PROP_FPS)


    isTrue, first_frame = capture.read()
    #Get the dominate colours of player jerseys to allocate object to team 1 or team 2
    team1_info, team2_info, all_colours = extract_team_colours_from_frame(first_frame, model)
    #Track statistics
    team_counts = defaultdict(int)
    #Target matrix for Homography
    target_width = 9.144 #meters of 1 10 yard line to a yard line 10 yards apart
    target_height = 26.79192 #meters distance between sideline numbers

    #Get user to select the source co-ordinates
    source_values = create_source(first_frame)
    source = np.array(source_values, dtype=np.float32)
    target = np.array([[0, 0], [target_width, 0], [target_width, target_height], [0, target_height]], dtype=np.float32)
    #Create BirdsEyeView object to create homography and calaulate metrics
    transformation = BirdsEyeView(source, target)
    #Process video 
    while True:
        isTrue, frame = capture.read()
        if not isTrue:
            break
        #Update the homography 
        transformation.update_matrix(frame)
        #Calculate time with the video
        current_time = frame_count/fps
        #Process the frame with the custom trained model to detect player, helmet and jersey objects. Track using Bytetrack
        results = model.track(source=frame, show=False, persist=True, verbose=False, conf=0.6, iou=0.6, tracker="bytetrack.yaml")  
        #results = model.track(source=frame, show=False, persist=True, verbose=True, conf=0.6, iou=0.6, tracker="botsort.yaml")  
        result = results[0]
        boxes = result.boxes
        #Transform the points of the bounding box 
        transformed_coords = transformation.transform_points(boxes.xyxy.cpu().numpy().astype(np.float32))
        #Process each detected bounding box
        for i, box in enumerate(boxes):
            cls = int(box.cls[0])    
            if cls == 2: #Person class is 2

                bbox = box.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = map(int, bbox)
                x,y = transformed_coords[i]
                track_id = int(box.id[0]) if box.id is not None else None
                #Assign a lable ID for player identification during video
                label = f"ID:{track_id}"

                #Calculate speed, acceleration and G-force of ther player using their transformed X and Y co-ordinates
                speed = transformation.calculate_speed(track_id, (x, y), current_time)
                acceleration = transformation.calculate_acceleration(track_id)
                g_force = transformation.calculate_GForce(track_id)
                
                #Estimate the pose of the player bounding box and retrieve the posiions of the node and ears
                nose, left_ear, right_ear = pose_estimation(frame, bbox)
                if nose is None or left_ear is None or right_ear is None:
                    continue
                #Calculate the angular displacement, angular_velocity and angular_acceleration of the player
                angle = transformation.calculate_angle(track_id, nose, left_ear, right_ear, current_time)
                angular_velocity = transformation.calculate_anglular_velocity(track_id, current_time)
                angular_acceleration = transformation.calculate_anglular_acceleration(track_id)
                #Attach label with values for display purposes

                #label = f"ID:{track_id} Angle{abs(round(angle))} Rad"
                #label = f"ID:{track_id} Angle Velocity{round(angular_velocity)} Rad"
                #label = f"ID:{track_id} Angle Acceleration {abs(round(angular_acceleration))} Rad^2"
                
                label = f"ID:{track_id} Speed{round(speed)} MpS"
                #label = f"ID:{track_id} Acceleration:{round(acceleration)} MS^2"
                #label = f"ID:{track_id} G-Force:{round(g_force)} G"
                 
               
                #Get dominant colour of the player bounidng box and classify the player to team 1 or team 2pppppp
                colour = get_dominant_colour(frame, bbox)
                team_label, box_colour = classify_team_by_colour(colour, team1_info, team2_info)
                team_counts[team_label] += 1
                # Draw background rectangle and place text over it
                cv.rectangle(frame,(x1, y1 - 20),(x2+100, y1),box_colour,-1)
                cv.putText(frame,label,(x1, y1 - 5),cv.FONT_HERSHEY_SIMPLEX,0.5,(255, 255, 255),2)  

        #Resize the frame
        frameResized = rescaleFrame(frame, scale=0.75)
        key = cv.waitKey(1) & 0xFF
        cv.imshow("Team Classification", frameResized)
        #Pause if 'p' is pressed
        if key  == ord('p'):
            cv.waitKey(0)
        #Stop if 'q' is pressed
        if key == ord('q'):
            break
        frame_count+= 1
    capture.release()
    cv.destroyAllWindows()
    

read_video()