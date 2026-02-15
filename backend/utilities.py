import cv2 as cv
import numpy as np
from time import time
from perspective_transformation import BirdsEyeView
import math 
from ultralytics import YOLO
from sklearn.cluster import KMeans
from collections import defaultdict, deque
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


#model is a pretrained Yolo Model
#model = YOLO('../models/CoLab_T4/CoLab_T4GPU.pt')
#model = YOLO('../models/AMDCPUv1TrainYOLON/weights/AMDCPUv1TrainYOLOn.pt')
#model = YOLO('../models/AMDGPUv1TrainYOLON/weights/AMDGPUv1TrainYOLOn.pt')

model = YOLO('../models/AMDGPUv2TrainYOLOS/weights/AMDGPUTrainYOLOs.pt')
#model = YOLO('yolo11s-pose.pt')
#Create Pose Landmarker object
# BaseOptions = mp.tasks.BaseOptions
# PoseLandmarker = mp.tasks.vision.PoseLandmarker
# PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
# FaceLandmarker = mp.tasks.vision.FaceLandmarker
# FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
# VisionRunningMode = mp.tasks.vision.RunningMode
# pose_options = PoseLandmarkerOptions(base_options=BaseOptions(model_asset_path="../models/PoseDetection/pose_landmarker_full.task"), running_mode=VisionRunningMode.IMAGE)
# face_options = FaceLandmarkerOptions(base_options=BaseOptions(model_asset_path="../models/FaceDetection/face_landmarker.task"), running_mode=VisionRunningMode.IMAGE)
# pose_landmarker = PoseLandmarker.create_from_options(pose_options)
# face_landmarker = FaceLandmarker.create_from_options(face_options)


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
        if cls == 0:
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

# def calculate_angular_acceleration():
#     angular_acceleration = calculate_angular_velocity()/get_time()
#     #Green      
#     #Yellow     
#     #Red        

# def calculate_angular_displacement(angle_a, angle_B):
#     angular_displacement = angle_a - angle_B
#     return angular_displacement

# def calculate_angular_velocity():
#     angular_velocity = calculate_angular_displacement()
#     return angular_velocity


# def get_camera_movement(frames):
#     camera_movement = [[0,0]*len(frames)]
#     grey = cv.cvtColor(frames[0],cv.COLOR_BGR2GRAY)

# def pose_estimation(frame, bbox):
#     x1, y1, x2, y2 = map(int, bbox)
#     playerImage = frame[y1:y2, x1:x2]
#     playerImage = cv.cvtColor(playerImage, cv.COLOR_BGR2RGB)
#     mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=playerImage)
#     pose_landmarker_result = pose_landmarker.detect(mp_image)
#     print("Pose landmarks:", pose_landmarker_result.pose_landmarks) 
#     return pose_landmarker_result
    #face_landmarker_result = face_landmarker.detect(mp_image)
    #print("Face landmarks:", len(face_landmarker_result.face_landmarks[0]) if face_landmarker_result.face_landmarks else 0)
    # #Render Detections
    # mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
    #                           mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),
    #                           mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2))
# def draw_pose_on_full_frame(frame, bbox, pose_result):
#     if not pose_result.pose_landmarks:
#         return
#     x1, y1, x2, y2 = map(int, bbox)
#     w = x2 - x1
#     h = y2 - y1
#     landmarks = pose_result.pose_landmarks[0]
#     #Draw each joint
#     for lm in landmarks:
#         cx = x1 + int(lm.x * w)
#         cy = y1 + int(lm.y * h)
#         cv.circle(frame, (cx, cy), 3, (0, 255, 0), -1)

def read_video():
    #Main video processing
    capture = cv.VideoCapture('../dataset/videos/CJStroudConcussion.mp4')
    #capture = cv.VideoCapture('C:/Users/Conor/Videos/ConcussionAssessment/ConcussionHits/MarquiseGoodwinConcussion.mp4')
    #Extract team colours from first frame

    # Get video properties
    fps = capture.get(cv.CAP_PROP_FPS)
    frame_count = int(capture.get(cv.CAP_PROP_FRAME_COUNT))


    isTrue, first_frame = capture.read()
    team1_info, team2_info, all_colours = extract_team_colours_from_frame(first_frame, model)
    #Track statistics
    team_counts = defaultdict(int)
    #Target is full NFL field size
    #Source is the area within the videp
    target_width = 109.7 #meters of NFL Field endzone to endzone
    target_height = 48.8 #meters of NFL Field sideline to sideline
    source = np.array([[200, 50],[1720, 50],[1880, 1000],[40, 1000]])
    target = np.array([[0, 0], [target_width-1, 0], [target_width-1, target_height-1], [0, target_height-1]])
    transformation = BirdsEyeView(source, target)
    #MediaPipe Pose estimation confidence of 50% for detecting and tracking
    #Process video
    while True:
        isTrue, frame = capture.read()
        if not isTrue:
            break
        current_time = time()
        results = model.track(source=frame, show=False, persist=True, verbose=True, conf=0.5, iou=0.5, tracker="bytetrack.yaml")  
        result = results[0]
        boxes = result.boxes

        
        #Transform the points of the bounding box 
        transformed_coords = transformation.transform_points(boxes.xyxy.cpu().numpy().astype(np.float32))
        #Process each detected person
        for i, box in enumerate(boxes):
            cls = int(box.cls[0])
            if cls == 0: #Person class is 2
                bbox = box.xyxy[0].cpu().numpy()
                #pose_result = pose_estimation(frame, bbox)
                #draw_pose_on_full_frame(frame, bbox, pose_result)
                x1, y1, x2, y2 = map(int, bbox)
                x,y = transformed_coords[i]
                track_id = int(box.id[0]) if box.id is not None else None
                #Save the transformed co-ordinates of the box at its id position
                label = f"ID:{track_id}"
                speed = transformation.calculate_speed(track_id, (x, y), current_time)
                acceleration = transformation.calculate_acceleration(track_id)
                g_force = transformation.calculate_GForce(track_id)
                label = f"ID:{track_id} Speed{round(speed)} Kph | Acceleration:{round(acceleration)} MS^2 | G-Force:{round(g_force)} G"

                #Get dominant colour
                colour = get_dominant_colour(frame, bbox)
                
                #Classify team
                team_label, box_colour = classify_team_by_colour(colour, team1_info, team2_info)
                team_counts[team_label] += 1

                cv.rectangle(frame, (x1, y1), (x2, y2), box_colour, 3)
                
                #Add label with background for better visibility
                label_text = f"{team_label} {label}"

                # Draw background rectangle
                cv.rectangle(frame,(x1, y1 - 5),(x2+300, y1-20),box_colour,-1)
                
                # Draw text on top of background
                cv.putText(frame,label_text,(x1, y1 - 5),cv.FONT_HERSHEY_SIMPLEX,0.5,(255, 255, 255),2)
            
        #Resize the frame
        frameResized = rescaleFrame(frame, scale=0.75)
        cv.imshow("Team Classification", frameResized)
        #Stop if 'd' is pressed
        if cv.waitKey(20) & 0xFF == ord('d'):
            break

    capture.release()
    cv.destroyAllWindows()
    

read_video()