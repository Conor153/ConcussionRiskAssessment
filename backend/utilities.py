import cv2 as cv
import torch
from torchvision import ops
from ultralytics import YOLO
import numpy as np
import subprocess
import os
from time import time
from perspective_transformation import BirdsEyeView 
import math 
from sklearn.cluster import KMeans
from collections import defaultdict, deque

#model is a pretrained Yolo Model
#model = YOLO('../models/CoLab_T4/CoLab_T4GPU.pt')
#model = YOLO('../models/AMDCPUv1TrainYOLON/weights/AMDCPUv1TrainYOLOn.pt')
#model = YOLO('../models/AMDGPUv1TrainYOLON/weights/AMDGPUv1TrainYOLOn.pt')

#Models
model = YOLO('../models/AMDGPUv2TrainYOLOS/weights/AMDGPUTrainYOLOs.pt')
pose_model = YOLO('../models/PoseDetection/yolo11s-pose.pt')

#Use GPU if available
if torch.cuda.is_available():
    model.to('cuda:0')
    pose_model.to('cuda:0')
print(f"Model device: {model.device}, Pose device: {pose_model.device}")

def pose_estimation(frame, bbox):
    """Pose Estimation Function
    Used to Identify the keypoints of the body on the players bounding box
    The key points are mapped to the identified position
    Returned is the nose, and ears"""
    
    #Get the corners of the bounding box
    x1, y1, x2, y2 = map(int, bbox)
    player_box = frame[y1:y2, x1:x2]

    #Return left and right ear as None
    if player_box.size == 0:
        return None, None
    
    #Connect the nose to the left and right ears
    connections = [(0,3),(0,4),(3,4)]
    #Run the model ove rthe bouning box
    player_pose = pose_model(player_box, verbose=False)
    
    #Return left and right ear as None
    if player_pose is None or len(player_pose) == 0:
        return None, None
    #Return left and right ear as None
    if player_pose[0].keypoints is None:
        return None, None
    
    keypoints = player_pose[0].keypoints.xy.cpu().numpy()
    
    #If no keypoints have been detected return None for both left and right ear
    if len(keypoints) == 0:
        return None, None
    #Take the first 5 keypoints which are of the head dimesnsions 
    for person_kpts in keypoints:
        person_kpts = person_kpts[0:5]
        #Draw a connecting line between the keypoints
        for start, end in connections:
            if start < len(person_kpts) and end < len(person_kpts):
                pt1, pt2 = person_kpts[start], person_kpts[end]
                if pt1[0] > 0 and pt1[1] > 0 and pt2[0] > 0 and pt2[1] > 0:
                    cv.line(frame, (int(pt1[0])+x1, int(pt1[1])+y1),
                            (int(pt2[0])+x1, int(pt2[1])+y1), (245, 66, 230), 3)
        #Map the keypoint onto the players body
        for pt in person_kpts:
            if pt[0] > 0 and pt[1] > 0:
                cv.circle(frame, (int(pt[0])+x1, int(pt[1])+y1), 5, (245, 117, 66), -1)

    #Return the Left ear and Right ear
    return person_kpts[3], person_kpts[4]

def sort_source(source_coordinates):
    """A function which sorts the source co-ordinates from frame 1"""
    #Sort the sourc co-ordinates into following order
    #Top-Left
    #Top-Right
    #Bottom Right
    #Bottom Left
    source_coordinates = sorted(source_coordinates, key=lambda y: (y[1]))
    if source_coordinates[0][0]>source_coordinates[1][0]:
        swap = source_coordinates[0]
        source_coordinates[0] = source_coordinates[1]
        source_coordinates[1] = swap
    if source_coordinates[2][0]<source_coordinates[3][0]:
        swap = source_coordinates[2]
        source_coordinates[2] = source_coordinates[3]
        source_coordinates[3] = swap
    return source_coordinates

def process_video(video_path, source_points):
    """ Main video processing thread"""

    #Capture the uploaded video
    capture = cv.VideoCapture(video_path)
    
    #Get video properties
    frame_count = 0
    fps = capture.get(cv.CAP_PROP_FPS)
    frame_width = int(capture.get(cv.CAP_PROP_FRAME_WIDTH))
    frame_height = int(capture.get(cv.CAP_PROP_FRAME_HEIGHT))

    #Format output video into mp4v
    fourcc = cv.VideoWriter_fourcc(*"mp4v")
    #Set the output of 
    outPath = 'processed_videos/processed_mp4_video.mp4'
    #Create an output to write the file to
    output = cv.VideoWriter(outPath, fourcc, fps/4, (frame_width, frame_height))

    #Risk Result dictionary to store peak risk values for identified IDs
    risk_result = {}

    #Target matrix for Homography
    target_width = 9.144 #meters of 1 10 yard line to a yard line 10 yards apart
    target_height = 26.79192 #meters distance between sideline numbers

    #Sort the source co-ordinates
    source_values = sort_source(source_points)
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
        results = model.track(source=frame, show=False, persist=True, verbose=False, conf=0.4, iou=0.4, tracker="../models/Trackers/bytetrack.yaml")  
        result = results[0]
        boxes = result.boxes
        #Transform the points of the bounding box 
        transformed_coords = transformation.transform_points(boxes.xyxy.cpu().numpy().astype(np.float32))
        #Process each detected bounding box
        for i, box in enumerate(boxes):
            cls = int(box.cls[0])    
            #Track helmet class which is 0
            if cls == 0:
                helmet_box = box.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = map(int, helmet_box)
                x,y = transformed_coords[i]
                track_id = int(box.id[0]) if box.id is not None else None
                #Assign a lable ID for player identification during video
                label = f"ID:{track_id}"
                #Calculate speed, acceleration and G-force of ther player using their transformed X and Y co-ordinates
                transformation.calculate_speed(track_id, (x, y), current_time)
                transformation.calculate_acceleration(track_id)
                g_force = transformation.calculate_GForce(track_id)            
                
                #IoU to get matching helemt and player box
                best_iou = 0
                best_player_box = None
                for j, bbox in enumerate(boxes):
                    player_cls = int(bbox.cls[0])  
                    #Track player class which is 2
                    if player_cls == 2:
                        player_box = bbox.xyxy[0].cpu().numpy()
                        px1, py1, px2, py2 = map(int, player_box)
                        helmet_box = torch.tensor([[x1, y1, x2, y2]], dtype=torch.float)
                        player_box = torch.tensor([[px1, py1, px2, py2]], dtype=torch.float)
                        iou = ops.box_iou(player_box, helmet_box)
                        if iou > best_iou:
                            best_iou = iou
                            best_player_box = bbox.xyxy[0].cpu().numpy()             
                if best_player_box is None:
                    continue
                left_ear, right_ear = pose_estimation(frame, best_player_box)
                if left_ear is None or right_ear is None:
                    continue
                
                #Transform left and right ears with the homography
                left_ear = transformation.transform_point(left_ear)
                right_ear = transformation.transform_point(right_ear)
                
                #Calculate the angular displacement, angular_velocity and angular_acceleration of the player
                transformation.calculate_angle(track_id, left_ear, right_ear, current_time)
                transformation.calculate_angular_velocity(track_id, current_time)
                angular_acceleration = transformation.calculate_angular_acceleration(track_id)
               
                #Attach label with values for display purposes
                label = f"ID:{track_id} G-Force:{round(g_force)} G | AA {abs(round(angular_acceleration))}"
                risk, box_colour = transformation.calculate_risk(g_force, angular_acceleration)
                
                #Draw Bounding Box with labels
                cv.rectangle(frame,(x1, y1),(x2, y2),box_colour,2)
                cv.putText(frame,label,(x1, y1 - 5),cv.FONT_HERSHEY_SIMPLEX,0.5,(255, 255, 255),2)  

                #If the track_id is not in the dictionary
                if track_id not in risk_result:
                    #Add it to dictionary with intial values
                    risk_result[track_id] = {
                        "track_id": track_id, 
                        "g_force" : round(g_force), 
                        "angular_acceleration": abs(round(angular_acceleration)), 
                        "risk": risk}
                #Else if the track_id is in the dictionary
                else:
                    #If the risk is red update the risk and add a pause to the video
                    if risk == "RED":
                        risk_result[track_id].update({"risk": risk})
                        for i in range(int(fps * 2)):
                            output.write(frame)
                    #If the risk is Yellow and red is not already recorded for the player update the result to yellow  
                    elif risk == "YELLOW" and risk_result[track_id].get("risk") != "RED":
                        risk_result[track_id].update({"risk": risk})

                    #If higher G-force recorded update the value in the dictionary
                    if risk_result[track_id].get("g_force") < round(g_force):
                        risk_result[track_id].update({"g_force": round(g_force)}) 
                    #If higher angular acceleration recorded update the value in the dictionary
                    if risk_result[track_id].get("angular_acceleration") < abs(round(angular_acceleration)): 
                            risk_result[track_id].update({"angular_acceleration": abs(round(angular_acceleration))})


        #Write the frame to create video
        output.write(frame)
           
        #Increase frame counter for time calculation
        frame_count+= 1

    #Release capture and output
    capture.release()
    output.release()

    #Convert file using ffmeg so that it is displayable in browser as type H.264
    videoPath = 'processed_videos/processed_video.mp4'
    subprocess.run([
        "ffmpeg", "-i", outPath,
        "-vcodec", "libx264",
        "-y", videoPath
    ])

    #Return the video and risk results
    return videoPath, list(risk_result.values())