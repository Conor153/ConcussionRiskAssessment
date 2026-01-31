import cv2 as cv
import numpy as np
import math 
from ultralytics import YOLO
from sklearn.cluster import KMeans
from collections import defaultdict



#mode is a pretrained Yolo Model
model = YOLO('yolo11n.pt')

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
        return "Team 1", (0, 255, 255)
    else:
        return "Team 2", (255, 255, 0)

#Function to calculate the velocity 
def get_velocity(frame, bbox):
    displacement = get_displacement(frame, bbox)
    time = get_time()
    velocity = displacement/time
    return velocity

#Function to calculate the displacement
def get_displacement(frame, bbox):
    #Get X and Y from the Player box
    x1, y1, x2, y2 = map(int, bbox)
    #Get the centre of both X and Y at begining and at the end
    initial_X_position, initial_Y_position = get_centre_of_bbox(x1, y1, x2, y2)
    final_X_position, final_Y_position = get_centre_of_bbox(x1, y1, x2, y2)
    displacement = math.sqrt(math.pow(final_X_position - initial_X_position) + math.pow(final_Y_position - initial_Y_position))
    return displacement

def get_centre_of_bbox(x1, y1, x2, y2):
    return (x1+x2)/2, (y1+y2)/2
    
    
#Function to calculate the Time
def get_time():
    time = frame_count/fps
    return time

#Function to calculate the acceleration
def get_acceleration():
    acceleration = get_velocity()/get_time()
    return acceleration 

def calculate_GForce():
    frame, bbox
    gForce = get_acceleration()/9.81
    #Green      
    #Yellow     
    #Red        

def calculate_angular_acceleration():
    angular_acceleration = calculate_angular_velocity()/get_time()
    #Green      
    #Yellow     
    #Red        

def calculate_angular_displacement(angle_a, angle_B):
    angular_displacement = angle_a - angle_B
    return angular_displacement

def calculate_angular_velocity():
    angular_velocity = calculate_angular_displacement()
    return angular_velocity


def get_camera_movement(frames):
    camera_movement = [[0,0]*len(frames)]
    grey = cv.cvtColor(frames[0],cv.COLOR_BGR2GRAY)



def read_video():
    #Main video processing
    isTrue, capture = cv.VideoCapture('../videos/CJStroudConcussion.mp4')
    frames = []
    #Extract team colours from first frame
    first_frame = capture.read()
    team1_info, team2_info, all_colours = extract_team_colours_from_frame(first_frame, model)
    #Track statistics
    team_counts = defaultdict(int)
    #Process video
    while True:
        isTrue, frame = capture.read()
        if not isTrue:
            break
        # Run Object Detection detection
        #trackingID = model.track(source=frame, show=True, tracker="bytetrack.yaml")
        results = model(frame)
        result = results[0]
        boxes = result.boxes
        #Process each detected person
        for box in boxes:
            cls = int(box.cls[0])
            if cls == 0:
                bbox = box.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = map(int, bbox)
                
                #Get dominant colour
                colour = get_dominant_colour(frame, bbox)
                
                #Classify team
                team_label, box_colour = classify_team_by_colour(colour, team1_info, team2_info)
                team_counts[team_label] += 1
                
                #Draw bounding box
                x_centre_position, y_centre_position = get_centre_of_bbox(x1, y1, x2, y2)
                #cv.ellipse(frame, (x_centre_position, y_centre_position), (int(width), int(width)35%),0,0,360 box_colour, 3)
                cv.rectangle(frame, (x1, y1), (x2, y2), box_colour, 3)
                
                #Add label
                label = f"{team_label} {trackerID}"
                cv.putText(frame, label, (x1, y1-10),cv.FONT_HERSHEY_SIMPLEX, 0.5, box_colour, 2)
        
        frames.append(frame)
        #Resize the frame
        frameResized = rescaleFrame(frame, scale=0.75)
        cv.imshow("Team Classification", frameResized)
        #Stop if 'd' is pressed
        if cv.waitKey(20) & 0xFF == ord('d'):
            break

    capture.release()
    cv.destroyAllWindows()
    return frames
    