import unittest
import numpy as np
import cv2 as cv
from ultralytics import YOLO
from perspective_transformation import BirdsEyeView 

class TestConcussionRisk(unittest.TestCase):
        

    def setUp(self):
        self.model = YOLO('../models/AMDGPUv2TrainYOLOS/weights/AMDGPUTrainYOLOs.pt')
        self.pose_model = YOLO('../models/PoseDetection/yolo11n-pose.pt')
        target_width = 9.144 #meters of 1 10 yard line to a yard line 10 yards apart
        target_height = 26.79192 #meters distance between sideline numbers
        source = np.array([[0, 0], [target_width, 0], [target_width, target_height], [0, target_height]], dtype=np.float32)
        target = np.array([[0, 0], [target_width, 0], [target_width, target_height], [0, target_height]], dtype=np.float32)
        self.transformation = BirdsEyeView(source, target)

    #Calculations Test
    def test_speed(self):
        #As only co-ordinate is stored. Player has no speed
        result = self.transformation.calculate_speed(1,(1.0,13.0), 0.0)
        self.assertEqual(result, 0)
        #Frame 1
        result = self.transformation.calculate_speed(1,(0.95,13.05), 0.016666666666666666)
        self.assertEqual(round(result, 2), 4.24)
        #Frame 2
        result = self.transformation.calculate_speed(1,(0.92, 13.09), 0.03333333333333333)
        self.assertEqual(round(result, 2), 3.61)
        #Frame 3
        result = self.transformation.calculate_speed(1,(0.87, 13.13), 0.05)
        self.assertEqual(round(result, 2), 3.68)
        #Frame 4
        result = self.transformation.calculate_speed(1,(0.90, 13.19), 0.06666666666666667)
        self.assertEqual(round(result, 2), 2.97)
        #Frame 5
        result = self.transformation.calculate_speed(1,(0.95, 13.24), 0.08333333333333333)
        self.assertEqual(round(result, 2), 3.06)
        #Frame 6
        result = self.transformation.calculate_speed(1,(1, 13.29), 0.1)
        self.assertEqual(round(result, 2), 2.90)
        #Frame 8
        result = self.transformation.calculate_speed(1,(1.06, 13.33), 0.11666666666666667)
        self.assertEqual(round(result, 2), 3.01)
        #Frame 9
        result = self.transformation.calculate_speed(1,(1.1, 13.39), 0.13333333333333333)
        self.assertEqual(round(result, 2), 3.50)
        #Frame 10
        result = self.transformation.calculate_speed(1,(1.13, 13.45), 0.15)
        self.assertEqual(round(result, 2), 4.12)
        #Frame 11
        result = self.transformation.calculate_speed(1,(1.2, 13.5), 0.16666666666666666)
        self.assertEqual(round(result, 2), 4.31)
        #Frame 12
        result = self.transformation.calculate_speed(1,(1.27, 13.56), 0.18333333333333332)
        self.assertEqual(round(result, 2), 4.53)
        #Frame 13
        result = self.transformation.calculate_speed(1,(1.34, 13.64), 0.2)
        self.assertEqual(round(result, 2), 4.88)
        #Frame 14
        result = self.transformation.calculate_speed(1,(1.42, 13.72), 0.21666666666666667)
        self.assertEqual(round(result, 2), 5.31)
        #Frame 15
        result = self.transformation.calculate_speed(1,(1.5, 13.78), 0.23333333333333333)
        self.assertEqual(round(result, 2), 5.59)
        
    #Fix
    def test_acceleration(self):
        #Frame 1
        self.transformation.calculate_speed(1,(1.0,13.0), 0.0)
        result = self.transformation.calculate_acceleration(1)
        self.assertEqual(result, 0)
        #Frame 2
        self.transformation.calculate_speed(1,(0.95,13.05), 0.016666666666666666)
        result = self.transformation.calculate_acceleration(1)
        self.assertEqual(round(result), 0)
        #Frame 3
        self.transformation.calculate_speed(1,(0.92, 13.09), 0.03333333333333333)
        result = self.transformation.calculate_acceleration(1)
        self.assertEqual(round(result), 108)
        #Frame 4
        self.transformation.calculate_speed(1,(0.87, 13.13), 0.05)
        result = self.transformation.calculate_acceleration(1)
        self.assertEqual(round(result), -17)
        #Frame 5
        self.transformation.calculate_speed(1,(0.90, 13.19), 0.06666666666666667)
        result = self.transformation.calculate_acceleration(1)
        self.assertEqual(round(result), -19)
        #Frame 6
        self.transformation.calculate_speed(1,(0.95, 13.24), 0.08333333333333333)
        result = self.transformation.calculate_acceleration(1)
        self.assertEqual(round(result), -19)
        #Frame 7
        self.transformation.calculate_speed(1,(1, 13.29), 0.1)
        result = self.transformation.calculate_acceleration(1)
        self.assertEqual(round(result), -2)
        #Frame 8
        self.transformation.calculate_speed(1,(1.06, 13.33), 0.11666666666666667)
        result = self.transformation.calculate_acceleration(1)
        self.assertEqual(round(result), -2)
        #Frame 9
        self.transformation.calculate_speed(1,(1.1, 13.39), 0.13333333333333333)
        result = self.transformation.calculate_acceleration(1)
        self.assertEqual(round(result), 18)
        #Frame 10
        self.transformation.calculate_speed(1,(1.13, 13.45), 0.15)
        result = self.transformation.calculate_acceleration(1)
        self.assertEqual(round(result), 33)
        #Frame 11
        self.transformation.calculate_speed(1,(1.2, 13.5), 0.16666666666666666)
        result = self.transformation.calculate_acceleration(1)
        self.assertEqual(round(result), 24)
        #Frame 12
        self.transformation.calculate_speed(1,(1.27, 13.56), 0.18333333333333332)
        result = self.transformation.calculate_acceleration(1)
        self.assertEqual(round(result), 12)
        #Frame 13
        self.transformation.calculate_speed(1,(1.34, 13.64), 0.2)
        result = self.transformation.calculate_acceleration(1)
        self.assertEqual(round(result), 17)
        #Frame 14
        self.transformation.calculate_speed(1,(1.42, 13.72), 0.21666666666666667)
        result = self.transformation.calculate_acceleration(1)
        self.assertEqual(round(result), 23)
        #Frame 15
        self.transformation.calculate_speed(1,(1.5, 13.78), 0.23333333333333333)
        result = self.transformation.calculate_acceleration(1)
        self.assertEqual(round(result), 21)

    #Fix
    def test_calculating_green_gForce(self):
        #Frame 1
        self.transformation.calculate_speed(1,(1.0,13.0), 0.0)#
        self.transformation.calculate_acceleration(1)#
        result = self.transformation.calculate_GForce(1)
        self.assertEqual(result, 0)
        #Frame 2
        self.transformation.calculate_speed(1,(0.95,13.05), 0.016666666666666666)#
        self.transformation.calculate_acceleration(1)#
        result = self.transformation.calculate_GForce(1)
        self.assertEqual(round(abs(result)), 0)
        #Frame 3
        self.transformation.calculate_speed(1,(0.92, 13.09), 0.03333333333333333)#
        self.transformation.calculate_acceleration(1)#
        result = self.transformation.calculate_GForce(1)
        self.assertEqual(round(abs(result)), 11)
        #Frame 4
        self.transformation.calculate_speed(1,(0.87, 13.13), 0.05)#
        self.transformation.calculate_acceleration(1)#
        result = self.transformation.calculate_GForce(1)
        self.assertEqual(round(abs(result)), 2)
        #Frame 5
        self.transformation.calculate_speed(1,(0.90, 13.19), 0.06666666666666667)#
        self.transformation.calculate_acceleration(1)#
        result = self.transformation.calculate_GForce(1)
        self.assertEqual(round(abs(result)), 2)
        #Frame 6
        self.transformation.calculate_speed(1,(0.95, 13.24), 0.08333333333333333)#
        self.transformation.calculate_acceleration(1)#
        result = self.transformation.calculate_GForce(1)
        self.assertEqual(round(abs(result)), 2)
        #Frame 7
        self.transformation.calculate_speed(1,(1, 13.29), 0.1)#
        self.transformation.calculate_acceleration(1)#
        result = self.transformation.calculate_GForce(1)
        self.assertEqual(round(abs(result)), 0)
        #Frame 8
        self.transformation.calculate_speed(1,(1.06, 13.33), 0.11666666666666667)#
        self.transformation.calculate_acceleration(1)#
        result = self.transformation.calculate_GForce(1)
        self.assertEqual(round(abs(result)), 0)
        #Frame 9
        self.transformation.calculate_speed(1,(1.1, 13.39), 0.13333333333333333)#
        self.transformation.calculate_acceleration(1)#
        result = self.transformation.calculate_GForce(1)
        self.assertEqual(round(abs(result)), 2)
        #Frame 10
        self.transformation.calculate_speed(1,(1.13, 13.45), 0.15)#
        self.transformation.calculate_acceleration(1)#
        result = self.transformation.calculate_GForce(1)
        self.assertEqual(round(abs(result)), 3)
        #Frame 11
        self.transformation.calculate_speed(1,(1.2, 13.5), 0.16666666666666666)#
        self.transformation.calculate_acceleration(1)#
        result = self.transformation.calculate_GForce(1)
        self.assertEqual(round(abs(result)), 2)
        #Frame 12
        self.transformation.calculate_speed(1,(1.27, 13.56), 0.18333333333333332)#
        self.transformation.calculate_acceleration(1)#
        result = self.transformation.calculate_GForce(1)
        self.assertEqual(round(abs(result)), 1)
        #Frame 13
        self.transformation.calculate_speed(1,(1.34, 13.64), 0.2)#
        self.transformation.calculate_acceleration(1)#
        result = self.transformation.calculate_GForce(1)
        self.assertEqual(round(abs(result)), 2)
        #Frame 14
        self.transformation.calculate_speed(1,(1.42, 13.72), 0.21666666666666667)#
        self.transformation.calculate_acceleration(1)#
        result = self.transformation.calculate_GForce(1)
        self.assertEqual(round(abs(result)), 2)
        #Frame 15
        self.transformation.calculate_speed(1,(1.5, 13.78), 0.23333333333333333)#
        self.transformation.calculate_acceleration(1)#
        result = self.transformation.calculate_GForce(1)
        self.assertEqual(round(abs(result)), 2)
    
    #   #Fix
    def test_calculating_yellow_gForce(self):
        #Frame 1
        self.transformation.calculate_speed(1,(1.0,13.0), 0.0)
        self.transformation.calculate_acceleration(1)
        result = self.transformation.calculate_GForce(1)
        self.assertEqual(result, 0)
        #Frame 2
        self.transformation.calculate_speed(1,(0.95,13.05), 0.016666666666666666)#4.24
        self.transformation.calculate_acceleration(1)#254.50
        result = self.transformation.calculate_GForce(1)
        self.assertEqual(round(abs(result)), 26)
        #Frame 3
        self.transformation.calculate_speed(1,(0.96, 13.09), 0.03333333333333333)#2.96
        self.transformation.calculate_acceleration(1)#-76.78
        result = self.transformation.calculate_GForce(1)
        self.assertEqual(round(abs(result)), -8)
        #Frame 4
        self.transformation.calculate_speed(1,(1.08,13.12), 0.05)#2.88
        self.transformation.calculate_acceleration(1)#-4.8
        result = self.transformation.calculate_GForce(1)
        self.assertEqual(round(abs(result)), 0)
        #Frame 5
        self.transformation.calculate_speed(1,(1.12,13.2), 0.06666666666666667)#4.53
        self.transformation.calculate_acceleration(1)#98.98
        result = self.transformation.calculate_GForce(1)
        self.assertEqual(round(abs(result)), 10)
        #Frame 6
        self.transformation.calculate_speed(1,(1.23, 13.7), 0.08333333333333333)#13.34
        self.transformation.calculate_acceleration(1)#528.49 
        result = self.transformation.calculate_GForce(1)#Yellow
        #colour = self.transformation.calculate_risk(result, 0)
        self.assertEqual(round(abs(result)), 54)
        #Frame 7
        self.transformation.calculate_speed(1,(1.3, 13.92), 0.1)#9.49
        self.transformation.calculate_acceleration(1)#-97.18
        result = self.transformation.calculate_GForce(1)
        self.assertEqual(round(abs(result)), 10)
        #Frame 8
        self.transformation.calculate_speed(1,(1.35, 14), 0.11666666666666667)#10.31
        self.transformation.calculate_acceleration(1)#49.19
        result = self.transformation.calculate_GForce(1)
        self.assertEqual(round(abs(result)), 5)
        #Frame 9
        self.transformation.calculate_speed(1,(1.35, 14), 0.13333333333333333)#9.90
        self.transformation.calculate_acceleration(1)#-24.6
        result = self.transformation.calculate_GForce(1)
        self.assertEqual(round(abs(result)), -3)
        #Frame 10
        self.transformation.calculate_speed(1,(1.35, 14), 0.15)#9.40
        self.transformation.calculate_acceleration(1)#29.99
        result = self.transformation.calculate_GForce(1)
        self.assertEqual(round(abs(result)), 3)


    def test_calculating_red_gForce(self):
        #Frame 1
        self.transformation.calculate_speed(1,(1.0,13.0), 0.0)
        self.transformation.calculate_acceleration(1)
        result = self.transformation.calculate_GForce(1)
        self.assertEqual(result, 0)
        #Frame 2
        self.transformation.calculate_speed(1,(0.95,13.05), 0.016666666666666666)#4.24
        self.transformation.calculate_acceleration(1)#254.50
        result = self.transformation.calculate_GForce(1)
        self.assertEqual(round(abs(result)), 26)
        #Frame 3
        self.transformation.calculate_speed(1,(0.96, 13.09), 0.03333333333333333)#2.96
        self.transformation.calculate_acceleration(1)#-76.78
        result = self.transformation.calculate_GForce(1)
        self.assertEqual(round(abs(result)), -8)
        #Frame 4
        self.transformation.calculate_speed(1,(1.08,13.12), 0.05)#2.88
        self.transformation.calculate_acceleration(1)#-4.8
        result = self.transformation.calculate_GForce(1)
        self.assertEqual(round(abs(result)), 0)
        #Frame 5
        self.transformation.calculate_speed(1,(1.12,13.2), 0.06666666666666667)#5.25
        self.transformation.calculate_acceleration(1)#98.98
        result = self.transformation.calculate_GForce(1)
        self.assertEqual(round(abs(result)), 10)
        #Frame 6
        self.transformation.calculate_speed(1,(1.23, 14), 0.08333333333333333)#19.8
        self.transformation.calculate_acceleration(1)#916.18
        result = self.transformation.calculate_GForce(1)#Red
        #colour = self.transformation.calculate_risk(result, 0)
        self.assertEqual(round(abs(result)), 93)
        #Frame 7
        self.transformation.calculate_speed(1,(1.23, 14), 0.1)#10.26
        self.transformation.calculate_acceleration(1)#-97.18
        result = self.transformation.calculate_GForce(1)
        self.assertEqual(round(abs(result)), 10)
        #Frame 8
        self.transformation.calculate_speed(1,(1.23, 14), 0.11666666666666667)#19.8
        self.transformation.calculate_acceleration(1)#49.19
        result = self.transformation.calculate_GForce(1)
        self.assertEqual(round(abs(result)), 5)
        #Frame 9
        self.transformation.calculate_speed(1,(1.23, 14), 0.13333333333333333)#9.49
        self.transformation.calculate_acceleration(1)#
        result = self.transformation.calculate_GForce(1)
        self.assertEqual(round(abs(result)), -3)
        #Frame 10
        self.transformation.calculate_speed(1,(1.23, 14), 0.15)#8.93
        self.transformation.calculate_acceleration(1)#
        result = self.transformation.calculate_GForce(1)
        self.assertEqual(round(abs(result)), 3)
        #Frame 11
        self.transformation.calculate_speed(1,(1.23, 14), 0.16666666666666666)#8.08
        self.transformation.calculate_acceleration(1)#
        result = self.transformation.calculate_GForce(1)
        self.assertEqual(round(abs(result)), 3)
        #Frame 12
        self.transformation.calculate_speed(1,(1.23, 14), 0.18333333333333332)#0
        self.transformation.calculate_acceleration(1)#
        result = self.transformation.calculate_GForce(1)
        self.assertEqual(round(abs(result)), 3)


    # def test_calculating_angular_orientation(self):
    #     angle = self.transformation.calculate_angle(1, [0,0], [100,100], [110,100], 0.0)
    #     # angle = self.transformation.calculate_angle(1, [0,0], [90,100], [100,100], 0.016666666666666666)
    #     # angle = self.transformation.calculate_angle(1, [0,0], [88,96], [98,100], 0.03333333333333333)

    #     # angle = self.transformation.calculate_angle(1, [0,0], [100,100], [110,100], 0.05)
    #     # angle = self.transformation.calculate_angle(1, [0,0], [100,100], [110,100], 0.06666666666666667)
    #     # angle = self.transformation.calculate_angle(1, [0,0], [100,100], [110,100], 0.08333333333333333)

    #     # angle = self.transformation.calculate_angle(1, [0,0], [100,100], [110,100], 0.1)
    #     # angle = self.transformation.calculate_angle(1, [0,0], [100,100], [110,100], 0.11666666666666667)
    #     # angle = self.transformation.calculate_angle(1, [0,0], [100,100], [110,100], 0.13333333333333333)

    #     # angle = self.transformation.calculate_angle(1, [0,0], [100,100], [110,100], 0.15)
    #     # angle = self.transformation.calculate_angle(1, [0,0], [100,100], [110,100], 0.16666666666666666)
    #     # angle = self.transformation.calculate_angle(1, [0,0], [100,100], [110,100], 0.18333333333333332)

    #     # angle = self.transformation.calculate_angle(1, [0,0], [100,100], [110,100], 0.2)
    #     # angle = self.transformation.calculate_angle(1, [0,0], [100,100], [110,100], 0.21666666666666667)
    #     # angle = self.transformation.calculate_angle(1, [0,0], [100,100], [110,100], 0.23333333333333334)


    # def test_calculating_angular_velocity():

    # def test_calculating_angular_acceleration():

    def test_thresholds_green_g_force(self):

        result = self.transformation.calculate_risk(0,0)
        self.assertEqual(result, ("GREEN", (0,255,0)))

        result = self.transformation.calculate_risk(20,0)
        self.assertEqual(result, ("GREEN", (0,255,0)))

        result = self.transformation.calculate_risk(48,0)
        self.assertEqual(result, ("GREEN", (0,255,0)))

        result = self.transformation.calculate_risk(48,3700)
        self.assertNotEqual(result, ("GREEN", (0,255,0)))

        result = self.transformation.calculate_risk(49,0)
        self.assertNotEqual(result, ("GREEN", (0,255,0)))

        result = self.transformation.calculate_risk(80,0)
        self.assertNotEqual(result, ("GREEN", (0,255,0)))

    def test_thresholds_yellow_g_force(self):

        result = self.transformation.calculate_risk(0,0)
        self.assertNotEqual(result, ("YELLOW", (0,255,255)))

        result = self.transformation.calculate_risk(48,0)
        self.assertNotEqual(result, ("YELLOW", (0,255,255)))

        result = self.transformation.calculate_risk(49,0)
        self.assertEqual(result, ("YELLOW", (0,255,255)))

        result = self.transformation.calculate_risk(68,0)
        self.assertEqual(result, ("YELLOW", (0,255,255)))

        result = self.transformation.calculate_risk(79,0)
        self.assertEqual(result, ("YELLOW", (0,255,255)))

        result = self.transformation.calculate_risk(80,0)
        self.assertNotEqual(result, ("YELLOW", (0,255,255)))

        result = self.transformation.calculate_risk(49, 6000)
        self.assertNotEqual(result, ("YELLOW", (0,255,255)))

    def test_thresholds_red_g_force(self):
        result = self.transformation.calculate_risk(0,0)
        self.assertNotEqual(result, ("RED", (0,0,255)))

        result = self.transformation.calculate_risk(48,0)
        self.assertNotEqual(result, ("RED", (0,0,255)))

        result = self.transformation.calculate_risk(49,0)
        self.assertNotEqual(result, ("RED", (0,0,255)))

        result = self.transformation.calculate_risk(79,0)
        self.assertNotEqual(result, ("RED", (0,0,255)))

        result = self.transformation.calculate_risk(80,0)
        self.assertEqual(result, ("RED", (0,0,255)))

        result = self.transformation.calculate_risk(100,0)
        self.assertEqual(result, ("RED", (0,0,255)))

        result = self.transformation.calculate_risk(80,4000)
        self.assertEqual(result, ("RED", (0,0,255)))

    def test_thresholds_green_angular_acceleration(self):
        result = self.transformation.calculate_risk(0,0)
        self.assertEqual(result, ("GREEN", (0,255,0)))

        result = self.transformation.calculate_risk(0,0)
        self.assertEqual(result, ("GREEN", (0,255,0)))

        result = self.transformation.calculate_risk(0,0)
        self.assertEqual(result, ("GREEN", (0,255,0)))

        result = self.transformation.calculate_risk(0, 3700)
        self.assertNotEqual(result, ("GREEN", (0,255,0)))

        result = self.transformation.calculate_risk(49,0)
        self.assertNotEqual(result, ("GREEN", (0,255,0)))

        result = self.transformation.calculate_risk(80,0)
        self.assertNotEqual(result, ("GREEN", (0,255,0)))

    def test_thresholds_yellow_angular_angular_acceleration(self):

        result = self.transformation.calculate_risk(0,3500)
        self.assertNotEqual(result, ("YELLOW", (0,255,255)))

        result = self.transformation.calculate_risk(0,3511)
        self.assertNotEqual(result, ("YELLOW", (0,255,255)))

        result = self.transformation.calculate_risk(0,3512)
        self.assertEqual(result, ("YELLOW", (0,255,255)))

        result = self.transformation.calculate_risk(48,3512)
        self.assertEqual(result, ("YELLOW", (0,255,255)))

        result = self.transformation.calculate_risk(68,3512)
        self.assertEqual(result, ("YELLOW", (0,255,255)))

        result = self.transformation.calculate_risk(79,5874)
        self.assertEqual(result, ("YELLOW", (0,255,255)))

        result = self.transformation.calculate_risk(80,5874)
        self.assertNotEqual(result, ("YELLOW", (0,255,255)))

        result = self.transformation.calculate_risk(70,5875)
        self.assertNotEqual(result, ("YELLOW", (0,255,255)))

        result = self.transformation.calculate_risk(50, 6000)
        self.assertNotEqual(result, ("YELLOW", (0,255,255)))

    def test_thresholds_red_angular_angular_acceleration(self):
        result = self.transformation.calculate_risk(0,0)
        self.assertNotEqual(result, ("RED", (0,0,255)))

        result = self.transformation.calculate_risk(48,0)
        self.assertNotEqual(result, ("RED", (0,0,255)))

        result = self.transformation.calculate_risk(49,2000)
        self.assertNotEqual(result, ("RED", (0,0,255)))

        result = self.transformation.calculate_risk(79,4800)
        self.assertNotEqual(result, ("RED", (0,0,255)))

        result = self.transformation.calculate_risk(60,5874)
        self.assertNotEqual(result, ("RED", (0,0,255)))

        result = self.transformation.calculate_risk(0,5875)
        self.assertEqual(result, ("RED", (0,0,255)))

        result = self.transformation.calculate_risk(50,6000)
        self.assertEqual(result, ("RED", (0,0,255)))

        result = self.transformation.calculate_risk(80,6000)
        self.assertEqual(result, ("RED", (0,0,255)))

    #Video Data Testing
    #Video Used
    def test_model_helmet_detection(self):
        capture = cv.VideoCapture("../dataset/videos/video12.mp4")
        helmet_count = 0
        isTrue, first_frame = capture.read()
        detections = self.model.track(source=first_frame, show=False, persist=True, verbose=False, conf=0.4, iou=0.4, tracker="../models/Trackers/bytetrack.yaml")  
        result = detections[0]
        boxes = result.boxes
        for box in boxes:
            cls = int(box.cls[0]) 
            if cls == 0:
                helmet_count += 1

    def test_model_jersey_detection(self):
        capture = cv.VideoCapture("../dataset/videos/video12.mp4")
        jersey_count = 0
        isTrue, first_frame = capture.read()
        detections = self.model.track(source=first_frame, show=False, persist=True, verbose=False, conf=0.4, iou=0.4, tracker="../models/Trackers/bytetrack.yaml")  
        result = detections[0]
        boxes = result.boxes
        for box in boxes:
            cls = int(box.cls[0]) 
            if cls == 1:
                jersey_count += 1

        self.assertEqual(jersey_count, 10)
       
    def test_model_player_detection(self):
        capture = cv.VideoCapture("../dataset/videos/video12.mp4")
        player_count = 0
        isTrue, first_frame = capture.read()
        detections = self.model.track(source=first_frame, show=False, persist=True, verbose=False, conf=0.4, iou=0.4, tracker="../models/Trackers/bytetrack.yaml")  
        result = detections[0]
        boxes = result.boxes
        for box in boxes:
            cls = int(box.cls[0]) 
            if cls == 2:
                player_count += 1
        self.assertEqual(player_count, 10)

    
    

if __name__ == '__main__':
    unittest.main()