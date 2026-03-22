import unittest
import numpy as np
import torch
import cv2 as cv
from ultralytics import YOLO
from perspective_transformation import BirdsEyeView 

class TestConcussionRisk(unittest.TestCase):
        

    def setUp(self):
        self.model = YOLO('../models/AMDGPUv2TrainYOLOS/weights/AMDGPUTrainYOLOs.pt')
        target_width = 9.144 #meters of 1 10 yard line to a yard line 10 yards apart
        target_height = 26.79192 #meters distance between sideline numbers
        source = np.array([[0, 0], [target_width, 0], [target_width, target_height], [0, target_height]], dtype=np.float32)
        target = np.array([[0, 0], [target_width, 0], [target_width, target_height], [0, target_height]], dtype=np.float32)
        self.transformation = BirdsEyeView(source, target)

    #Calculations Test
    def test_speed(self):
        """
        Test to ensure the correct calculations are conducted 
        for speed
        """
        #Frame 1
        result = self.transformation.calculate_speed(1,(1.0,13.0), 0.0)
        self.assertEqual(result, 0)
        #Frame 2
        result = self.transformation.calculate_speed(1,(0.95,13.05), 0.016666666666666666)
        self.assertEqual(round(result, 2), 4.24)
        #Frame 3
        result = self.transformation.calculate_speed(1,(0.92, 13.09), 0.03333333333333333)
        self.assertEqual(round(result, 2), 3.61)
        #Frame 4
        result = self.transformation.calculate_speed(1,(0.87, 13.13), 0.05)
        self.assertEqual(round(result, 2), 3.68)
        #Frame 5
        result = self.transformation.calculate_speed(1,(0.90, 13.19), 0.06666666666666667)
        self.assertEqual(round(result, 2), 2.97)
        #Frame 6
        result = self.transformation.calculate_speed(1,(0.95, 13.24), 0.08333333333333333)
        self.assertEqual(round(result, 2), 3.06)
        #Frame 7
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
        
    def test_acceleration(self):
        """
        Test to ensure the correct calculations are conducted 
        for acceleration
        """
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
        self.assertEqual(round(result), -38)
        #Frame 4
        self.transformation.calculate_speed(1,(0.87, 13.13), 0.05)
        result = self.transformation.calculate_acceleration(1)
        self.assertEqual(round(result), 4)
        #Frame 5
        self.transformation.calculate_speed(1,(0.90, 13.19), 0.06666666666666667)
        result = self.transformation.calculate_acceleration(1)
        self.assertEqual(round(result), -42)
        #Frame 6
        self.transformation.calculate_speed(1,(0.95, 13.24), 0.08333333333333333)
        result = self.transformation.calculate_acceleration(1)
        self.assertEqual(round(result), 5)
        #Frame 7
        self.transformation.calculate_speed(1,(1, 13.29), 0.1)
        result = self.transformation.calculate_acceleration(1)
        self.assertEqual(round(result), -10)
        #Frame 8
        self.transformation.calculate_speed(1,(1.06, 13.33), 0.11666666666666667)
        result = self.transformation.calculate_acceleration(1)
        self.assertEqual(round(result), 6)
        #Frame 9
        self.transformation.calculate_speed(1,(1.1, 13.39), 0.13333333333333333)
        result = self.transformation.calculate_acceleration(1)
        self.assertEqual(round(result), 29)
        #Frame 10
        self.transformation.calculate_speed(1,(1.13, 13.45), 0.15)
        result = self.transformation.calculate_acceleration(1)
        self.assertEqual(round(result), 37)
        #Frame 11
        self.transformation.calculate_speed(1,(1.2, 13.5), 0.16666666666666666)
        result = self.transformation.calculate_acceleration(1)
        self.assertEqual(round(result), 11)
        #Frame 12
        self.transformation.calculate_speed(1,(1.27, 13.56), 0.18333333333333332)
        result = self.transformation.calculate_acceleration(1)
        self.assertEqual(round(result), 13)
        #Frame 13
        self.transformation.calculate_speed(1,(1.34, 13.64), 0.2)
        result = self.transformation.calculate_acceleration(1)
        self.assertEqual(round(result), 21)
        #Frame 14
        self.transformation.calculate_speed(1,(1.42, 13.72), 0.21666666666666667)
        result = self.transformation.calculate_acceleration(1)
        self.assertEqual(round(result), 26)
        #Frame 15
        self.transformation.calculate_speed(1,(1.5, 13.78), 0.23333333333333333)
        result = self.transformation.calculate_acceleration(1)
        self.assertEqual(round(result), 17)

    def test_calculating_green_gForce(self):
        """
        Test to ensure the correct calculations are conducted that result 
        in a green G-Force risk
        """
        #Frame 1
        self.transformation.calculate_speed(1,(1.0,13.0), 0.0)#
        self.transformation.calculate_acceleration(1)#0
        result = self.transformation.calculate_GForce(1)
        self.assertEqual(result, 0)
        #Frame 2
        self.transformation.calculate_speed(1,(0.95,13.05), 0.016666666666666666)#
        self.transformation.calculate_acceleration(1)#0
        result = self.transformation.calculate_GForce(1)
        self.assertEqual(round(abs(result)), 0)
        #Frame 3
        self.transformation.calculate_speed(1,(0.92, 13.09), 0.03333333333333333)#
        self.transformation.calculate_acceleration(1)#-38
        result = self.transformation.calculate_GForce(1)
        self.assertEqual(round(abs(result)), 4)
        #Frame 4
        self.transformation.calculate_speed(1,(0.87, 13.13), 0.05)#-42
        self.transformation.calculate_acceleration(1)#4
        result = self.transformation.calculate_GForce(1)
        self.assertEqual(round(abs(result)), 0)
        #Frame 5
        self.transformation.calculate_speed(1,(0.90, 13.19), 0.06666666666666667)#
        self.transformation.calculate_acceleration(1)#-42
        result = self.transformation.calculate_GForce(1)
        self.assertEqual(round(abs(result)), 4)
        #Frame 6
        self.transformation.calculate_speed(1,(0.95, 13.24), 0.08333333333333333)#
        self.transformation.calculate_acceleration(1)#5
        result = self.transformation.calculate_GForce(1)
        self.assertEqual(round(abs(result)), 1)
        #Frame 7
        self.transformation.calculate_speed(1,(1, 13.29), 0.1)#
        self.transformation.calculate_acceleration(1)#-10
        result = self.transformation.calculate_GForce(1)
        self.assertEqual(round(abs(result)), 1)
        #Frame 8
        self.transformation.calculate_speed(1,(1.06, 13.33), 0.11666666666666667)#
        self.transformation.calculate_acceleration(1)#6
        result = self.transformation.calculate_GForce(1)
        self.assertEqual(round(abs(result)), 1)
        #Frame 9
        self.transformation.calculate_speed(1,(1.1, 13.39), 0.13333333333333333)#
        self.transformation.calculate_acceleration(1)#29
        result = self.transformation.calculate_GForce(1)
        self.assertEqual(round(abs(result)), 3)
        #Frame 10
        self.transformation.calculate_speed(1,(1.13, 13.45), 0.15)#
        self.transformation.calculate_acceleration(1)#37
        result = self.transformation.calculate_GForce(1)
        self.assertEqual(round(abs(result)), 4)
        colour = self.transformation.calculate_risk(result,0)
        self.assertEqual(colour, ("GREEN", (0,255,0)))
        #Frame 11
        self.transformation.calculate_speed(1,(1.2, 13.5), 0.16666666666666666)#
        self.transformation.calculate_acceleration(1)#11
        result = self.transformation.calculate_GForce(1)
        self.assertEqual(round(abs(result)), 1)
        #Frame 12
        self.transformation.calculate_speed(1,(1.27, 13.56), 0.18333333333333332)#
        self.transformation.calculate_acceleration(1)#13
        result = self.transformation.calculate_GForce(1)
        self.assertEqual(round(abs(result)), 1)
        #Frame 13
        self.transformation.calculate_speed(1,(1.34, 13.64), 0.2)#
        self.transformation.calculate_acceleration(1)#21
        result = self.transformation.calculate_GForce(1)
        self.assertEqual(round(abs(result)), 2)
        #Frame 14
        self.transformation.calculate_speed(1,(1.42, 13.72), 0.21666666666666667)#
        self.transformation.calculate_acceleration(1)#26
        result = self.transformation.calculate_GForce(1)
        self.assertEqual(round(abs(result)), 3)
        #Frame 15
        self.transformation.calculate_speed(1,(1.5, 13.78), 0.23333333333333333)#
        self.transformation.calculate_acceleration(1)#17
        result = self.transformation.calculate_GForce(1)
        self.assertEqual(round(abs(result)), 2)
    
    def test_calculating_yellow_gForce(self):
        """
        Test to ensure the correct calculations are conducted that result 
        in a yellow G-Force risk
        """
        #Frame 1
        self.transformation.calculate_speed(1,(1.0,13.0), 0.0)
        self.transformation.calculate_acceleration(1)
        result = self.transformation.calculate_GForce(1)
        self.assertEqual(result, 0)
        #Frame 2
        self.transformation.calculate_speed(1,(0.95,13.05), 0.016666666666666666)#4.24
        self.transformation.calculate_acceleration(1)#254.50
        result = self.transformation.calculate_GForce(1)
        self.assertEqual(round(abs(result)), 0)
        #Frame 3
        self.transformation.calculate_speed(1,(0.96, 13.09), 0.03333333333333333)#2.96
        self.transformation.calculate_acceleration(1)#-76.78
        result = self.transformation.calculate_GForce(1)
        self.assertEqual(round(abs(result)), 8)
        #Frame 4
        self.transformation.calculate_speed(1,(1.08,13.12), 0.05)#16.18
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
        colour = self.transformation.calculate_risk(result,0)
        self.assertEqual(colour, ("YELLOW", (0,255,255)))
        #Frame 7
        self.transformation.calculate_speed(1,(1.3, 13.92), 0.1)#9.68
        self.transformation.calculate_acceleration(1)#-219.60
        result = self.transformation.calculate_GForce(1)
        self.assertEqual(round(abs(result)), 22)
        #Frame 8
        self.transformation.calculate_speed(1,(1.35, 14), 0.11666666666666667)#10.31
        self.transformation.calculate_acceleration(1)#49.19
        result = self.transformation.calculate_GForce(1)
        self.assertEqual(round(abs(result)), 4)
        #Frame 9
        self.transformation.calculate_speed(1,(1.35, 14), 0.13333333333333333)#9.90
        self.transformation.calculate_acceleration(1)#-24.6
        result = self.transformation.calculate_GForce(1)
        self.assertEqual(round(abs(result)), 2)
        #Frame 10
        self.transformation.calculate_speed(1,(1.35, 14), 0.15)#9.20
        self.transformation.calculate_acceleration(1)#42
        result = self.transformation.calculate_GForce(1)
        self.assertEqual(round(abs(result)), 4)


    def test_calculating_red_gForce(self):
        """
        Test to ensure the correct calculations are conducted that result 
        in a red G-Force risk
        """
        #Frame 1
        self.transformation.calculate_speed(1,(1.0,13.0), 0.0)
        self.transformation.calculate_acceleration(1)
        result = self.transformation.calculate_GForce(1)
        self.assertEqual(result, 0)
        #Frame 2
        self.transformation.calculate_speed(1,(0.95,13.05), 0.016666666666666666)#4.24
        self.transformation.calculate_acceleration(1)#254.50
        result = self.transformation.calculate_GForce(1)
        self.assertEqual(round(abs(result)), 0)
        #Frame 3
        self.transformation.calculate_speed(1,(0.96, 13.09), 0.03333333333333333)#2.96
        self.transformation.calculate_acceleration(1)#-76.78
        result = self.transformation.calculate_GForce(1)
        self.assertEqual(round(abs(result)), 8)
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
        self.transformation.calculate_speed(1,(1.23, 14), 0.08333333333333333)#18.98
        self.transformation.calculate_acceleration(1)#872.98
        result = self.transformation.calculate_GForce(1)#Red
        self.assertEqual(round(abs(result)), 88)
        colour = self.transformation.calculate_risk(result,0)
        self.assertEqual(colour, ("RED", (0,0,255)))
        #Frame 7
        self.transformation.calculate_speed(1,(1.23, 14), 0.1)#10.26
        self.transformation.calculate_acceleration(1)#-97.18
        result = self.transformation.calculate_GForce(1)
        self.assertEqual(round(abs(result)), 53)#Yellow
        #Frame 8
        self.transformation.calculate_speed(1,(1.23, 14), 0.11666666666666667)#9.90
        self.transformation.calculate_acceleration(1)#58.347
        result = self.transformation.calculate_GForce(1)
        self.assertEqual(round(abs(result)), 2)
        #Frame 9
        self.transformation.calculate_speed(1,(1.23, 14), 0.13333333333333333)#9.49
        self.transformation.calculate_acceleration(1)#-24.6
        result = self.transformation.calculate_GForce(1)
        self.assertEqual(round(abs(result)), 3)
        #Frame 10
        self.transformation.calculate_speed(1,(1.23, 14), 0.15)#8.93
        self.transformation.calculate_acceleration(1)#-33.6
        result = self.transformation.calculate_GForce(1)
        self.assertEqual(round(abs(result)), 3)
        #Frame 11
        self.transformation.calculate_speed(1,(1.23, 14), 0.16666666666666666)#8.08
        self.transformation.calculate_acceleration(1)#-51
        result = self.transformation.calculate_GForce(1)
        self.assertEqual(round(abs(result)), 5)
        #Frame 12
        self.transformation.calculate_speed(1,(1.23, 14), 0.18333333333333332)#0
        self.transformation.calculate_acceleration(1)#-484.8
        result = self.transformation.calculate_GForce(1)
        self.assertEqual(round(abs(result)), 49)#Yellow


    def test_calculating_angular_orientation(self):
        """
        Test to ensure the correct calculations are conducted 
        for angular orientation
        """
        #Frame 1
        angle = self.transformation.calculate_angle(1, [7.00, 20.00], [7.10, 20.00], 0.0)
        self.assertEqual(angle, 0)
        #Frame 2
        angle = self.transformation.calculate_angle(1, [7.00, 20.02], [7.10, 20.04], 0.016666666666666666)
        self.assertEqual(round(angle, 5), 0.19740)
        #Frame 3
        angle = self.transformation.calculate_angle(1, [7.05, 20.05], [7.15, 20.08], 0.03333333333333333)
        self.assertEqual(round(angle, 5), 0.29146)
        #Frame 4
        angle = self.transformation.calculate_angle(1, [7.08, 20.08], [7.18, 20.10], 0.05)
        self.assertEqual(round(angle, 5), 0.19740)
        #Frame 5
        angle = self.transformation.calculate_angle(1, [7.11, 20.11], [7.21, 20.15], 0.06666666666666667)
        self.assertEqual(round(angle, 5), 0.38051)
        #Frame 6
        angle = self.transformation.calculate_angle(1, [7.14, 20.14], [7.22, 20.17], 0.08333333333333333)
        self.assertEqual(round(angle, 5), 0.35877)
        #Frame 7
        angle = self.transformation.calculate_angle(1, [7.15, 20.17], [7.27, 20.19], 0.1)
        self.assertEqual(round(angle, 5), 0.16515)
        #Frame 8
        angle = self.transformation.calculate_angle(1, [7.20, 20.20], [7.29, 20.20], 0.11666666666666667)
        self.assertEqual(round(angle, 5), 0)
        #Frame 9
        angle = self.transformation.calculate_angle(1, [7.23, 20.22], [7.33, 20.24], 0.13333333333333333)
        self.assertEqual(round(angle, 5), 0.19740)
        #Frame 10
        angle = self.transformation.calculate_angle(1, [7.24, 20.24], [7.36, 20.22], 0.15)
        self.assertEqual(round(angle, 5), -0.16515)
        #Frame 11
        angle = self.transformation.calculate_angle(1, [7.29, 20.26], [7.39, 20.24], 0.16666666666666666)
        self.assertEqual(round(angle, 5), -0.19740)
        #Frame 12
        angle = self.transformation.calculate_angle(1, [7.32, 20.28], [7.42, 20.30], 0.18333333333333332)#
        self.assertEqual(round(angle, 5), 0.19740)
        #Frame 13
        angle = self.transformation.calculate_angle(1, [7.34, 20.30], [7.45, 20.32], 0.2)
        self.assertEqual(round(angle, 5), 0.17985)
        #Frame 14
        angle = self.transformation.calculate_angle(1, [7.38, 20.33], [7.48, 20.30], 0.21666666666666667)
        self.assertEqual(round(angle, 5), -0.29146)
        #Frame 15
        angle = self.transformation.calculate_angle(1, [7.41, 20.34], [7.51, 20.33], 0.23333333333333334)
        self.assertEqual(round(angle, 5), -0.09967)

    def test_calculating_angular_velocity(self):
        """
        Test to ensure the correct calculations are conducted 
        for angular velocity
        """
        #Frame 1
        self.transformation.calculate_angle(1, [7.00, 20.00], [7.10, 20.00], 0.0)
        angular_velocity = self.transformation.calculate_angular_velocity(1, 0.0)
        self.assertEqual(angular_velocity, 0)
        #Frame 2
        self.transformation.calculate_angle(1, [7.00, 20.02], [7.10, 20.04], 0.016666666666666666)
        angular_velocity = self.transformation.calculate_angular_velocity(1, 0.016666666666666666)
        self.assertEqual(round(angular_velocity,2), 11.84)
        #Frame 3
        self.transformation.calculate_angle(1, [7.05, 20.05], [7.15, 20.08], 0.03333333333333333)
        angular_velocity = self.transformation.calculate_angular_velocity(1, 0.03333333333333333)
        self.assertEqual(round(angular_velocity,2), 8.74)
        #Frame 4
        self.transformation.calculate_angle(1, [7.08, 20.08], [7.18, 20.10], 0.05)
        angular_velocity = self.transformation.calculate_angular_velocity(1, 0.05)
        self.assertEqual(round(angular_velocity,2), 3.95)
        #Frame 5
        self.transformation.calculate_angle(1, [7.11, 20.11], [7.21, 20.15], 0.06666666666666667)
        angular_velocity = self.transformation.calculate_angular_velocity(1, 0.06666666666666667)
        self.assertEqual(round(angular_velocity,2), 3.66)
        #Frame 6
        self.transformation.calculate_angle(1, [7.14, 20.14], [7.22, 20.17], 0.08333333333333333)
        angular_velocity = self.transformation.calculate_angular_velocity(1, 0.08333333333333333)
        self.assertEqual(round(angular_velocity,2), 1.35)
        #Frame 7
        self.transformation.calculate_angle(1, [7.15, 20.17], [7.27, 20.19], 0.1)
        angular_velocity = self.transformation.calculate_angular_velocity(1, 0.1)
        self.assertEqual(round(angular_velocity,2), 1.65)
        #Frame 8
        self.transformation.calculate_angle(1, [7.20, 20.20], [7.29, 20.20], 0.11666666666666667)
        angular_velocity = self.transformation.calculate_angular_velocity(1, 0.11666666666666667)
        self.assertEqual(round(angular_velocity,2), -1.97)
        #Frame 9
        self.transformation.calculate_angle(1, [7.23, 20.22], [7.33, 20.24], 0.13333333333333333)
        angular_velocity = self.transformation.calculate_angular_velocity(1, 0.13333333333333333)
        self.assertEqual(round(angular_velocity,2), -0.94)
        #Frame 10
        self.transformation.calculate_angle(1, [7.24, 20.24], [7.36, 20.22], 0.15)
        angular_velocity = self.transformation.calculate_angular_velocity(1, 0.15)
        self.assertEqual(round(angular_velocity,2), -3.63)
        #Frame 11
        self.transformation.calculate_angle(1, [7.29, 20.26], [7.39, 20.24], 0.16666666666666666)
        angular_velocity = self.transformation.calculate_angular_velocity(1, 0.16666666666666666)
        self.assertEqual(round(angular_velocity,2), -5.78)
        #Frame 12
        self.transformation.calculate_angle(1, [7.32, 20.28], [7.42, 20.30], 0.18333333333333332)
        angular_velocity = self.transformation.calculate_angular_velocity(1, 0.18333333333333332)
        self.assertEqual(round(angular_velocity,2), -1.61)
        #Frame 13
        self.transformation.calculate_angle(1, [7.34, 20.30], [7.45, 20.32], 0.2)
        angular_velocity = self.transformation.calculate_angular_velocity(1, 0.2)
        self.assertEqual(round(angular_velocity,2), 0.15)
        #Frame 14
        self.transformation.calculate_angle(1, [7.38, 20.33], [7.48, 20.30], 0.21666666666666667)
        angular_velocity = self.transformation.calculate_angular_velocity(1, 0.21666666666666667)
        self.assertEqual(round(angular_velocity,2), -2.91)
        #Frame 15
        self.transformation.calculate_angle(1, [7.41, 20.34], [7.51, 20.33], 0.23333333333333334)
        angular_velocity = self.transformation.calculate_angular_velocity(1, 0.23333333333333334)
        self.assertEqual(round(angular_velocity,2), -2.97)

    def test_green_angular_acceleration(self):
        """
        Test to ensure the correct calculations are conducted that result 
        in a green angular acceleration risk
        """
        #Frame 1
        self.transformation.calculate_angle(1, [7.00, 20.00], [7.10, 20.00], 0.0)
        self.transformation.calculate_angular_velocity(1, 0.0)
        angular_acceleration = self.transformation.calculate_angular_acceleration(1)
        self.assertEqual(abs(round(angular_acceleration)), 0)
        #Frame 2
        self.transformation.calculate_angle(1, [7.00, 20.02], [7.10, 20.04], 0.016666666666666666)
        self.transformation.calculate_angular_velocity(1, 0.016666666666666666)
        angular_acceleration = self.transformation.calculate_angular_acceleration(1)
        self.assertEqual(abs(round(angular_acceleration)), 0)
        #Frame 3
        self.transformation.calculate_angle(1, [7.05, 20.05], [7.15, 20.08], 0.03333333333333333)
        self.transformation.calculate_angular_velocity(1, 0.03333333333333333)
        angular_acceleration = self.transformation.calculate_angular_acceleration(1)
        self.assertEqual(abs(round(angular_acceleration)), 262)
        result = self.transformation.calculate_risk(0,angular_acceleration)
        self.assertEqual(result, ("GREEN", (0,255,0)))
        #Frame 4
        self.transformation.calculate_angle(1, [7.08, 20.08], [7.18, 20.10], 0.05)
        self.transformation.calculate_angular_velocity(1, 0.05)
        angular_acceleration = self.transformation.calculate_angular_acceleration(1)
        self.assertEqual(abs(round(angular_acceleration)), 79)
        #Frame 5
        self.transformation.calculate_angle(1, [7.11, 20.11], [7.21, 20.15], 0.06666666666666667)
        self.transformation.calculate_angular_velocity(1, 0.06666666666666667)
        angular_acceleration = self.transformation.calculate_angular_acceleration(1)
        self.assertEqual(abs(round(angular_acceleration)), 164)
        #Frame 6
        self.transformation.calculate_angle(1, [7.14, 20.14], [7.22, 20.17], 0.08333333333333333)
        self.transformation.calculate_angular_velocity(1, 0.08333333333333333)
        angular_acceleration = self.transformation.calculate_angular_acceleration(1)
        self.assertEqual(abs(round(angular_acceleration)), 148)
        #Frame 7
        self.transformation.calculate_angle(1, [7.15, 20.17], [7.27, 20.19], 0.1)
        self.transformation.calculate_angular_velocity(1, 0.1)
        angular_acceleration = self.transformation.calculate_angular_acceleration(1)
        self.assertEqual(abs(round(angular_acceleration)), 46)
        #Frame 8
        self.transformation.calculate_angle(1, [7.20, 20.20], [7.29, 20.20], 0.11666666666666667)
        self.transformation.calculate_angular_velocity(1, 0.11666666666666667)
        angular_acceleration = self.transformation.calculate_angular_acceleration(1)
        self.assertEqual(abs(round(angular_acceleration)), 113)
        #Frame 9
        self.transformation.calculate_angle(1, [7.23, 20.22], [7.33, 20.24], 0.13333333333333333)
        self.transformation.calculate_angular_velocity(1, 0.13333333333333333)
        angular_acceleration = self.transformation.calculate_angular_acceleration(1)
        self.assertEqual(abs(round(angular_acceleration)), 46)
        #Frame 10
        self.transformation.calculate_angle(1, [7.24, 20.24], [7.36, 20.22], 0.15)
        self.transformation.calculate_angular_velocity(1, 0.15)
        angular_acceleration = self.transformation.calculate_angular_acceleration(1)
        self.assertEqual(abs(round(angular_acceleration)), 106)
        #Frame 11
        self.transformation.calculate_angle(1, [7.29, 20.26], [7.39, 20.24], 0.16666666666666666)
        self.transformation.calculate_angular_velocity(1, 0.16666666666666666)
        angular_acceleration = self.transformation.calculate_angular_acceleration(1)
        self.assertEqual(abs(round(angular_acceleration)), 76)
        #Frame 12
        self.transformation.calculate_angle(1, [7.32, 20.28], [7.42, 20.30], 0.18333333333333332)
        self.transformation.calculate_angular_velocity(1, 0.18333333333333332)
        angular_acceleration = self.transformation.calculate_angular_acceleration(1)
        self.assertEqual(abs(round(angular_acceleration)), 13)
        #Frame 13
        self.transformation.calculate_angle(1, [7.34, 20.30], [7.45, 20.32], 0.2)
        self.transformation.calculate_angular_velocity(1, 0.2)
        angular_acceleration = self.transformation.calculate_angular_acceleration(1)
        self.assertEqual(abs(round(angular_acceleration)), 75)
        #Frame 14
        self.transformation.calculate_angle(1, [7.38, 20.33], [7.48, 20.30], 0.21666666666666667)
        self.transformation.calculate_angular_velocity(1, 0.21666666666666667)
        angular_acceleration = self.transformation.calculate_angular_acceleration(1)
        self.assertEqual(abs(round(angular_acceleration)), 57)
        #Frame 15
        self.transformation.calculate_angle(1, [7.41, 20.34], [7.51, 20.33], 0.23333333333333334)
        self.transformation.calculate_angular_velocity(1, 0.23333333333333334)
        angular_acceleration = self.transformation.calculate_angular_acceleration(1)
        self.assertEqual(abs(round(angular_acceleration)), 27)

    def test_yellow_angular_acceleration(self):
        """
        Test to ensure the correct calculations are conducted that result 
        in a yellow angular acceleration risk
        """
        #Frame 1
        self.transformation.calculate_angle(1, [7.00, 20.00], [6.70, 20.00], 0.0)#3.14159
        self.transformation.calculate_angular_velocity(1, 0.0)#0
        angular_acceleration = self.transformation.calculate_angular_acceleration(1)
        self.assertEqual(abs(round(angular_acceleration)), 0)
        #Frame 2
        self.transformation.calculate_angle(1, [7.02, 20.02], [6.77, 20.02], 0.016666666666666666)#3.14159
        self.transformation.calculate_angular_velocity(1, 0.016666666666666666)#0
        angular_acceleration = self.transformation.calculate_angular_acceleration(1)
        self.assertEqual(abs(round(angular_acceleration)), 0)
        #Frame 3
        self.transformation.calculate_angle(1, [7.07, 20.04], [6.80, 19.86], 0.033333333333333333)#-2.55359
        self.transformation.calculate_angular_velocity(1, 0.033333333333333333)#-170.87
        angular_acceleration = self.transformation.calculate_angular_acceleration(1)
        self.assertEqual(abs(round(angular_acceleration)), 5126)#Yellow
        result = self.transformation.calculate_risk(0,angular_acceleration)
        self.assertEqual(result, ("YELLOW", (0,255,255)))
        #Frame 4
        self.transformation.calculate_angle(1, [7.06, 20.07], [6.84, 19.89], 0.05)#-2.45586
        self.transformation.calculate_angular_velocity(1, 0.05)#-111.95
        angular_acceleration = self.transformation.calculate_angular_acceleration(1)
        self.assertEqual(abs(round(angular_acceleration)), 2239)
        #Frame 5
        self.transformation.calculate_angle(1, [7.08, 20.10], [6.86, 19.91], 0.066666666666666667)#-2.42924
        self.transformation.calculate_angular_velocity(1, 0.066666666666666667)#-111.42
        angular_acceleration = self.transformation.calculate_angular_acceleration(1)
        self.assertEqual(abs(round(angular_acceleration)), 2228)
        #Frame 6
        self.transformation.calculate_angle(1, [7.11, 20.13], [6.89, 19.95], 0.083333333333333333)#-2.45586
        self.transformation.calculate_angular_velocity(1, 0.083333333333333333)#1.95
        angular_acceleration = self.transformation.calculate_angular_acceleration(1)
        self.assertEqual(abs(round(angular_acceleration)), 3456)
        #Frame 7
        self.transformation.calculate_angle(1, [7.12, 20.12], [6.90, 19.99], 0.1)#-2.60788
        self.transformation.calculate_angular_velocity(1, 0.1)#-57.49
        angular_acceleration = self.transformation.calculate_angular_acceleration(1)
        self.assertEqual(abs(round(angular_acceleration)), 1089)
        #Frame 8
        self.transformation.calculate_angle(1, [7.17, 20.14], [6.86, 20.02], 0.116666666666666667)#-2.77230
        self.transformation.calculate_angular_velocity(1, 0.116666666666666667)#-59.14
        angular_acceleration = self.transformation.calculate_angular_acceleration(1)
        self.assertEqual(abs(round(angular_acceleration)), 1046)
        #Frame 9
        self.transformation.calculate_angle(1, [7.16, 20.18], [6.87, 20.06], 0.133333333333333333)#-2.74925
        self.transformation.calculate_angular_velocity(1, 0.133333333333333333)#-1.96
        angular_acceleration = self.transformation.calculate_angular_acceleration(1)
        self.assertEqual(abs(round(angular_acceleration)), 78)
        #Frame 10
        self.transformation.calculate_angle(1, [7.18, 20.21], [6.89, 20.11], 0.15)#-2.8095
        self.transformation.calculate_angular_velocity(1, 0.15)#-3.55
        angular_acceleration = self.transformation.calculate_angular_acceleration(1)
        self.assertEqual(abs(round(angular_acceleration)), 1079)
        #Frame 11
        self.transformation.calculate_angle(1, [7.20, 20.22], [6.92, 20.14], 0.166666666666666667)#-2.86329
        self.transformation.calculate_angular_velocity(1, 0.166666666666666667)#-4.34
        angular_acceleration = self.transformation.calculate_angular_acceleration(1)
        self.assertEqual(abs(round(angular_acceleration)), 1096)
        #Frame 12
        self.transformation.calculate_angle(1, [7.22, 20.25], [6.94, 20.16], 0.183333333333333333)#-2.83059
        self.transformation.calculate_angular_velocity(1, 0.183333333333333333)#-3.75
        angular_acceleration = self.transformation.calculate_angular_acceleration(1)
        self.assertEqual(abs(round(angular_acceleration)), 36)

    def test_red_angular_acceleration(self):
        """
        Test to ensure the correct calculations are conducted that result 
        in a red angular acceleration risk
        """
        #Frame 1
        self.transformation.calculate_angle(1, [7.00, 20.00], [6.30, 20.00], 0.0)#3.14159
        self.transformation.calculate_angular_velocity(1, 0.0)#0
        angular_acceleration = self.transformation.calculate_angular_acceleration(1)
        self.assertEqual(abs(round(angular_acceleration)), 0)
        #Frame 2
        self.transformation.calculate_angle(1, [7.02, 20.02], [6.77, 19.85], 0.016666666666666666)#-2.54442
        self.transformation.calculate_angular_velocity(1, 0.016666666666666666)#-341.15
        angular_acceleration = self.transformation.calculate_angular_acceleration(1)
        self.assertEqual(abs(round(angular_acceleration)), 0)
        #Frame 3
        self.transformation.calculate_angle(1, [7.04, 20.04], [6.79, 19.87], 0.03333333333333333)#-2.54442
        self.transformation.calculate_angular_velocity(1, 0.03333333333333333)#-170.58
        angular_acceleration = self.transformation.calculate_angular_acceleration(1)
        self.assertEqual(abs(round(angular_acceleration)), 5117)
        #Frame 4
        self.transformation.calculate_angle(1, [7.06, 20.06], [6.81, 19.88], 0.05)#-2.51757
        self.transformation.calculate_angular_velocity(1, 0.05)#-113.18
        angular_acceleration = self.transformation.calculate_angular_acceleration(1)
        self.assertEqual(abs(round(angular_acceleration)), 2264)
        #Frame 5
        self.transformation.calculate_angle(1, [7.05, 20.07], [6.80, 19.89], 0.066666666666666667)#-2.51757
        self.transformation.calculate_angular_velocity(1, 0.066666666666666667)#0.537
        angular_acceleration = self.transformation.calculate_angular_acceleration(1)
        self.assertEqual(abs(round(angular_acceleration)), 6834)
        result = self.transformation.calculate_risk(0,angular_acceleration)
        self.assertEqual(result, ("RED", (0,0,255)))
        #Frame 6
        self.transformation.calculate_angle(1, [7.11, 20.11], [6.84, 19.92], 0.083333333333333333)#-2.52839
        self.transformation.calculate_angular_velocity(1, 0.083333333333333333)#0.3206
        angular_acceleration = self.transformation.calculate_angular_acceleration(1)
        self.assertEqual(abs(round(angular_acceleration)), 3418)
        #Frame 7
        self.transformation.calculate_angle(1, [7.14, 20.13], [6.88, 19.94], 0.1)#-2.51051
        self.transformation.calculate_angular_velocity(1, 0.1)#-56.521
        angular_acceleration = self.transformation.calculate_angular_acceleration(1)
        self.assertEqual(abs(round(angular_acceleration)), 1133)
        #Frame 8
        self.transformation.calculate_angle(1, [7.15, 20.16], [6.95, 19.96], 0.116666666666666667)#-2.35619
        self.transformation.calculate_angular_velocity(1, 0.116666666666666667)#1.88
        angular_acceleration = self.transformation.calculate_angular_acceleration(1)
        self.assertEqual(abs(round(angular_acceleration)), 27)
        #Frame 9
        self.transformation.calculate_angle(1, [7.12, 20.17], [6.94, 19.99], 0.133333333333333333)#-2.35619
        self.transformation.calculate_angular_velocity(1, 0.133333333333333333)#1.88
        angular_acceleration = self.transformation.calculate_angular_acceleration(1)
        self.assertEqual(abs(round(angular_acceleration)), 31)
        #Frame 10
        self.transformation.calculate_angle(1, [7.15, 20.18], [6.96, 20.01], 0.15)#-2.41169
        self.transformation.calculate_angular_velocity(1, 0.15)#1.0588
        angular_acceleration = self.transformation.calculate_angular_acceleration(1)
        self.assertEqual(abs(round(angular_acceleration)), 1152)
        #Frame 11
        self.transformation.calculate_angle(1, [7.17, 20.21], [6.99, 20.02], 0.166666666666666667)#-2.32917
        self.transformation.calculate_angular_velocity(1, 0.166666666666666667)#1.88
        angular_acceleration = self.transformation.calculate_angular_acceleration(1)
        self.assertEqual(abs(round(angular_acceleration)), 0)
        #Frame 12
        self.transformation.calculate_angle(1, [7.19, 20.22], [6.99, 20.04], 0.183333333333333333)#-2.40878
        self.transformation.calculate_angular_velocity(1, 0.183333333333333333)#1.1961
        angular_acceleration = self.transformation.calculate_angular_acceleration(1)
        self.assertEqual(abs(round(angular_acceleration)), 14)

    def test_thresholds_green_g_force(self):
        """
        Test to ensure the correct risk colour result is returned
        Tests Green G-Force against threshold boundaries
        """
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
        """
        Test to ensure the correct risk colour result is returned
        Tests Yellow G-Force against threshold boundaries
        """
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
        """
        Test to ensure the correct risk colour result is returned
        Tests Yellow G-Force against threshold boundaries
        """
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
        """
        Test to ensure the correct risk colour result is returned
        This tests for Green result using angular acceleration
        """
        result = self.transformation.calculate_risk(0,0)
        self.assertEqual(result, ("GREEN", (0,255,0)))

        result = self.transformation.calculate_risk(0,3511)
        self.assertEqual(result, ("GREEN", (0,255,0)))

        result = self.transformation.calculate_risk(0,3512)
        self.assertNotEqual(result, ("GREEN", (0,255,0)))

        result = self.transformation.calculate_risk(0, 3700)
        self.assertNotEqual(result, ("GREEN", (0,255,0)))

        result = self.transformation.calculate_risk(49,0)
        self.assertNotEqual(result, ("GREEN", (0,255,0)))

        result = self.transformation.calculate_risk(80,0)
        self.assertNotEqual(result, ("GREEN", (0,255,0)))

    def test_thresholds_yellow_angular_angular_acceleration(self):
        """
        Test to ensure the correct risk colour result is returned
        This tests for Yellow result using angular acceleration
        """
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
        """
        Test to ensure the correct risk colour result is returned
        This tests for Red result using angular acceleration
        """
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

    def test_model_helmet_detection(self):
        """
        Test to check if model detects helemts
        """
        if torch.cuda.is_available():
            self.model.to('cuda:0')
            print(f"Model device: {self.model.device}")
        capture = cv.VideoCapture("../dataset/videos/video1.mp4")
        helmet_count = 0
        isTrue, first_frame = capture.read()
        detections = self.model.track(source=first_frame, show=False, persist=True, verbose=False, conf=0.4, iou=0.4, tracker="../models/Trackers/bytetrack.yaml")  
        result = detections[0]
        boxes = result.boxes
        for box in boxes:
            cls = int(box.cls[0]) 
            if cls == 0:
                helmet_count += 1
        self.assertGreater(helmet_count, 0)

    def test_model_jersey_detection(self):
        """
        Test to check if model detects jerseys in rain environment
        """
        if torch.cuda.is_available():
            self.model.to('cuda:0')
            print(f"Model device: {self.model.device}")
        capture = cv.VideoCapture("../dataset/videos/rainvideo1.mp4")
        jersey_count = 0
        isTrue, first_frame = capture.read()
        detections = self.model.track(source=first_frame, show=False, persist=True, verbose=False, conf=0.4, iou=0.4, tracker="../models/Trackers/bytetrack.yaml")  
        result = detections[0]
        boxes = result.boxes
        for box in boxes:
            cls = int(box.cls[0]) 
            if cls == 1:
                jersey_count += 1

        self.assertGreater(jersey_count, 0)
       
    def test_model_player_detection(self):
        """
        Test to check if model detects players in a snow environment
        """
        if torch.cuda.is_available():
            self.model.to('cuda:0')
            print(f"Model device: {self.model.device}")
        capture = cv.VideoCapture("../dataset/videos/snowvideo.mp4")
        player_count = 0
        isTrue, first_frame = capture.read()
        detections = self.model.track(source=first_frame, show=False, persist=True, verbose=False, conf=0.4, iou=0.4, tracker="../models/Trackers/bytetrack.yaml")  
        result = detections[0]
        boxes = result.boxes
        for box in boxes:
            cls = int(box.cls[0]) 
            if cls == 2:
                player_count += 1
        self.assertGreater(player_count, 0)
    
if __name__ == '__main__':
    unittest.main()