import unittest
import numpy as np
from perspective_transformation import BirdsEyeView 

class TestConcussionRisk(unittest.TestCase):
        
    target_width = 9.144 #meters of 1 10 yard line to a yard line 10 yards apart
    target_height = 26.79192 #meters distance between sideline numbers
    source = np.array([[0, 0], [target_width, 0], [target_width, target_height], [0, target_height]], dtype=np.float32)
    target = np.array([[0, 0], [target_width, 0], [target_width, target_height], [0, target_height]], dtype=np.float32)
    transformation = BirdsEyeView(source, target)

    #Calculations Test
    def test_speed(self):

        #As only co-ordinate is stored. Player has no speed
        result = self.transformation.calculate_speed(1,(1.0,13.0), 0.0)
        self.assertEqual(result, 0)
        #First speed 
        result = self.transformation.calculate_speed(1,(0.95,13.05), 0.016666666666666666)
        self.assertEqual(round(result, 2), 4.24)
        #Between most recent and 4 frames prior
        result = self.transformation.calculate_speed(1,(0.92, 13.09), 0.03333333333333333)
        self.assertEqual(round(result, 2), 3.61)

        result = self.transformation.calculate_speed(1,(0.87, 13.13), 0.05)
        self.assertEqual(round(result, 2), 3.68)
        #Between most recent and 7 frames prior
        result = self.transformation.calculate_speed(1,(0.90, 13.19), 0.06666666666666667)
        self.assertEqual(round(result, 2), 2.97)
        result = self.transformation.calculate_speed(1,(0.95, 13.24), 0.08333333333333333)
        self.assertEqual(round(result, 2), 3.06)

        result = self.transformation.calculate_speed(1,(1, 13.29), 0.1)
        self.assertEqual(round(result, 2), 2.90)
        #Post 7 frames
        result = self.transformation.calculate_speed(1,(1.06, 13.33), 0.11666666666666667)
        self.assertEqual(round(result, 2), 3.00)
        result = self.transformation.calculate_speed(1,(1.1, 13.39), 0.13333333333333333)
        self.assertEqual(round(result, 2), 3.50)

        result = self.transformation.calculate_speed(1,(1.13, 13.45), 0.15)
        self.assertEqual(round(result, 2), 4.12)
        result = self.transformation.calculate_speed(1,(1.2, 13.5), 0.16666666666666666)
        self.assertEqual(round(result, 2), 4.31)
        result = self.transformation.calculate_speed(1,(1.27, 13.56), 0.18333333333333332)
        self.assertEqual(round(result, 2), 4.53)

        result = self.transformation.calculate_speed(1,(1.34, 13.64), 0.2)
        self.assertEqual(round(result, 2), 4.88)
        result = self.transformation.calculate_speed(1,(1.42, 13.72), 0.21666666666666667)
        self.assertEqual(round(result, 2), 5.31)
        result = self.transformation.calculate_speed(1,(1.5, 13.78), 0.23333333333333333)
        self.assertEqual(round(result, 2), 5.59)
        

    def test_acceleration(self):
        #Frame 1
        self.transformation.calculate_speed(1,(1.0,13.0), 0.0)
        result = self.transformation.calculate_acceleration(1)
        self.assertEqual(result, 0)

        #Frame 2
        self.transformation.calculate_speed(1,(0.95,13.05), 0.016666666666666666)
        result = self.transformation.calculate_acceleration(1)
        self.assertEqual(round(result), 254)


        self.transformation.calculate_speed(1,(0.92, 13.09), 0.03333333333333333)
        result = self.transformation.calculate_acceleration(1)
        self.assertEqual(round(result), 11)

        #Frame 4
        self.transformation.calculate_speed(1,(0.87, 13.13), 0.05)
        result = self.transformation.calculate_acceleration(1)
        self.assertEqual(round(result), 7)

        self.transformation.calculate_speed(1,(0.90, 13.19), 0.06666666666666667)
        result = self.transformation.calculate_acceleration(1)
        self.assertEqual(round(result), -25)

        self.transformation.calculate_speed(1,(0.95, 13.24), 0.08333333333333333)
        result = self.transformation.calculate_acceleration(1)
        self.assertEqual(round(result), -11)

        #Frame 7
        self.transformation.calculate_speed(1,(1, 13.29), 0.1)
        result = self.transformation.calculate_acceleration(1)
        self.assertEqual(round(result), 30)
        #Frame 8
        self.transformation.calculate_speed(1,(1.06, 13.33), 0.11666666666666667)
        result = self.transformation.calculate_acceleration(1)
        self.assertEqual(round(result), -12)
        #Frame 9
        self.transformation.calculate_speed(1,(1.1, 13.39), 0.13333333333333333)
        result = self.transformation.calculate_acceleration(1)
        self.assertEqual(round(result), -1)
        #Frame 10
        self.transformation.calculate_speed(1,(1.13, 13.45), 0.15)
        result = self.transformation.calculate_acceleration(1)
        self.assertEqual(round(result), 4)
        #Frame 11
        self.transformation.calculate_speed(1,(1.2, 13.5), 0.16666666666666666)
        result = self.transformation.calculate_acceleration(1)
        self.assertEqual(round(result), 13)
        #Frame 12
        self.transformation.calculate_speed(1,(1.27, 13.56), 0.18333333333333332)
        result = self.transformation.calculate_acceleration(1)
        self.assertEqual(round(result), 15)
        #Frame 13
        self.transformation.calculate_speed(1,(1.34, 13.64), 0.2)
        result = self.transformation.calculate_acceleration(1)
        self.assertEqual(round(result), 20)
        #Frame 14
        self.transformation.calculate_speed(1,(1.42, 13.72), 0.21666666666666667)
        result = self.transformation.calculate_acceleration(1)
        self.assertEqual(round(result), 23)
        #Frame 15
        self.transformation.calculate_speed(1,(1.5, 13.78), 0.23333333333333333)
        result = self.transformation.calculate_acceleration(1)
        self.assertEqual(round(result), 21)

    def test_green_gForce(self):
        #Frame 1
        self.transformation.calculate_speed(1,(1.0,13.0), 0.0)
        self.transformation.calculate_acceleration(1)
        result = self.transformation.calculate_GForce(1)
        self.assertEqual(result, 0)

        #Frame 2
        self.transformation.calculate_speed(1,(0.95,13.05), 0.016666666666666666)
        self.transformation.calculate_acceleration(1)
        result = self.transformation.calculate_GForce(1)
        self.assertEqual(round(abs(result)), 26)


        self.transformation.calculate_speed(1,(0.92, 13.09), 0.03333333333333333)
        self.transformation.calculate_acceleration(1)
        result = self.transformation.calculate_GForce(1)
        self.assertEqual(round(abs(result)), 1)

        #Frame 4
        self.transformation.calculate_speed(1,(0.87, 13.13), 0.05)
        self.transformation.calculate_acceleration(1)
        result = self.transformation.calculate_GForce(1)
        self.assertEqual(round(abs(result)), 1)

        self.transformation.calculate_speed(1,(0.90, 13.19), 0.06666666666666667)
        self.transformation.calculate_acceleration(1)
        result = self.transformation.calculate_GForce(1)
        self.assertEqual(round(abs(result)), 3)

        self.transformation.calculate_speed(1,(0.95, 13.24), 0.08333333333333333)
        self.transformation.calculate_acceleration(1)
        result = self.transformation.calculate_GForce(1)
        self.assertEqual(round(abs(result)), 1)

        #Frame 7
        self.transformation.calculate_speed(1,(1, 13.29), 0.1)
        self.transformation.calculate_acceleration(1)
        result = self.transformation.calculate_GForce(1)
        self.assertEqual(round(abs(result)), 3)
        #Frame 8
        self.transformation.calculate_speed(1,(1.06, 13.33), 0.11666666666666667)
        self.transformation.calculate_acceleration(1)
        result = self.transformation.calculate_GForce(1)
        self.assertEqual(round(abs(result)), 1)
        #Frame 9
        self.transformation.calculate_speed(1,(1.1, 13.39), 0.13333333333333333)
        self.transformation.calculate_acceleration(1)
        result = self.transformation.calculate_GForce(1)
        self.assertEqual(round(abs(result)), 1)
        #Frame 10
        self.transformation.calculate_speed(1,(1.13, 13.45), 0.15)
        self.transformation.calculate_acceleration(1)
        result = self.transformation.calculate_GForce(1)
        self.assertEqual(round(abs(result)), 0)
        #Frame 11
        self.transformation.calculate_speed(1,(1.2, 13.5), 0.16666666666666666)
        self.transformation.calculate_acceleration(1)
        result = self.transformation.calculate_GForce(1)
        self.assertEqual(round(abs(result)), 1)
        #Frame 12
        self.transformation.calculate_speed(1,(1.27, 13.56), 0.18333333333333332)
        self.transformation.calculate_acceleration(1)
        result = self.transformation.calculate_GForce(1)
        self.assertEqual(round(abs(result)), 1)
        #Frame 13
        self.transformation.calculate_speed(1,(1.34, 13.64), 0.2)
        self.transformation.calculate_acceleration(1)
        result = self.transformation.calculate_GForce(1)
        self.assertEqual(round(abs(result)), 2)
        #Frame 14
        self.transformation.calculate_speed(1,(1.42, 13.72), 0.21666666666666667)
        self.transformation.calculate_acceleration(1)
        result = self.transformation.calculate_GForce(1)
        self.assertEqual(round(abs(result)), 2)
        #Frame 15
        self.transformation.calculate_speed(1,(1.5, 13.78), 0.23333333333333333)
        self.transformation.calculate_acceleration(1)
        result = self.transformation.calculate_GForce(1)
        self.assertEqual(round(abs(result)), 2)
    
    def test_yellow_gForce(self):
        #Frame 1
        self.transformation.calculate_speed(1,(1.0,13.0), 0.0)
        self.transformation.calculate_acceleration(1)
        result = self.transformation.calculate_GForce(1)
        self.assertEqual(result, 0)

        #Frame 2
        self.transformation.calculate_speed(1,(0.95,13.05), 0.016666666666666666)
        self.transformation.calculate_acceleration(1)
        result = self.transformation.calculate_GForce(1)
        self.assertEqual(round(abs(result)), 26)


        self.transformation.calculate_speed(1,(0.92, 13.09), 0.03333333333333333)
        self.transformation.calculate_acceleration(1)
        result = self.transformation.calculate_GForce(1)
        self.assertEqual(round(abs(result)), 1)

        #Frame 4
        self.transformation.calculate_speed(1,(0.87, 13.13), 0.05)
        self.transformation.calculate_acceleration(1)
        result = self.transformation.calculate_GForce(1)
        self.assertEqual(round(abs(result)), 1)

        self.transformation.calculate_speed(1,(0.90, 13.19), 0.06666666666666667)
        self.transformation.calculate_acceleration(1)
        result = self.transformation.calculate_GForce(1)
        self.assertEqual(round(abs(result)), 3)

        self.transformation.calculate_speed(1,(0.95, 13.24), 0.08333333333333333)
        self.transformation.calculate_acceleration(1)
        result = self.transformation.calculate_GForce(1)
        self.assertEqual(round(abs(result)), 1)

        #Frame 7
        self.transformation.calculate_speed(1,(1, 13.29), 0.1)
        self.transformation.calculate_acceleration(1)
        result = self.transformation.calculate_GForce(1)
        self.assertEqual(round(abs(result)), 3)
        #Frame 8
        self.transformation.calculate_speed(1,(1.06, 13.33), 0.11666666666666667)
        self.transformation.calculate_acceleration(1)
        result = self.transformation.calculate_GForce(1)
        self.assertEqual(round(abs(result)), 1)
        #Frame 9
        self.transformation.calculate_speed(1,(1.1, 13.39), 0.13333333333333333)
        self.transformation.calculate_acceleration(1)
        result = self.transformation.calculate_GForce(1)
        self.assertEqual(round(abs(result)), 1)
        #Frame 10
        self.transformation.calculate_speed(1,(1.13, 13.45), 0.15)
        self.transformation.calculate_acceleration(1)
        result = self.transformation.calculate_GForce(1)
        self.assertEqual(round(abs(result)), 0)
        #Frame 11
        self.transformation.calculate_speed(1,(1.2, 13.5), 0.16666666666666666)
        self.transformation.calculate_acceleration(1)
        result = self.transformation.calculate_GForce(1)
        self.assertEqual(round(abs(result)), 1)
        #Frame 12
        self.transformation.calculate_speed(1,(1.27, 13.56), 0.18333333333333332)
        self.transformation.calculate_acceleration(1)
        result = self.transformation.calculate_GForce(1)
        self.assertEqual(round(abs(result)), 1)
        #Frame 13
        self.transformation.calculate_speed(1,(1.34, 13.64), 0.2)
        self.transformation.calculate_acceleration(1)
        result = self.transformation.calculate_GForce(1)
        self.assertEqual(round(abs(result)), 2)
        #Frame 14
        self.transformation.calculate_speed(1,(1.42, 13.72), 0.21666666666666667)
        self.transformation.calculate_acceleration(1)
        result = self.transformation.calculate_GForce(1)
        self.assertEqual(round(abs(result)), 2)
        #Frame 15
        self.transformation.calculate_speed(1,(1.5, 13.78), 0.23333333333333333)
        self.transformation.calculate_acceleration(1)
        result = self.transformation.calculate_GForce(1)
        self.assertEqual(round(abs(result)), 2)


    # def test_red_gForce():


    def test_angular_orientation(self):
        angle = self.transformation.calculate_angle(1, [0,0], [100,100], [110,100], 0.0)
        angle = self.transformation.calculate_angle(1, [0,0], [90,100], [100,100], 0.016666666666666666)
        angle = self.transformation.calculate_angle(1, [0,0], [88,96], [98,100], 0.03333333333333333)

        angle = self.transformation.calculate_angle(1, [0,0], [100,100], [110,100], 0.05)
        angle = self.transformation.calculate_angle(1, [0,0], [100,100], [110,100], 0.06666666666666667)
        angle = self.transformation.calculate_angle(1, [0,0], [100,100], [110,100], 0.08333333333333333)

        angle = self.transformation.calculate_angle(1, [0,0], [100,100], [110,100], 0.1)
        angle = self.transformation.calculate_angle(1, [0,0], [100,100], [110,100], 0.11666666666666667)
        angle = self.transformation.calculate_angle(1, [0,0], [100,100], [110,100], 0.13333333333333333)

        angle = self.transformation.calculate_angle(1, [0,0], [100,100], [110,100], 0.15)
        angle = self.transformation.calculate_angle(1, [0,0], [100,100], [110,100], 0.16666666666666666)
        angle = self.transformation.calculate_angle(1, [0,0], [100,100], [110,100], 0.18333333333333332)

        angle = self.transformation.calculate_angle(1, [0,0], [100,100], [110,100], 0.2)
        angle = self.transformation.calculate_angle(1, [0,0], [100,100], [110,100], 0.21666666666666667)
        angle = self.transformation.calculate_angle(1, [0,0], [100,100], [110,100], 0.23333333333333334)

    # def test_angular_velocity():

    # def test_angular_acceleration():
    # def test_green_angular_acceleration():
    # def test_yellow_angular_angular_acceleration():
    # def test_red_angular_angular_acceleration():
    

if __name__ == '__main__':
    unittest.main()