import unittest
import math
import numpy as np
import utils


class UtilsTest(unittest.TestCase):
    def setUp(self):
        pass

    def test_angle_hypotenuse_to_dxy(self):
        '''
        Test mainly correct vectorization
        :return:
        '''

        # arrange
        robot1_sensor_angles = np.array([
            math.radians(-60),  # -1.0471975511965976
            math.radians(-30),  # -0.5235987755982988
            math.radians(0),    # 0.0
            math.radians(30),   # 0.5235987755982988
            math.radians(60)    # 1.0471975511965976
        ])
        robot2_sensor_angles = np.array([
            math.radians(-90),  # -1.5707963267948966
            math.radians(-60),
            math.radians(-30),
            math.radians(0),
            math.radians(30)
        ])
        sensor_angles = np.column_stack((robot1_sensor_angles.T, robot2_sensor_angles.T))

        hypotenuse = 1          # lets make it simple

        # act
        dxy = utils.angle_hypotenuse_to_dxy(sensor_angles, hypotenuse)

        # assert
        self.assertEquals(dxy.shape, (5, 2, 2))
        # continue with actual value test

    def test_calc_coordinates(self):
        # arrange
        robot1_sensor_angles = np.array([
            math.radians(-60),  # -1.0471975511965976
            math.radians(-30),  # -0.5235987755982988
            math.radians(0),  # 0.0
            math.radians(30),  # 0.5235987755982988
            math.radians(60)  # 1.0471975511965976
        ])
        robot2_sensor_angles = np.array([
            math.radians(-90),  # -1.5707963267948966
            math.radians(-60),
            math.radians(-30),
            math.radians(0),
            math.radians(30)
        ])
        sensor_angles = np.column_stack((robot1_sensor_angles.T, robot2_sensor_angles.T))

        hypotenuse = 1  # lets make it simple

        source1 = np.array([1, 0]).T
        source2 = np.array([0, 2]).T

        source = np.column_stack((source1, source2))

        # act
        xy = utils.calc_coordinates(sensor_angles, hypotenuse, source)

        # assert
        self.assertEquals(xy.shape, (5, 2, 2))
        self.assertEquals(xy[2, 0, 0], 2)


if __name__ == "__main__":
    unittest.main()
