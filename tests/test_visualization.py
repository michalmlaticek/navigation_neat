import visualization as viz
import numpy as np
import utils
import math

map_size = 250
border_size = 46
map = np.ones((map_size, map_size))
map[100:151, 100:151] = 0.
map[0:border_size, :] = 0.
map[:, 0:border_size] = 0.
map[map_size - border_size:, :] = 0
map[:, map_size - border_size:] = 0

robot_body = utils.build_robot_body(10) + [[[65, 65]]]

sensor_angles = np.array(
    [[math.radians(-60), math.radians(-30), math.radians(0), math.radians(30), math.radians(60)]]).T

sensor_start_points = utils.calc_coordinates(sensor_angles, np.array([[10]]), [[[65, 65]]])  # (1, ?, 2)
sensor_end_points = utils.calc_coordinates(sensor_angles, np.array([[30]]), [[[65, 65]]])  # (1, ?, 2)
sensor_lines = utils.calc_line_coordinates(sensor_start_points, sensor_end_points)

viz.draw(robot_body, sensor_lines, map)
