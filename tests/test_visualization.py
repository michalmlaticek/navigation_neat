import FrameFactory as viz
import numpy as np
import utils
import math
from array2gif import write_gif

map_size = 250
border_size = 2
map = np.ones((map_size, map_size)).astype(int)
map[100:151, 100:151] = 0
map[0:border_size, :] = 0
map[:, 0:border_size] = 0
map[map_size - border_size:, :] = 0
map[:, map_size - border_size:] = 0

map_object = utils.Map(map,  np.array([[[40, 40]]]), np.array([[[210, 210]]]))

robot_body = utils.build_robot_body(10) + [[[65, 65]]]

sensor_angles = np.array(
    [[math.radians(-60), math.radians(-30), math.radians(0), math.radians(30), math.radians(60)]]).T

sensor_start_points = utils.calc_coordinates(sensor_angles, np.array([[10]]), [[[65, 65]]])  # (1, ?, 2)
sensor_end_points = utils.calc_coordinates(sensor_angles, np.array([[30]]), [[[65, 65]]])  # (1, ?, 2)
sensor_lines = utils.calc_line_coordinates(sensor_start_points, sensor_end_points)

img = viz.get_image_zxy(robot_body, sensor_lines, map_object)

write_gif(img, "map.gif", fps=5)

print(str(img))
