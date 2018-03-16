from collections import namedtuple
import numpy as np
from bresenham import bresenham
import math

SimulationConf = namedtuple("SimulationConf", "robot init_rotation start_point end_point map step_count animate")

RobotConf = namedtuple("RobotConf", "radius sensor_angles sensor_len max_speed body")


def angle_hypotenuse_to_dxy(angles, hypotenuse):
    '''

    :param angles: 2d matrix of radian angles
    :param hypotenuse: 2d matrix of sensor lengths
    :return: 3d matrix of x, y deltas
    '''
    sin_mat = np.sin(angles)
    cos_mat = np.cos(angles)
    dxy = np.multiply(np.dstack((cos_mat, sin_mat)), np.expand_dims(hypotenuse, axis=2))

    return dxy


def calc_coordinates(angles, hypotenuse, start_coordinates):
    '''

    :param angles:
    :param len:
    :param start_coordinates:
    :return:
    '''
    dxy = angle_hypotenuse_to_dxy(angles, hypotenuse)
    xy = dxy + start_coordinates
    return xy


def calc_line_coordinates(start_coordinates, end_coordinates):
    rint_start = np.rint(start_coordinates).astype(int)
    rint_end = np.rint(end_coordinates).astype(int)
    coordinates = []
    for col in range(0, rint_start.shape[1]):
        col_coordinates = []
        for row in range(0, rint_start.shape[0]):
            sensor_coordinates = list(bresenham(
                rint_start[row, col, 0],
                rint_start[row, col, 1],
                rint_end[row, col, 0],
                rint_end[row, col, 1]
            ))
            # remove the ones that are outside of map boundaries
            sensor_coordinates[:] = [c for c in sensor_coordinates if (0 <= c[0] < 250 and 0 <= c[1] < 250)]
            col_coordinates.append(sensor_coordinates)
        coordinates.append(col_coordinates)

    return coordinates


def read_sensors(sensor_lines, sensor_len, map):
    readings = np.zeros((len(sensor_lines[0]), len(sensor_lines))) + sensor_len
    for robot in range(0, len(sensor_lines)):
        for sensor in range(0, len(sensor_lines[robot])):
            for line_point in sensor_lines[robot][sensor]:
                if map[line_point[0], line_point[1]] == 0.0:
                    # calc distance from start of the sensor line to point of contact
                    readings[sensor, robot] = point_distance(sensor_lines[robot][sensor][0], line_point)
                    break

    return readings


def point_distance(source, target):
    return math.sqrt(((target[0] - source[0]) ** 2) + ((target[1] - source[1]) ** 2))


def point_dist_vec(source, target):
    st_dxy = np.subtract(source, target)
    st_dxy_power2 = np.power(st_dxy, 2)
    sum_power = np.sum(st_dxy_power2, axis=2)
    return np.sqrt(sum_power)


# TODO: test -pi / +pi behavior
def calc_angle_error(source, target, robot_angles):
    # move target to source base
    moved_target = np.subtract(target, source)
    # calc angle to target from [0, 0]
    target_angles = np.expand_dims(np.arctan2(moved_target[0, :, 1], moved_target[0, :, 0]), axis=0)
    return np.subtract(robot_angles, target_angles)


def normalize_sensor_readings(sensor_readings, sensor_len):
    return np.subtract(1, np.divide(sensor_readings, sensor_len))


def normalize_target_distance(target_distance, max_distance):
    return np.divide(target_distance, max_distance)


def normalize_angle_error(angle_errors):
    return np.divide(np.add(angle_errors, math.pi), 2 * math.pi)


def normalize_angle_error_for_fit(angle_error):
    return angle_error / math.pi


def build_robot_body(radius):
    body = []
    for x in range(-radius, radius):
        for y in range(-radius, radius):
            if x ** 2 + y ** 2 <= radius ** 2:
                body.append([x, y])

    return np.expand_dims(np.array(body), axis=1)
