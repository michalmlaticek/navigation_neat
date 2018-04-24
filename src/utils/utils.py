from collections import namedtuple
import numpy as np
from bresenham import bresenham
import math

SimulationConf = namedtuple("SimulationConf", "robot init_rotation map step_count animate")


def to_minus_pi_pi(angles):
    angles = (angles + np.pi) % (2 * np.pi) - np.pi
    return angles


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


def point_distance(source, target):
    return math.sqrt(((target[0] - source[0]) ** 2) + ((target[1] - source[1]) ** 2))


def point_dist_vec(source, target):
    st_dxy = np.subtract(source, target)
    st_dxy_power2 = np.power(st_dxy, 2)
    sum_power = np.sum(st_dxy_power2, axis=2)
    return np.sqrt(sum_power)


# TODO: test -pi / +pi behavior
def calc_angle_error(sources, targets, source_angles):
    # move target to source base
    rebased_target = np.subtract(targets, sources)
    # calc angle to target from [0, 0]
    target_angles = np.expand_dims(np.arctan2(rebased_target[0, :, 1], rebased_target[0, :, 0]), axis=0)
    return np.subtract(source_angles, target_angles)
