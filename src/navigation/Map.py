from PIL import Image
import numpy as np


class Map:

    def __init__(self, plane, height, width, start, target):
        self.plane = plane
        self.height = height
        self.width = width
        self.start_point = start
        self.end_point = target


class MapFactory:
    @staticmethod
    def init_basic(width, height, border_size, obst_width, obst_height, start, target):
        start = np.array([[start]])
        target = np.array([[target]])
        plane = np.ones((height, width))
        mid_height = int(height / 2)
        mid_width = int(width / 2)
        obst_height = int(obst_height / 2)
        obst_width = int(obst_width / 2)
        plane[mid_height - obst_height:mid_height + obst_height, mid_width - obst_width: mid_width + obst_width] = 0.
        plane[0:border_size, :] = 0.
        plane[:, 0:border_size] = 0.
        plane[height - border_size:, :] = 0.
        plane[:, width - border_size:] = 0.

        return Map(plane, height, width, start, target)

    @staticmethod
    def create_from_pic(path, start, target):
        map_img = Image.open(path)
        plane = np.array(map_img)[:, :, 0] / 255
        height = plane.shape[0]
        width = plane.shape[1]
        start = np.array([[start]])
        target = np.array([[target]])

        return Map(plane, height, width, start, target)
