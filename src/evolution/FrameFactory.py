import numpy as np


class FrameFactory:

    @staticmethod
    def get_image(robot_bodies, sensor_lines, simulation_map, collis_ind=None):
        local_map = np.multiply(simulation_map.plane.astype(int), 255)
        local_map = np.dstack((local_map, local_map, local_map))
        for r in range(0, robot_bodies.shape[1]):
            # draw robot body
            for b in range(0, robot_bodies.shape[0]):
                if 0 <= robot_bodies[b, r, 0] < simulation_map.plane.shape[0] and 0 <= robot_bodies[b, r, 1] < \
                        simulation_map.plane.shape[1]:
                    if collis_ind[0, r] == 1.0:
                        local_map[robot_bodies[b, r, 0], robot_bodies[b, r, 1], 0] = 255
                        local_map[robot_bodies[b, r, 0], robot_bodies[b, r, 1], 1] = 0
                        local_map[robot_bodies[b, r, 0], robot_bodies[b, r, 1], 2] = 0
                    else:
                        local_map[robot_bodies[b, r, 0], robot_bodies[b, r, 1], 0] = r + 2
                        local_map[robot_bodies[b, r, 0], robot_bodies[b, r, 1], 1] = r + 2
                        local_map[robot_bodies[b, r, 0], robot_bodies[b, r, 1], 2] = r + 2

            # draw robot sensors
            for s in range(0, len(sensor_lines[r])):
                for lx, ly in sensor_lines[r][s]:
                    if 0 <= lx < simulation_map.plane.shape[0] and 0 <= ly < simulation_map.plane.shape[1]:
                        local_map[lx, ly] = r + 2

        # start point as blue
        local_map[simulation_map.start_point[0, 0, 0], simulation_map.start_point[0, 0, 1], 0] = 0
        local_map[simulation_map.start_point[0, 0, 0], simulation_map.start_point[0, 0, 1], 1] = 0
        local_map[simulation_map.start_point[0, 0, 0], simulation_map.start_point[0, 0, 1], 2] = 255

        # end point as green
        local_map[simulation_map.end_point[0, 0, 0], simulation_map.end_point[0, 0, 1], 0] = 0
        local_map[simulation_map.end_point[0, 0, 0], simulation_map.end_point[0, 0, 1], 1] = 255
        local_map[simulation_map.end_point[0, 0, 0], simulation_map.end_point[0, 0, 1], 2] = 0

        return local_map

    @staticmethod
    def get_image_zxy(robot_bodies, sensor_lines, simulation_map):
        xyz = FrameFactory.get_image(robot_bodies, sensor_lines, simulation_map)
        return FrameFactory.to_zxy(xyz)

    @staticmethod
    def to_zxy(xyz):
        xzy = np.swapaxes(xyz, 1, 2)
        zxy = np.swapaxes(xzy, 0, 1)
        return zxy
