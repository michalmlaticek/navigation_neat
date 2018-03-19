import numpy as np


def get_image(robot_bodies, sensor_lines, map):
    local_map = map.plane.copy()
    # local_map = np.multiply(local_map.astype(int), 255)
    for r in range(0, robot_bodies.shape[1]):
        # draw robot body
        for b in range(0, robot_bodies.shape[0]):
            if 0 <= robot_bodies[b, r, 0] < map.plane.shape[0] and 0 <= robot_bodies[b, r, 1] < map.plane.shape[1]:
                local_map[robot_bodies[b, r, 0], robot_bodies[b, r, 1]] = r + 2

        # draw robot sensors
        for s in range(0, len(sensor_lines[r])):
            for lx, ly in sensor_lines[r][s]:
                if 0 <= lx < map.plane.shape[0] and 0 <= ly < map.plane.shape[1]:
                    local_map[lx, ly] = r + 2

    map_3d = np.dstack((local_map, local_map, local_map))

    # start point as blue
    map_3d[map.start_point[0, 0, 0], map.start_point[0, 0, 1], 0] = 0
    map_3d[map.start_point[0, 0, 0], map.start_point[0, 0, 1], 1] = 0
    map_3d[map.start_point[0, 0, 0], map.start_point[0, 0, 1], 2] = 255

    # end point as green
    map_3d[map.end_point[0, 0, 0], map.end_point[0, 0, 1], 0] = 0
    map_3d[map.end_point[0, 0, 0], map.end_point[0, 0, 1], 1] = 255
    map_3d[map.end_point[0, 0, 0], map.end_point[0, 0, 1], 2] = 0

    return map_3d.astype(int)


def get_image_zxy(robot_bodies, sensor_lines, map):
    xyz = get_image(robot_bodies, sensor_lines, map)
    return to_zxy(xyz)


def to_zxy(xyz):
    xzy = np.swapaxes(xyz, 1, 2)
    zxy = np.swapaxes(xzy, 0, 1)
    return zxy
