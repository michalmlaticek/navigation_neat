import numpy as np


def get_image(robot_bodies, sensor_lines, map, target):
    local_map = map.copy()
    for r in range(0, robot_bodies.shape[1]):
        # draw robot body
        for b in range(0, robot_bodies.shape[0]):
            local_map[robot_bodies[b, r, 0], robot_bodies[b, r, 1]] = (r + 2) / 255

        # draw robot sensors
        for s in range(0, len(sensor_lines[r])):
            for lx, ly in sensor_lines[r][s]:
                local_map[lx, ly] = (r + 2) / 255

    map_3d = np.dstack((local_map, local_map, local_map))
    map_3d[target[0, 0, 0], target[0, 0, 1], 0] = 1.
    map_3d[target[0, 0, 0], target[0, 0, 1], 1] = 0.
    map_3d[target[0, 0, 0], target[0, 0, 1], 2] = 0.

    return map_3d
