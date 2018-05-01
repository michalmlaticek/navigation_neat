import numpy as np

class Robot:

    def __init__(self, radius, sensor_angles_deg, sensor_len, max_speed):
        self.radius = radius
        self.sensor_angles = np.radians(np.array(sensor_angles_deg, ndmin=2).T)
        self.sensor_len = sensor_len
        self.max_speed = max_speed
        self.body = self._build_robot_body()

    def _build_robot_body(self):
        body = []
        for x in range(-self.radius, self.radius):
            for y in range(-self.radius, self.radius):
                if x ** 2 + y ** 2 <= self.radius ** 2:
                    body.append([x, y])

        return np.expand_dims(np.array(body), axis=1)
