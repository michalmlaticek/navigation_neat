import utils as utils
import numpy as np
import math
import logging


class SimulationConf:
    def __init__(self, robot, init_rotation, sim_map, step_count, pop_size, animate):
        self.robot = robot
        self.init_rotation = init_rotation
        self.sim_map = sim_map
        self.step_count = step_count
        self.pop_size = pop_size
        self.animate = animate


class Simulation:

    def __init__(self, conf):
        logger = logging.getLogger('Simulation')
        logger.info('Initializing')

        self.conf = conf

        # for readability
        self.pop_size = conf.pop_size
        self.sim_map = conf.sim_map
        self.robot = conf.robot
        self.max_distance = math.sqrt((self.sim_map.plane.shape[0] ** 2) + (self.sim_map.plane.shape[1] ** 2))
        self.s_to_t_distance = utils.point_distance(conf.sim_map.start_point[0, 0, :].tolist(),
                                                    conf.sim_map.end_point[0, 0, :].tolist())

        self.fitnesses = None
        self.net_ids = None
        self.robot_angles = None
        self.sensor_angles = None
        self.robot_positions = None
        self.robot_bodies = None
        self.sensor_lines = None
        self.angle_errors = None
        self.normalized_angle_errors = None
        self.target_distances = None
        self.normalized_target_distances = None
        self.max_distance_from_start = None
        self.nets = None
        self.collision_counters = None

    def reset(self):
        self.fitnesses = np.full((1, self.pop_size), -10000.0)  # init fitness to -10000

        # array of population indexes
        self.net_ids = np.arange(self.pop_size).reshape(1, self.pop_size)

        # initialize robot angles (1, pop_size)
        self.robot_angles = np.full((1, self.pop_size), utils.to_minus_pi_pi(self.conf.init_rotation))
        # initialize sensor angles of robots
        self.sensor_angles = np.zeros(
            (len(self.robot.sensor_angles), self.pop_size)) + utils.to_minus_pi_pi(self.robot.sensor_angles)

        # initialize robot positions (1, pop_size, 2)
        self.robot_positions = np.full((1, self.pop_size, 2), self.sim_map.start_point)
        self.robot_bodies = np.add(self.robot.body, np.rint(self.robot_positions)).astype(int)
        self.sensor_lines = self.__update_sensor_lines()

        # calculate angle errors and then normalize them
        self.angle_errors = self.__update_angle_errors()
        self.normalized_angle_errors = self.__normalize_angle_errors()

        # calculate target distances and then normalize them
        self.target_distances = self.__update_target_distances()
        self.normalized_target_distances = self.__normalize_target_distances()
        self.max_distance_from_start = np.zeros((1, self.pop_size))
        self.collision_counters = np.zeros((1, self.pop_size))

    def simulate(self, nets, step_count, step_callback=None, callback_args=None):
        '''

        :param step_count:
        :param step_callback:
        :param callback_args:
        :return:
        '''

        self.nets = nets

        for i in range(0, step_count):
            self.step()
            if step_callback is not None:
                if callback_args is None:
                    step_callback()
                else:
                    step_callback(*callback_args)

        self.__set_fits()

        return self.fitnesses, self.target_distances, self.collision_counters

    def step(self):
        net_inputs = self.__create_net_inputs()
        net_outputs = self.__eval_nets(net_inputs)
        (delta_angles, robot_speeds) = self.__extract_outputs(net_outputs)

        self.__rotate(delta_angles)  # rotate robos
        self.__translate(robot_speeds)  # move robots based on speed in the new direction
        self.__update_angle_errors()
        self.__normalize_angle_errors()
        self.__update_target_distances()
        self.__normalize_target_distances()
        self.__update_sensor_lines()
        self.__handle_collisions()

    def __update_angle_errors(self):
        self.angle_errors = utils.calc_angle_error(self.robot_positions, self.sim_map.end_point, self.robot_angles)
        self.angle_errors = utils.to_minus_pi_pi(self.angle_errors)
        return self.angle_errors

    def __normalize_angle_errors(self):
        self.normalized_angle_errors = np.divide(np.add(self.angle_errors, math.pi), 2 * math.pi)
        return self.normalized_angle_errors

    def __update_target_distances(self):
        self.target_distances = utils.point_dist_vec(self.robot_positions, self.sim_map.end_point)
        return self.target_distances

    def __normalize_target_distances(self):
        self.normalized_target_distances = np.divide(self.target_distances, self.max_distance)
        return self.normalized_target_distances

    def __update_sensor_lines(self):
        sensor_start_points = utils.calc_coordinates(self.sensor_angles,
                                                     np.array([[self.robot.radius]]),
                                                     self.robot_positions)
        sensor_end_points = utils.calc_coordinates(self.sensor_angles,
                                                   np.array([[self.robot.sensor_len]]),
                                                   self.robot_positions)
        self.sensor_lines = utils.calc_line_coordinates(sensor_start_points, sensor_end_points)
        return self.sensor_lines

    def __create_net_inputs(self):
        sensor_readings = self.__read_sensors()
        angle_errors = self.__normalize_angle_errors()

        net_inputs = np.vstack((sensor_readings, self.normalized_target_distances, angle_errors))
        return net_inputs

    def __eval_nets(self, net_inputs):
        net_outputs = np.zeros((2, len(self.nets)))
        for i in range(0, len(self.nets)):
            net_outputs[:, i] = self.nets[i].activate(list(net_inputs[:, i]))
        return net_outputs

    def __extract_outputs(self, net_outputs):
        delta_angles = np.subtract(np.multiply(net_outputs[[0], :], 2 * math.pi), math.pi)
        robot_speeds = np.multiply(net_outputs[[1], :], self.robot.max_speed)  # actually distance because t = 1
        return delta_angles, robot_speeds

    def __rotate(self, delta_angles):
        self.robot_angles = utils.to_minus_pi_pi(np.add(self.robot_angles, delta_angles))
        self.sensor_angles = utils.to_minus_pi_pi(np.add(self.sensor_angles, delta_angles))
        return self.robot_angles, self.sensor_angles

    def __translate(self, speeds):
        self.robot_positions = utils.calc_coordinates(self.robot_angles, speeds, self.robot_positions)
        self.robot_bodies = np.add(self.robot.body, np.rint(self.robot_positions)).astype(int)
        return self.robot_positions

    def __is_collision(self, robot_body, map):
        for idx in range(0, robot_body.shape[0]):
            if (robot_body[idx, 0] < 0
                    or robot_body[idx, 0] >= 250
                    or robot_body[idx, 1] < 0
                    or robot_body[idx, 1] >= 250
                    or map[robot_body[idx, 0], robot_body[idx, 1]] == 0):
                return True
        return False

    def __read_sensors(self):
        readings = np.zeros((len(self.sensor_lines[0]), len(self.sensor_lines))) + self.robot.sensor_len
        for robot in range(0, len(self.sensor_lines)):
            for sensor in range(0, len(self.sensor_lines[robot])):
                for line_point in self.sensor_lines[robot][sensor]:
                    if self.sim_map.plane[line_point[0], line_point[1]] != 1:  # 1 means free
                        # calc distance from start of the sensor line to point of contact
                        readings[sensor, robot] = utils.point_distance(self.sensor_lines[robot][sensor][0], line_point)
                        break
        readings = self.__normalize_sensor_readings(readings)
        return readings

    def __normalize_sensor_readings(self, sensor_readings):
        return np.subtract(1, np.divide(sensor_readings, self.robot.sensor_len))

    def __handle_collisions(self):
        for i in range(0, self.net_ids.shape[1]):
            is_collision = self.__is_collision(self.robot_bodies[:, i, :], self.sim_map.plane)
            if is_collision:
                self.collision_counters[0, i] += 1

    def __set_fits(self):
        self.fitnesses = -1 * (np.add(self.target_distances, self.collision_counters * 10))