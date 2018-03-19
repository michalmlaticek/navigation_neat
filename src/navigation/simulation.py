import utils
import numpy as np
import math


class Simulation:

    def __init__(self, nets, simulation_config):
        self.nets = nets
        self.simulation_config = simulation_config

        # for readability
        self.pop_size = len(self.nets)
        self.map = simulation_config.map
        self.robot = simulation_config.robot
        self.max_distance = math.sqrt(self.map.plane.shape[0] * self.map.plane.shape[1])

        self.fitnesses = np.full((1, self.pop_size), -100.0)  # init to -100

        # array of population indexes
        self.net_ids = np.arange(self.pop_size).reshape(1, self.pop_size)

        # initialize robot angles (1, pop_size)
        self.robot_angles = np.full((1, self.pop_size), utils.to_minus_pi_pi(self.simulation_config.init_rotation))
        # initialize sensor angles of robots
        self.sensor_angles = np.zeros(
            (len(self.robot.sensor_angles), self.pop_size)) + utils.to_minus_pi_pi(self.robot.sensor_angles)

        # initialize robot positions (1, pop_size, 2)
        self.robot_positions = np.full((1, self.pop_size, 2), self.map.start_point)
        self.robot_bodies = np.add(self.robot.body, np.rint(self.robot_positions)).astype(int)
        self.sensor_lines = self.__update_sensor_lines()

        # calculate angle errors and then normalize them
        self.angle_errors = self.__update_angle_errors()
        self.normalized_angle_errors = self.__normalize_angle_errors()

        # calculate target distances and then normalize them
        self.target_distances = self.__update_target_distances()
        self.normalized_target_distances = self.__normalize_target_distances()

    def simulate(self, step_count, step_callback=None, callback_args=None):
        '''

        :param step_count:
        :param step_callback:
        :param callback_args:
        :return:
        '''

        # calculate second level help variables

        for i in range(0, step_count):
            print("step: {}".format(i))
            self.step()
            if step_callback is not None:
                step_callback(*callback_args)

        self.__handle_remaining()

        return self.fitnesses

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

        self.__handle_collisions_or_targets()

    def __update_angle_errors(self):
        self.angle_errors = utils.calc_angle_error(self.robot_positions, self.map.end_point, self.robot_angles)
        return self.angle_errors

    def __normalize_angle_errors(self):
        self.normalized_angle_errors = np.divide(np.add(self.angle_errors, math.pi), 2 * math.pi)
        return self.normalized_angle_errors

    def __update_target_distances(self):
        self.target_distances = utils.point_dist_vec(self.robot_positions, self.map.end_point)
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

    def __is_goal(self, robot_position, target):
        return np.array_equal(np.rint(robot_position).astype(int), target)

    def __fitness_value(self, norm_target_distance, norm_angle_error, alfa, beta):
        # adjust angle error to consider only absolute value of angle error
        angle_error = math.fabs(norm_angle_error - 0.5)  # at the same time, this reduces slightly the angle importance
        return (alfa * beta * (angle_error + norm_target_distance)) * -1  # we are maximizing to 0

    def __deactivate_nets(self, ids_to_delete):

        to_delete = np.array(ids_to_delete)

        self.net_ids = np.delete(self.net_ids, to_delete, 1)
        self.robot_positions = np.delete(self.robot_positions, to_delete, 1)
        self.robot_angles = np.delete(self.robot_angles, to_delete, 1)
        self.nets = np.delete(self.nets, to_delete)
        self.sensor_angles = np.delete(self.sensor_angles, to_delete, 1)

    def __read_sensors(self):
        readings = np.zeros((len(self.sensor_lines[0]), len(self.sensor_lines))) + self.robot.sensor_len
        for robot in range(0, len(self.sensor_lines)):
            for sensor in range(0, len(self.sensor_lines[robot])):
                for line_point in self.sensor_lines[robot][sensor]:
                    if self.map.plane[line_point[0], line_point[1]] != 1:  # 1 means free
                        # calc distance from start of the sensor line to point of contact
                        readings[sensor, robot] = utils.point_distance(self.sensor_lines[robot][sensor][0], line_point)
                        break
        normalized_readings = self.__normalize_sensor_readings(readings)
        return normalized_readings

    def __normalize_sensor_readings(self, sensor_readings):
        return np.subtract(1, np.divide(sensor_readings, self.robot.sensor_len))

    def __handle_collisions_or_targets(self):
        nets_to_deactivate = []
        for i in range(0, self.net_ids.shape[1]):
            is_collision = self.__is_collision(self.robot_bodies[:, i, :], self.map.plane)
            if is_collision:
                self.fitnesses[0, self.net_ids[0, i]] = self.__fitness_value(
                    self.normalized_target_distances[0, i],
                    self.normalized_angle_errors[0, i],
                    alfa=1,
                    beta=2)
                nets_to_deactivate.append(i)
                continue

            is_target = self.__is_goal(self.robot_positions[:, [i], :], self.map.end_point)
            if is_target:
                self.__fitnesses[0, self.net_ids[0, i]] = 0.
                nets_to_deactivate.append(i)
                continue

        self.__deactivate_nets(nets_to_deactivate)

    def __handle_remaining(self):
        for i in range(0, self.net_ids.shape[1]):
            self.fitnesses[0, self.net_ids[0, i]] = self.__fitness_value(
                self.normalized_target_distances[0, i],
                self.normalized_angle_errors[0, i],
                alfa=1,
                beta=1)
