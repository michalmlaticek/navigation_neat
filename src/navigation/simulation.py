import utils
import numpy as np
import math
import visualization as viz
from matplotlib import pyplot as plt


def simulate(nets, simulation_conf):
    '''
    Run simulation
    :param nets:
    :param simulation_conf:
    :return:
    '''

    ##############################################################
    # Internal methods
    ##############################################################
    def eval_nets(nets, net_inputs):
        net_outputs = np.zeros((2, len(nets)))
        for i in range(0, len(nets)):
            net_outputs[:, i] = nets[i].activate(list(net_inputs[:, i]))
        return net_outputs

    def is_collision(robot_body, map):
        for idx in range(0, robot_body.shape[0]):
            if (robot_body[idx, 0] < 0
                    or robot_body[idx, 0] >= 250
                    or robot_body[idx, 1] < 0
                    or robot_body[idx, 1] >= 250
                    or map[robot_body[idx, 0], robot_body[idx, 1]] == 0):
                return True

    def is_goal(robot_position, target):
        return np.array_equal(np.rint(robot_position).astype(int), target)

    def fitness_value(norm_distance, norm_angle_error, alfa, beta):
        return (alfa * beta * (math.fabs(norm_angle_error) + norm_distance)) * -1  # we are maximizing to 0

    def deactivate_nets(ids_to_delete):
        nonlocal net_ids
        nonlocal robot_positions
        nonlocal robot_angles
        nonlocal nets
        nonlocal sensor_angles

        to_delete = np.array(ids_to_delete)

        net_ids = np.delete(net_ids, to_delete, 1)
        robot_positions = np.delete(robot_positions, to_delete, 1)
        robot_angles = np.delete(robot_angles, to_delete, 1)
        nets = np.delete(nets, to_delete)
        sensor_angles = np.delete(sensor_angles, to_delete, 1)

    def correct_angle(angle):
        if angle > math.pi:
            angle -= 2 * math.pi
        elif angle < -math.pi:
            angle += 2 * math.pi

        return angle

    ##############################################################

    ##############################################################
    # For visualization
    ##############################################################
    fig = plt.Figure()
    im = plt.imshow(np.dstack((simulation_conf.map, simulation_conf.map, simulation_conf.map)))
    ##############################################################

    fitnesses = np.zeros((1, len(nets))) - 1  # init to -1

    # initialize first level help variables
    net_ids = np.arange(len(nets)).reshape(1, len(
        nets))  # mark the initial net ids (we're going to delete the ones that collide)
    robot_positions = np.full((1, len(nets), 2), simulation_conf.start_point)  # (1, ?, 2)
    robot_angles = np.full((1, len(nets)), simulation_conf.init_rotation)  # (1, ?)
    sensor_angles = np.zeros((
        len(simulation_conf.robot.sensor_angles), len(nets))) + simulation_conf.robot.sensor_angles  # (sensor_count, ?)

    # calculate second level help variables
    sensor_start_points = utils.calc_coordinates(sensor_angles, np.array([[simulation_conf.robot.radius]]),
                                                 robot_positions)  # (1, ?, 2)
    sensor_end_points = utils.calc_coordinates(sensor_angles, np.array([[simulation_conf.robot.sensor_len]]),
                                               robot_positions)  # (1, ?, 2)
    sensor_lines = utils.calc_line_coordinates(sensor_start_points, sensor_end_points)
    robot_distances = utils.point_dist_vec(robot_positions, simulation_conf.end_point)
    normalized_robot_distances = utils.normalize_target_distance(robot_distances,
                                                                 math.sqrt(2 * (simulation_conf.map.shape[0] ** 2)))
    angle_errors = utils.calc_angle_error(robot_positions, simulation_conf.end_point, robot_angles)

    vcorrect_angle = np.vectorize(correct_angle)

    for i in range(0, simulation_conf.step_count):
        sensor_readings = utils.read_sensors(sensor_lines, simulation_conf.robot.sensor_len, simulation_conf.map)

        normalized_sensor_readings = utils.normalize_sensor_readings(sensor_readings, simulation_conf.robot.sensor_len)
        normalized_angle_errors = utils.normalize_angle_error(angle_errors)

        net_inputs = np.vstack((normalized_sensor_readings, normalized_robot_distances))
        net_inputs = np.vstack((net_inputs, normalized_angle_errors))

        net_outputs = eval_nets(nets, net_inputs)

        delta_angles = np.subtract(np.multiply(np.expand_dims(net_outputs[0, :], axis=0), 2 * math.pi), math.pi)
        robot_speeds = np.multiply(np.expand_dims(net_outputs[1, :], axis=0),
                                   simulation_conf.robot.max_speed)  # actually distance because t = 1
        robot_angles = vcorrect_angle(np.add(robot_angles, delta_angles))
        sensor_angles = vcorrect_angle(np.add(sensor_angles, delta_angles))
        robot_positions = utils.calc_coordinates(robot_angles, robot_speeds, robot_positions)
        sensor_start_points = utils.calc_coordinates(sensor_angles, np.array([[simulation_conf.robot.radius]]),
                                                     robot_positions)
        sensor_end_points = utils.calc_coordinates(sensor_angles, np.array([[simulation_conf.robot.sensor_len]]),
                                                   robot_positions)
        sensor_lines = utils.calc_line_coordinates(sensor_start_points, sensor_end_points)

        robot_distances = utils.point_dist_vec(robot_positions, simulation_conf.end_point)
        normalized_robot_distances = utils.normalize_target_distance(robot_distances,
                                                                     math.sqrt(simulation_conf.map.shape[0] ** 2))
        angle_errors = utils.calc_angle_error(robot_positions, simulation_conf.end_point, robot_angles)

        nets_to_deactivate = []
        robot_bodies = np.add(simulation_conf.robot.body, np.rint(robot_positions)).astype(int)
        for n in range(0, net_ids.shape[1]):
            isCollision = is_collision(robot_bodies[:, n, :], simulation_conf.map)
            if isCollision:
                fitnesses[0, net_ids[0, n]] = fitness_value(normalized_robot_distances[0, n],
                                                            utils.normalize_angle_error_for_fit(angle_errors[0, n]), 1,
                                                            2)
                nets_to_deactivate.append(n)
                continue

            isTarget = is_goal(robot_positions[:, [n], :], simulation_conf.end_point)
            if isTarget:
                fitnesses[0, net_ids[0, n]] = 0
                nets_to_deactivate.append(n)
                continue

        deactivate_nets(nets_to_deactivate)

        if simulation_conf.animate:
            bodies = np.rint(robot_positions + simulation_conf.robot.body).astype(int)
            img = viz.get_image(bodies, sensor_lines, simulation_conf.map, simulation_conf.end_point)
            im.set_data(img)
            plt.draw()
            plt.pause(0.001)

    for id in range(0, net_ids.shape[1]):
        fitnesses[0, net_ids[0, id]] = fitness_value(normalized_robot_distances[0, id],
                                                     utils.normalize_angle_error_for_fit(angle_errors[0, id]),
                                                     1,
                                                     1)

    return fitnesses[0, :]
