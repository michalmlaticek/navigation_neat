class SimulationConf:
    def __init__(self, id, robot, init_rotation, map, step_count, pop_size, animate, log_folder):
        self.id = id
        self.robot = robot
        self.init_rotation = init_rotation
        self.map = map
        self.step_count = step_count
        self.pop_size = pop_size
        self.animate = animate
        self.log_folder = log_folder
