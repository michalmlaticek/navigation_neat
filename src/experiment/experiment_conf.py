class ExperimentConf:
    def __init__(self, experiment_id, run_id, neat_conf, simulation, gen_count, log_folder):
        self.experiment_id = experiment_id
        self.run_id = run_id
        self.neat_conf = neat_conf
        self.simulation = simulation
        self.gen_count = gen_count
        self.log_folder = log_folder
