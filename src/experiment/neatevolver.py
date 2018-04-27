import logging
import neat
from LogReporter import LogReporter
from MyCheckpoint import MyCheckpointer
import visualization as viz
import matplotlib.pyplot as plt
import dill


class NeatEvolver:
    logger = logging.getLogger('NeatEvolver')

    def __init__(self, experiment_conf):
        self.logger.info("Initializing evolver")
        self.neat_conf = experiment_conf.neat_conf
        self.gen_count = experiment_conf.gen_count
        self.simulation = experiment_conf.simulation
        self.simulation.reset()
        self.generation = -1
        self.log_folder = experiment_conf.log_folder

    def evolve(self, init_pop=None):
        self.logger.info("Starting evolution")
        if init_pop is None:
            pop = neat.Population(self.neat_conf)
            pop.add_reporter(neat.StatisticsReporter())
            pop.add_reporter(LogReporter(True))
            pop.add_reporter(
                MyCheckpointer(folder_path=self.log_folder))
        else:
            pop = init_pop

        winner = pop.run(self._eval_population, self.gen_count)
        self.logger.info("winner: {}".format(winner))

        self.logger.info("End evolution")

        return winner

    def _eval_population(self, genomes, neat_config):
        self.generation += 1
        self.logger.info("Evolving generation: {}".format(self.generation))

        fig = plt.Figure()
        img = plt.imshow(viz.get_image(self.simulation.robot_bodies,
                                       self.simulation.sensor_lines,
                                       self.simulation.sim_map))
        if self.simulation.conf.animate:
            plt.draw()
            plt.pause(0.001)

        nets = []
        for genome_id, genome in genomes:
            nets.append(neat.nn.FeedForwardNetwork.create(genome, neat_config))

        self.simulation.reset()
        if self.simulation.conf.animate:
            data = self.simulation.simulate(nets, self.simulation.conf.step_count, step_callback=self.draw,
                                            callback_args=[img])
        else:
            data = self.simulation.simulate(nets, self.simulation.conf.step_count)
        self.logger.info("Fitness values: {}".format(data[0]))
        gen_data = "{}/out-data-gen-{}".format(self.log_folder, self.generation)
        self.logger.info("Saving simulation outputs to: {}".format(gen_data))
        with open(gen_data, 'wb') as f:
            dill.dump(data, f)

        # update fitness of genomes
        i = 0
        for genome_id, genome in genomes:
            genome.fitness = data[0][0, i]
            i += 1

        return genomes

    def draw(self, img):
        img.set_data(viz.get_image(self.simulation.robot_bodies,
                                   self.simulation.sensor_lines,
                                   self.simulation.sim_map))
        plt.draw()
        plt.pause(0.001)
