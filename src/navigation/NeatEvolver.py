import logging
import neat
from LogReporter import LogReporter
from MyCheckpoint import MyCheckpointer
import visualization as viz
import matplotlib.pyplot as plt


class NeatEvolver:
    logger = logging.getLogger('NeatEvolver')

    def __init__(self, neat_conf, generation_count, simulation):
        self.logger.info("Initializing evolver")
        self.neat_conf = neat_conf
        self.generation_count = generation_count
        self.simulation = simulation
        self.simulation.reset()
        self.generation = -1
        if simulation.conf.animate:
            self.fig = plt.Figure()
            self.img = plt.imshow(viz.get_image(self.simulation.robot_bodies,
                                                self.simulation.sensor_lines,
                                                self.simulation.map))
            plt.draw()
            plt.pause(0.001)

    def evolve(self):
        self.logger.info("Starting evolution")
        pop = neat.Population(self.neat_conf)
        # pop = neat.Checkpointer.restore_checkpoint("0321_1130/neat-checkpoint-171")

        pop.add_reporter(neat.StatisticsReporter())
        pop.add_reporter(LogReporter(True))
        pop.add_reporter(
            MyCheckpointer(folder_path=self.simulation.conf.log_folder))

        winner = pop.run(self._eval_population, self.generation_count)
        self.logger.info("winner: {}".format(winner))

        self.logger.info("End evolution")

        return winner

    def _eval_population(self, genomes, neat_config):
        self.generation += 1
        self.logger.info("Evolving generation: {}".format(self.generation))

        nets = []
        for genome_id, genome in genomes:
            nets.append(neat.nn.FeedForwardNetwork.create(genome, neat_config))

        self.simulation.reset()
        if self.simulation.conf.animate:
            fits = self.simulation.simulate(nets, self.simulation.conf.step_count, step_callback=self.draw)
        else:
            fits = self.simulation.simulate(nets, self.simulation.conf.step_count)
        self.logger.info("Fitness values: {}".format(fits))

        # update fitness of genomes
        i = 0
        for genome_id, genome in genomes:
            genome.fitness = fits[0, i]
            i += 1

        return genomes

    def draw(self):
        self.img.set_data(viz.get_image(self.simulation.robot_bodies,
                                        self.simulation.sensor_lines,
                                        self.simulation.map))
        plt.draw()
        plt.pause(0.001)
