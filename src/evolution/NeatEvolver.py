import logging
import neat
from neat.reporting import ReporterSet
from LogReporter import LogReporter
from MyCheckpoint import MyCheckpointer
import dill


class NeatEvolver:
    logger = logging.getLogger('NeatEvolver')

    def __init__(self, neat_conf, fitness, gen_count, log_folder):
        self.logger.info("Initializing evolver")
        self.neat_conf = neat_conf
        self.gen_count = gen_count
        self.fitness = fitness

        self.fitness.reset()
        self.generation = -1
        self.log_folder = log_folder

        self.pop = neat.Population(self.neat_conf)

    def set_init_pop(self, init_pop):
        self.pop = init_pop

        self.generation = init_pop.generation

    def _init_reporters(self):
        stat_reporter = neat.StatisticsReporter()
        log_reporter = LogReporter(True)
        checkpoint_reporter = MyCheckpointer(folder_path=self.log_folder)

        self.pop.reporters = ReporterSet()
        self.pop.add_reporter(stat_reporter)
        self.pop.add_reporter(log_reporter)
        self.pop.add_reporter(checkpoint_reporter)
        self.pop.species.reporters = self.pop.reporters

    def evolve(self):
        self._init_reporters()
        winner = self.pop.run(self._eval_population, self.gen_count)
        return winner, self.generation

    def _eval_population(self, genomes, neat_config):
        self.generation += 1
        self.logger.info("Evolving generation: {}".format(self.generation))

        nets = []
        for genome_id, genome in genomes:
            nets.append(neat.nn.FeedForwardNetwork.create(genome, neat_config))

        self.logger.info("Net count: {}".format(len(nets)))
        self.fitness.reset(len(nets))
        data = self.fitness.simulate(nets)
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
