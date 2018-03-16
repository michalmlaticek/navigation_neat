import neat
import simulation


def eval_population(genomes, neat_config):
    nets = []
    simul_config = neat_config.simul_config
    for genome in genomes:
        nets.append(neat.nn.FeedForwardNetwork(genome, neat_config))

    fitnesses = simulation.simul(nets, simul_config)

    i = 0
    for genome in genomes:
        genome.fitness = fitnesses[i]
        i += 1

    return genomes


def eval_genome(genome, neat_config, simul_config):
    nets = [neat.nn.FeedForwardNetwork.create(genome, neat_config)]

    genome.fitness = simulation.simul(nets, simul_config)[0]

    return genome
