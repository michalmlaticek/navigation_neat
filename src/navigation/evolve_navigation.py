import utils
import math
import numpy as np
import os
import neat
import fitness
import simulation

'''
Initialize test environment:
    1. map
    2. robot properties
    3. put it together in experiment config
'''

map_size = 250
border_size = 46

map = np.ones((map_size, map_size))
map[100:151, 100:151] = 0.
map[0:border_size, :] = 0.
map[:, 0:border_size] = 0.
map[map_size - border_size:, :] = 0.
map[:, map_size - border_size:] = 0.

sensor_angles = np.array(
    [[math.radians(-60), math.radians(-30), math.radians(0), math.radians(30), math.radians(60)]]).T

robot_prop = utils.RobotConf(radius=10.0,
                             sensor_angles=sensor_angles,
                             sensor_len=20.0,
                             max_speed=10.0,
                             body=utils.build_robot_body(10))

simulation_conf = utils.SimulationConf(robot=robot_prop,
                                       init_rotation=0.0,
                                       start_point=np.array([[[65, 65]]]),
                                       end_point=np.array([[[175, 175]]]),
                                       map=map,
                                       step_count=500,
                                       animate=True)

# Load the config file, which is assumed to live in
# the same directory as this script.
local_dir = os.path.dirname(__file__)
config_path = os.path.join(local_dir, 'config')
config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                     neat.DefaultSpeciesSet, neat.DefaultStagnation,
                     config_path)


def eval_population(genomes, neat_config):
    nets = []
    for genome_id, genome in genomes:
        # genome.fitness = 1.0
        nets.append(neat.nn.FeedForwardNetwork.create(genome, neat_config))

    fitnesses = simulation.simulate(nets, simulation_conf)

    i = 0
    for genome_id, genome in genomes:
        genome.fitness = fitnesses[i]
        i += 1

    return genomes


pop = neat.Population(config)
stats = neat.StatisticsReporter()
pop.add_reporter(stats)
pop.add_reporter(neat.StdOutReporter(True))
pop.add_reporter(neat.Checkpointer(1))

winner = pop.run(eval_population, 300)

# Display the winning genome.
print('\nBest genome:\n{!s}'.format(winner))
