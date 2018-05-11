from MyCheckpoint import MyCheckpointer
import dill
from neat.six_util import iteritems
import neat
import matplotlib.pyplot as plt
import FrameFactory as viz

###############################################
# TODO: populate with correct values
# experiment_path = 'dist_collision100_angleErr/1525298334'
# experiment_path = 'dist_collision100/1525424125'
experiment_path = 'meandist_collision100/1525807840'
gen = 1023  # Set to None for winner
top_x = 20  # Set to None for all
###############################################

log_path = '../../logs/{}'.format(experiment_path)
# log_path = 'C:/_user/_other/diplo/code/navigation_neat/logs/pure_distance/1524777070'


nets = []
if gen is not None:
    # Load population to run
    pop = MyCheckpointer.restore_checkpoint('{}/pop-gen-{}'.format(log_path, gen))
    genomes = list(iteritems(pop.population))

    # sort
    if top_x is not None:
        def cond(item):
            return item[1].fitness


        sorted_genoms = sorted(genomes, reverse=True, key=cond)
        genomes = sorted_genoms[0:top_x]

    neat_conf = pop.config
    # initialize nets
    for genome_id, genome in genomes:
        nets.append(neat.nn.FeedForwardNetwork.create(genome, neat_conf))
else:
    config_path = '{}/neat_config'.format(log_path)
    neat_conf = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                            neat.DefaultSpeciesSet, neat.DefaultStagnation,
                            config_path)
    # winner
    with open('{}/winner-gen-1499'.format(log_path), 'rb') as f:
        winner = dill.load(f)
    nets.append(neat.nn.FeedForwardNetwork.create(winner, neat_conf))
# Load configuration used
exp_conf_path = '{}/experiment_conf'.format(log_path)
with open(exp_conf_path, 'rb') as f:
    exp_conf = dill.load(f)

# Load simulation
fitness = exp_conf[2]
fitness.reset(pop_size=len(nets))
fitness.animate = True

fitness.simulate(nets)

to_break = True
