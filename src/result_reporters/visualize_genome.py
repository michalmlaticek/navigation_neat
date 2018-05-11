import neat
from neat.six_util import iteritems
from MyCheckpoint import MyCheckpointer
from GenomeVisualizer import GenomeVisualizer

root_path = '../logs/meandist_collision100/1525807840'
generation = 1029
top_x = 5

config_path = '{}/neat_config'.format(root_path)
pop_path = '{}/pop-gen-{}'.format(root_path, generation)

neat_conf = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                        neat.DefaultSpeciesSet, neat.DefaultStagnation,
                        config_path)

pop = MyCheckpointer.restore_checkpoint(pop_path)
genomes = list(iteritems(pop.population))


def cond(item):
    return item[1].fitness


sorted_genoms = sorted(genomes, reverse=True, key=cond)
genomes = sorted_genoms[0:top_x]

for i in range(0, len(genomes)):
    genome_viz_file = '{}/genome-gen-{}-top-{}-viz'.format(root_path, generation, i)
    GenomeVisualizer.draw_net(neat_conf, genomes[i][1], False, filename=genome_viz_file)
