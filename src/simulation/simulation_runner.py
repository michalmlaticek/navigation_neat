from MyCheckpoint import MyCheckpointer
import dill
from neat.six_util import iteritems
import neat
import matplotlib.pyplot as plt
import FrameFactory as viz

###############################################
# TODO: populate with correct values
experiment_path = 'pure_distance/1524949367'
gen = 999
###############################################

log_path = '../../logs/{}'.format(experiment_path)
# log_path = 'C:/_user/_other/diplo/code/navigation_neat/logs/pure_distance/1524777070'

# Load population to run
pop = MyCheckpointer.restore_checkpoint('{}/gen-{}'.format(log_path, gen))
genomes = list(iteritems(pop.population))
neat_conf = pop.config
# initialize nets
nets = []
for genome_id, genome in genomes:
    nets.append(neat.nn.FeedForwardNetwork.create(genome, neat_conf))

# Load configuration used
exp_conf_path = '{}/experiment_conf'.format(log_path)
with open(exp_conf_path, 'rb') as f:
    exp_conf = dill.load(f)

# Load simulation
simulation = exp_conf.simulation
simulation.reset()

# prepare fig
fig = plt.Figure()
img = plt.imshow(viz.get_image(simulation.robot_bodies,
                               simulation.sensor_lines,
                               simulation.sim_map))


def draw(img, sim):
    img.set_data(viz.get_image(sim.robot_bodies,
                               sim.sensor_lines,
                               sim.sim_map))
    plt.draw()
    plt.pause(0.001)



simulation.simulate(nets, simulation.conf.step_count, step_callback=draw, callback_args=[img, simulation])

to_break = True
