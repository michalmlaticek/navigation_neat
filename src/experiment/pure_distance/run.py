import os
import logging
import init_logging as log
from SimulationMap import MapFactory
from Robot import Robot
from dist_collision100_angleErr.fitness_collision_angleerr import Fitness, SimulationConf
from NeatEvolver import NeatEvolver
import neat
import time
import random
from pathlib import Path
from experiment_conf import ExperimentConf
import dill
import math

# TODO: set values
experiment_id = "pure_distance"
run_id = round(time.time())  # or some other num value

gen_count = 1000

# Initialize random seed
random.seed(run_id)

log_path = Path('../../logs/{}/{}'.format(experiment_id, run_id))
log_path.mkdir(parents=True, exist_ok=True)  # create path if not exist

# Initialize logging
log.init(log_path, 'experiment.log')
logger = logging.getLogger('run')
logger.info("Starting experiment: {} - with id: {}".format(experiment_id, run_id))

###################################################################################################
# Load the neat neat_config file, which is assumed to live in the same directory as this script.
config_path = os.path.join(os.path.dirname(__file__), 'neat_config')
neat_conf = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                        neat.DefaultSpeciesSet, neat.DefaultStagnation,
                        config_path)


def roundedlogsig(x):
    return round(1 / (1 + math.exp(-x)), 4)


neat_conf.genome_config.add_activation('roundedlogsig', roundedlogsig)
logger.info("NEAT Config: {}".format(neat_conf))
###################################################################################################

###################################################################################################
# Intialize map object
sim_map = MapFactory.create_from_pic('../../data/maps/map_4_path_2.png', [40, 40], [210, 210])
robot = Robot(radius=15,
              sensor_angles_deg=[-60, -40, -20, 0, 20, 40, 60],
              sensor_len=35.0,
              max_speed=15.0)
###################################################################################################

###################################################################################################
# Initialize Simulator
simulation_conf = SimulationConf(robot=robot,
                                 init_rotation=0.0,
                                 sim_map=sim_map,
                                 step_count=250,
                                 pop_size=neat_conf.pop_size,
                                 animate=False)
logger.info("Simulation neat_config: {}".format(simulation_conf))

simulation = Fitness(simulation_conf)
###################################################################################################

###################################################################################################
# Initialize & Save Experiment conf
experiment_conf = ExperimentConf(experiment_id, run_id, neat_conf, simulation, gen_count, log_path)
exp_conf_file = '{}/experiment_conf'.format(log_path)
logger.info("Saving experiment configuration to: {}".format(exp_conf_file))
with open(exp_conf_file, 'wb') as f:
    dill.dump(experiment_conf, f)
###################################################################################################

###################################################################################################
# Run evolver

evolver = NeatEvolver(experiment_conf)
winner = evolver.evolve()

# pop = MyCheckpointer.restore_checkpoint('../../logs/pure_distance/1524842042/gen-299'.format())
# logger.info('Restoring population: ../../logs/pure_distance/1524842042/gen-299')
# winner = evolver.evolve(pop)

# Display the winning genome.
print('\nBest genome:\n{!s}'.format(winner))
logging.info('Finished')
