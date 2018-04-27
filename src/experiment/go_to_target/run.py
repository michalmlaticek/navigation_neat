import os
import logging
import init_logging as log
from map import Map, MapFactory
from Robot import Robot
from simulation_allow_collision import Simulation, SimulationConf
from neatevolver import NeatEvolver
import neat
import time
import random
from pathlib import Path
from experiment_conf import ExperimentConf
import dill

# TODO: set values
experiment_id = "go_to_target"
run_id = round(time.time())  # or some other num value

gen_count = 300

# Initialize random seed
random.seed(run_id)

log_path = Path('../../logs/{}/{}'.format(experiment_id, run_id))
log_path.mkdir(parents=True, exist_ok=True)  # create path if not exist

# Initialize logging
log.init(log_path, 'experiment.log')
logger = logging.getLogger('run')
logger.info("Starting experiment: {} - with id: {}".format(experiment_id, run_id))

###################################################################################################
# Load the neat config file, which is assumed to live in the same directory as this script.
config_path = os.path.join(os.path.dirname(__file__), 'config')
neat_conf = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                        neat.DefaultSpeciesSet, neat.DefaultStagnation,
                        config_path)
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
logger.info("Simulation config: {}".format(simulation_conf))

simulation = Simulation(simulation_conf)
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

# Display the winning genome.
print('\nBest genome:\n{!s}'.format(winner))
logging.info('Finished')
