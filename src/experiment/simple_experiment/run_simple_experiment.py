import os
import logging
import init_logging as log
from map import Map, MapFactory
from Robot import Robot
from simulation import Simulation
from SimulationConf import SimulationConf
from neatevolver import NeatEvolver
import neat
from datetime import datetime
import random
from pathlib import Path

# Define run ID
run_id = datetime.now().timestamp()

# Initialize random seed
random.seed(run_id)

log_path = Path('../../logs/simple_experiment/{}'.format(run_id))
log_path.mkdir(parents=True, exist_ok=True)

# Initialize logging
log.init(log_path, 'experiment.log')
logger = logging.getLogger('simple_experiment')
logger.info("Starting experiment with id: {}".format(run_id))

# Load the config file, which is assumed to live in
# the same directory as this script.
local_dir = os.path.dirname(__file__)
config_path = os.path.join(local_dir, 'config')
neat_config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                          neat.DefaultSpeciesSet, neat.DefaultStagnation,
                          config_path)
logger.info("NEAT Config: {}".format(neat_config))

map = MapFactory.init_basic(250, 250, 2, 30, 60, [40, 40], [210, 210])
robot = Robot(radius=15,
              sensor_angles_deg=[-60, -40, -20, 0, 20, 40, 60],
              sensor_len=35.0,
              max_speed=15.0)

simulation_conf = SimulationConf(id=run_id,
                                 robot=robot,
                                 init_rotation=0.0,
                                 map=map,
                                 step_count=500,
                                 pop_size=neat_config.pop_size,
                                 animate=True,
                                 log_folder=log_path)
logger.info("Simulation config: {}".format(simulation_conf))

simulation = Simulation(simulation_conf)

evolver = NeatEvolver(neat_config, 500, simulation)
winner = evolver.evolve()

# Display the winning genome.
print('\nBest genome:\n{!s}'.format(winner))
logging.info('Finished')
