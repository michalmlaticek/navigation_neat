import os
import logging
import init_logging as log
from SimulationMap import MapFactory
from Robot import Robot
from pure_distance.fitness import Simulation, SimulationConf
from NeatEvolver import NeatEvolver
import neat
from datetime import datetime
import random
from pathlib import Path

# Define run ID
run_id = datetime.now().timestamp()

# Initialize random seed
random.seed(run_id)

log_path = Path('../../logs/map_4_path_2_pure_dist/{}'.format(run_id))
log_path.mkdir(parents=True, exist_ok=True)

# Initialize logging
log.init(log_path, 'experiment.log')
logger = logging.getLogger('simple_experiment')
logger.info("Starting experiment with id: {}".format(run_id))

# Load the neat_config file, which is assumed to live in
# the same directory as this script.
local_dir = os.path.dirname(__file__)
config_path = os.path.join(local_dir, 'neat_config')
neat_config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                          neat.DefaultSpeciesSet, neat.DefaultStagnation,
                          config_path)
logger.info("NEAT Config: {}".format(neat_config))

map = MapFactory.create_from_pic('../../data/maps/map_4_path_2.png', [40, 40], [210, 210])
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
logger.info("Simulation neat_config: {}".format(simulation_conf))

simulation = Simulation(simulation_conf)

evolver = NeatEvolver(neat_config, 500, simulation)
winner = evolver.evolve()

# Display the winning genome.
print('\nBest genome:\n{!s}'.format(winner))
logging.info('Finished')
