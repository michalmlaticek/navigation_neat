import os
import logging
import init_logging as log
from SimulationMap import MapFactory
from Robot import Robot
from fitness_collision_angleerr import Fitness
from NeatEvolver import NeatEvolver
import neat
import time
import random
from pathlib import Path
import dill
from hardlogsig import hardlogsig
from shutil import copy2
from GenomeVisualizer import GenomeVisualizer


def run():
    ###################################################################################################
    ###################################################################################################
    ###################################################################################################
    # TODO: set values
    experiment_id = "dist_collision100_angleErr"
    run_id = round(time.time())  # or some other num value
    gen_count = 1500
    step_count = 300
    animate = False
    map_path = '../../data/maps/15_cells/map_4_path_2.png'
    ###################################################################################################
    ###################################################################################################
    ###################################################################################################

    log_path = Path('../../logs/{}/{}'.format(experiment_id, run_id))
    log_path.mkdir(parents=True, exist_ok=True)  # create path if not exist
    log.init(log_path, 'experiment.log')
    logger = logging.getLogger('run')

    # Initialize random seed
    random.seed(run_id)

    ###################################################################################################
    # Load the neat neat_config file, which is assumed to live in the same directory as this script.
    # copy neat neat_config file to log folder (to make sure we know whit what config the experiment run was run)
    config_path = '{}/neat_config'.format(log_path)
    copy2('dist_collision100_angleErr/neat_config', config_path)
    # config_path = os.path.join(os.path.dirname(__file__), 'neat_config')
    neat_conf = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                            neat.DefaultSpeciesSet, neat.DefaultStagnation,
                            config_path)
    # setting custom activation function
    # neat_conf.genome_config.add_activation('hardlogsig', hardlogsig)
    logger.info("NEAT Config: {}".format(neat_conf))
    ###################################################################################################

    ###################################################################################################
    # Initialize simulation map
    if map_path is None:
        sim_map = MapFactory.init_basic(250, 250, 2, 30, 70)
    else:
        sim_map = MapFactory.create_from_pic(map_path, [40, 40], [210, 210])
    ###################################################################################################

    ###################################################################################################
    # Initialize robot
    robot = Robot(radius=15,
                  sensor_angles_deg=[-60, -40, -20, 0, 20, 40, 60],
                  sensor_len=35.0,
                  max_speed=15.0)
    ###################################################################################################

    ###################################################################################################
    # Initialize fitness
    fitness = Fitness(robot=robot,
                      init_rotation=0.0,
                      sim_map=sim_map,
                      step_count=step_count,
                      pop_size=neat_conf.pop_size,
                      animate=animate)
    ###################################################################################################

    ###################################################################################################
    # Save Experiment conf
    experiment_conf = (experiment_id, run_id, fitness, gen_count, log_path)
    exp_conf_file = '{}/experiment_conf'.format(log_path)
    logger.info("Saving experiment configuration to: {}".format(exp_conf_file))
    with open(exp_conf_file, 'wb') as f:
        dill.dump(experiment_conf, f)
    ###################################################################################################

    ###################################################################################################
    # Run evolver
    evolver = NeatEvolver(neat_conf, fitness, gen_count, log_path)
    (winner, generation) = evolver.evolve()
    ###################################################################################################

    ###################################################################################################
    # Final steps:
    # - save winning genome
    winner_file = '{}/winner-gen-{}'.format(log_path, generation)
    logger.info('Saving winning genome to: {}'.format(winner_file))
    with open(winner_file, 'wb') as f:
        dill.dump(winner, f)
    # - display the winning genome
    logger.info('\nBest genome:\n{!s}'.format(winner))
    # - save winner net visualization
    winner_viz_file = '{}-viz'.format(winner_file)
    GenomeVisualizer.draw_net(neat_conf, winner, False, filename=winner_viz_file)
    logger.info('End of experiment')


if __name__ == "__main__":
    run()
