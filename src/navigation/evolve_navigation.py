import utils
import math
import numpy as np
import os
import neat
from simulation import Simulation
import visualization as viz
from PIL import Image
import matplotlib.pyplot as plt
# import matplotlib.animation as animation
from array2gif import write_gif

'''
Initialize test environment:
    1. map
    2. robot properties
    3. put it together in experiment config
'''

map_size = 250
border_size = 46

# map = np.ones((map_size, map_size))
# map[100:151, 100:151] = 0.
# map[0:border_size, :] = 0.
# map[:, 0:border_size] = 0.
# map[map_size - border_size:, :] = 0.
# map[:, map_size - border_size:] = 0.

map_img = Image.open("../map_generator/maps5/map_4_path_1_flip.png")
map = np.array(map_img)[:, :, 0]

training_worlds = []
training_worlds.append(utils.Map(np.array(map_img)[:, :, 0], np.array([[[40, 40]]]), np.array([[[210, 210]]])))

# for map_id in range(0, 50):
#     for path_id in range(1, 4):
#         map_img = Image.open("../map_generator/maps5/map_{}_path_{}.png".format(map_id, path_id))
#         training_worlds.append(
#             utils.Map(np.array(map_img)[:, :, 0], np.array([[[40, 40]]]), np.array([[[210, 210]]])))
#         map_img = Image.open("../map_generator/maps5/map_{}_path_{}_flip.png".format(map_id, path_id))
#         training_worlds.append(
#             utils.Map(np.array(map_img)[:, :, 0], np.array([[[40, 210]]]), np.array([[[210, 40]]])))

sensor_angles = np.array(
    [[math.radians(-60), math.radians(-30), math.radians(0), math.radians(30), math.radians(60)]]).T

robot_prop = utils.RobotConf(radius=10.0,
                             sensor_angles=sensor_angles,
                             sensor_len=35.0,
                             max_speed=10.0,
                             body=utils.build_robot_body(10))

simulation_conf = utils.SimulationConf(robot=robot_prop,
                                       init_rotation=0.0,
                                       map=training_worlds[0],  # just testing
                                       step_count=10,
                                       animate=True)
log_folder = "0319_2200"
curr_gen = 0


def save(frames, sim):
    frame = viz.get_image(sim.robot_bodies, sim.sensor_lines, sim.map)
    frames.append(viz.to_zxy(frame))
    return frame


def save_and_draw(frames, sim, img):
    frame = save(frames, sim)
    img.set_data(frame)
    plt.draw()
    plt.pause(0.0001)


# Load the config file, which is assumed to live in
# the same directory as this script.
local_dir = os.path.dirname(__file__)
config_path = os.path.join(local_dir, 'config')
config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                     neat.DefaultSpeciesSet, neat.DefaultStagnation,
                     config_path)


def eval_population(genomes, neat_config):
    global curr_gen
    nets = []
    gif_frames = []
    for genome_id, genome in genomes:
        nets.append(neat.nn.FeedForwardNetwork.create(genome, neat_config))

    simulation = Simulation(nets, simulation_conf)

    if simulation_conf.animate:
        fig = plt.Figure()
        im = plt.imshow(viz.get_image(simulation.robot_bodies, simulation.sensor_lines, simulation_conf.map))
        simulation.simulate(simulation_conf.step_count, step_callback=save_and_draw,
                            callback_args=(gif_frames, simulation, im))
    else:
        simulation.simulate(simulation_conf.step_count, step_callback=save,
                            callback_args=(gif_frames, simulation))

    # update fitens of genomes
    i = 0
    for genome_id, genome in genomes:
        genome.fitness = simulation.fitnesses[0, i]
        i += 1

    # write gif
    write_gif(gif_frames, "{}/generation_{}.gif".format(log_folder, curr_gen), fps=5)
    curr_gen += 1

    return genomes


pop = neat.Population(config)
stats = neat.StatisticsReporter()
pop.add_reporter(stats)
pop.add_reporter(neat.StdOutReporter(True))
pop.add_reporter(neat.Checkpointer(1))

winner = pop.run(eval_population, 300)

# Display the winning genome.
print('\nBest genome:\n{!s}'.format(winner))
