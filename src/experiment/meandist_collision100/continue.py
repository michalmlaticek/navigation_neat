from MyCheckpoint import MyCheckpointer
import logging
from pathlib import Path
import init_logging as log
from NeatEvolver import NeatEvolver
import dill
from SimulationMap import MapFactory
from GenomeVisualizer import GenomeVisualizer

###################################################################################################
###################################################################################################
###################################################################################################
# Population to load
experiment_id = "dist_collision100_angleErr"
orig_run_id = '1525210220'
gen_num = 1

# Set new_run_id to run_id if you want to continue to save and write to original log folder
new_run_id = orig_run_id
# or assign a new time stamp value if you want to create a new run_id folder
# new_run_id = round(time.time())

# keep the next param as None if you want to maintain the same settings as the original experiment run
# if you want to change it, than set to desired appropriate values
new_gen_count = 2
new_step_count = None
new_animate = True
(new_map_path, new_start_point, new_end_point) = (None, None, None)  # set all or nothing
###################################################################################################
##################################################################################################
###################################################################################################

orig_log_path = Path('../../logs/{}/{}'.format(experiment_id, orig_run_id))
new_log_path = Path('../../logs/{}/{}'.format(experiment_id, new_run_id))
new_log_path.mkdir(parents=True, exist_ok=True)  # create path if not exist
log.init(new_log_path, 'experiment.log')
logger = logging.getLogger('run')

logger.info('*******************************************************************')
logger.info('Continuing experiment: {}'.format(experiment_id))
logger.info('*******************************************************************')
logger.info('Original run_id: {}'.format(orig_run_id))
logger.info('Loading generation: {}'.format(gen_num))
logger.info('New run id: {} (is same = {}'.format(new_run_id, orig_run_id == new_run_id))
logger.info('Other settings: \ngen_count = {}\nstep_count = {}\nanimate = {}\nmap_path = {}'
            .format(new_gen_count, new_step_count, new_animate, new_map_path))

logger.info('Loading original configuration')
with open('{}/experiment_conf'.format(orig_log_path), 'rb') as f:
    (experiment_id, run_id, fitness, gen_count, log_path) = dill.load(f)
if new_gen_count is not None:
    gen_count = new_gen_count
if new_step_count is not None:
    fitness.step_count = new_step_count
if new_animate is not None:
    fitness.animate = new_animate
if new_map_path is not None:
    fitness.sim_map = MapFactory.create_from_pic(new_map_path, new_start_point, new_end_point)

init_pop = MyCheckpointer.restore_checkpoint('{}/pop-gen-{}'.format(orig_log_path, gen_num))

###################################################################################################
# Continue from existing population
evolver = NeatEvolver(init_pop.config, fitness, gen_count, new_log_path)
evolver.set_init_pop(init_pop)
(winner, generation) = evolver.evolve()

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
GenomeVisualizer.draw_net(init_pop.config, winner, False, filename=winner_file)
logger.info('End of experiment')
