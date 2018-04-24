"""Uses `pickle` to save and restore populations (and other aspects of the simulation state)."""
from __future__ import print_function

from pathlib import Path
import random
import dill
from neat.population import Population
from neat.reporting import BaseReporter

import logging

class MyCheckpointer(BaseReporter):
    """
    A reporter class that performs checkpointing using `pickle`
    to save and restore populations (and other aspects of the simulation state).
    """

    def __init__(self, folder_path=None, filename_prefix='gen-'):
        """
        Saves the current state (at the end of a generation) every ``generation_interval`` generations or
        ``time_interval_seconds``, whichever happens first.

        :param generation_interval: If not None, maximum number of generations between save intervals
        :type generation_interval: int or None
        :param time_interval_seconds: If not None, maximum number of seconds between checkpoint attempts
        :type time_interval_seconds: float or None
        :param str filename_prefix: Prefix for the filename (the end will be the generation number)
        """

        self.logger = logging.getLogger('MyCheckpoint')
        self.folder_path = folder_path

        self.filename_prefix = filename_prefix

        self.current_generation = None

    def start_generation(self, generation):
        self.current_generation = generation

    def end_generation(self, config, population, species_set):
            self.save_checkpoint(config, population, species_set, self.current_generation)

    def save_checkpoint(self, config, population, species_set, generation):
        """ Save the current simulation state. """
        self.logger.info("Saving checkpoint - generation: {}".format(generation))
        filename = '{0}/{1}{2}'.format(self.folder_path, self.filename_prefix, generation)
        print("Saving checkpoint to {0}".format(filename))

        with open(filename, 'wb') as f:
            data = (generation, config, population, species_set, random.getstate())
            dill.dump(data, f)

    @staticmethod
    def restore_checkpoint(filename):
        """Resumes the simulation from a previous saved point."""
        with open(filename) as f:
            generation, config, population, species_set, rndstate = dill.load(f)
            random.setstate(rndstate)
            return Population(config, (population, species_set, generation))
