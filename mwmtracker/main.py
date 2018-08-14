# -*- coding: utf-8 -*-

""" Code used for object tracking. Primary purpose is for tracking a mouse in
the Morris Water Maze experiment.

main.py: this contains the main code for running the tracking software.

"""

##############################################################################
__author__ = "Chris Cadonic"
__credits__ = ["Chris Cadonc"]
__license__ = "GPL"
__version__ = "1.0.1"
__maintainer__ = "Chris Cadonic"
__email__ = "chriscadonic@gmail.com"
__status__ = "Development"
##############################################################################


import os
import cv2
import util
import numpy as np
import yaml
from tracker import Tracker


def main():
    """
    Main code for running the tracking software.

    :return:
    """

    # get current directory to work relative to current file path
    curdir = os.path.dirname(__file__)

    # Load configuration for system
    yaml_file = os.path.join(curdir, 'config.yaml')
    with open(yaml_file, "r") as f:
        config = yaml.load(f)

    datadir = os.path.join(curdir, config['datadir'])
    outdir = os.path.join(curdir, config['outputdir'])

    # TODO: set up code for running processing using the selected tracker
    # Determine which tracker system should load
    tracker = Tracker(config)

    # initialize tracker and videos
    #all_vids = util.load_files(datadir)
    #tracker.initialize_tracker(all_vids)


if __name__ == '__main__':
    main()