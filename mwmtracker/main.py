# -*- coding: utf-8 -*-

""" Code used for object tracking. Primary purpose is for tracking a mouse in
the Morris Water Maze experiment.

main.py: this contains the main code for running the tracking software.

"""

##############################################################################
__author__ = "Chris Cadonic"
__credits__ = ["Chris Cadonic"]
__license__ = "GPL"
__version__ = "1.0.1"
__maintainer__ = "Chris Cadonic"
__email__ = "chriscadonic@gmail.com"
__status__ = "Development"
##############################################################################


import os
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

    # update data directories to current working dir
    config['datadir'] = os.path.join(curdir, config['datadir'])
    config['templatedir'] = os.path.join(curdir, config['templatedir'])
    config['outputdir'] = os.path.join(curdir, config['outputdir'])
    config['imagedir'] = os.path.join(curdir, config['imagedir'])
    config['tracking_excel'] = os.path.join(config['outputdir'], config[
        'tracking_excel'])
    config['raw_data'] = os.path.join(config['outputdir'], config[
        'raw_data'])

    # Determine which tracker system should load
    tracker = Tracker(config)
    tracker.initialize_tracker()

    # initialize tracker and videos
    tracker.process_videos()


if __name__ == '__main__':
    main()