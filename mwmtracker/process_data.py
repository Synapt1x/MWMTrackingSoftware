# -*- coding: utf-8 -*-

""" Code used for manipulating and outputting data.

process_data.py: this contains code for loading, formatting, processing and
exporting data to its final state.

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
import yaml
from data_processor import Data_processor


def process_data():
    """
    Code for running data processor for formatting and saving data.

    :return:
    """

    # get current directory to work relative to current file path
    curdir = os.path.dirname(__file__)

    # Load configuration for system
    yaml_file = os.path.join(curdir, 'config.yaml')
    with open(yaml_file, "r") as f:
        config = yaml.load(f)

    config = {**config, **config[config['tracker']]}
    config['datadir'] = os.path.join(curdir, config['datadir'])
    config['templatedir'] = os.path.join(curdir, config['templatedir'])
    config['outputdir'] = os.path.join(curdir, config['outputdir'])
    config['imagedir'] = os.path.join(curdir, config['imagedir'])
    config['tracking_excel'] = os.path.join(config['outputdir'], config[
        'tracking_excel'])
    config['raw_data'] = os.path.join(config['outputdir'], config[
        'raw_data'])

    data = Data_processor(config['tracking_excel'],
                          config['raw_data'],
                          config['cm_scale'])

    if config['add_tracking_times']:
        data.merge_tracking_data_times(config['datadir'])

    # from previous tracking effort
    target_bounds = [610, 243, 654, 286]  # form: [min_x, min_y, max_x, max_y]
    quadrants = [736, 368]  # form: [mid_x, mid_y]

    data.compute_annulus_crossing_index(target_bounds, quadrants)

    data.fix_prelim_data()


if __name__ == '__main__':
    process_data()