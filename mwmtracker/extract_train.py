# -*- coding: utf-8 -*-

""" Code used for extracting training data for training a custom CNN model.

extract_train.py: this contains code for running the util.extract_train_data
function specifically for populating training data.

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
import numpy as np
import util
import sys
import yaml


def train_model():
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

    config = {**config, **config[config['tracker']]}
    config['datadir'] = os.path.join(curdir, config['datadir'])
    config['h'] = config['img_size']
    config['w'] = config['img_size']

    data_vids = util.load_files(config['datadir'])
    num_vids = len(data_vids)

    if len(sys.argv) > 1:
         pickle = config['testpickle'] if sys.argv[1] == '-test' else \
             config['trainpickle']

    for num in range(config['num_train_vids']):

        print("num:", num, "total num:", config['num_train_vids'])

        img_size = config['img_size']
        i = np.random.randint(num_vids - 1)

        video = cv2.VideoCapture(data_vids[i])

        util.extract_train_data(img_size, video, config['traindir'],
                                pickle=pickle)


if __name__ == '__main__':
    train_model()