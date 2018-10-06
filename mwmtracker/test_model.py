# -*- coding: utf-8 -*-

""" Code used for testing a specified CNN model, using valid training data
through extraction.

test_cnn.py: this contains the code for loading a video and testing the model
on classifying a selected ROI as mouse or not.

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
import sys
import cv2
import numpy as np
import util
import yaml


def test_cnn():
    """
    Main code for testing the designated neural network model.

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

    config['load_weights'] = True

    if config['tracker'] == 'yolo':
        # import and create yolo tracker
        from cnns.yolo import Yolo as Model

        # initialize model tracker
        model = Model(config)
        model.initialize()

    else:
        # default to importing and creating custom cnn tracker
        from cnns.custom_cnn import CustomModel as Model

        # initialize model tracker
        model = Model(config)
        model.initialize()

    data_vids = util.load_files(config['datadir'])
    num_vids = len(data_vids)

    for num in range(config['num_train_vids']):
        print("num:", num, "total num:", config['num_train_vids'])

        img_size = config['img_size']
        i = np.random.randint(num_vids - 1)

        video = cv2.VideoCapture(data_vids[i])

        util.test_model(model, img_size, video, config['traindir'])


if __name__ == '__main__':
    test_cnn()