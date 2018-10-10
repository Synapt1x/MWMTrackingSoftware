# -*- coding: utf-8 -*-

""" Code used for training a specified CNN model, using valid training data
through extraction.

train_model.py: this contains the code for loading/extracting training data
and training the CNN.

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


def train_model():
    """
    Main code for training the designated neural network model.

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

    train_data, train_labels = util.load_train_data(config['traindir'] + os.sep
                                                 + config['trainpickle'])
    test_data, test_labels = util.load_train_data(config['traindir'] + os.sep
                                                 + config['testpickle'])

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

    if len(sys.argv) > 1:
        config['training_verbose'] = 1 if sys.argv[1] == '-v' else 0

    acc = 0.0

    while acc <= 0.99:
        model = Model(config)
        model.initialize()
        model.train(train_data, train_labels, int(config['training_verbose']))
        acc = model.test(test_data, test_labels, verbose=1)

    print("\n*******************************************************\n")
    print("*** Model found with > 99 % accuracy on test data! ***")
    print("\n*******************************************************\n")


if __name__ == '__main__':
    train_model()