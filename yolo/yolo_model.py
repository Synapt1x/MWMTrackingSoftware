#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Code used for object tracking. Primary purpose is for tracking a mouse in
the Morris Water Maze experiment.

cnn.py: this file contains the code for implementing a convolutional neural
network, specifically implementing the YOLO algorithm, to detect a mouse
location during swimming during the tracking.

"""

import tensorflow as tf


class Yolo_Model:
    """
    Convolutional Neural Network class
    """

    def __init__(self):
        """constructor"""

        self.model = self.create_model()

    def initialize_parameters(self):
        """
        initialize parameters for YOLO model

        """

    def create_model(self):
        """
        create model

        :return: return the model
        """

        #TODO: Build YOLO model using tf

    def train(self):
        """
        train the neural network
        """

        #TODO: Train YOLO model

        pass

    def query(self):
        """
        query the neural network to find output
        :return:
        """

        return


if __name__ == '__main__':
    print("Please run the file 'main.py'")
