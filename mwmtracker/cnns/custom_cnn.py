# -*- coding: utf-8 -*-

""" Code used for object tracking. Primary purpose is for tracking a mouse in
the Morris Water Maze experiment.

cnn.py: this file contains the code for implementing a convolutional neural
network, specifically implementing a custom convolutional neural network,
to detect a mouse location during swimming during the tracking.

"""

import tensorflow as tf


class CustomModel:
    """
    Convolutional Neural Network class
    """

    def __init__(self, config):
        """constructor"""

        self.config = config
        self.model = self.create_model()

    def initialize(self):
        """
        initialize parameters for YOLO model

        """

        #TODO: Initialize model
        pass

    def create_model(self):
        """
        create model

        :return: return the model
        """

        #TODO: Build model
        pass

    def train(self, train_data, train_labels):
        """
        Train the model using the provided training data along with the
        appropriate labels.

        :param train_data: (ndarray) - training images as a tensor of shape
                                     [n, w, h, c] for n images of size w x h
                                     with c color channels
        :param train_labels: (ndarray) - training image labels for each of n
                                       images
        :return:
        """

        #TODO: code training of model
        pass


    def query(self):
        """
        query the neural network to find output
        :return:
        """

        return



if __name__ == '__main__':
    print("Please run the file 'main.py'")
