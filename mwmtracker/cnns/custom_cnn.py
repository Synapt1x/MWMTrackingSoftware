# -*- coding: utf-8 -*-

""" Code used for object tracking. Primary purpose is for tracking a mouse in
the Morris Water Maze experiment.

cnn.py: this file contains the code for implementing a convolutional neural
network, specifically implementing a custom convolutional neural network,
to detect a mouse location during swimming during the tracking.

"""

import tensorflow as tf
import numpy as np
import pickle
import os


class CustomModel:
    """
    Convolutional Neural Network class
    """

    def __init__(self, config):
        """constructor"""

        self.config = config
        self.h = config['h']
        self.w = config['w']

        self.input_shape = (self.h, self.w, 3)
        self.model = None

    def initialize(self):
        """
        initialize parameters for YOLO model

        """

        self.model = self.create_model()

    def create_model(self):
        """
        create model

        :return: return the model
        """

        model = tf.keras.Sequential()

        # define input layer
        model.add(tf.keras.layers.Conv2D(16, (3, 3), padding='same',
                                         activation='relu',
                                         input_shape=self.input_shape))

        # define hidden convolutional layers
        model.add(tf.keras.layers.Conv2D(32, (3, 3), padding='same',
                                         activation='relu'))
        model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
        model.add(tf.keras.layers.Conv2D(32, (3, 3), padding='same',
                                         activation='relu'))
        model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
        model.add(tf.keras.layers.Conv2D(32, (3, 3), padding='same',
                                         activation='relu'))
        model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

        # define the fully connected output layer
        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(128, activation='relu'))
        model.add(tf.keras.layers.Dropout(0.25))
        model.add(tf.keras.layers.Dense(1,
                                        kernel_initializer='normal',
                                        activation='sigmoid'))

        return model

    def view_model(self):
        """
        Print summary of the model
        :return:
        """

        self.model.summary()

    def compile_model(self):
        """
        Compile the model using the parameters and optimizer specified in
        yaml config file.
        :return:
        """

        self.model.compile(optimizer=self.config['optimizer'],
                           loss='binary_crossentropy',
                           metrics=['accuracy'])

    def split_train_data(self, train_data, train_labels):
        """
        Split training data into train/validation data for use in k-folds
        cross-validation.

        :param train_data:
        :param train_labels:
        :return:
        """

        orig_total = train_data.shape[0]
        num_valid = int(orig_total * self.config['validation_proportion'])
        num_train = orig_total - num_valid

        # randomly choose indices
        rand_indices = np.random.choice(range(num_valid),
                                        size=num_valid,
                                        replace=False)

        # create mask for separating train data into train / valid
        mask = np.ones(orig_total, dtype=bool)
        mask[rand_indices] = False

        valid_data = train_data[rand_indices, :, :, :] / 255.
        valid_labels = train_labels[rand_indices].reshape(num_valid, 1)

        train_data = train_data[mask, :, :, :] / 255.
        train_labels = train_labels[mask].reshape(num_train, 1)

        return train_data, train_labels, valid_data, valid_labels

    def train(self, train_data, train_labels, verbose):
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

        # compile the model if not done so
        self.compile_model()

        # extract validation data
        train_data, train_labels, valid_data, valid_labels = \
            self.split_train_data(train_data, train_labels)

        epochs = self.config['num_epochs']
        batch_size = self.config['batch_size']

        # create checkpointer for saving weights to output file
        checkpoint_func = tf.keras.callbacks.ModelCheckpoint(
            filepath=self.config['traindir']+os.sep+self.config[
                'output_weights'], verbose=1, save_best_only=True)

        # fit the model
        with tf.device('/device:GPU:0'):
            history = self.model.fit(train_data, train_labels,
                           validation_data=(valid_data, valid_labels),
                           epochs=epochs, batch_size=batch_size,
                           callbacks=[checkpoint_func], verbose=verbose)

        with open(self.config['traindir']+os.sep+'history', 'wb') as \
                file:
            pickle.dump(history.history, file)

    def query(self):
        """
        query the neural network to find output
        :return:
        """

        return


if __name__ == '__main__':
    print("Please run the file 'main.py'")
