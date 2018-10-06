# -*- coding: utf-8 -*-

""" Code used for object tracking. Primary purpose is for tracking a mouse in
the Morris Water Maze experiment.

cnn.py: this file contains the code for implementing a convolutional neural
network, specifically implementing a custom convolutional neural network,
to detect a mouse location during swimming during the tracking.

"""

import tensorflow as tf
from tensorflow.keras.layers import Convolution2D, MaxPooling2D, \
    ZeroPadding2D, Flatten, Dense, Dropout
import numpy as np
from sklearn.utils import class_weight
import pickle
import os


class CustomModel:
    """
    Convolutional Neural Network class
    """

    def __init__(self, config):
        """constructor"""

        self.config = config
        self.h = config['img_size']
        self.w = config['img_size']

        self.input_shape = (self.h, self.w, 3)
        self.model = None

    def initialize(self):
        """
        initialize parameters for YOLO model

        """

        if self.config['load_weights']:
            self.model = tf.keras.models.load_model(self.config['traindir'] +
                                                    os.sep +
                                                self.config['fitted_weights'])
        else:
            self.model = self.create_model()

    def create_model(self):
        """
        create model

        :return: return the model
        """

        model = tf.keras.Sequential()

        # define input layer
        model.add(Convolution2D(64, (3, 3), padding='same',
                                         activation='relu',
                                         input_shape=self.input_shape))

        # define hidden convolutional layers
        model.add(Convolution2D(64, (2, 2), padding='same',
                                         activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Convolution2D(128, (2, 2), padding='same',
                                         activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Convolution2D(256, (2, 2), padding='same',
                                         activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(ZeroPadding2D(2))
        model.add(Convolution2D(512, (3, 3), activation='relu'))
        model.add(ZeroPadding2D(2))
        model.add(Convolution2D(512, (3, 3), activation='relu'))
        model.add(MaxPooling2D((2, 2), strides=(2, 2)))

        # define the fully connected output layer
        model.add(Flatten())
        model.add(Dense(2048, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(2048, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(1, kernel_initializer='normal',
                                        activation='sigmoid'))

        model.summary()

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

        orig_total = 80 # train_data.shape[0]
        num_valid = 20  #int(orig_total * self.config['validation_proportion'])
        num_train = 60  #orig_total - num_valid
        valid_labels = np.array([0] * num_valid)

        # while np.all(valid_labels == 0) or np.all(valid_labels == 1):
        #     # randomly choose indices
        #     rand_indices = np.random.choice(range(num_train),
        #                                     size=num_valid,
        #                                     replace=False)
        #
        #     # create mask for separating train data into train / valid
        #     mask = np.ones(orig_total, dtype=bool)
        #     mask[rand_indices] = False
        #
        #     valid_data = train_data[rand_indices, :, :, :] / 255.
        #     valid_labels = train_labels[rand_indices].reshape(num_valid, 1)
        #
        #     new_train_data = train_data[mask, :, :, :] / 255.
        #     new_train_labels = train_labels[mask].reshape(num_train, 1)

        valid_data = np.vstack((train_data[:10, :, :, :], train_data[-10:, :,
                                                        :, :])) / 255.
        valid_labels = np.hstack((train_labels[:10], train_labels[-10:]))

        new_train_data = np.vstack((train_data[10:40, :, :, :],
                                    train_data[-40:-10, :, :, :])) / 255.
        new_train_labels = np.hstack((train_labels[10:40],
                                      train_labels[-40:-10]))

        # reshuffle data sets to ensure randomization
        valid_shuffle = np.random.permutation(num_valid)
        train_shuffle = np.random.permutation(num_train)

        valid_data = valid_data[valid_shuffle]
        valid_labels = valid_labels[valid_shuffle]

        train_data = new_train_data[train_shuffle]
        train_labels = new_train_labels[train_shuffle]


        valid_labels = valid_labels.reshape(num_valid, 1)
        train_labels = train_labels.reshape(num_train, 1)



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
        batch_size = 5  #self.config['batch_size']

        # define class weights for unevenly represented data
        num_class = np.unique(train_labels[:, 0])
        class_weights = class_weight.compute_class_weight('balanced',
                                                          num_class,
                                                          train_labels[:, 0])

        # if augmentation is to be used
        if self.config['augmentation']:
            datagen = tf.keras.preprocessing.image.ImageDataGenerator(
                rotation_range=45,
                vertical_flip=True,
                horizontal_flip=True)
            datagen.fit(train_data)

        # create checkpointer for saving weights to output file
        checkpoint_func = tf.keras.callbacks.ModelCheckpoint(
            filepath=self.config['traindir']+os.sep+self.config[
                'output_weights'], verbose=1, save_best_only=True)

        # fit the model
        with tf.device('/device:GPU:0'):
            if self.config['augmentation']:
                history = self.model.fit_generator(datagen.flow(train_data,
                                                   train_labels,
                                                   batch_size=batch_size),
                                                   steps_per_epoch=train_data.shape[0] // batch_size,
                                                   epochs=epochs,
                                                   verbose=verbose,
                                                   callbacks=[checkpoint_func],
                                                   validation_data=(
                                                       valid_data,
                                                       valid_labels),
                                                   class_weight=class_weights)
            else:
                history = self.model.fit(train_data, train_labels,
                                         validation_data=(
                                         valid_data, valid_labels),
                                         epochs=epochs, batch_size=batch_size,
                                         callbacks=[checkpoint_func],
                                         verbose=verbose,
                                         class_weight=class_weights)

        with open(self.config['traindir']+os.sep+'history', 'wb') as \
                file:
            pickle.dump(history.history, file)

    def query(self, frame):
        """
        query the neural network to find output using a simple sliding window
        approach and distance limiting based on previous location
        :return:
        """

        stride = self.config['window_stride']
        min_h, max_h, min_w, max_w = self.config['miny'], self.config['maxy'], \
                                     self.config['minx'], self.config['maxx']

        imgs = []
        coords = []

        for i in range(min_h, max_h - self.h, stride):

            for j in range(min_w, max_w - self.w, stride):

                test_img = frame[i: i + self.h, j: j + self.w]

                imgs.append(test_img)
                coords.append((i, j))

        imgs = np.array(imgs)

        predictions = self.model.predict(imgs)
        if all(predictions == 1.0) or all(predictions == 0.0):
            return False, None, None

        first_best = np.argmax(predictions)
        best_i, best_j = coords[first_best]

        return True, best_i + self.h // 2, best_j + self.w // 2


if __name__ == '__main__':
    print("Please run the file 'main.py'")
