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
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import numpy as np
from sklearn.utils import class_weight
from sklearn.model_selection import train_test_split
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
        self.initialized = False

    def initialize(self):
        """
        initialize parameters for YOLO model

        """

        if not self.initialized:
            if self.config['load_weights']:
                self.model = load_model(self.config['traindir'] +
                                        os.sep + self.config['fitted_weights'])
            else:
                self.model = self.create_model()

            self.initialized = True

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
        model.add(Dropout(self.config['dropout']))
        model.add(Dense(2048, activation='relu'))
        model.add(Dropout(self.config['dropout']))
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

    def load_weights(self, filename=None):
        """
        Load fitted weights for the model

        :return:
        """

        if filename is None:
            self.model = load_model(self.config['traindir'] +
                                    os.sep + self.config['fitted_weights'])
        else:
            self.model = load_model(filename)

    def compile_model(self):
        """
        Compile the model using the parameters and optimizer specified in
        yaml config file.
        :return:
        """

        opt_name = self.config['optimizer']
        if opt_name == 'adam':
            optimizer = tf.keras.optimizers.Adam(lr=1e-4)

        self.model.compile(optimizer=optimizer,
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

        seed = self.config['seed']
        test_size = self.config['validation_proportion']

        # randomly split data into train and validation sets
        train_data, valid_data, train_labels, valid_labels = train_test_split(
            train_data / 255., train_labels, test_size=test_size,
            random_state=seed)

        # reshape labels to fix rank
        train_labels = train_labels.reshape(train_labels.shape[0], 1)
        valid_labels = valid_labels.reshape(valid_labels.shape[0], 1)

        return train_data, train_labels, valid_data, valid_labels

    def train(self, train_data, train_labels, verbose=1):
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
        checkpoint_func = ModelCheckpoint(
            filepath=self.config['traindir']+os.sep+self.config[
                'output_weights'], verbose=1, save_best_only=True)

        if self.config['early_stop']:
            early_stopping = EarlyStopping(monitor='val_loss', min_delta=0,
                                           patience=self.config[
                                               'early_stop_epochs'],
                                           verbose=self.config[
                                               'early_stop_verbose'],
                                           mode='auto')
            callbacks = [checkpoint_func, early_stopping]
        else:
            callbacks = [checkpoint_func]

        # fit the model
        with tf.device('/device:GPU:0'):
            if self.config['augmentation']:
                history = self.model.fit_generator(datagen.flow(train_data,
                                                   train_labels,
                                                   batch_size=batch_size),
                                                   steps_per_epoch=train_data.shape[0] // batch_size,
                                                   epochs=epochs,
                                                   verbose=verbose,
                                                   callbacks=callbacks,
                                                   validation_data=(
                                                       valid_data,
                                                       valid_labels),
                                                   class_weight=class_weights)
            else:
                history = self.model.fit(train_data, train_labels,
                                         validation_data=(
                                         valid_data, valid_labels),
                                         epochs=epochs, batch_size=batch_size,
                                         callbacks=callbacks,
                                         verbose=verbose,
                                         class_weight=class_weights)

        with open(self.config['traindir']+os.sep+'history', 'wb') as \
                file:
            pickle.dump(history.history, file)

    def test(self, test_data, test_labels, verbose=1):
        """
        Test the model using the reserved test data along with the
        appropriate labels.

        :param test_data: (ndarray) - testing images as a tensor of shape
                                     [n, w, h, c] for n images of size w x h
                                     with c color channels
        :param test_labels: (ndarray) - testing image labels for each of n
                                       images
        :return:
        """

        predictions = self.model.predict(test_data)
        correct_predictions = 0

        for i, predict in enumerate(predictions):
            predict = predict[0]
            predict_class = 1 if predict >= 0.5 else 0
            true_val = test_labels[i]

            if predict_class == true_val:
                correct_predictions += 1

        acc = float(correct_predictions) / len(predictions)

        if verbose:
            print("\n=======================================================\n")
            print("*** Testing accuracy:", acc, "***")
            print("\n=======================================================\n")

        return acc

    def single_query(self, test_img, thresh=None):
        """
        Query the neural network to find output using a single valid input
        test image.

        :param test_img:
        :return:
        """

        h, w, c = test_img.shape

        if h == self.h and w == self.w:

            prediction = self.model.predict(np.expand_dims(test_img,
                                                           0)).astype(
                np.float32)

            if thresh is not None:
                predict_class = 1 if prediction[0][0] >= 0.5 else 0
            else:
                predict_class = prediction[0][0]

            return True, predict_class
        else:
            return False, None

    def query(self, frame):
        """
        Query the neural network to find output using a simple sliding window
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
