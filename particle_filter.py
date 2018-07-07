#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Code used for object tracking. Primary purpose is for tracking a mouse in
the Morris Water Maze experiment.

particle_filter.py: this contains the code for the Particle Filter
implementation for use in the tracking software.

"""
import numpy as np


class ParticleFilter:
    """
    Particle filter class
    """

    def __init__(self, num_particles=200):
        """constructor"""

        self.num_particles = num_particles
        self.particles = None

    def initialize_particles(self, w, h):
        """
        initialize particles to randomize n-vector for each particle

        :param w: width of the video
        :param h: height of the video
        """

        x_vals = np.random.uniform(0., w, size=self.num_particles)
        y_vals = np.random.uniform(0., h, size=self.num_particles)

        x_dot_vals = np.random.normal(0., 0.1, size=self.num_particles)
        y_dot_vals = np.random.normal(0., 0.1, size=self.num_particles)

        self.particles = np.vstack((x_vals, y_vals, x_dot_vals, y_dot_vals))

        return

    def calc_error(self):
        """
        Calculate error and update weights for particles
        """

    def resample(self):
        """
        resample particles
        """



        pass

    def query(self):
        """
        query the particle filter for the estimated location
        :return: est_vec: (ndarray) - array for average particle vector
        """

        return


if __name__ == '__main__':
    print("Please run the file 'main.py'")
