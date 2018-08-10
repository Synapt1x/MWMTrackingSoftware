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

    def __init__(self, num_particles=1000):
        """constructor"""

        self.num_particles = num_particles
        self.particles = None
        self.full_frame = np.array([], dtype=np.uint8)

        self.template = np.array([], dtype=np.uint8)
        self.template_hog = None

    def initialize_particles(self, h, w, dist_noise=None, vel_noise=None):
        """
        initialize particles to randomize n-vector for each particle

        :param w: width of the video
        :param h: height of the video
        """

        self.max_h = h
        self.max_w = w

        #TODO: Need more work on playing with noise
        if dist_noise is None:
            self.dist_noise = min(h, w) / 20
        else:
            self.dist_noise = dist_noise
        self.error_noise = 0.5
        self.vel_noise = 0.1

        x_vals = np.random.uniform(0., w, size=(self.num_particles, 1))
        y_vals = np.random.uniform(0., h, size=(self.num_particles, 1))
        #y_vals = np.random.normal(90., 10, size=(self.num_particles, 1))
        #x_vals = np.random.normal(225., 10, size=(self.num_particles, 1))

        x_dot_vals = np.random.normal(0., 0.1, size=(self.num_particles, 1))
        y_dot_vals = np.random.normal(0., 0.1, size=(self.num_particles, 1))

        self.particles = np.hstack((y_vals, x_vals, x_dot_vals, y_dot_vals))
        self.weights = np.zeros(shape=self.num_particles)

        return

    def calc_error(self, start=False):
        """
        Calculate error and update weights for particles
        """

        temp_h, temp_w = self.template.shape

        means = np.zeros(self.particles.shape[1])
        cov = np.array([[self.dist_noise, 0., 0., 0.],
                        [0., self.dist_noise, 0., 0.],
                        [0., 0., self.vel_noise, 0.],
                        [0., 0., 0., self.vel_noise]])
        error = np.random.multivariate_normal(means, cov,
                                              size=self.num_particles)
        self.particles += error

        if not start:
            self.initialize_particles(self.max_h, self.max_w)

        #TODO: Add error check for determining if template likely found

        for p_i in range(len(self.particles)):

            i, j, dx, dy = self.particles[p_i]

            # apply motion model with random movement noise
            i += dx + np.random.normal(0, self.dist_noise)
            j += dy + np.random.normal(0, self.dist_noise)

            #TODO: add interpolation to scale continuous values to ints

            i, j = int(i), int(j)

            # extract from full frame the comparison frame
            try:
                comp_frame = self.full_frame[i: i + temp_h,
                                             j: j + temp_w]
                #TODO: Change method of comparison to HOG and/or segmenter
                diff = self.template - comp_frame
                mse = np.sqrt((np.sum(np.square(diff))) / (temp_h * temp_w))

                weight = np.exp(-mse / (2 * self.error_noise ** 2))
                self.weights[p_i] = weight  #1 / mse

            except ValueError:

                self.weights[p_i] = 0.0

        self.weights /= np.sum(self.weights)

    def resample(self):
        """
        resample particles
        """

        print("Max weight:", np.max(self.weights))

        #TODO: Implement sampling wheel eo speed up resampling
        new_particles = np.random.choice(range(self.num_particles),
                                         self.num_particles,
                                         p=self.weights)
        self.particles = self.particles[np.array(new_particles), :]

        #TODO: Need to add particle randomization using noise parameters

    def query(self):
        """
        query the particle filter for the estimated location
        :return: est_vec: (ndarray) - array for average particle vector
        """

        avg_x, avg_y, dx, dy = np.mean(self.particles, axis=0)

        return (avg_x, avg_y)


if __name__ == '__main__':
    print("Please run the file 'main.py'")
