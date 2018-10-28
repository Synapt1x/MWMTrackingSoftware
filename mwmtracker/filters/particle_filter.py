# -*- coding: utf-8 -*-

""" Code used for object tracking. Primary purpose is for tracking a mouse in
the Morris Water Maze experiment.

particle_filter.py: this contains the code for the Particle Filter
implementation for use in the tracking software.

"""
import numpy as np
import cv2
from detectors.simple_detector import SimpleDetector


class ParticleFilter:
    """
    Particle filter class
    """

    def __init__(self, config):
        """constructor"""

        self.num_particles = config['num_particles']
        self.config = {**config, **config[config['detector']]}
        self.particles = None
        self.full_frame = np.array([], dtype=np.uint8)
        self.template_vals = np.array([])

        self.template_hog = None
        self.detector = None

    def initialize(self, h, w, dist_noise=None, vel_noise=None, detector=None):
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
        self.vel_noise = 0.5

        # x_vals = np.random.uniform(0., w, size=(self.num_particles, 1))
        # y_vals = np.random.uniform(0., h, size=(self.num_particles, 1))
        y_vals = np.random.normal(h // 2., h // 6, size=(self.num_particles,
                                                         1))
        x_vals = np.random.normal(w // 2, w // 6, size=(self.num_particles,
                                                         1))
        x_vals[x_vals < 0] = 0
        x_vals[x_vals > w] = w
        y_vals[y_vals < 0] = 0
        y_vals[y_vals > h] = h

        x_dot_vals = np.random.normal(0., 0.1, size=(self.num_particles, 1))
        y_dot_vals = np.random.normal(0., 0.1, size=(self.num_particles, 1))

        self.particles = np.hstack((y_vals, x_vals, x_dot_vals, y_dot_vals))
        self.weights = np.zeros(shape=self.num_particles)

        # initialize the detector used to measure error during each frame
        if self.config['detector'] in ['Canny', 'template']:
            self.detector = SimpleDetector(self.config[self.config['detector']],
                                           self.config['detector'])
        elif self.config['detector'] == 'cnn':
            # import and create custom cnn tracker
            from cnns.custom_cnn import CustomModel as Detector

            # initialize model tracker
            self.detector = Detector(self.config)
            self.detector.initialize()

        return

    def calc_error(self, i, j):
        """
        Calculate error and update weights for particles
        """

        err = self.template_vals[i, j]

        if err == 1.0:
            return 0.0

        return np.exp(-err / (2 * self.error_noise ** 2))

    def process_frame(self, template=None):

        if self.config['detector'] == 'template':
            temp_h, temp_w = template.shape[:2]
        else:
            temp_h, temp_w = self.config['img_size'], self.config['img_size']
            half_h, half_w = temp_h // 2, temp_w // 2

        means = np.zeros(self.particles.shape[1])
        cov = np.array([[self.dist_noise, 0., 0., 0.],
                        [0., self.dist_noise, 0., 0.],
                        [0., 0., self.vel_noise, 0.],
                        [0., 0., 0., self.vel_noise]])
        error = np.random.multivariate_normal(means, cov,
                                              size=self.num_particles)
        self.particles += error

        if self.config['detector'] in ['Canny', 'template']:
            valid, self.template_vals = self.detector.detect(self.full_frame,
                                                        template, True)
            self.template_vals = self.template_vals[:-1, :-1]
            self.template_vals[self.template_vals > 0.1] = 1.0

            # cv2.imshow('TEMPLATE VALS', self.template_vals)
            # cv2.waitKey(0)

        # TODO: Add error check for determining if template likely found

        for p_i in range(len(self.particles)):

            i, j, dx, dy = self.particles[p_i]

            # apply motion model with random movement noise
            i += dx + np.random.normal(0, self.dist_noise)
            j += dy + np.random.normal(0, self.dist_noise)

            # TODO: add interpolation to scale continuous values to ints

            i, j = int(i), int(j)

            # extract from full frame the comparison frame
            if 0 + half_h < i < self.max_h - temp_h and 0 + half_w \
                    < j < self.max_w - half_w:
                comp_frame = self.full_frame[i - half_h: i + half_h,
                                             j - half_w: j + temp_w]

                # TODO: Change method of comparison to HOG and/or segmenter
                # diff = template - comp_frame
                # mse = np.sqrt((np.sum(np.square(diff))) / (temp_h * temp_w))

                # weight = np.exp(-err / (2 * self.error_noise ** 2))
                weight = self.calc_error(i, j)
                self.weights[p_i] = weight  # 1 / mse

            else:

                self.weights[p_i] = 0.0

        self.weights /= np.sum(self.weights)

    def resample(self):
        """
        resample particles
        """

        print("Max weight:", np.max(self.weights))

        #TODO: Implement sampling wheel to speed up resampling
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

        return avg_x, avg_y


if __name__ == '__main__':
    print("Please run the file 'main.py'")
