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

    def initialize(self, max_h, max_w, h=None, w=None, start_h=0, start_w=0,
                   dist_noise=None, vel_noise=None, detector=None,
                   mouse_params=None):
        """
        initialize particles to randomize n-vector for each particle

        :param w: width of the video
        :param h: height of the video
        """

        self.max_h = max_h
        self.max_w = max_w

        if h is None:
            h = max_h
        if w is None:
            w = max_w

        #TODO: Need more work on playing with noise
        if dist_noise is None:
            self.dist_noise = min(h, w) / 4
        else:
            self.dist_noise = dist_noise
        self.error_noise = 0.5
        self.vel_noise = 0.5

        # x_vals = np.random.uniform(0., w, size=(self.num_particles, 1))
        # y_vals = np.random.uniform(0., h, size=(self.num_particles, 1))
        y_vals = np.random.normal(start_h + h // 2,
                                  h // 10,
                                  size=(self.num_particles, 1)).astype(np.int)
        x_vals = np.random.normal(start_w + w // 2,
                                   w // 10,
                                   size=(self.num_particles, 1)).astype(np.int)
        x_vals[x_vals < start_w] = start_w
        x_vals[x_vals > (start_w + w)] = start_w + w
        y_vals[y_vals < start_h] = start_h
        y_vals[y_vals > (start_h + h)] = start_h + h

        x_dot_vals = np.random.normal(0., 0.1, size=(self.num_particles, 1))
        y_dot_vals = np.random.normal(0., 0.1, size=(self.num_particles, 1))

        self.particles = np.hstack((y_vals, x_vals, x_dot_vals, y_dot_vals))
        self.weights = np.zeros(shape=self.num_particles)

        # initialize the detector used to measure error during each frame
        if self.config['detector'] in ['canny', 'template']:
            self.detector = SimpleDetector(self.config[self.config['detector']],
                                           self.config['detector'],
                                           mouse_params=mouse_params)
        elif self.config['detector'] == 'cnn':
            # import and create custom cnn tracker
            from cnns.custom_cnn import CustomModel as Detector

            # initialize model tracker
            self.detector = Detector(self.config[self.config['detector']])
            if not self.detector.initialized:
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

    def process_frame(self, frame=None, template=None,
                      start_h=None, start_w=None):

        all_locs = None

        if self.config['detector'] == 'template':
            temp_h, temp_w = template.shape[:2]
            half_h, half_w = round(temp_h / 2), round(temp_w / 2)
        elif self.config['detector'] == 'cnn':
            temp_h, temp_w = self.config['img_size'], self.config['img_size']
            half_h, half_w = round(temp_h / 2), round(temp_w / 2)
        else:
            half_h, half_w = self.config['bound_size'] // 2, self.config[
                'bound_size'] // 2

        means = np.zeros(self.particles.shape[1])
        cov = np.array([[self.dist_noise, 0., 0., 0.],
                        [0., self.dist_noise, 0., 0.],
                        [0., 0., self.vel_noise, 0.],
                        [0., 0., 0., self.vel_noise]])
        error = np.random.multivariate_normal(means, cov,
                                              size=self.num_particles)
        self.particles += error

        if self.config['detector'] == 'template':
            valid, self.template_vals = self.detector.detect(self.full_frame,
                                                        template, True)
            self.template_vals = self.template_vals[:-1, :-1]
            #self.template_vals[self.template_vals > 0.1] = 1.0

        if self.config['detector'] == 'canny':
            if frame is not None:
                valid, all_locs, _ = self.detector.detect(frame,
                                                          keep_all_locs=True,
                                                          check_color=False)
                if not valid:
                    valid, all_locs, _ = self.detector.detect(self.full_frame,
                                                              keep_all_locs=True)
            else:
                valid, all_locs, _ = self.detector.detect(self.full_frame,
                                                          keep_all_locs=True)

        for p_i in range(len(self.particles)):

            j, i, dx, dy = self.particles[p_i]

            # apply motion model with random movement noise
            i += dy + np.random.normal(0, self.dist_noise)
            j += dx + np.random.normal(0, self.dist_noise)

            # TODO: add interpolation to scale continuous values to ints

            i, j = int(i), int(j)

            # extract from full frame the comparison frame
            if 0 + half_h < i < self.max_h - half_h and 0 + half_w \
                    < j < self.max_w - half_w:
                comp_frame = frame[j - half_h: j + half_h,
                                   i - half_w: i + half_w]

                if self.config['detector'] == 'cnn':
                    valid, err = self.detector.single_query(comp_frame)
                elif self.config['detector'] == 'template':
                    valid = True
                    err = self.template_vals[j, i]
                else:
                    if start_h is not None:
                        test_i = i - start_h
                        test_j = j - start_w
                        valid, err = self.detector.calc_err(test_j,
                                                            test_i, all_locs)
                    else:
                        valid, err = self.detector.calc_err(i, j, all_locs)

                if valid:
                    weight = 1 - np.exp(-err / (2 * self.error_noise ** 2))
                    #weight = 1 / err
                    #weight = np.exp(-err)
                    if abs(err) < 1E-9:
                        weight = 0.0
                    # weight = self.calc_error(i, j)
                else:
                    weight = 0.0
                self.weights[p_i] = weight  # 1 / mse

            else:

                self.weights[p_i] = 0.0

        if np.sum(self.weights) < 1E-9:
            self.weights[:] = 1. / len(self.weights)
        else:
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
