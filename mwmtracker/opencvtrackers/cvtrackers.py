# -*- coding: utf-8 -*-

""" Code used for object tracking using previously developed implementations
as part of opencv 3.4.1.

cvtrackers.py: this contains the code for implementing opencv as a default
tracking setup.

"""
import numpy as np
import cv2

(major_ver, minor_ver, _) = cv2.__version__.split('.')


class CVTracker:

    def __init__(self, config):
        self.config = config
        self.alg = config['algorithm']


if __name__ == '__main__':
    print("Please run the file 'main.py'")
