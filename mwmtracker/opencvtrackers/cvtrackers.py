# -*- coding: utf-8 -*-

""" Code used for object tracking using previously developed implementations
as part of opencv 3.4.1.

cvtrackers.py: this contains the code for implementing opencv as a default
tracking setup.

"""
import numpy as np
import cv2

(major_ver, minor_ver, _) = cv2.__version__.split('.')
alg_funcs = {'kcf': cv2.TrackerKCF_create,
             'boosting': cv2.TrackerBoosting_create,
             'mil': cv2.TrackerMIL_create,
             'tld': cv2.TrackerTLD_create,
             'medianflow': cv2.TrackerTLD_create,
             'goturn': cv2.TrackerGOTURN_create,
             'mosse': cv2.TrackerMOSSE_create}


class CVTracker:

    def __init__(self, config):
        self.config = config
        self.alg = config['algorithm']

    def initialize(self):
        if int(major_ver) < 3:
            self.model = cv2.Tracker_create(self.alg)
        else:
            self.model = alg_funcs[self.alg]()


if __name__ == '__main__':
    print("Please run the file 'main.py'")
