# -*- coding: utf-8 -*-

""" Code used for object tracking. Primary purpose is for tracking a mouse in
the Morris Water Maze experiment.

tracker.py: this houses the primary functionality for storing, processing and
tracking videos using a specific tracking model.

"""
import util
import cv2
import numpy as np


TEMPLATE_JUMP = 25


class Tracker:

    def __init__(self, config):
        """ Constructor """

        # store configuration yaml
        self.config = config

        # initialize data storage
        self.train_vids = []
        self.validation_vids = []
        self.test_vids = []
        self.data = {}

        # running parameters for the tracker
        self.vid_num = -1
        self.num_vids = 0
        self.current_vid = None
        self.template = None

        # initialize model for tracking
        self.init_model(model_type = config['tracker'])

    def init_model(self, model_type='yolo', model_config=None):
        """
        Build, initialize, and store model in Tracker object.

        args: model_type - (str) specific model to be loaded default 'yolo'
              model_config - (dict) dict outlining parameters for model
        """

        # use defaults if no config file is passed in
        if model_config == None:
            model_config = self.config[model_type]

        if model_type == 'pfilter':
            # import and create particle filter
            from filters.particle_filter import ParticleFilter as Model

        elif model_type == 'yolo':
            # import and create yolo tracker
            from cnns.yolo import Yolo as Model

        elif model_type == 'opencv':
            # import and create opencv tracker
            from opencvtrackers.cvtrackers import CVTracker as Model

        # initialize model tracker
        self.model = Model(model_config)
        self.model.initialize()

    def initialize_tracker(self):
        """
        Initialize tracker and load relevant data.
        """

        # load videos and template if available
        self.data['train_vids'] = util.load_files(self.config['datadir'])
        self.data['template_img'] = util.load_files(self.config['templatedir'])

        # assess nature of videos loaded
        self.num_vids = len(self.data['train_vids'])
        if len(self.data['template_img']) > 0:
            self.template_defined = True

        # load initial video
        if self.vid_num < (self.num_vids - 1):
            self.load_next_vid()

    def load_next_vid(self):
        """
        Load next video.
        """

        self.vid_num += 1
        vid_name = self.data['train_vids'][self.vid_num]
        self.current_vid = cv2.VideoCapture(vid_name)

    def extract_template(self):
        """
        Prompt user to select ROI for initial template of the mouse.
        """

        valid, frame = self.current_vid.read()
        INIT_RECT = (0, 0, 0, 0)
        rect = INIT_RECT

        while valid and rect == INIT_RECT:
            # ask user for bounding box
            rect = cv2.selectROI('Template', frame)

            if rect != INIT_RECT:
                break
            else:
                # load next jump frame
                [self.current_vid.read() for _ in range(TEMPLATE_JUMP)]
                valid, frame = self.current_vid.read()

        if rect != INIT_RECT:
            self.template = frame[rect[1] : (rect[1] + rect[3]),
                            rect[0] : (rect[0] + rect[2]), :]

        cv2.destroyAllWindows()

    def process_videos(self):
        """
        Process each video in the train video directory.
        """

        # check if template has been defined
        if not self.template and self.num_vids > 0:
            self.extract_template()

        cv2.imshow('template chosen', self.template)
        cv2.waitKey(0)

        # loop over each video in training set
        for vid_i in range(self.num_vids):

            init_vid = False

            # process current video
            valid, frame = self.current_vid.read()

            while valid:
                if not init_vid:
                    self.process_initial_frame(frame)
                else:
                    self.process_frame()

                valid, frame = self.current_vid.readFrame()

            self.load_next_vid()

    def update_template(self):
        """
        Update template to track mouse using adaptive template.
        """

        #TODO: Adapt template

    def process_initial_frame(self, frame):
        """
        Process initial frame to find ideal location for template.
        """

        #TODO: process first frame


    def process_frame(self):
        """
        Process frame using selected tracker model.
        """

        #TODO: Process frame


if __name__ == '__main__':
    print('Please run the program by running main.py')
