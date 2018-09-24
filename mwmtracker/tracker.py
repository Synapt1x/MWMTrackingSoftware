# -*- coding: utf-8 -*-

""" Code used for object tracking. Primary purpose is for tracking a mouse in
the Morris Water Maze experiment.

tracker.py: this houses the primary functionality for storing, processing and
tracking videos using a specific tracking model.

"""
import util
import cv2
import os
import numpy as np
from data_processor import Data_processor


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
        self.data = {'data': Data_processor(config['outputExcel']),
                     'x': [],
                     'y': [],
                     't': []}

        # running parameters for the tracker
        self.vid_num = -1
        self.num_vids = 0
        self.locations = []
        self.current_pos = []
        self.current_vid = None
        self.current_vid_name = ''
        self.template = None
        self.w, self.h = 0, 0
        self.max_frames = 0
        self.max_length = 0
        self.t = 0

        # initialize model for tracking
        self.init_model(model_type=config['tracker'])

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
        self.current_vid_name = vid_name.split(os.sep)[-1].split('.')[0]
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
            self.template_rect = rect
            self.template = frame[rect[1] : (rect[1] + rect[3]),
                            rect[0] : (rect[0] + rect[2]), :]
            self.w, self.h = self.template.shape[:2]

        cv2.destroyAllWindows()

    def process_videos(self):
        """
        Process each video in the train video directory.
        """

        # check if template has been defined
        if not self.template and self.num_vids > 0:
            self.extract_template()

        #cv2.imshow('template chosen', self.template)
        #cv2.waitKey(0)

        # loop over each video in training set
        for vid_i in range(self.num_vids):

            init_vid = False

            # process current video
            valid, frame = self.current_vid.read()

            # while frames have successfully been extracted
            while valid:
                if init_vid:
                    self.process_frame(frame)
                else:
                    init_vid = self.process_initial_frame(frame)

                self.t = self.current_vid.get(cv2.CAP_PROP_POS_MSEC) / 1000

                valid, frame = self.current_vid.read()

            # save data to pandas dataframe and write to excel
            self.data['data'].save_frame(self.current_vid_name,
                                         self.data['x'],
                                         self.data['y'],
                                         self.data['t'])
            self.load_next_vid()

        self.data['data'].save_to_excel(self.config['datadir'].split(
            os.sep)[-1])

    def extract_detect_img(self):
        """
        Provided a bounding box, return an image of the detection
        """

        #TODO: Finish extraction of image

    def update_template(self):
        """
        Update template to track mouse using adaptive template.
        """

        # if template not defined,, extract one
        if not self.template_defined:
            self.extract_template()
            return

        # morph template otherwise
        detection = self.extract_detect_img()
        self.template = self.config['alpha'] * detection\
                        + (1 - self.config['alpha']) * self.template

    def process_initial_frame(self, frame):
        """
        Process initial frame to find ideal location for template.
        """

        # If template is not defined then extract a new one
        if self.template is None:
            self.extract_template()

        # find max correlation with template to find likely location
        template_vals = cv2.matchTemplate(frame, self.template,
                                          eval(self.config['template_ccorr']))
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(template_vals)

        # initialize model and save centroid location to data
        self.model.model.init(frame, self.template_rect)

        # use thresholding to determine if template is located
        if max_val > self.config['template_thresh']:
            w, h = self.w // 2, self.h // 2
            self.current_pos = (max_loc[0] + w, max_loc[1] + h)

            # initialize tracking data for current video
            self.data['x'].append(self.current_pos[0])
            self.data['y'].append(self.current_pos[1])
            self.data['t'].append(self.t)

            if self.config['verbose']:
                cv2.circle(img=frame, center=self.current_pos, radius=2,
                           color=[0, 255, 0], thickness=2)
            return True
        else:
            return False

        # draw bounding box if a match if found

    def process_frame(self, frame):
        """
        Process frame using selected tracker model.
        """

        valid, box = self.model.model.update(frame)
        if valid:
            # Tracking success
            UL_corner = (int(box[0]), int(box[1]))
            BR_corner = (int(box[0] + box[2]), int(box[1] + box[3]))
            cv2.rectangle(frame, UL_corner, BR_corner, (255,0,0), 2, 1)

            self.data['x'].append(self.current_pos[0])
            self.data['y'].append(self.current_pos[1])
            self.data['t'].append(self.t)
        else:
            cv2.putText(frame, "Tracking failure detected", (100,80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75,(0,0,255),2)

if __name__ == '__main__':
    print('Please run the program by running main.py')
