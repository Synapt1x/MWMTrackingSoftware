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
from video_processor import Video_processor


TEMPLATE_JUMP = 25


class Tracker:

    def __init__(self, config):
        """ Constructor """

        # store configuration yaml
        self.config = config

        # initialize video writer
        self.video_writer = Video_processor()

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
        self.orig_template = None
        self.template_defined = False
        self.w, self.h = 0, 0
        self.max_frames = 0
        self.max_length = 0
        self.t = 0
        self.pool_rect = []

        # initialize model for tracking
        self.init_model(model_type=config['tracker'])

        # initialize first frame
        self.first_frame = None

    def init_model(self, model_type='yolo', model_config=None):
        """
        Build, initialize, and store model in Tracker object.

        args: model_type - (str) specific model to be loaded default 'yolo'
              model_config - (dict) dict outlining parameters for model
        """

        # use defaults if no config file is passed in
        if model_config == None:
            self.config = {**self.config, **self.config[model_type]}
        else:
            self.config = model_config

        if model_type == 'pfilter':
            # import and create particle filter
            from filters.particle_filter import ParticleFilter as Model

            # initialize model tracker
            self.model = Model(self.config)

        elif model_type == 'yolo':
            # import and create yolo tracker
            from cnns.yolo import Yolo as Model

            # initialize model tracker
            self.model = Model(self.config)
            self.model.initialize()

        elif model_type == 'cnn':
            # import and create custom cnn tracker
            from cnns.custom_cnn import CustomModel as Model

            # initialize model tracker
            self.model = Model(self.config)
            self.model.initialize()

        elif model_type == 'opencv':
            # import and create opencv tracker
            from opencvtrackers.cvtrackers import CVTracker as Model

            # initialize model tracker
            self.model = Model(self.config)
            self.model.initialize()

        elif model_type in ['canny', 'template']:
            # import and create simple object detector
            from detectors.simple_detector import SimpleDetector as Model

            # initialize model
            self.model = Model(self.config)

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
            self.template = cv2.imread(self.data['template_img'][0])
            self.orig_template = self.template.copy()

        # load first frame of first video
        vid_name = self.data['train_vids'][self.vid_num]
        first_vid = cv2.VideoCapture(vid_name)
        valid, frame = first_vid.read()
        h, w = frame.shape[:-1]

        # reset counter and close vid
        self.vid_num = -1
        first_vid.release()

        # if the pool bounding option is set then ask user to bound the pool
        if self.config['boundPool']:

            if valid:
                self.pool_rect = cv2.selectROI(
                    'Draw a Bounding box around the pool', frame)

            cv2.destroyAllWindows()

        # initialize tracker with height and width if needed
        if self.config['tracker'] == 'pfilter':
            self.model.initialize(h, w)

        # load initial video
        if self.vid_num < (self.num_vids - 1):
            self.load_next_vid()

    def load_next_vid(self):
        """
        Load next video.
        """

        # increase vid counter and get next video name
        self.vid_num += 1
        vid_name = self.data['train_vids'][self.vid_num]
        self.current_vid_name = vid_name.split(os.sep)[-1].strip().split('.')[0]

        self.current_vid = cv2.VideoCapture(vid_name)
        print('Processing video: ', self.current_vid_name.split(os.sep)[-1])

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

        print("Do you want to save this template image?")

        cv2.destroyAllWindows()

        while(1):
            cv2.imshow('Chosen template', self.template)
            button = cv2.waitKey(100)

            if button == 32 or button == 121:
                print("Saving image...")
                cv2.imwrite(self.config['templatedir'] + os.sep +
                            self.current_vid_name + '.jpg', self.template)
                break
            elif button == -1:
                continue
            else:
                break

        cv2.destroyAllWindows()

    def process_videos(self):
        """
        Process each video in the train video directory.
        """

        # if the config indicates we need to extract train videos
        # if self.config['extract_data']:
        #     util.extract_train_data(self.config['traindir'],
        #                             self.config['img_size'],
        #                             self.current_vid)
        #
        # if self.config['load_pickle']:
        #     self.data['train_data'], self.data['train_labels'] = \
        #         util.load_train_data()

        # check if template has been defined
        if not self.template_defined and self.num_vids > 0:
            self.extract_template()

        # loop over each video in training set
        for vid_i in range(self.num_vids):

            init_vid = False

            # process current video
            valid, frame = self.current_vid.read()

            # while frames have successfully been extracted
            while valid:

                frame = util.resize_frame(frame, self.config['resize'])

                if self.config['tracker'] == 'pfilter':
                    self.model.full_frame = frame

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

            # save image to folder
            img_name = self.config['outputdir'] + os.sep \
                       + self.current_vid_name.split('.')[0] + '.jpg'
            self.video_writer.save_image(self.data['x'], self.data['y'],
                                         self.first_frame,
                                         img_name)

            self.data['x'], self.data['y'], self.data['t'] = [], [], []

            self.current_vid.release()  # release the current video

            if vid_i < self.num_vids - 1:
                self.template = self.orig_template.copy()
                self.load_next_vid()

        self.data['data'].save_to_excel(self.config['datadir'].split(
            os.sep)[-1])

    def extract_detect_img(self, frame, i, j):
        """
        Provided a bounding box, return an image of the detection
        """

        h, w = self.template.shape[:2]

        return frame[i: i + h, j: j + w]

    def update_template(self, frame, i, j):
        """
        Update template to track mouse using adaptive template.
        """

        # if template not defined,, extract one
        if not self.template_defined:
            self.extract_template()
            return

        # morph template otherwise
        detection = self.extract_detect_img(frame, i, j)
        # cv2.imshow('detection', detection)
        # cv2.waitKey(0)
        self.template = (self.config['alpha'] * detection\
                        + (1 - self.config['alpha']) *
                         self.template).astype(np.uint8)

        cv2.imshow('template', self.template)
        cv2.waitKey(1)

    def detect_loc(self, frame):
        """
        Use the chosen method to detect the location of the mouse
        :param frame: (ndarray) - image as numpy array
        :return:
        """

        if self.config['tracker'] == 'template' or self.config['tracker'] ==\
                'canny':

            return self.model.detect(frame, self.config)

        elif self.config['tracker'] == 'pfilter':

            self.model.process_frame(self.template)
            self.model.resample()
            x, y = self.model.query()

            return True, int(x), int(y)
        return False, None, None

    def process_initial_frame(self, frame):
        """
        Process initial frame to find ideal location for template.
        """

        # If template is not defined then extract a new one
        if self.template is None:
            self.extract_template()

        # detect location of the mouse
        valid, x, y = self.detect_loc(frame)

        # save the first frame
        self.first_frame = frame

        if valid:

            # initialize tracking data for current video
            self.data['x'].append(x)
            self.data['y'].append(y)
            self.data['t'].append(self.t)

            self.update_template(frame, x, y)

            if self.config['verbose']:
                cv2.circle(img=frame, center=(x, y), radius=2,
                           color=[0, 255, 0], thickness=2)
            return True
        else:
            return False

        # draw bounding box if a match if found

    def process_frame(self, frame):
        """
        Process frame using selected tracker model.
        """

        #valid, box = self.model.model.update(frame)

        valid, x, y = self.detect_loc(frame)

        if valid:
            # tracking success
            self.data['x'].append(x)
            self.data['y'].append(y)
            self.data['t'].append(self.t)

            self.update_template(frame, x, y)

            self.current_pos = (x, y)

            # UL_corner = (int(box[0]), int(box[1]))
            # BR_corner = (int(box[0] + box[2]), int(box[1] + box[3]))
            # cv2.rectangle(frame, UL_corner, BR_corner, (255,0,0), 2, 1)
        else:
            cv2.putText(frame, "Tracking failure detected", (100, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)


if __name__ == '__main__':
    print('Please run the program by running main.py')
