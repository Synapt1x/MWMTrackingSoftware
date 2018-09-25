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

        elif model_type == 'opencv':
            # import and create opencv tracker
            from opencvtrackers.cvtrackers import CVTracker as Model

            # initialize model tracker
            self.model = Model(self.config)
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
            self.template = cv2.imread(self.data['template_img'][0])

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

        # check if template has been defined
        if not self.template_defined and self.num_vids > 0:
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

            # save image to folder
            img_name = self.config['outputdir'] + os.sep \
                       + self.current_vid_name.split('.')[0] + '.jpg'
            self.video_writer.save_image(self.data['x'], self.data['y'],
                                         self.first_frame,
                                         img_name)

            self.data['x'], self.data['y'], self.data['t'] = [], [], []

            self.current_vid.release()  # release the current video

            if vid_i < self.num_vids - 1:
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

    def detect_loc(self, frame):
        """
        Use the chosen method to detect the location of the mouse
        :param frame: (ndarray) - image as numpy array
        :return:
        """

        if self.config['tracker'] == 'template_match':

            max_detection = 0

            if self.config['boundPool']:
                frame = frame[int(self.pool_rect[1]):
                              int(self.pool_rect[1] + self.pool_rect[3]),
                        int(self.pool_rect[0]):
                        int(self.pool_rect[0] + self.pool_rect[2])]

            for rotation in [0, 45, 90, 135, 180, 225]:

                # rotate the template to check for other orientations
                rotation_mtx = cv2.getRotationMatrix2D((self.w // 2, self.h //2),
                                                       rotation, 1)
                new_width = int((self.h * np.abs(rotation_mtx[0, 1]))
                                + (self.w * np.abs(rotation_mtx[0, 0])))
                new_height = int((self.h * np.abs(rotation_mtx[0, 0]))
                                 + (self.w * np.abs(rotation_mtx[0, 1])))

                rotation_mtx[0, 2] += (new_width / 2) - self.w // 2
                rotation_mtx[1, 2] += (new_height / 2) - self.h // 2

                new_template = cv2.warpAffine(self.template, rotation_mtx,
                                              (new_height, new_width))

                # test current rotation using template matching
                template_vals = cv2.matchTemplate(frame, new_template,
                                                  eval(self.config['template_ccorr']))
                min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(template_vals)

                # if max val is found
                if max_val > self.config['template_thresh'] and max_val > \
                        max_detection:
                    max_detection = max_val
                    w, h = new_width // 2, new_height // 2
                    x_val, y_val = max_loc[0] + w, max_loc[1] + h

            if max_detection == 0:
                return False, None, None

            if self.config['boundPool']:
                x_val += self.pool_rect[0]
                y_val += self.pool_rect[1]

            return True, x_val, y_val

        elif self.config['tracker'] == 'canny':

            if self.config['boundPool']:
                frame = frame[int(self.pool_rect[1]):
                              int(self.pool_rect[1] + self.pool_rect[3]),
                        int(self.pool_rect[0]):
                        int(self.pool_rect[0] + self.pool_rect[2])]

            # use Canny detector to find edges and find contours to find all
            # detected shapes
            edge_frame = cv2.Canny(frame, threshold1=self.config['threshold1'],
                                   threshold2=self.config['threshold2'])

            contour_frame, contours, hierarchy = cv2.findContours(edge_frame,
                                        cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            found = False

            for contour in contours:
                # extract moments and features of detected contour
                moments = cv2.moments(contour)
                area = cv2.contourArea(contour)
                arc_length = cv2.arcLength(contour, True)

                if moments['m00'] == 0:
                    continue

                areaCheck = self.config['minArea'] < area < self.config[
                    'maxArea']
                arcCheck = self.config['minArcLength'] < arc_length < \
                           self.config['maxArcLength']

                if areaCheck and arcCheck:
                    print("area:", arc_length)
                    found = True
                    x = int(moments['m10'] / moments['m00'])
                    y = int(moments['m01'] / moments['m00'])
                    break
                else:
                    continue

            if found:
                if self.config['boundPool']:
                    x += self.pool_rect[0]
                    y += self.pool_rect[1]

                return True, x, y

            else:

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

            self.current_pos = (x, y)

            # UL_corner = (int(box[0]), int(box[1]))
            # BR_corner = (int(box[0] + box[2]), int(box[1] + box[3]))
            # cv2.rectangle(frame, UL_corner, BR_corner, (255,0,0), 2, 1)
        else:
            cv2.putText(frame, "Tracking failure detected", (100, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)


if __name__ == '__main__':
    print('Please run the program by running main.py')
