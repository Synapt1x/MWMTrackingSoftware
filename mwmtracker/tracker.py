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
mouse_loc = ()
mouse_params = {}


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
        self.data = {'data': Data_processor(config['outputExcel'],
                                            self.config['cm_scale']),
                     'x': [],
                     'y': [],
                     't': []}

        if self.config['saveIDs']:
            self.data['data'].write_ids(self.config['datadir'])

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

        elif model_type == 'kalman':

            self.model = cv2.KalmanFilter()
            self.model.measurementMatrix = np.array([[1, 0, 0, 0],
                                                     [0, 1, 0, 0]], np.float32)
            self.model.transitionMatrix = np.array([[1, 0, 1, 0],
                                                    [0, 1, 0, 1],
                                                    [0, 0, 1, 0],
                                                    [0, 0, 0, 1]], np.float32)
            self.model.processNoiseCov = np.eye(4, 4, dtype=np.float32)

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

        # loop over each video in training set
        for vid_i in range(self.num_vids):

            if not self.data['data'].dist_data.empty:
                vid_num = int(self.data['train_vids'][vid_i].split(" ")[-1].split(
                    ".")[0])
                if vid_num in np.array(self.data['data'].dist_data['vid num'],
                                       dtype=np.int):

                    self.current_vid.release()  # release the current video

                    if vid_i < self.num_vids - 1:
                        self.template = self.orig_template.copy()
                        self.load_next_vid()
                    continue

            init_vid = False

            frame_num = 0

            # process current video
            valid, frame = self.current_vid.read()

            if valid:
                if self.config['boundPool']:
                    self.pool_rect = cv2.selectROI(
                        'Draw a Bounding box around the pool', frame)

                    cv2.destroyAllWindows()
                    self.config['minx'] = self.pool_rect[0]
                    self.config['maxx'] = self.pool_rect[0] + \
                                          self.pool_rect[2]
                    self.config['miny'] = self.pool_rect[1]
                    self.config['maxy'] = self.pool_rect[1] + \
                                          self.pool_rect[3]

            # while frames have successfully been extracted
            while valid:
                if self.config['tracker'] != 'kalman':
                    if 'config' in self.model.config:
                        frame = util.resize_frame(frame, self.model.config['resize'])

                frame_num += 1

                print('frame num:', frame_num)

                if self.config['tracker'] == 'pfilter':
                    self.model.full_frame = frame

                if self.config['boundPool']:

                    orig_frame = frame.copy()
                    frame = frame[self.config['miny']: self.config['maxy'],
                                  self.config['minx']: self.config['maxx']]

                    if self.config['maskPool']:
                        mask = np.zeros(shape=frame.shape,
                                        dtype=frame.dtype)
                        mid_x = (self.config['minx'] + self.config['maxx']) \
                                // 2 - self.config['minx']
                        mid_y = (self.config['miny'] + self.config['maxy']) \
                                // 2 - self.config['miny']
                        mask = cv2.circle(mask, center=(mid_x, mid_y),
                                          radius=max(mid_x, mid_y),
                                          color=(255, 255, 255),
                                          thickness=cv2.FILLED)
                        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
                        frame = cv2.bitwise_and(frame, frame, mask=mask)
                        frame[np.where((frame == [0, 0, 0]).all(axis=2))] = [
                            164, 164, 164]

                if init_vid:
                    self.process_frame(frame)
                else:
                    init_vid = self.process_initial_frame(frame)
                    if self.config['boundPool']:
                        self.first_frame = orig_frame
                    else:
                        self.first_frame = frame

                self.t = self.current_vid.get(cv2.CAP_PROP_POS_MSEC) / 1000

                [self.current_vid.read() for _ in range(self.config[
                                                            'frame_skip'])]
                frame_num += self.config['frame_skip']

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

        self.data['data'].save_to_excel()

    def extract_detect_img(self, frame, j, i, h=None, w=None):
        """
        Provided a bounding box, return an image of the detection
        """

        if h == None or w == None:
            h, w = self.template.shape[:2]

        return frame[i - h // 2: i + h // 2,
                     j - w // 2: j + w // 2]

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

    def detect_loc(self, frame, start_h=None, start_w=None):
        """
        Use the chosen method to detect the location of the mouse
        :param frame: (ndarray) - image as numpy array
        :return:
        """

        if self.config['tracker'] == 'template' or self.config['tracker'] ==\
                'canny':

            return self.model.detect(frame, self.config)

        elif self.config['tracker'] == 'pfilter':

            #util.display_particles(frame, self.model.particles)

            self.model.process_frame(frame, start_h=start_h, start_w=start_w)
            self.model.resample()
            x, y = self.model.query()

            return True, int(y), int(x)

        elif self.config['tracker'] == 'cnn':

            valid, x, y = self.model.query(frame)

            if not valid:
                return False, None, None

            return True, int(x), int(y)

        return False, None, None

    def process_initial_frame(self, frame):
        """
        Process initial frame to find ideal location for template.
        """
        global mouse_params

        # If template is not defined then extract a new one
        if self.template is None:
            self.extract_template()

        if self.config['getMouseParams']:
            cv2.namedWindow("Click on the mouse")
            cv2.setMouseCallback("Click on the mouse", get_mouse_params,
                {'size': self.config['prev_bounds'],
                 'canny_params': self.config['canny'],
                 'frame': frame})
            while True:
                cv2.imshow("Click on the mouse", frame)
                key = cv2.waitKey(1) & 0xFF

                if key == ord('s'):
                    cv2.destroyWindow("Click on the mouse")
                    return False
                # if the 'c' key or space bar is pressed, break from the loop
                if key == ord('c') or key == 32:
                    break
            cv2.destroyWindow("Click on the mouse")
        else:
            mouse_params['area'] = 500
            mouse_params['circle_area'] = 2500
            mouse_params['arc_length'] = 600

        if self.config['tracker'] == 'canny':
            # re-initialize with initial mouse selection
            self.model.config = {**self.model.config, **mouse_params}

        if self.config['tracker'] == 'pfilter':
            roi = cv2.selectROI("Select initial bounds for particles",
                                frame)
            cv2.destroyWindow("Select initial bounds for particles")
            self.model.initialize(frame.shape[0], frame.shape[1],
                                  h=roi[3], w=roi[2],
                                  start_h=roi[1], start_w=roi[0],
                                  mouse_params=mouse_params)

        # detect location of the mouse
        if self.config['boundLoc']:
            roi = cv2.selectROI("Select bounds for mouse location",
                                frame)
            cv2.destroyWindow("Select bounds for mouse location")
            first_img = frame[int(roi[1]):int(roi[1]+roi[3]),
                              int(roi[0]):int(roi[0]+roi[2])]
            if self.config['getMouseParams']:
                valid, x, y = True, mouse_params['initial_x'], mouse_params[
                    'initial_y']
                if self.config['boundPool']:
                    x += self.config['minx']
                    y += self.config['miny']
            else:
                valid, x, y = self.detect_loc(first_img,
                                              start_h=int(roi[1]),
                                              start_w=int(roi[0]))
            if self.config['tracker'] != 'pfilter':
                if valid:
                    x += roi[0]
                    y += roi[1]
                    if self.config['boundPool']:
                        x += self.config['minx']
                        y += self.config['miny']
        else:
            valid, x, y = self.detect_loc(frame)

        if not valid:
            self.ask_xy(frame)
            x, y = mouse_loc

        if self.config['boundPool']:
            x += self.config['minx']
            y += self.config['miny']

        # tracking success
        self.data['x'].append(x)
        self.data['y'].append(y)
        self.data['t'].append(self.t)
        self.current_pos = [x, y]

        return valid

    def ask_xy(self, frame):

        while True:
            cv2.imshow("Cannot detect. Please click mouse location.", frame)
            cv2.setMouseCallback("Cannot detect. Please click mouse location.",
                                 fail_detect_click)
            key = cv2.waitKey(1) & 0xFF

            # if the 'c' key or space bar is pressed, break from the loop
            if key == ord("c") or key == 32:
                break

        self.config['frame_skip'] = 19

    def prev_dist(self, x, y):

        x_dist = (x - self.data['x'][-1]) ** 2
        y_dist = (y - self.data['y'][-1]) ** 2

        dist = np.sqrt(x_dist + y_dist)

        return dist

    def process_frame(self, frame):
        """
        Process frame using selected tracker model.
        """

        global mouse_loc

        self.config['frame_skip'] = 2

        # detect location of the mouse
        if self.config['boundLoc']:
            prev_x, prev_y = self.data['x'][-1], self.data['y'][-1]
            if self.config['boundPool']:
                prev_x -= self.config['minx']
                prev_y -= self.config['miny']
            h, w = self.config['prev_bounds'], self.config['prev_bounds']
            new_rect = self.extract_detect_img(frame, prev_x, prev_y, h, w)
            valid, x, y = self.detect_loc(new_rect, start_h=prev_x - h // 2,
                                          start_w=prev_y - w // 2)
            if self.config['tracker'] != 'pfilter':
                if valid:
                    x += prev_x - int(w * 0.75)
                    y += prev_y - int(h * 0.75)
                    if self.config['boundPool']:
                        x += self.config['minx']
                        y += self.config['miny']
        else:
            valid, x, y = self.detect_loc(frame)

        if not valid:
            self.ask_xy(frame)
            x, y = mouse_loc

        if self.config['boundPool']:
            x += self.config['minx']
            y += self.config['miny']

        if self.prev_dist(x, y) > self.config['dist_error']:
            self.ask_xy(frame)
            x, y = mouse_loc
            if self.config['boundPool']:
                x += self.config['minx']
                y += self.config['miny']

        # tracking success
        self.data['x'].append(x)
        self.data['y'].append(y)
        self.data['t'].append(self.t)

        self.current_pos = [x, y]


def fail_detect_click(event, x, y, flags, param):
    global mouse_loc

    # if the left mouse button was clicked, record location
    if event == cv2.EVENT_LBUTTONDOWN:

        mouse_loc = (x, y)


def get_mouse_params(event, x, y, flags, param):
    global mouse_params

    size = 48
    canny_params = param['canny_params']
    frame = param['frame']
    frame = frame[y - size: y + size,
                  x - size: x + size]

    # if the left mouse button was clicked, record location
    if event == cv2.EVENT_LBUTTONDOWN:
        from detectors.simple_detector import SimpleDetector

        temp_detector = SimpleDetector(config=canny_params,
                                       model_type='canny')
        mouse_params = temp_detector.get_params(frame, canny_params, size,
                                                size)
        print("Mouse params circ area:", mouse_params['circle_area'])
        mouse_params['initial_x'] = x
        mouse_params['initial_y'] = y
        mouse_params['col'] = param['frame'][y, x]


if __name__ == '__main__':
    print('Please run the program by running main.py')
