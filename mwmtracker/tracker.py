# -*- coding: utf-8 -*-

""" Code used for object tracking. Primary purpose is for tracking a mouse in
the Morris Water Maze experiment.

tracker.py: this houses the primary functionality for storing, processing and
tracking videos using a specific tracking model.

"""
import util


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

        # initialize setup and video loading
        self.initialize_tracker()


    def initialize_tracker(self):
        """
        Initialize tracker and load relevant data.

        """

        self.data['train_vids'] = util.load_files(self.config['datadir'])
        self.data['template_img'] = util.load_files(self.config['templatedir'])



if __name__ == '__main__':
    print('Please run the program by running main.py')
