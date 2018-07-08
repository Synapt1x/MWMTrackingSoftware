#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Code used for object tracking. Primary purpose is for tracking a mouse in
the Morris Water Maze experiment.

main.py: this contains the main code for running the tracking software.

"""

##############################################################################
__author__ = "Chris Cadonic"
__credits__ = ["Chris Cadonc"]
__license__ = "GPL"
__version__ = "1.0.1"
__maintainer__ = "Chris Cadonic"
__email__ = "chriscadonic@gmail.com"
__status__ = "Development"
##############################################################################


from config import Configuration
from cnn import Network
from video_processor import VideoProcessor
from particle_filter import ParticleFilter
import cv2
import util
import numpy as np


def main():
    """
    Main code for running the tracking software.

    :return:
    """

    # Load configuration for
    config = Configuration()

    # create a particle filter for tracking
    pfilter = ParticleFilter(config.num_particles)

    # load video processor for extracting frames during tracking
    vid_reader = VideoProcessor()
    image_generator = vid_reader.frame_generator(config.test_file)

    # load files and parse
    train_videos = util.load_files(config.training_dir)

    # load template files and parse
    templates = util.load_files(config.template_dir)

    # if template dir is empty
    if len(templates) == 0:
        templates = util.acquire_template(train_videos[0])
        pfilter.template = templates[0]
    pfilter.template = cv2.imread(templates[0]).astype(np.uint8)

    pfilter.template = 0.12 * pfilter.template[:, :, 0] +\
                       0.58 * pfilter.template[:, :, 1] +\
                       0.3 * pfilter.template[:, :, 2]

    # get first frame of video and the properties of the video
    frame = image_generator.__next__()

    if frame is not None:
        # extract properties of video
        h, w, d = frame.shape

        # initialize particles in filter based on frame size
        pfilter.initialize_particles(h, w)

        # create video writer for writing out video
        video_out = vid_reader.create_writer(config.test_out, (w, h),
                                             config.framerate)

        # template = cv2.imread('template.jpg')

        frame_num = 1

        while frame is not None:
            print("Processing frame ", frame_num)

            # pad the border of frame to be stored as current processing img
            pfilter.full_frame = util.pad_frame(frame, pfilter.template)

            # update particle filter and get the estimated location
            pfilter.calc_error()
            pfilter.resample()
            avg_x, avg_y = pfilter.query()

            frame = cv2.circle(frame, center=(int(avg_y), int(avg_x)),
                               radius=5, color=(255, 0, 0), thickness=3)

            video_out.write(frame)

            frame = image_generator.__next__()

            frame_num += 1

        video_out.release()

    else:
        #TODO: change to output GUI message
        print("Error loading video! Please ensure at least one test video is "
              "located in the testVids directory.")

    return


if __name__ == '__main__':
    main()