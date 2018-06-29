#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Code used for object tracking. Primary purpose is for tracking a mouse in
the Morris Water Maze experiment.

util.py: this contains all utility functions for use in the tracking software.

"""
import os
import cv2
import numpy as np
from video_processor import VideoProcessor


def load_files(dirname='train_files'):
    """
    Load files from a specified folder passed in as a string 'dirname'.

    :param dirname: string - directory name to load files for parsing
    :return: output_files: list - list of video files found in dirname directory
    """

    # get absolute path to directory
    cur_path = os.path.dirname(os.path.realpath(__file__))
    dir = os.path.join(cur_path, dirname)

    # extract all files from directory
    output_files = os.listdir(dir)
    output_files = [os.path.join(dir, file) for file in output_files]

    return output_files


def acquire_template(vidname):
    """
    Ask user to acquire template(s) from a selected frame from set of videos.

    :param videos: list - list of video files to be parsed
    :return: templates: list - list of ndarray images for template(s)
    """

    # load the first video
    new_processor = VideoProcessor()
    image_generator = new_processor.frame_generator(vidname)

    #TODO: get input from user to select template image

    '''while ret:
        ret, frame = init_video.read()

        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break'''

    return []


def show_frame(frame, frame_num=0, save_img=False, output_name=''):
    """
    Show video frame to the screen, and also allow for additional options to
    save the image to a file.
    
    :param frame: ndarray - the image frame to be shown
    :param save_img: bool - whether or not to save frame to an output image
    :param output_name: string - file name for output image to be saved 
    :return: 
    """

    title = 'Frame number: ' + str(frame_num)

    cv2.imshow(title, frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    print("Please run the file 'main.py'")
