#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Code used for object tracking. Primary purpose is for tracking a mouse in
the Morris Water Maze experiment.

util.py: this contains all utility functions for use in the tracking software.

"""
import os
import cv2
import numpy as np
import argparse
from video_processor import VideoProcessor
from math import *


# global vals
topleft = []  # point for left clipping location of mouse for template
botright = []


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


def get_mouse_loc(event, x, y, flags, param):

    global topleft, botright

    # if image was left-clicked; udpate global val for [x, y] mouse centroid
    if event == cv2.EVENT_LBUTTONDOWN:
        topleft = [x, y]
    elif event == cv2.EVENT_LBUTTONUP:
        botright = [x, y]


def acquire_template(vidname):
    """
    Ask user to acquire template(s) from a selected frame from set of videos.

    :param videos: list - list of video files to be parsed
    :return: templates: list - list of ndarray images for template(s)
    """
    global topleft, botright

    #TODO: Add to func args: dirname
    template_dir = '/home/synapt1x/MWMTracker/templates/'
    template_file = template_dir + 'template1.png'

    # initialize boolean for whether template has been identified
    template = None

    # get height, width of template

    # load the first video
    new_processor = VideoProcessor()
    image_generator = new_processor.frame_generator(vidname)

    # run to 30th frame and start checking for template at frame 31
    [image_generator.__next__() for _ in range(50)]
    frame = image_generator.__next__()

    # initialize first frame showing
    frame_copy = frame.copy()
    cv2.namedWindow('template')
    cv2.setMouseCallback('template', get_mouse_loc)
    found_template = None

    print("Please left click on the location of the mouse")

    if frame is not None:
        # extract properties of video
        h, w, d = frame.shape
        #temp_h = h // 10
        #temp_w = w // 10
        temp_h = 12
        temp_w = 16

        frame_jump = 50  # number of frames to skip through video

        # while there are still remaining frames left in the video
        while frame is not None and not found_template:

            cv2.imshow('template', frame)
            key = cv2.waitKey(1) & 0xFF

            if key == 32:  # space bar pressed
                [image_generator.__next__() for _ in range(frame_jump)]
                frame = image_generator.__next__()
                continue
            elif key == 27:  # ESC pressed
                print("Exiting template discovery...")
                break
            else:  # else if any other key is pressed
                # check if both crop points were defined
                if len(topleft) == 2 and len(botright) == 2:
                    add_h = (botright[1] - topleft[1]) % 4
                    add_w = (botright[0] - topleft[0]) % 4
                    template = frame[topleft[1]: botright[1] + add_h,
                               topleft[0]: botright[0] + add_w]
                    print("Is this a good template of the mouse? y/n")
                    cv2.imshow('suggested template', template)
                    ans = cv2.waitKey(0) & 0xFF

                    # check answer regarding quality of template
                    if ans == ord('y') or ans == ord('Y'):
                        # save if good
                        print('Saving template and continuing with analysis')
                        cv2.imwrite(template_file, template)
                        break
                    elif ans == ord('n') or ans == ord('n'):
                        # continue skipping through frames looking for template
                        print('Moving to next frame to re-select mouse')
                        cv2.destroyWindow('suggested template')
                        topleft = []
                        botright = []
                        template = np.array([])

                        [image_generator.__next__() for _ in range(frame_jump)]
                        frame = image_generator.__next__()
                        continue
                    elif ans == 27:  # ESC -> quit template extraction
                        print('Exiting template creation process...')
                        break

        cv2.destroyAllWindows()

    return [template]


def pad_frame(frame, template, padding=cv2.BORDER_REPLICATE):
    """
    Pad the frame using the template to determine the size of the padding.
    Default padding is replicate, though additional padding methods can be
    passed.

    :param frame: (ndarray) - image of the entire video frame
    :param template: (ndarray) - template image of the object to be tracked
    :param padding: (const) - cv2 constant indicating padding method desired
                            default: cv2.BORDER_REPLICATE
    :return:
    """

    # create padded image using template size to replicate borders to stretch
    t_h, t_w = template.shape
    frame = 0.12 * frame[:, :, 0] + \
            0.58 * frame[:, :, 1] + \
            0.3 * frame[:, :, 2]
    padded_img = cv2.copyMakeBorder(frame, t_h // 2, t_h // 2,
                                    t_w // 2, t_w // 2,
                                    padding)

    return padded_img


def process_template(template, feature='hog'):
    """
    process template image to extract feature

    :param feature: (string) - string indicating feature extraction method
    :return:
    """

    if feature == 'hog':
        #TODO: Complete HOG descriptor for image
        win_size = template.shape
        block_size = (4, 4)
        block_stride = (2, 2)
        cell_size = (4, 4)
        nbins = 6

        hog = cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size,
                                nbins)
        hog.compute(template)

    elif feature == 'segment':
        #TODO: Implement method of segmenting the object and extracting params
        extraction = None


def convolve(frame, template):
    """
    Convolve template with the current frame to determine the maximum
    likelihood for a match given the provided template.

    :param frame: (ndarray) - image of the entire video frame
    :param template: (ndarray) - template image of the object to be tracked
    :return:
    """

    t_h, t_w = template.shape
    padded_img = cv2.copyMakeBorder(frame, t_h // 2, t_h // 2,
                                    t_w // 2, t_w // 2,
                                    cv2.BORDER_REPLICATE)

    return padded_img


def display_particles(frame, particles):
    """
    Display particles onto frame using cv2 circles.

    :param frame: (ndarray) - cv2 image in uint8.
    :param particles: (ndarray) - array of particles: rows are particles
    :return:
    """

    for (x, y, dx, dy) in particles:
        frame = cv2.circle(frame, (int(y), int(x)), 1, (0, 255, 0), 1)

    return frame


if __name__ == '__main__':
    print("Please run the file 'main.py'")
