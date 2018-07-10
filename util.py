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


# global vals
crop_pnt = []  # point for left clipping location of mouse for template


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

    global crop_pnt

    # if image was left-clicked; udpate global val for [x, y] mouse centroid
    if event == cv2.EVENT_LBUTTONDOWN:
        crop_pnt = [x, y]


def acquire_template(vidname):
    """
    Ask user to acquire template(s) from a selected frame from set of videos.

    :param videos: list - list of video files to be parsed
    :return: templates: list - list of ndarray images for template(s)
    """
    global crop_pnt

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
        temp_h = h // 10
        temp_w = w // 10

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
                if len(crop_pnt) == 2:  # check if a click point was registered
                    template = frame[crop_pnt[1] - temp_h: crop_pnt[1] +
                                                           temp_h,
                               crop_pnt[0] - temp_w: crop_pnt[0] + temp_w, :]
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
                        crop_pnt = []
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
    padded_img = cv2.copyMakeBorder(frame, t_h // 2, t_h // 2,
                                    t_w // 2, t_w // 2,
                                    padding)

    padded_img = 0.12 * padded_img[:, :, 0] +\
                 0.58 * padded_img[:, :, 1] +\
                 0.3 * padded_img[:, :, 2]

    return padded_img


def show_frame(frame, frame_num=0, save_img=False, output_name=''):
    """
    Show video frame to the screen, and also allow for additional options to
    save the image to a file.
    
    :param frame: ndarray - the image frame to be shown
    :param save_img: bool - whether or not to save frame to an output image
    :param output_name: string - file name for output image to be saved 
    :return: 
    """

    #tODO: Need to fix show image

    title = 'Frame number: ' + str(frame_num)

    cv2.imshow(title, frame)
    cv2.waitKey()
    cv2.destroyWindow(title)


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
