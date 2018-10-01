# -*- coding: utf-8 -*-

""" Code used for object tracking. Primary purpose is for tracking a mouse in
the Morris Water Maze experiment.

util.py: this contains all utility functions for use in the tracking software.

"""
import os
import cv2
import numpy as np
import argparse
import pickle
from video_processor import Video_processor
from math import *


# global vals
topleft = []  # point for left clipping location of mouse for template
botright = []
all_imgs = []
labels = []
frame_skip = 10


def load_files(dirname='train_files'):
    """
    Load files from a specified folder passed in as a string 'dirname'.

    :param dirname: string - directory name to load files for parsing
    :return: output_files: list - list of video files found in dirname directory
    """

    # extract all files from directory
    output_files = os.listdir(dirname)
    output_files = [os.path.join(dirname, file) for file in output_files]

    return output_files


def get_mouse_loc(event, x, y, flags, param):

    global topleft, botright

    # if image was left-clicked; udpate global val for [x, y] mouse centroid
    if event == cv2.EVENT_LBUTTONDOWN:
        topleft = [x, y]
    elif event == cv2.EVENT_LBUTTONUP:
        botright = [x, y]


def acquire_template(dirname, vidname):
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


def resize_frame(img, factor):
    """
    Resize the input img by the specified factor.

    :param img:
    :param factor:
    :return:
    """

    # extract shape and get new output shape
    h, w = img.shape[:2]
    new_h, new_w = int(h * factor), int(w * factor)

    # resize and store resized image
    new_img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    return new_img


def display_particles(frame, particles):
    """
    Display particles onto frame using cv2 circles.

    :param frame: (ndarray) - cv2 image in uint8.
    :param particles: (ndarray) - array of particles: rows are particles
    :return:
    """

    for (x, y, dx, dy) in particles:
        frame = cv2.circle(frame, (int(y), int(x)), 1, (255, 0, 0), 1)

    return frame


def get_rois(event, x, y, flags, param):
    """
    Extract image ROIs from a click on the image

    :param event:
    :param x:
    :param y:
    :param flags:
    :param param:
    :return:
    """

    if event == cv2.EVENT_LBUTTONDOWN:

        img_size, frame = param

        h, w = frame.shape[:2]

        # store single pixel shifts of the select location
        for i in [-2, -1, 0, 1, 2]:
            for j in [-2, -1, 0, 1, 2]:
                pos_img = frame[y - img_size // 2 - i: y + img_size // 2 - i,
                            x - img_size // 2 - j: x + img_size // 2 - j]
                all_imgs.append(pos_img)
                labels.append(1)

        # if there are enough negative examples already
        if len(labels) < 2000:

            # iterate over all other locations in image to generate negative
            #  images
            for i in range(1, h // img_size):
                for j in range(1, w // img_size):

                    neg_i, neg_j = i * img_size, j * img_size

                    if abs(neg_i - x) > 2 * img_size and \
                            abs(neg_j - y) > 2 * img_size:
                        img = frame[neg_i - img_size // 2: neg_i + img_size // 2,
                                    neg_j - img_size // 2: neg_j + img_size // 2]

                        all_imgs.append(img)
                        labels.append(0)


def load_train_data(pickle_name='data/trainData/train_data.pickle',
                    as_array=True):
    """
    Load training data from the specified pickle file and return it as numpy
    arrays.
    :param pickle_name:
    :return:
    """

    # load all imgs and their labels from the pickle file
    with open(pickle_name, 'rb') as file:
        all_data = pickle.load(file)
    all_imgs, labels = all_data

    if as_array:
        all_imgs = np.array(all_imgs)
        labels = np.array(labels)

    return all_imgs, labels


def save_train_data(filename):
    """

    :return:
    """

    with open(filename, 'wb') as file:
        pickle.dump([all_imgs, labels], file)


def extract_train_data(img_size=48, video=None, output_dir=None):
    """
    Interactively extract training data from a provided video
    :param output_dir:
    :param img_size:
    :return:
    """

    global all_imgs, labels

    # exit if no video was passed in
    if video is None:
        return

    filename = output_dir + os.sep + 'train_data.pickle'
    all_imgs, labels = load_train_data(filename, as_array=False)

    valid, frame = video.read()

    cv2.namedWindow("Click on mouse")
    cv2.setMouseCallback("Click on mouse", get_rois, [img_size, frame])

    # while frames have successfully been extracted
    while valid:

        # show frame to extract ROIs from
        cv2.imshow("Click on mouse", frame)
        key = cv2.waitKey(0) & 0xFF
        if key == ord('q') or key == ord('Q'):
            break

        [video.read() for _ in range(frame_skip)]

        valid, frame = video.read()

        if not valid:
            break

    cv2.destroyAllWindows()

    # update pickle file
    save_train_data(filename)


if __name__ == '__main__':
    print("Please run the file 'main.py'")
