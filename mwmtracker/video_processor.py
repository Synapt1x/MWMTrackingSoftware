# -*- coding: utf-8 -*-

""" Code used for object tracking. Primary purpose is for tracking a mouse in
the Morris Water Maze experiment.

video_processor.py: this contains the code for generating a video
reader/writer for processing video frames during tracking.

"""
import cv2
import numpy as np


class Video_processor:
    """
    Video Processor class
    """

    def __init__(self):
        """constructor"""

        self.writer = self.create_writer

    def create_writer(self, filename, frame_size, fps=20):
        """Opens and returns a video for writing.
        
        :param filename: string - Filename for output video
        :param frame_size: tuple - width, height of output video resolution
        :param fps: int - frames per second of output video
        
        :return: 
        """

        # assign fourcc codec of video writer
        fourcc = cv2.VideoWriter_fourcc(*'MP4V')

        # create video writer
        videoWriter = cv2.VideoWriter(filename, fourcc, fps, frame_size)

        return videoWriter

    def frame_generator(self, filename=None):
        """
        A frame generator that yields a frame every next() call.
        Will return 'None' if there are no frames left in video.

        :return:
        """

        # if video name is provided; then load new video into processor
        if filename is not None:
            video = cv2.VideoCapture(filename)
            self.video = video
        else:
            video = self.video

        # while video is still opened
        while video.isOpened():
            # get next
            ret, frame = video.read()

            if ret:
                yield frame
            else:
                break

        video.release()
        yield None

    def save_image(self, x_vals, y_vals, frame, img_name):
        """

        :param x_vals:
        :param y_vals:
        :param frame:
        :param img_name:
        :return:
        """

        prev_x, prev_y = 0, 0

        # loop over detection points and draw a line between points
        for x, y in zip(x_vals, y_vals):

            # if this is the first point, draw an initial circle for drop point
            if prev_x == 0 and prev_y == 0:
                cv2.circle(frame, center=(prev_x, prev_y), radius=1,
                           color=(0, 255, 0), thickness=1)
            else:
                cv2.line(frame, (prev_x, prev_y), (x, y),
                         color=(0, 255, 0), thickness=1)

            prev_x, prev_y = x, y

        # save the image
        cv2.imwrite(img_name, frame)


if __name__ == '__main__':
    print("Please run the file 'main.py'")
