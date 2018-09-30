# -*- coding: utf-8 -*-

""" Code used for object tracking, specifically using simple detection
approaches to detect the object in a given frame.

detectors.py: this contains code for implementing simple object detection
approaches, such as template matching, or canny edge detection + convex
object separation.

"""
import numpy as np
import cv2


class SimpleDetector:
    """
    Simple detector class containing methods for detecting an object
    location in a frame.
    """

    def __init__(self, config=None, model_type='canny'):

        self.model_type = model_type
        self.config = config

        self.config = config

    def detect(self, frame=None, template=None, detector=False):

        if frame is None or template is None:
            return False, None, None

        if self.model_type == 'canny':
            return self.canny_detect(frame, self.config['threshold1'],
                                     self.config['threshold2'],
                                     self.config['min_area'],
                                     self.config['max_area'],
                                     self.config['min_arc_length'],
                                     self.config['max_arc_length'])
        elif self.model_type == 'template':
            return self.template_match(frame,
                                       self.config['template_ccorr'],
                                       self.config['template_thresh'],
                                       template=template,
                                       detector=detector)

    def canny_detect(self, frame, thresh1=60, thresh2=320, min_area=40,
                     max_area=600, min_arc=60, max_arc=180):

        #TODO: Adapt output for pfilter detector

        # use Canny detector to find edges and find contours to find all
        # detected shapes
        edge_frame = cv2.Canny(frame, threshold1=thresh1, threshold2=thresh2)

        contour_frame, contours, hierarchy = cv2.findContours(edge_frame,
                                                      cv2.RETR_TREE,
                                                      cv2.CHAIN_APPROX_SIMPLE)

        found = False

        for contour in contours:
            # extract moments and features of detected contour
            moments = cv2.moments(contour)
            area = cv2.contourArea(contour)
            arc_length = cv2.arcLength(contour, True)

            if moments['m00'] == 0:
                continue

            areaCheck = min_area < area < max_area
            arcCheck = min_arc < arc_length < max_arc

            if areaCheck and arcCheck:
                found = True
                x = int(moments['m10'] / moments['m00'])
                y = int(moments['m01'] / moments['m00'])
                break
            else:
                continue

        if found:

            return True, x, y

        else:

            return False, None, None

    def template_match(self, frame, match_method='template_ccorr',
                       template_thresh=0.8, template=None, detector=False):

        max_detection = 0
        h, w = template.shape[:2]
        pad_frame = cv2.copyMakeBorder(frame, h // 2, h // 2,
                                       w // 2, w // 2, cv2.BORDER_REPLICATE)

        for rotation in [0]:  #[0, 45, 90, 135, 180, 225]:
            if rotation != 0:
                # rotate the template to check for other orientations
                rotation_mtx = cv2.getRotationMatrix2D((w // 2, h // 2),
                                                       rotation, 1)
                new_width = int((h * np.abs(rotation_mtx[0, 1]))
                                + (w * np.abs(rotation_mtx[0, 0])))
                new_height = int((h * np.abs(rotation_mtx[0, 0]))
                                 + (w * np.abs(rotation_mtx[0, 1])))

                rotation_mtx[0, 2] += (new_width / 2) - w // 2
                rotation_mtx[1, 2] += (new_height / 2) - h // 2

                new_template = cv2.warpAffine(template, rotation_mtx,
                                              (new_height, new_width))
            else:
                new_template = template
                new_width = w
                new_height = h

            # test current rotation using template matching
            template_vals = cv2.matchTemplate(pad_frame, new_template,
                                              eval(match_method))
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(template_vals)

            # if max val is found
            if max_val > template_thresh and max_val > \
                    max_detection:
                max_detection = max_val
                w, h = new_width // 2, new_height // 2
                x_val, y_val = max_loc[0] + w, max_loc[1] + h
                if detector:
                    template_corr = template_vals

        if max_detection == 0:
            if detector:
                return False, []
            return False, None, None

        if detector:
            return True, template_corr
        return True, x_val, y_val


if __name__ == '__main__':
    print("Please run the file 'main.py'")
