# -*- coding: utf-8 -*-

""" Code used for processing the tracking data from the mouse tracker,
specifically by assembling a pandas dataframe and writing this to an excel
using XlsxWriter from pandas.

data_processor.py: this file contains the code implementing the functionality
for saving the output path data to an excel file.

"""
import os

import pandas as pd
import numpy as np
import cv2
import util


class Data_processor:
    """
    Data Processor class
    """

    def __init__(self, excelFilename='output.xlsx'):
        """constructor"""

        self.excelWriter = pd.ExcelWriter(excelFilename, engine='xlsxwriter')
        self.output_data = pd.DataFrame(columns=['vid num', 'x', 'y',
                                                 'dist', 't'])
        self.all_data = pd.DataFrame()

    def save_frame(self, col, x_locs, y_locs, t):
        """
        Update Pandas dataframe with current frame information.

        :param col:
        :param x_locs:
        :param y_locs:
        :param t:
        :return:
        """

        vid_name = np.repeat(col, len(x_locs))

        temp_df = pd.DataFrame(columns=['vid num', 'x', 'y', 'dist', 't'])

        temp_df['vid num'] = vid_name
        temp_df['x'] = x_locs
        temp_df['y'] = y_locs
        x_diff = np.power(temp_df['x'] - temp_df['x'].shift(1), 2)
        y_diff = np.power(temp_df['y'] - temp_df['y'].shift(1), 2)
        dists = np.sqrt(x_diff + y_diff)
        temp_df['dist'] = np.nancumsum(dists)
        temp_df['t'] = t

        self.output_data = pd.concat([self.output_data, temp_df])

    def save_to_excel(self, sheetName):
        """
        Save to the
        :return:
        """

        self.output_data.to_excel(self.excelWriter, sheetName)

        self.excelWriter.save()

    def write_ids(self, vid_folder=None, num_days=6, num_trials=4, n=10,
                  num_vids=480):
        """
        Save video IDs to a file

        :return:
        """

        cols = ['ID', 'Group', 'Day', 'Trial', 'Time', 'Found']
        base_df = pd.DataFrame({}, index=range(3, num_vids + 3), columns=cols)
        day_orders = {1: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                          11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
                      2: [2, 12, 3, 13, 8, 18, 4, 14, 7, 17,
                          10, 20, 1, 11, 9, 19, 6, 16, 5, 15],
                      3: [9, 19, 8, 18, 1, 11, 4, 14, 6, 16,
                          3, 13, 5, 15, 10, 20, 2, 12, 7, 17],
                      4: [1, 11, 10, 20, 2, 12, 9, 19, 3, 13,
                          4, 14, 8, 18, 7, 17, 6, 16, 5, 15],
                      5: [4, 14, 6, 16, 3, 13, 7, 17, 5, 15,
                          9, 19, 2, 12, 1, 11, 10, 20, 8, 18],
                      6: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                          11, 12, 13, 14, 15, 16, 17, 18, 19, 20]}

        vid_num = 3
        done = False

        for day in range(1, num_days + 1):

            if done:
                break

            trial = 1
            while trial < num_trials + 1:

                if done:
                    break

                run_num = 1
                while run_num < 2 * n + 1:

                    if vid_num >= num_vids:
                        done = True
                        break

                    mouse = day_orders[day][run_num - 1]
                    if mouse <= 10:
                        group = 'Control'
                    else:
                        group = 'Nilotinib'

                    if vid_folder is not None:
                        video_name = vid_folder + os.sep + 'Video ' + str(
                            vid_num) + '.mp4'
                        vid = cv2.VideoCapture(video_name)
                        valid, _ = vid.read()

                        if not valid:
                            vid_num += 1
                            continue

                        if int(vid.get(cv2.CAP_PROP_FPS)) == 0:
                            time = 0
                            found = False
                        else:
                            time = int(vid.get(cv2.CAP_PROP_FRAME_COUNT)) / int(
                                vid.get(cv2.CAP_PROP_FPS))
                            if time >= 90:
                                time = 90
                                found = False
                            else:
                                found = True
                    else:
                        time = 0
                        found = False

                    base_df.loc[vid_num] = [mouse, group, day, trial,
                                                 time, found]

                    vid_num += 1
                    run_num += 1

                    if vid_folder is not None:
                        video_name = vid_folder + os.sep + 'Video ' + str(
                            vid_num - 1) + '-1.mp4'
                        vid = cv2.VideoCapture(video_name)
                        valid, _ = vid.read()

                        if valid:

                            if int(vid.get(cv2.CAP_PROP_FPS)) == 0:
                                time = 0
                                found = False
                            else:
                                time = int(vid.get(cv2.CAP_PROP_FRAME_COUNT)) / int(
                                    vid.get(cv2.CAP_PROP_FPS))
                                if time >= 90:
                                    time = 90
                                    found = False
                                else:
                                    found = True

                            mouse = day_orders[day][run_num - 1]
                            if mouse <= 10:
                                group = 'Control'
                            else:
                                group = 'Nilotinib'
                            base_df.loc[(vid_num - 1) * 10] = [mouse, group,
                                                               day, trial,
                                                               time, found]

                            run_num += 1

        base_df.dropna(inplace=True)

        self.all_data = base_df.copy()

        return base_df


if __name__ == '__main__':
    print("Please run the file 'main.py'")
