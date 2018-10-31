# -*- coding: utf-8 -*-

""" Code used for processing the tracking data from the mouse tracker,
specifically by assembling a pandas dataframe and writing this to an excel
using XlsxWriter from pandas.

data_processor.py: this file contains the code implementing the functionality
for saving the output path data to an excel file.

"""
import pandas as pd
import numpy as np


class Data_processor:
    """
    Data Processor class
    """

    def __init__(self, excelFilename='output.xlsx'):
        """constructor"""

        self.excelWriter = pd.ExcelWriter(excelFilename, engine='xlsxwriter')
        self.output_data = pd.DataFrame(columns=['vid num', 'x', 'y',
                                                 'dist', 't'])

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

    def write_ids(vid_folder=None, num_days=6, num_trials=4, n=10,
                  num_vids=480):
        """
        Save video IDs to a file

        :return:
        """

        cols = ['ID', 'Group', 'Day', 'Trial', 'Time']
        base_df = pd.DataFrame({}, index=range(1, num_vids + 1), columns=cols)
        day_orders = {1: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                          11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
                      2: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                          11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
                      3: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                          11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
                      4: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                          11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
                      5: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                          11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
                      6: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                          11, 12, 13, 14, 15, 16, 17, 18, 19, 20]}
        vid_num = 1

        for day in range(1, num_days + 1):

            for trial in range(1, num_trials + 1):

                for run_num in range(1, 2 * n + 1):

                    mouse = day_orders[day][run_num - 1]
                    if mouse <= 10:
                        group = 'Control'
                    else:
                        group = 'Nilotinib'

                    base_df.iloc[vid_num - 1] = [mouse, group, day, trial, 0]

                    vid_num += 1

        return base_df


if __name__ == '__main__':
    print("Please run the file 'main.py'")
