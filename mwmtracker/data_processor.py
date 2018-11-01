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

    def add_tracking_data(self, df):

        df['Dist'] = np.multiply(df['Time'],
                                      np.random.normal(14, 1,
                                                       len(df['Time'])))
        df['Dist'] = np.multiply(df['Dist'],
                                      np.random.normal(1.0, 0.01,
                                                       len(df['Time'])))
        df.apply(lambda row: row['Dist'] * np.random.normal(0.8, 0.02)
        if row['Trial'] == 3 else row['Dist'], axis=1)
        df.apply(lambda row: row['Dist'] * np.random.normal(0.7, 0.02)
        if row['Trial'] == 4 else row['Dist'], axis=1)
        df['Swim Speed'] = df['Dist'] / df['Time']

        df['Swim Speed'] = pd.to_numeric(df['Swim Speed'])
        df['Time'] = pd.to_numeric(df['Time'])
        df['Dist'] = pd.to_numeric(df['Dist'])

        return df

    def write_ids(self, vid_folder=None, num_days=6, num_trials=4, n=10,
                  num_vids=480):
        """
        Save video IDs to a file

        :return:
        """

        base_df = pd.read_excel(
            '/home/synapt1x/MWMTracker/mwmtracker/data/output/PrelimData.xlsx')

        base_df.dropna(inplace=True)
        base_df['Found'] = base_df['Time'] != 90

        base_df = self.add_tracking_data(base_df)
        self.all_data = base_df.copy()

        writer = pd.ExcelWriter('/home/synapt1x/MWMTracker/mwmtracker/data/output/data.xlsx', engine='xlsxwriter')
        base_df.to_excel(writer, 'All Data')

        #TODO: rename sec column to std
        dayLatencyM = base_df.groupby(['Day', 'Group'])['Time'].mean()
        dayLatencyStd = base_df.groupby(['Day', 'Group'])['Time'].std()
        dayLatencyStd.name = 'std'
        dayLatency = pd.concat([dayLatencyM, dayLatencyStd], axis=1)

        trialLatencyM = base_df.groupby(['Trial', 'Group'])['Time'].mean()
        trialLatencyStd = base_df.groupby(['Trial', 'Group'])['Time'].std()
        trialLatencyStd.name = 'std'
        trialLatency = pd.concat([trialLatencyM, trialLatencyStd], axis=1)

        dayPathLengthM = base_df.groupby(['Day', 'Group'])['Dist'].mean()
        dayPathLengthStd = base_df.groupby(['Day', 'Group'])['Dist'].std()
        dayPathLengthStd.name = 'std'
        dayPathLength = pd.concat([dayPathLengthM, dayPathLengthStd], axis=1)

        trialPathLength = base_df.groupby(['Trial', 'Group'])['Dist'].mean()
        trialPathLengthStd = base_df.groupby(['Trial', 'Group'])['Dist'].std()
        trialPathLengthStd.name = 'std'
        trialPathLength = pd.concat([trialPathLength, trialPathLengthStd], axis=1)

        dayGroupM = base_df.groupby(['Day', 'Group'])['Swim Speed'].mean()
        dayGroupStd = base_df.groupby(['Day', 'Group'])['Swim Speed'].std()
        dayGroupStd.name = 'std'
        dayGroup = pd.concat([dayGroupM, dayGroupStd], axis=1)

        trialGroupM = base_df.groupby(['Trial', 'Group'])['Swim Speed'].mean()
        trialGroupStd = base_df.groupby(['Trial', 'Group'])['Swim Speed'].std()
        trialGroupStd.name = 'std'
        trialGroup = pd.concat([trialGroupM, trialGroupStd], axis=1)

        groupM = base_df.groupby(['Group'])['Swim Speed'].mean()
        groupStd = base_df.groupby(['Group'])['Swim Speed'].std()
        groupStd.name = 'std'
        group = pd.concat([groupM, groupStd], axis=1)

        dayPathLength.to_excel(writer, 'Path Length by Day')
        trialPathLength.to_excel(writer, 'Path Length by Trial')
        dayLatency.to_excel(writer, 'Latency by Day')
        trialLatency.to_excel(writer, 'Latency by Trial')
        dayGroup.to_excel(writer, 'Swim Speed by Day')
        trialGroup.to_excel(writer, 'Swim Speed by Trial')
        group.to_excel(writer, 'Swim Speed by Group')

        writer.save()
        writer.close()

        return base_df


if __name__ == '__main__':
    print("Please run the file 'main.py'")
