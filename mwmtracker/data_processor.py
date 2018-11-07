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

    def __init__(self, excelFilename='trackingData.xlsx', cm_scale=6.0):
        """constructor"""

        self.excelFilename = excelFilename
        self.initialize_writer()
        self.output_data = pd.DataFrame(columns=['vid num', 'x', 'y',
                                                 'dist', 't'])
        self.dist_data = pd.DataFrame(columns=['vid num', 'dist'])
        self.all_data = pd.DataFrame()
        self.dist_data = pd.DataFrame()

        self.cm_scale = cm_scale

    def initialize_writer(self):
        self.excelWriter = pd.ExcelWriter(self.excelFilename,
                                          engine='xlsxwriter')

    def save_frame(self, col, x_locs, y_locs, t):
        """
        Update Pandas dataframe with current frame information.

        :param col:
        :param x_locs:
        :param y_locs:
        :param t:
        :return:
        """

        col = int(col.split(" ")[-1])

        vid_name = np.repeat(col, len(x_locs))

        # store all data from x, y locs through tracking
        temp_df = pd.DataFrame(columns=['vid num', 'x', 'y', 'dist', 't'])

        temp_df['vid num'] = vid_name
        temp_df['x'] = x_locs
        temp_df['y'] = y_locs
        x_diff = np.power(temp_df['x'] - temp_df['x'].shift(1), 2)
        y_diff = np.power(temp_df['y'] - temp_df['y'].shift(1), 2)
        dists = np.sqrt(x_diff + y_diff) / self.cm_scale
        temp_df['dist'] = np.nancumsum(dists)
        temp_df['t'] = t

        # store all total dist data for each tracking effort
        dist_df = pd.DataFrame(columns=['vid num', 'dist'])
        dist_df = dist_df.append(pd.Series([col, temp_df.iloc[-1]['dist']],
                                 index=dist_df.columns), ignore_index=True)

        self.output_data = pd.concat([self.output_data, temp_df])
        self.dist_data = pd.concat([self.dist_data, dist_df])

        self.save_to_excel()

    def save_to_excel(self):
        """
        Save to the
        :return:
        """

        self.initialize_writer()

        self.output_data.to_excel(self.excelWriter, 'All Tracking Data')
        self.dist_data.to_excel(self.excelWriter, 'Dist Data')

        self.excelWriter.save()
        self.excelWriter.close()

    def add_tracking_data(self, df):

        #TODO: Load tracking data from tracking data excel

        return df

    def no_collapse_df(self, df, groups, output_var):

        out_df = df.groupby(['Day', 'Group'])['Dist'].mean()
        dfStd = df.groupby(['Day', 'Group'])['Dist'].std()
        dfSem = df.groupby(['Day', 'Group'])[
            'Dist'].sem()
        dfStd.name = 'std'
        dfSem.name = 'sem'
        final_df = pd.concat([df, dfStd, dfSem], axis=1).reset_index()

    def collapse_df(self, df, groups, output_var, columns):

        final_df = pd.DataFrame(columns=columns)

        out_df = df.groupby(groups)[output_var].mean().reset_index()
        out_df_std = df.groupby(['Day',
                                 'Group'])[output_var].std().reset_index()
        out_df_sem = df.groupby(['Day',
                                 'Group'])[output_var].sem().reset_index()

        control_out_df = out_df.where(out_df['Group'] ==
                                      'Control').dropna().reset_index()
        exp_out_df = out_df.where(out_df['Group'] ==
                                  'Nilotinib').dropna().reset_index()

        control_out_std = out_df_std.where(out_df['Group'] ==
                                           'Control').dropna().reset_index()
        exp_out_std = out_df_std.where(out_df['Group'] ==
                                       'Nilotinib').dropna().reset_index()

        control_out_sem = out_df_sem.where(out_df['Group'] ==
                                           'Control').dropna().reset_index()
        exp_out_sem = out_df_sem.where(out_df['Group'] ==
                                       'Nilotinib').dropna().reset_index()

        final_df[columns[0]] = control_out_df['Day']
        final_df[columns[1]] = control_out_df[output_var]
        final_df[columns[2]] = exp_out_df[output_var]
        final_df[columns[3]] = control_out_std[output_var]
        final_df[columns[4]] = exp_out_std[output_var]
        final_df[columns[5]] = control_out_sem[output_var]
        final_df[columns[6]] = exp_out_sem[output_var]

        return final_df

    def write_ids(self, vid_folder=None, num_days=6, num_trials=4, n=10,
                  num_vids=480):
        """
        Save video IDs to a file

        :return:
        """

        by_individual = False
        without_five = False

        base_df = pd.read_excel(
            '/home/synapt1x/MWMTracker/mwmtracker/data/output/PrelimData.xlsx')
        self.output_data = pd.read_excel(self.excelFilename)
        self.dist_data = pd.read_excel(self.excelFilename, sheet_name="Dist "
                                                                      "Data")

        base_df.dropna(inplace=True)
        base_df['Found'] = base_df['Time'] != 90
        base_df['Group'] = base_df['ID'].apply(lambda x: 'Nilotinib' if x in
                                                                        range(1, 11) else 'Control')

        #base_df = self.add_tracking_data(base_df)
        self.all_data = base_df.copy()

        writer = pd.ExcelWriter('/home/synapt1x/MWMTracker/mwmtracker/data/output/data.xlsx', engine='xlsxwriter')
        base_df.sort_values(['Day', 'Trial', 'ID'], inplace=True)
        base_df.to_excel(writer, 'All Data')

        trial_data = base_df.set_index('Trial')
        trial_data = trial_data.groupby(['Day', 'ID',
                                         'Group']).mean().reset_index()
        trial_data.to_excel(writer, 'Data by trial')

        if without_five:

            other_df = base_df.groupby(['Day', 'Group']).mean()
            other_df = other_df.drop(columns=['ID', 'Trial', 'Found'])

            other_df.to_excel(writer, 'Without Five Trial Averages')

        if by_individual:

            day_latency = self.collapse_df(base_df, ['Day', 'Group'],
                                           'Time', ['Day', 'Control Time',
                                                    'Nilotinib Time',
                                                    'Control std',
                                                    'Nilotinib std',
                                                    'Control sem',
                                                    'Nilotinib sem'])
            day_dists = self.collapse_df(base_df, ['Day', 'Group'],
                                         'Dist', ['Day', 'Control Dist',
                                                  'Nilotinib Dist',
                                                  'Control std',
                                                  'Nilotinib std',
                                                  'Control sem',
                                                  'Nilotinib sem'])
            day_speeds = self.collapse_df(base_df, ['Day', 'Group'],
                                         'Swim Speed', ['Day', 'Control '
                                                               'Swim Speed',
                                                        'Nilotinib Swim Speed',
                                                        'Control std',
                                                        'Nilotinib std',
                                                        'Control sem',
                                                        'Nilotinib sem'])
        else:
            day_latency = self.collapse_df(trial_data, ['Day', 'Group'],
                                           'Time', ['Day', 'Control Time',
                                                    'Nilotinib Time',
                                                    'Control std',
                                                    'Nilotinib std',
                                                    'Control sem',
                                                    'Nilotinib sem'])
            day_dists = self.collapse_df(trial_data, ['Day', 'Group'],
                                           'Dist', ['Day', 'Control Dist',
                                                    'Nilotinib Dist',
                                                    'Control std',
                                                    'Nilotinib std',
                                                    'Control sem',
                                                    'Nilotinib sem'])
            day_speeds = self.collapse_df(trial_data, ['Day', 'Group'],
                                           'Swim Speed', ['Day', 'Control '
                                                                 'Swim Speed',
                                                    'Nilotinib Swim Speed',
                                                    'Control std',
                                                    'Nilotinib std',
                                                    'Control sem',
                                                    'Nilotinib sem'])

        day_latency.to_excel(writer, 'Latency by Day')
        day_dists.to_excel(writer, 'Path Length by Day')
        day_speeds.to_excel(writer, 'Swim Speed by Day')

        writer.save()
        writer.close()

        return base_df


if __name__ == '__main__':
    print("Please run the file 'main.py'")
