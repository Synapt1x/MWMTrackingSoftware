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
        self.output_data = pd.DataFrame(columns=['vid num', 'x', 'y', 't'])

    def save_frame(self, col, x_locs, y_locs, t):

        vid_name = np.repeat(col, len(x_locs))

        temp_df = pd.DataFrame(columns=['vid num', 'x', 'y', 't'])

        temp_df['vid num'] = vid_name
        temp_df['x'] = x_locs
        temp_df['y'] = y_locs
        temp_df['t'] = t

        self.output_data = pd.concat([self.output_data, temp_df])

    def save_to_excel(self, sheetName):
        """
        Save to the
        :return:
        """

        self.output_data.to_excel(self.excelWriter, sheetName)

        self.excelWriter.save()


if __name__ == '__main__':
    print("Please run the file 'main.py'")
