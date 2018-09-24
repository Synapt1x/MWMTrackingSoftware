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

        self.output_data['vid num'] = vid_name
        self.output_data['x'] = x_locs
        self.output_data['y'] = y_locs
        self.output_data['t'] = t

    def save_to_excel(self, sheetName):
        """
        Save to the
        :return:
        """

        self.output_data.to_excel(self.excelWriter, sheetName)


if __name__ == '__main__':
    print("Please run the file 'main.py'")
