# -*- coding: utf-8 -*-

""" Code used for processing the tracking data from the mouse tracker,
specifically by assembling a pandas dataframe and writing this to an excel
using XlsxWriter from pandas.

data_processor.py: this file contains the code implementing the functionality
for saving the output path data to an excel file.

"""
import pandas as pd



class dataProcessor:
    """
    Data Processor class
    """

    def __init__(self, excelFilename='output.xlsx'):
        """constructor"""

        self.excelWriter = pd.ExcelWriter(excelFilename, engine='xlsxwriter')
        self.outputData = pd.DataFrame({})

    def save_to_excel(self, sheetName):
        """
        Save to the
        :return:
        """

        self.outputData.to_excel(self.excelWriter, sheetName)


if __name__ == '__main__':
    print("Please run the file 'main.py'")
