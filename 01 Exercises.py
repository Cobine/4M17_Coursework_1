import numpy as np
import pandas as pd
import os

def main():

    dir_name = 'C:\\Users\\Cobin\\Downloads\\2022-data\\2022-data\\Q1\\'
    array_A, array_b = openFile(dir_name, 'A1', 'b1')

def openFile(dir_name, data_A, data_b):
    array_A = pd.read_csv(os.path.join(dir_name, data_A + "." + 'csv'), sep=',', header=None)
    array_b = pd.read_csv(os.path.join(dir_name, data_b + "." + 'csv'), sep=',', header=None)
    return array_A, array_b


if __name__ == "__main__":
    main()