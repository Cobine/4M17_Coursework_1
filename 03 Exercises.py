from lp import lp
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

def main():

    dir_name = 'C:\\Users\\Cobin\\Desktop\\Personal\\Projects\\4M17_Coursework_1\\2022-data\\2022-data\\Q3'
    array_A, array_x0 = openFile(dir_name, 'A', 'x0')
    A = array_A
    x = array_x0
    col_norm = np.linalg.norm(A, axis=0)
    A_normalized = np.divide(A, np.transpose(col_norm))

    # Set the optimality tolerance of the linear programing solver
    tol_lp = 100
    b = A_normalized @ x

    x_lp = lp(A_normalized, b, tol_lp)
    x_lp[np.absolute(x_lp) <= 1e-4] = 0
    plt.figure(1)
    plt.title('Original Signal')
    plt.plot(x)
    plt.figure(2)
    plt.title('Reconstructed Signal')
    plt.plot(x_lp, color='red')
    plt.show()

def openFile(dir_name, data_A, data_b):
    array_A = pd.read_csv(os.path.join(dir_name, data_A + "." + 'csv'), sep=',', header=None).to_numpy()
    array_b = pd.read_csv(os.path.join(dir_name, data_b + "." + 'csv'), sep=',', header=None).to_numpy()
    return array_A, array_b


if __name__ == "__main__":
    main()

