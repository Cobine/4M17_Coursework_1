import numpy as np
import cvxpy as cp
import pandas as pd
import os
import functools
import time
import matplotlib.pyplot as plt

'''
Python Script which minimises the l1, l2, and linf norms for Ax-b given the matrices A and b
Uses cvxpy to implement the cost function and optimisation

Input: Takes as input two matrices array_A and array_b representing matrices A and b.
        Here imported from data and supplied as (A1,b1),...(A5,b5). Will run one set of these
        at a time.

Output: Runtime, optimal value, and histogram of the the residuals

For any changes, make these in def main(): and call the functions as necessary. 
'''

def main():

    dir_name = 'C:\\Users\\Cobin\\Desktop\\Personal\\Projects\\4M17_Coursework_1\\2022-data\\2022-data\\Q1'

    array_A, array_b = openFile(dir_name, 'A4', 'b4')
    l1_sol, l1_opt = l1(array_A,array_b)
    l2_sol, l2_opt = l2(array_A,array_b)
    linf_sol, linf_opt = linf(array_A,array_b)
    hist_plot(l1_opt,'l1')
    hist_plot(l2_opt,'l2')
    hist_plot(linf_opt,'l_inf')

def openFile(dir_name, data_A, data_b):
    array_A = pd.read_csv(os.path.join(dir_name, data_A + "." + 'csv'), sep=',', header=None).to_numpy()
    array_b = pd.read_csv(os.path.join(dir_name, data_b + "." + 'csv'), sep=',', header=None).to_numpy()
    return array_A, array_b

def timer(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        value = func(*args, **kwargs)
        end_time = time.perf_counter()
        run_time = end_time - start_time
        print("Finished {} in {} secs".format(repr(func.__name__), round(run_time, 3)))
        return value

    return wrapper

@timer
def l1(array_A, array_b):
    A = array_A
    b = np.ravel(array_b)
    # Define and solve the CVXPY problem.
    x = cp.Variable(np.shape(A)[1])
    cost = cp.norm1(A @ x - b)
    prob = cp.Problem(cp.Minimize(cost))
    prob.solve()

    # Print result.
    print("\nThe optimal value is", prob.value)
    print("The optimal x is")
    print(x.value)
 #   print("The norm of the residual is ", cp.norm(A @ x - b, p=2).value)
    return prob.value, x.value

@timer
def l2(array_A, array_b):
    A = array_A
    b = np.ravel(array_b)
    # Define and solve the CVXPY problem.
    x = cp.Variable(np.shape(A)[1])
    cost = cp.sum_squares(A @ x - b)
    prob = cp.Problem(cp.Minimize(cost))
    prob.solve()

    # Print result.
    print("\nThe optimal value is", prob.value)
    print("The optimal x is")
    print(x.value)
    return prob.value, x.value

@timer
def linf(array_A, array_b):
    A = array_A
    b = np.ravel(array_b)
    # Define and solve the CVXPY problem.
    x = cp.Variable(np.shape(A)[1])
    cost = cp.norm_inf(A @ x - b)
    prob = cp.Problem(cp.Minimize(cost))
    prob.solve()

    # Print result.
    print("\nThe optimal value is", prob.value)
    print("The optimal x is")
    print(x.value)
    #   print("The norm of the residual is ", cp.norm(A @ x - b, p=2).value)
    return prob.value, x.value

def hist_plot(plot_data, label):
    q25, q75 = np.percentile(plot_data, [25, 75])
    bin_width = 2 * (q75 - q25) * len(plot_data) ** (-1 / 3)
    bins = round((plot_data.max() - plot_data.min()) / bin_width)
    print("Freedman–Diaconis number of bins:", bins)
    plt.hist(plot_data,  bins=bins, density = True);
    plt.ylabel("Frequency")
    plt.xlabel(label)
    plt.title("Histogram");
    plt.xlim(-0.2, 0.2)
    plt.show()


if __name__ == "__main__":
    main()