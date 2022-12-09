import numpy as np
from pylab import rcParams
from matplotlib import pyplot as plt
from algorithms import log_barrier_for_qp, interior_point_for_qp

rcParams['figure.figsize'] = 10, 5

def main():
    n = 100
    m = 200
    p_helper = np.random.rand(n, n)
    P = np.dot(p_helper, p_helper.T)
    q = np.random.rand(n, 1)
    A = np.random.rand(m, n)
    b = np.random.rand(m, 1)

    gaps = log_barrier_for_qp(P, q, A, b, alpha=0.01, beta=0.5, mu=20, iterations=200)

    rcParams['figure.figsize'] = 10, 5
    plt.plot(list(range(1, len(gaps) + 1)), gaps)
    plt.yscale('log')
    plt.xlabel('Iterations')
    plt.ylabel('Duality gap')
    plt.title('Log barrier method applied on the random convex QP')

    surrogates, dual_residuals = interior_point_for_qp(P, q, A, b, alpha=0.01, beta=0.5, mu=10, iterations=200,
                                                       tol=1e-4, eps=1e-6)
    iterations = list(range(1, len(surrogates) + 1))

    rcParams['figure.figsize'] = 15, 5

    plt.subplot(1, 2, 1)
    plt.plot(iterations, surrogates)
    plt.yscale('log')
    plt.xlabel('Iterations')
    plt.ylabel('Surrogate duality gap')
    plt.title('Interior point method applied on the random convex QP')

    plt.subplot(1, 2, 2)
    plt.plot(iterations, dual_residuals)
    plt.yscale('log')
    plt.xlabel('Iteration number')
    plt.ylabel('Error in minimisation')
    plt.title('Interior point method applied on the random convex QP')

if __name__ == "__main__":
    main()


