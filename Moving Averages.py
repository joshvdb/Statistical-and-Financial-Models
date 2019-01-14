import numpy as np
from numba import jit


@jit(nopython=True, cache=False)
def sma(matrix, interval):
    """
    Function to implement a Simple Moving Average, optimized with Numba.

    :param matrix: np.array([float])
    :param interval: int
    :return: np.array([float])
    """

    # declare empty SMA numpy array
    s = np.zeros((matrix.shape[0] - interval))

    # calculate the value of each point in the Simple Moving Average array
    for t in range(0, s.shape[0]):
        s[t] = np.sum(matrix[t:t + interval])/interval

    return s


@jit(nopython=True, cache=False)
def ema(matrix, alpha):
    """
    Function to implement an Exponential Moving Average, optimized with Numba. The variable alpha represents the degree
    of weighting decrease, a constant smoothing factor between 0 and 1. A higher alpha discounts older observations faster.

    :param matrix: np.array([float])
    :param alpha: float
    :return: np.array([float])
    """

    # declare empty EMA numpy array
    e = np.zeros(matrix.shape[0])

    # set the value of the first element in the EMA array
    e[0] = matrix[0]

    # use the EMA formula to calculate the value of each point in the EMA array
    for t in range(1, matrix.shape[0]):
        e[t] = alpha*matrix[t] + (1 - alpha)*e[t - 1]

    return e


# declare SMA and EMA variables
interval = 2
alpha = 0.1
matrix = np.array([1.5, 1.8, 1.9, 2.2, 1.4, 1.8, 1.95, 1.87, 1.88, 1.89])

# print results
print(ema(matrix, alpha))
print(sma(matrix, interval))
