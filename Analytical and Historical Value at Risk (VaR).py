import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt

def Analytical_VaR(returns, weights, a, n):
    """
    Calculate the Analytical VaR for a portfolio comprised of returns.shape[0] assets. This takes in several parameters.
    The first is a numpy array - returns, where each row corresponds to the historical daily returns (given as a decimal
    value) of each asset. Next the numpy array, weights, represents the weight of each asset in the portfolio. The
    integer a is the percentage point from which we take the VaR - for example, a value of a = 0.05 represents the cdf
    up to 5% of daily portfolio returns. Finally the integer n represents the number of data points to use when plotting
    the Normal Distribution.

    :param returns: np.array([float])
    :param weights: np.array([float])
    :param a: float
    :param n: int
    :return: np.array([float]), np.array([float]), int, float
    """

    # generate the transform of the 1D numpy weights array
    W_T = [[x] for x in weights]

    # calculate the covariance matrix of the asset returns
    cov = np.cov(returns)

    # calculate the standard deviation
    sigma = np.sqrt(np.dot(np.dot(weights, cov), W_T))

    # calculate the mean return of each asset
    mean_asset_returns = [np.sum(x)/len(x) for x in returns]

    # calculate the mean return of the portfolio
    mu = np.sum(mean_asset_returns)/returns.shape[0]

    # set the plot range at 9 standard deviations from the mean
    plot_range = 9 * sigma

    # set the bottom value on the x-axis (of % daily returns)
    bottom = mu - plot_range

    # set the top value on the x-axis (of % daily returns)
    top = mu + plot_range

    # declare the numpy array of the range of x values for the normal distribution
    x = np.linspace(bottom, top, n)

    # calculate the pdf of the normal distribution
    pdf = 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(- (x - mu) ** 2 / (2 * sigma ** 2))

    # calculate the index of the nearest daily return in x corresponding to a%
    risk_range = (np.abs(x - a)).argmin()

    # use Simpsons rule to estimate the cdf of the Normal distribution up to a%
    Analytical_VaR = integrate.simps(pdf[0:risk_range], x[0:risk_range])

    # print the Analytical VaR
    print('Analytical VaR = ' + str(Analytical_VaR*100) + '% at ' + str(x[risk_range]*100) + '% of daily returns')

    return x, pdf, risk_range, Analytical_VaR

def Historical_VaR(returns, weights, a):
    """
    Calculate the Historical VaR for a portfolio comprised of returns.shape[0] assets. This takes in several parameters.
    The first is a numpy array - returns, where each row corresponds to the historical daily returns (given as a decimal
    value) of each asset. Next the numpy array, weights, represents the weight of each asset in the portfolio. Finally,
    the integer a is the percentage point from which we take the VaR - for example, a value of a = 0.05 represents the
    cdf up to 5% of daily portfolio returns.

    :param returns: np.array([float])
    :param weights: np.array([float])
    :param a: float
    :return: np.array([float]), float
    """

    # generate the transform of the 1D numpy weights array
    W_T = [[x] for x in weights]

    # calculate the weighted time-series of portfolio returns
    portfolio_returns = np.sum(np.multiply(returns, W_T), axis=0)

    # calculate the Historical VaR of the portfolio
    relevant_returns = 0
    for i in range(0, portfolio_returns.shape[0]):
        if portfolio_returns[i] < a:
            relevant_returns += 1

    Historical_VaR = relevant_returns/portfolio_returns.shape[0]

    return portfolio_returns, Historical_VaR;

def plot_Analytical_VaR(x, pdf, a):
    """
    Add a shading under the graph!
    :param x: np.array([float])
    :param pdf: np.array([float])
    :return:
    """
    plt.plot(x, pdf, linewidth=2, color='r', label='Distribution of Returns')
    plt.fill_between(x[0:a], pdf[0:a], facecolor='blue', label='Analytical VaR')
    plt.legend(loc='upper left')
    plt.xlabel('Daily Returns')
    plt.ylabel('Frequency')
    plt.title('Frequency vs Daily Returns')
    plt.show()

    return;

# declare the weight of each asset in the portfolio - each element in the row is the weight of the nth asset
W = np.array([0.25, 0.15, 0.2, 0.3, 0.05, 0.05])

# declare the returns of each asset in the portfolio - each row represents an asset, each element in a row is the
# daily return of that asset at a point in time
X = np.array([[0.1, 0.3, 0.4, 0.8, 0.9],
               [0.20, 0.90, 0.25, 0.1, 0.09],
               [0.1, 0.85, 0.45, 0.29, 0.9],
              [0.1, 0.82, 4.2, 0.26, 0.9],
              [0.1, 0.82, 0.43, 0.23, 0.9],
              [0.32, 0.24, 0.24, 0.1, 0.55]])

# calculate the Analytical VaR and the associated values
x, pdf, a, A_VaR = Analytical_VaR(X, W, -0.05, 100000)

# print the Analytical VaR
print(A_VaR)

# plot the Analytical VaR
plot_Analytical_VaR(x, pdf, a)

# calculate the overall historical portfolio returns, and the Historical VaR
portfolio_returns, H_VaR = Historical_VaR(X, W, 0.5)

# print the overall historical portfolio returns, and the Historical VaR
print(portfolio_returns, H_VaR)

# perform unit testing to verify the values of the Analytical and Historical VaR
import unittest

class MyTestCase(unittest.TestCase):
    def test_something(self):
        _, H_VaR = Historical_VaR(X, W, 0.5)
        _, _, _, A_VaR = Analytical_VaR(X, W, -0.05, 100000)
        self.assertEqual(H_VaR, 0.4)
        self.assertEqual(round(A_VaR, 4), 0.1237)

if __name__ == '__main__':
    unittest.main()
