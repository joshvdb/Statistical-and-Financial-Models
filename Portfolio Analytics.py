import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt

def Portfolio_Returns(portfolio, weights):
    """
    Calculate the total portfolio returns time-series array by multiplying the weights matrix (dimensions = 1*N) and the
    portfolio matrix (dimensions = N*t), for N securities in the portfolio and t returns per security. This function
    returns a 1*N matrix where each element in the matrix is the portfolio return fo a given time.

    :param portfolio: np.array([float])
    :param weights: np.array([float])
    :return: np.array([float])
    """

    # the portfolio returns are given by the dot product of the weights matrix and the portfolio matrix
    portfolio_returns = np.dot(weights, portfolio)

    return portfolio_returns

def Alpha(portfolio_returns, risk_free_rate, market_returns, beta):
    """
    Calculate the Alpha of the portfolio.

    :param portfolio_returns: np.array([float])
    :param risk_free_rate: np.array([float])
    :param market_returns: np.array([float])
    :param beta: float
    :return: float
    """

    # the portfolio Alpha is given by the below equation, as stated by the Capital Asset Pricing Model
    alpha = sum(portfolio_returns - risk_free_rate + beta*(market_returns - risk_free_rate))

    return alpha

def Beta(portfolio_returns, market_returns):
    """
    Calculate the Beta of the portfolio.

    :param portfolio_returns: np.array([float])
    :param market_returns: np.array([float])
    :return: float
    """

    # the portfolio Beta is given by the covariance of the returns of the portfolio and the returns of the market,
    # divided by the variance of the returns of the market
    beta = np.cov(np.array([portfolio_returns, market_returns]))[0][1]/np.cov([portfolio_returns, market_returns])[1][1]

    return beta

def Portfolio_Volatility(returns, weights):
    """
    Calculate the total portfolio volatility (the variance of the returns of the portfolio) using a covariance matrix.

    :param returns: np.array([float])
    :param weights: np.array([float])
    :return: float
    """

    # generate the transform of the 1D numpy weights array
    W_T = [[x] for x in weights]

    # calculate the covariance matrix of the asset returns
    covariance = np.cov(returns)

    # calculate the portfolio volatility
    portfolio_volatility = np.dot(np.dot(weights, covariance), W_T)[0]

    return portfolio_volatility

def Sharpe_Ratio(portfolio_returns, risk_free_rate, asset_returns, weights):
    """
    Calculate the Sharpe ratio of the portfolio.

    :param portfolio_returns: np.array([float])
    :param risk_free_rate: float
    :param asset_returns: np.array([float])
    :param weights: np.array([float])
    :return: float
    """

    # calculate the standard deviation of the returns of the portfolio
    portfolio_standard_deviation = np.sqrt(Portfolio_Volatility(asset_returns, weights))

    # calculate the Sharpe ratio of the portfolio
    sharpe_ratio = sum((portfolio_returns - risk_free_rate))/portfolio_standard_deviation

    return sharpe_ratio

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

    # calculate the standard deviation of the portfolio
    sigma = np.sqrt(Portfolio_Volatility(returns, weights))

    # calculate the mean return of each asset
    mean_asset_returns = [np.sum(x)/len(x) for x in returns]

    # calculate the mean return of the portfolio
    mu = np.sum(mean_asset_returns)/returns.shape[0]

    # set the plot range at 9 standard deviations from the mean in both the left and right directions
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

    # calculate the Historical VaR of the portfolio - check if the daily return value is less than a% - if it is, then
    # it is counted in the Historical VaR calculation
    relevant_returns = 0
    for i in range(0, portfolio_returns.shape[0]):
        if portfolio_returns[i] < a:
            relevant_returns += 1

    Historical_VaR = relevant_returns/portfolio_returns.shape[0]

    # print the Historical VaR
    print('Historical VaR = ' + str(Historical_VaR * 100) + '% at ' + str(a * 100) + '% of daily returns')

    return portfolio_returns, Historical_VaR;

def plot_Analytical_VaR(x, pdf, a):
    """
    Plot the normal distribution of portfolio returns - show the area under the normal curve that corresponds to the
    Analytical VaR of the portfolio.

    :param x: np.array([float])
    :param pdf: np.array([float])
    :param a: float
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

# instead of using a defined bin width, see if there is a built-in function to plot the frequency of each element of the
# returns. Pass an optional argument (bin_width = False) to allow the user to choose if they want to group returns into
# bins
def plot_Historical_VaR(x, a, bins):
    """
    Plot a histogram showing the distribution of portfolio returns - mark the cutoff point that corresponds to the
    Historical VaR of the portfolio.

    :param x: np.array([float])
    :param a: float
    :param bins: np.array([float])
    :return:
    """
    plt.hist(x, bins, label='Distribution of Returns')
    plt.axvline(x=a, ymin=0, color='r', label='Historical VaR cutoff point')
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

# calculate the returns of the portfolio
portfolio_returns = Portfolio_Returns(X, W)

print('Historical Portfolio Returns = ' + str(portfolio_returns))

# declare the historical returns of the benchmark index
market_returns = np.array([0.8,1.9,0.1003,10.9,0.1])

# calculate the
beta = Beta(portfolio_returns, market_returns)

print('Portfolio Beta = ' + str(beta))

print('Portfolio Volatility = ' + str(Portfolio_Volatility(portfolio_returns, W)))

print('Portfolio Alpha = ' + str(Alpha(portfolio_returns, 0.02, market_returns, beta)))

print('Portfolio Sharpe Ratio = ' + str(Sharpe_Ratio(portfolio_returns, 0.02, X, W)))

# calculate the Analytical VaR at -5% daily returns and the associated values
x, pdf, a, A_VaR = Analytical_VaR(X, W, -0.05, 100000)

# plot the Analytical VaR
plot_Analytical_VaR(x, pdf, a)

# calculate the overall historical portfolio returns, and the Historical VaR at 50% daily returns
portfolio_returns, H_VaR = Historical_VaR(X, W, 0.5)

# plot the Historical VaR
bins = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
plot_Historical_VaR(portfolio_returns, 0.5, bins)

# perform unit testing to verify the values of the Analytical and Historical VaR
import unittest

class VaR_Testing(unittest.TestCase):
    # define the VaR testing function
    def test_VaR(self):
        # calculate the Historical VaR
        _, H_VaR = Historical_VaR(X, W, 0.5)
        # calculate the Analytical VaR
        _, _, _, A_VaR = Analytical_VaR(X, W, -0.05, 100000)
        # test the Historical VaR - confirm that it has the correct value
        self.assertEqual(H_VaR, 0.4)
        # test the Analytical VaR - confirm that it has the correct value to 4 decimal places
        self.assertEqual(round(A_VaR, 4), 0.1237)

unittest.main()
