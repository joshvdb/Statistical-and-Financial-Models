import unittest
import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt


def portfolio_returns(returns, weights):
    """
    Calculate the total portfolio returns time-series array by multiplying the weights matrix (dimensions = 1*N) and the
    portfolio matrix (dimensions = N*t), for N securities in the portfolio and t returns per security. This function
    returns a 1*N matrix where each element in the matrix is the portfolio return fo a given time.

    :param returns: np.array([float])
    :param weights: np.array([float])
    :return: np.array([float])
    """

    # the portfolio returns are given by the dot product of the weights matrix and the portfolio matrix
    port_returns = np.dot(weights, returns)

    return port_returns


def alpha(port_returns, risk_free_rate, market_returns, B):
    """
    Calculate the Alpha of the portfolio.

    :param port_returns: np.array([float])
    :param risk_free_rate: np.array([float])
    :param market_returns: np.array([float])
    :param B: float
    :return: float
    """

    # the portfolio Alpha is given by the below equation, as stated by the Capital Asset Pricing Model
    A = np.sum(port_returns - risk_free_rate + B*(market_returns - risk_free_rate))

    return A


def beta(port_returns, market_returns):
    """
    Calculate the Beta of the portfolio.

    :param port_returns: np.array([float])
    :param market_returns: np.array([float])
    :return: float
    """

    # the portfolio Beta is given by the covariance of the returns of the portfolio and the returns of the market,
    # divided by the variance of the returns of the market
    B = np.cov(np.array([port_returns, market_returns]))[0][1]/np.cov([port_returns, market_returns])[1][1]

    return B


def portfolio_volatility(returns, weights):
    """
    Calculate the total portfolio volatility (the variance of the historical returns of the portfolio) using a
    covariance matrix.

    :param returns: np.array([float])
    :param weights: np.array([float])
    :return: float
    """

    # generate the transform of the 1D numpy weights array
    W_T = np.array([[x] for x in weights])

    # calculate the covariance matrix of the asset returns
    covariance = np.cov(returns)

    # calculate the portfolio volatility
    port_volatility = np.dot(np.dot(weights, covariance), W_T)[0]

    return port_volatility


def sharpe_ratio(port_returns, risk_free_rate, asset_returns, weights):
    """
    Calculate the Sharpe ratio of the portfolio.

    :param port_returns: np.array([float])
    :param risk_free_rate: float
    :param asset_returns: np.array([float])
    :param weights: np.array([float])
    :return: float
    """

    # calculate the standard deviation of the returns of the portfolio
    portfolio_standard_deviation = np.sqrt(portfolio_volatility(asset_returns, weights))

    # calculate the Sharpe ratio of the portfolio
    sr = np.sum((port_returns - risk_free_rate))/portfolio_standard_deviation

    return sr


def r_squared(port_returns, market_returns):
    """
    Calculate the R-Squared value of the portfolio.

    :param port_returns: np.array([float])
    :param market_returns: np.array([float])
    :return: float
    """

    # calculate the R-Squared value of the portfolio using the standard statistical definition of R-Squared
    rs = 1 - np.var(port_returns-market_returns)/np.var(market_returns)

    return rs


def analytical_var(returns, weights, a, n):
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
    sigma = np.sqrt(portfolio_volatility(returns, weights))

    # calculate the mean return of the portfolio
    mu = np.sum(portfolio_returns(returns, weights))/returns.shape[1]

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
    a_var = integrate.simps(pdf[0:risk_range], x[0:risk_range])

    # print the Analytical VaR
    print('Analytical VaR = ' + str(a_var * 100) + '% at ' + str(x[risk_range] * 100) + '% of daily returns')

    return x, pdf, risk_range, a_var


def historical_var(returns, weights, a):
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

    # calculate the total portfolio returns
    port_returns = portfolio_returns(returns, weights)

    # calculate the Historical VaR of the portfolio - check if the daily return value is less than a% - if it is, then
    # it is counted in the Historical VaR calculation
    relevant_returns = 0
    for i in range(0, port_returns.shape[0]):
        if port_returns[i] < a:
            relevant_returns += 1

    h_var = relevant_returns/port_returns.shape[0]

    # print the Historical VaR
    print('Historical VaR = ' + str(h_var * 100) + '% at ' + str(a * 100) + '% of daily returns')

    return h_var


def plot_analytical_var(x, pdf, a):
    """
    Plot the normal distribution of portfolio returns - show the area under the normal curve that corresponds to the
    Analytical VaR of the portfolio. The variable x is the range of daily returns of the portfolio (the x-axis), pdf
    is the Normal Distribution of the historical returns of the portfolio (the y-axis), and a is the cutoff value.

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


def plot_historical_var(x, a, bins):
    """
    Plot a histogram showing the distribution of portfolio returns - mark the cutoff point that corresponds to the
    Historical VaR of the portfolio. The variable x is the historical distribution of returns of the portfolio, a is
    the cutoff value, and the bins are the bins in which to stratify the historical returns.

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


# declare the weight of each asset in the portfolio - each element in the row corresponds to the weight of the Nth asset
W = np.array([0.25, 0.15, 0.2, 0.3, 0.05, 0.05])

# declare the returns of each asset in the portfolio - each row represents an asset, and each element in a row is the
# daily return of that asset at a point in time. Here we have 6 assets, with 5 historical returns each
X = np.array([[0.1, 0.3, 0.4, 0.8, 0.9],
               [0.20, 0.90, 0.25, 0.1, 0.09],
               [0.1, 0.85, 0.45, 0.29, 0.9],
              [0.1, 0.82, 4.2, 0.26, 0.9],
              [0.1, 0.82, 0.43, 0.23, 0.9],
              [0.32, 0.24, 0.24, 0.1, 0.55]])

# calculate the historical returns of the portfolio
port_returns = portfolio_returns(X, W)

# declare the historical returns of the benchmark index
market_returns = np.array([0.8,1.9,0.1003,10.9,0.1])

# calculate the Beta of the portfolio
B = beta(port_returns, market_returns)

# print the various portfolio analytics values
print('Historical Portfolio Returns = ' + str(port_returns))

print('Portfolio R-Squared = ' + str(r_squared(port_returns, market_returns)))

print('Portfolio Beta = ' + str(B))

print('Portfolio Volatility = ' + str(portfolio_volatility(port_returns, W)))

print('Portfolio Alpha = ' + str(alpha(port_returns, 0.02, market_returns, B)))

print('Portfolio Sharpe Ratio = ' + str(sharpe_ratio(port_returns, 0.02, X, W)))

# calculate the Analytical VaR at -5% daily returns and the associated values
x, pdf, a, A_VaR = analytical_var(X, W, -0.05, 100000)

# plot the Analytical VaR
plot_analytical_var(x, pdf, a)

# calculate  the Historical VaR at 50% daily returns
H_VaR = historical_var(X, W, 0.5)

# plot the Historical VaR
bins = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
plot_historical_var(port_returns, 0.5, bins)


# perform unit testing to verify the values of the Analytical and Historical VaR
class VaRTesting(unittest.TestCase):
    """
    Unit Testing class for the VaR functions.
    """
    def test_VaR(self):
        """
        Function to test the values of the Analytical and Historical VaR.

        :return:
        """

        # calculate the Historical VaR
        H_VaR = historical_var(X, W, 0.5)

        # calculate the Analytical VaR
        _, _, _, A_VaR = analytical_var(X, W, -0.05, 100000)

        # test the Historical VaR - confirm that it has the correct value
        self.assertEqual(H_VaR, 0.4)

        # test the Analytical VaR - confirm that it has the correct value to 6 decimal places
        self.assertEqual(round(A_VaR, 6), 0.080493)

unittest.main()
