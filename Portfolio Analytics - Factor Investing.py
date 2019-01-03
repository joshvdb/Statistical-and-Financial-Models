import unittest
import numpy as np


def excess_returns(factor_exposures, factor_returns, specific_returns):
    """
    Calculate the excess returns of the overall portfolio, based on the Arbitrage Pricing Theory (APT).

    :param factor_exposures: np.array([float])
    :param factor_returns: np.array([float])
    :param specific_returns: np.array([float])
    :return: np.array([float])
    """

    K = np.array([np.sum(x) for x in factor_returns])

    e_returns = np.dot(factor_exposures, K) + specific_returns

    return e_returns


def specific_covariance(asset_returns):
    """
    Calculate the specific covariance matrix of the assets in the portfolio. This is used in the calculation of the
    overall portfolio volatility.

    :param asset_returns: np.array([float])
    :return: np.array([float])
    """

    covariance = np.cov(asset_returns)
    s_covariance = np.zeros((covariance.shape[0], covariance.shape[0]))

    for x in range(0, covariance.shape[0]):
        s_covariance[x][x] = covariance[x][x]

    return s_covariance


def portfolio_factor_volatility(factor_exposures, factor_returns, asset_returns, weights):
    """
    Calculate the volatility of the portfolio, based on the Arbitrage Pricing Theory (APT).

    :param factor_exposures: np.array([float])
    :param factor_returns: np.array([float])
    :param asset_returns: np.array([float])
    :param weights: np.array([float])
    :return: float
    """

    # generate the transform of the 1D numpy weights array
    X_T = factor_exposures.T

    # calculate the factor returns of the Factor returns
    F = np.cov(factor_returns)

    # calculate the portfolio volatility
    portfolio_covariance_matrix = np.dot(np.dot(factor_exposures, F), X_T) + specific_covariance(asset_returns)

    # generate the transform of the 1D numpy weights array
    W_T = np.array([[x] for x in weights])

    # calculate the portfolio volatility
    portfolio_volatility = np.dot(np.dot(weights, portfolio_covariance_matrix), W_T)[0]

    return portfolio_volatility


# declare the returns of each asset in the portfolio - each row represents an asset, and each element in a row is the
# daily return of that asset at a point in time. Here we have 5 assets, with 6 historical returns each
X = np.array([[0.1, 0.3, 0.4, 0.8, 0.9,0.8],
               [0.20, 0.90, 0.25, 0.1, 0.09,0.1],
               [0.1, 0.85, 0.45, 0.29, 0.9,0.3],
              [0.1, 0.82, 4.2, 0.26, 0.9,0.1],
              [0.1, 0.82, 0.43, 0.23, 0.9,0.1]])

# declare the exposure of each asset to each factor - each row represents an asset, and each element in a row is the
# exposure that the asset has to that factor. Here we have 5 assets and 6 factors
factor_exposures = np.array([[0.1, 0.3, 0.4, 0.2, 0.3, 0.09],
               [0.20, 0.90, 0.25, 0.1, 0.1,0.87],
               [0.1, 0.85, 0.45, 0.29, 0.2,0.19],
              [0.1, 0.82, 4.2, 0.26, 0.3,0.89],
              [0.1, 0.82, 0.43, 0.23, 0.4,0.98]])

# declare the historical returns for each factor - each row represents a factor, and each element in a row is the
# the daily return of that factor at a point in time. Here we have 6 factors, with 9 historical returns each
factor_returns = np.array([[0.12, 0.3, 0.4, 0.2, 0.3, 0.09, 0.2, 0.3, 0.09],
               [0.20, 0.90, 0.25, 0.1, 0.1, 0.87, 0.2, 0.3, 0.09],
               [0.1, 0.85, 0.90, 0.29, 0.2, 0.19, 0.2, 0.3, 0.09],
              [0.1, 0.82, 4.13, 0.26, 0.3, 0.89, 0.2, 0.3, 0.09],
              [0.1, 0.82, 0.57, 0.23, 0.4, 0.98, 0.2, 0.3, 0.09],
              [0.32, 0.24, 0.19, 0.21, 0.91, 0.2134, 0.2, 0.3, 0.09]])

# declare the weight of each asset in the portfolio - each element in the row is the weight of the nth asset
weights = np.array([0.25, 0.15, 0.2, 0.3, 0.05])

# declare the specific returns of each asset
specific_returns = np.array([0.1, 0.90, 0.45, 0.26, 0.9])

# print the excess returns and the portfolio volatility
print('Excess Returns of the portfolio = ' + str(excess_returns(factor_exposures, factor_returns, specific_returns)))

print('Volatility of the portfolio = ' + str(portfolio_factor_volatility(factor_exposures, factor_returns, X, weights)))


# perform unit testing to verify the values of the excess returns and the portfolio volatility
class VaRTesting(unittest.TestCase):
    """
    Unit Testing class for the excess returns and the portfolio volatility.
    """
    def test_VaR(self):
        """
        Function to test the values of the excess returns and the portfolio volatility

        :return:
        """

        # calculate the excess returns
        e_returns = excess_returns(factor_exposures, factor_returns, specific_returns)

        # calculate the portfolio volatility
        port_volatility = portfolio_factor_volatility(factor_exposures, factor_returns, X, weights)

        # test the excess returns - confirm that they have the correct value to 6 decimal places
        for i in range(0, e_returns.shape[0]):
            self.assertEqual(round(e_returns[i], 6), [5.216606, 8.192858, 7.914546, 21.361926, 10.636432][i])

        # test the portfolio volatility that it has the correct value to 6 decimal places
        self.assertEqual(round(port_volatility, 6), 0.936432)

unittest.main()
