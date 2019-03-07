import numpy as np
from numba import jit
from scipy import integrate
import statsmodels.api as sm
from datetime import datetime
import matplotlib.pyplot as plt
import pandas_datareader.data as web


@jit(nopython=True, cache=False)
def sma(matrix, interval):
    """
    Function to implement a Simple Moving Average (SMA), optimized with Numba.

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
    Function to implement an Exponential Moving Average (EMA), optimized with Numba. The variable alpha represents the
    degree of weighting decrease, a constant smoothing factor between 0 and 1. A higher alpha discounts older
    observations faster.

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


@jit(nopython=True, cache=False)
def twap(high, low, open, close, interval):
    """
    Function to implement a Time-Weighted Average Price (TWAP), optimized with Numba.

    :param high: np.array([float])
    :param low: np.array([float])
    :param open: np.array([float])
    :param close: np.array([float])
    :param interval: int
    :return: np.array([float])
    """

    # calculate prices data for each day
    prices = (high + low + open + close) / 4

    # declare empty TWAP numpy array
    p = np.zeros((prices.shape[0] - interval))

    # calculate the value of each point in the TWAP array
    for t in range(0, p.shape[0]):
        p[t] = np.sum(prices[t:t + interval]) / interval

    return p


@jit(nopython=True, cache=False)
def vwap(high, low, close, volumes, interval):
    """
    Function to implement a Volume-Weighted Average Price (VWAP), optimized with Numba.

    :param high: np.array([float])
    :param low: np.array([float])
    :param close: np.array([float])
    :param volumes: np.array([float])
    :param interval: int
    :return: np.array([float])
    """

    # calculate prices data for each day
    prices = (high + low + close) / 3

    # declare empty VWAP numpy array
    p = np.zeros((prices.shape[0] - interval))

    # calculate the value of each point in the VWAP array
    for t in range(0, p.shape[0]):
        p[t] = np.sum(prices[t:t + interval]*volumes[t:t + interval]) / np.sum(volumes[t:t + interval])

    return p


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


def alpha_rf(port_returns, risk_free_rate, market_returns, b):
    """
    Calculate the Alpha of the portfolio.

    :param port_returns: np.array([float])
    :param risk_free_rate: np.array([float])
    :param market_returns: np.array([float])
    :param b: float
    :return: float
    """

    # the portfolio Alpha is given by the below equation, as stated by the Capital Asset Pricing Model
    alpha = np.mean(port_returns) - risk_free_rate + b*(np.mean(market_returns) - risk_free_rate)

    return alpha


def portfolio_analytics(port_returns, market_returns):
    """
    Perform a regression on the portfolio returns and benchmark returns to calculate the Alpha, beta, and R-squared of
    the portfolio. This function will also return a numpy array containing the regression prediction values.

    :param port_returns: np.array([float])
    :param market_returns: np.array([float])
    :return: float, float, float, np.array([float])
    """

    # add the intercept to the model
    x2 = sm.add_constant(market_returns)

    # train the model
    estimator = sm.OLS(port_returns, x2)
    model = estimator.fit()

    # get portfolio analytics
    alpha, beta = model.params
    r_squared = model.rsquared
    regression = model.predict()

    return alpha, beta, r_squared, regression


def portfolio_volatility(returns, weights):
    """
    Calculate the total portfolio volatility (the variance of the historical returns of the portfolio) using a
    covariance matrix.

    :param returns: np.array([float])
    :param weights: np.array([float])
    :return: float
    """

    # generate the transform of the 1D numpy weights array
    w_T = np.array([[x] for x in weights])

    # calculate the covariance matrix of the asset returns
    covariance = np.cov(returns)

    # calculate the portfolio volatility
    port_volatility = np.dot(np.dot(weights, covariance), w_T)[0]

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
    sr = (np.mean(port_returns) - risk_free_rate)/portfolio_standard_deviation

    return sr


def tracking_error(port_returns, market_returns):
    """
    Calculate the Tracking Error of the portfolio relative to the benchmark.

    :param port_returns: np.array([float])
    :param market_returns: np.array([float])
    :return: float
    """

    return np.std(port_returns - market_returns)


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


def historical_var(returns, weights, a, num_plot_points):
    """
    Calculate the Historical VaR for a portfolio comprised of returns.shape[0] assets. This takes in several parameters.
    The first is a numpy array - returns, where each row corresponds to the historical daily returns (given as a decimal
    value) of each asset. Next the numpy array, weights, represents the weight of each asset in the portfolio. Finally,
    the integer a is the percentage point from which we take the VaR - for example, a value of a = 0.05 represents the
    cdf up to 5% of daily portfolio returns. The variable num_plot_points is the number of plot points to use on the
    horizontal axis (the %daily return) when plotting the Historical VaR.

    :param returns: np.array([float])
    :param weights: np.array([float])
    :param a: float
    :param num_plot_points: int
    :return: float, np.array([float])
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

    # sort the array of the portfolio returns in ascending order
    sorted_returns = sorted(port_returns, reverse=False)

    # create a numpy array of the bins to use for plotting the Historical VaR, based on the maximum and minimum values
    # of the portfolio returns, and the number of plot points to include
    bins = np.linspace(sorted_returns[0], sorted_returns[-1], num_plot_points)

    return h_var, bins


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


def plot_historical_returns(port_returns, market_returns):
    """
    Function to plot the historical portfolio returns.

    :param port_returns: np.array([float])
    :param market_returns: np.array([float])
    :return:
    """

    # define x-axis data points
    x = np.linspace(0, port_returns.shape[0], port_returns.shape[0])

    plt.plot(x, port_returns, linewidth=1, color='b', label='Portfolio Returns')
    plt.plot(x, market_returns, linewidth=1, color='r', label='Benchmark Returns')
    plt.legend(loc='upper left')
    plt.xlabel('Time (days)')
    plt.ylabel('Daily Return')
    plt.title('Daily Return vs Time')
    plt.show()


def plot_returns_regression(port_returns, market_returns, regression):
    """
    Function to plot the Returns Regression.

    :param port_returns: np.array([float])
    :param market_returns: np.array([float])
    :param regression: np.array([float])
    :return:
    """

    plt.scatter(market_returns, port_returns, marker='.', linewidth=1, color='b', label='Actual Returns')
    plt.plot(market_returns, regression, linewidth=1, color='k', label='Returns Regression')
    plt.legend(loc='upper left')
    plt.xlabel('Benchmark Daily Return')
    plt.ylabel('Portfolio Daily Return')
    plt.title('Returns Regression')
    plt.show()


def plot_equity_prices(ticker, prices):
    """
    Function to plot the prices of a single equity.

    :param ticker: str
    :param prices: np.array([float])
    :return:
    """

    # define x-axis data points
    x = np.linspace(0, prices.shape[0], prices.shape[0])

    plt.plot(x, prices[ticker], linewidth=1, color='b', label=ticker)
    plt.legend(loc='upper left')
    plt.xlabel('Time (days)')
    plt.ylabel('Price')
    plt.title('Price vs Time: ' + ticker)
    plt.show()


def equity_price_analytics(ticker, high_prices, low_prices, open_prices, close_prices, volume, alpha, interval):
    """
    Function to calculate the mean price, EMA, SMA, TWAP and VWAP of a single equity.

    :param ticker: str
    :param high_prices: pd.DataFrame([float])
    :param low_prices: pd.DataFrame([float])
    :param open_prices: pd.DataFrame([float])
    :param close_prices: pd.DataFrame([float])
    :param volume: pd.DataFrame([float])
    :param alpha: float
    :param interval: int
    :return: np.array([float]), np.array([float]), np.array([float]), np.array([float]), np.array([float])
    """

    # get price and volume data in numpy array form
    high = np.array(high_prices[ticker])
    low = np.array(low_prices[ticker])
    open = np.array(open_prices[ticker])
    close = np.array(close_prices[ticker])
    volume = np.array(volume[ticker])

    # calculate price analytics
    e_ma = ema(close, alpha)
    s_ma = sma(close, interval)
    t_wap = twap(high, low, open, close, interval)
    v_wap = vwap(high, low, close, volume, interval)
    mean_prices = (high + low + open + close) / 4

    return e_ma, s_ma, t_wap, v_wap, mean_prices


def plot_equity_price_analytics(ticker, e_ma, s_ma, t_wap, v_wap, mean_prices):
    """
    Function to plot the mean price, EMA, SMA, TWAP and VWAP of a single equity.

    :param ticker: str
    :param e_ma: np.array([float])
    :param s_ma: np.array([float])
    :param t_wap: np.array([float])
    :param v_wap: np.array([float])
    :param mean_prices: np.array([float])
    :return:
    """

    # define x-axis data points
    x = np.linspace(0, mean_prices.shape[0], mean_prices.shape[0])

    plt.plot(x, mean_prices, linewidth=1, color='k', label='Mean Price')
    plt.plot(x, e_ma, linewidth=1, color='r', label='EMA Price')
    plt.plot(x[interval:], s_ma, linewidth=1, color='b', label='SMA Price')
    plt.plot(x[interval:], t_wap, linewidth=1, color='g', label='TWAP Price')
    plt.plot(x[interval:], v_wap, linewidth=1, color='m', label='VWAP Price')
    plt.legend(loc='upper left')
    plt.xlabel('Time (days)')
    plt.ylabel('Price')
    plt.title('Price vs Time: ' + ticker)
    plt.show()


def plot_equity_returns(ticker, returns):
    """
    Function to plot the returns of a single equity.

    :param ticker: str
    :param returns: np.array([float])
    :return:
    """

    # define x-axis data points
    x = np.linspace(0, returns.shape[0], returns.shape[0])

    plt.plot(x, returns[ticker], linewidth=1, color='b', label=ticker)
    plt.legend(loc='upper left')
    plt.xlabel('Time (days)')
    plt.ylabel('Daily Return')
    plt.title('Return vs Time for: ' + ticker)
    plt.show()


def get_data(data):
    """
    Function to convert pandas DataFrames into numpy array's of shape (n*m), where n = the number of equities and m =
    the number of days for which we have price or return data.

    :param data: pd.DataFrame([float])
    :return: np.array([float])
    """

    np_data = np.array(data)
    array = []

    for i in range(0, np_data.shape[1]):
        array.append(np_data[:, i])

    return np.array(array)


# declare equities and time data
start = datetime(2018, 1, 1)
end = datetime(2019, 3, 7)
equities = ['AAPL', 'GOOGL', 'BLK', 'IBM']

# get all price and volume data from Yahoo Finance
high_prices = web.DataReader(equities, 'yahoo', start, end)['High']
low_prices = web.DataReader(equities, 'yahoo', start, end)['Low']
open_prices = web.DataReader(equities, 'yahoo', start, end)['Open']
close_prices = web.DataReader(equities, 'yahoo', start, end)['Close']
volumes = web.DataReader(equities, 'yahoo', start, end)['Volume']

# get the S&P500 benchmark price data from Yahoo Finance
underlying = web.DataReader(['^GSPC'], 'yahoo', start, end)['Close']

# calculate the daily returns of the equities and the S&P500 benchmark
equity_returns = close_prices.div(close_prices.iloc[0])
benchmark_returns = underlying.div(underlying.iloc[0])

# plot the prices of a single equity
plot_equity_prices('AAPL', close_prices)

# plot the returns of a single equity
plot_equity_returns('AAPL', (equity_returns - 1))

# declare the weight of each asset in the portfolio - each element in the row corresponds to the weight of the Nth asset
w = np.array([0.15, 0.6, 0.2, 0.05])

# get historical return data for all equities
eq_returns = get_data(equity_returns) - 1

# calculate the historical returns of the portfolio
port_returns = portfolio_returns(eq_returns, w)

# declare the historical returns of the benchmark index
market_returns = np.array(benchmark_returns)[:, 0] - 1

# calculate portfolio analytics
alpha, beta, r_squared, regression = portfolio_analytics(port_returns, market_returns)

# plot historical returns
plot_historical_returns(port_returns, market_returns)

# plot Returns Regression
plot_returns_regression(port_returns, market_returns, regression)

# print the various portfolio analytics values
print('Portfolio R-Squared = ' + str(r_squared))

print('Portfolio Beta = ' + str(beta))

print('Portfolio Volatility = ' + str(portfolio_volatility(eq_returns, w)))

print('Portfolio Alpha (from the regression) = ' + str(alpha))

print('Portfolio Alpha (based on risk-free rate) = ' + str(alpha_rf(port_returns, 0.02, market_returns, beta)))

print('Portfolio Sharpe Ratio = ' + str(sharpe_ratio(port_returns, 0.02, eq_returns, w)))

print('Portfolio Tracking Error = ' + str(tracking_error(port_returns, market_returns)))

# calculate the Analytical VaR at -5% daily returns and the associated values
x, pdf, a, A_VaR = analytical_var(eq_returns, w, -0.02, 100000)

# plot the Analytical VaR
plot_analytical_var(x, pdf, a)

# calculate  the Historical VaR at 50% daily returns
H_VaR = historical_var(eq_returns, w, -0.02, 100)

# plot the Historical VaR
plot_historical_var(port_returns, -0.02, H_VaR[1])

# declare SMA and EMA variables
interval = 20
alpha = 0.3

# define the ticker of the stock to look at
ticker = 'AAPL'

# calculate price analytics
e_ma, s_ma, t_wap, v_wap, mean_prices = equity_price_analytics(ticker, high_prices, low_prices, open_prices, close_prices, volumes, alpha, interval)

# plot price analytics
plot_equity_price_analytics(ticker, e_ma, s_ma, t_wap, v_wap, mean_prices)
