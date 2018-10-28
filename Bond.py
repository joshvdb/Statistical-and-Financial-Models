# use the Newton-Raphson method to sove for Yield to Maturity
# import the required Python packages
from matplotlib import *
from numpy import arange

# declare the Bond class
class Bond:
    def __init__(self, price, coupon, face_value, n):
        """
        Function to initialize the YTM object by setting the basic bond variables - market price, coupon, face value and number of
        payments remaining until the bond reaches maturity.
        """
        self.price = price
        self.coupon = coupon
        self.face_value = face_value
        self.n = n

    def calculate_YTM(self):
        """
        Function to calculate the Yield to Maturity (YTM) of a bond.
        """

        # define the bond pricing formula, with price moved to the right-hand side, setting the equation to zero
        # in order to use it in the Newton-Raphson process. This is a function of the Semi-Annual YTM
        def f(ytm):
            return (((self.coupon * (1 - (1 / ((1 + ytm) ** self.n)))) / ytm) + (
                        self.face_value / ((1 + ytm) ** self.n)) - self.price);

        # define the derivative of the function f(ytm)
        def dfdx(ytm):
            dfdx = self.coupon / (((1 + ytm) ** self.n) * ytm ** 2) + (self.coupon * self.n) / (
                        ((1 + ytm) ** (self.n + 1)) * ytm) - (
                               (self.n * self.face_value) / (1 + ytm) ** (self.n + 1)) - self.coupon / ytm ** 2
            return dfdx

        # function that uses the the Newton-Raphson process to find a root of function f(ytm)
        def froot(f, dfdx):
            # find the initial starting point for the Newton-Raphson process - we define x1 as the
            # range of possible values for the YTM - between 0.0001 and 1.0 (we use 0.0001 in
            # order to prevent obtaining a NaN value if we used x = 0)
            x1 = arange(0.0001, 1.0, 0.0001)
            y1 = f(x1)

            # find the value in x1 that is closest to 0 - start at the index: initial = 0
            initial = 0

            # search through the values of f(x) - use the smallest possible absolute value as the initial guess
            for i in range(0, len(y1)):
                if abs(y1[initial]) > abs(y1[i]):
                    initial = i

            # set the initial guess value for YTM, based on the index value of 'initial', found above
            x = x1[initial]

            # Define variables for the Newton-Raphson process
            err = 1.0e-6
            nmax = 30
            error = 1.0
            n = 0

            # Calculate the root approximation, using the Newton-Raphson process
            while ((abs(error) > abs(err)) and (n < nmax)):
                n = n + 1
                error = - f(x) / dfdx(x)
                x = x + error
            return x

        # Find approximation for the root of f(ytm) - the value of the Semi-Annual Yield to Maturity of the bond
        YTM = froot(f, dfdx)

        # print the value of the Semi-Annual Yield to Maturity of the bond, to 2 decimal places
        print ('Yield to Maturity (Semi-Annual) = {0:.2f}%'.format((YTM * 100)))

        # return the value of the Semi-Annual Yield to Maturity
        return YTM;

# instantiate a bond with market price = $95.92, coupon = 2.5%, face value = $100 and number of payments remaining = 5
bond = Bond(95.92,2.5,100.0,5.0)

# calcualte the Yield to Maturity of the bond
bond.calculate_YTM()
