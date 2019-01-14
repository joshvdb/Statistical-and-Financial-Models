# class to define and run analytics on a bond
# import the required Python packages
from numpy import arange

# declare the Bond class
class Bond:
    def __init__(self, price, coupon, call_price, face_value, t, years_to_maturity, num_coupon_periods):
        """
        Function to initialize the bond object by setting the basic bond variables.

        :param t: float - number of years left until the call date of the bond
        :param price: float - market price of the bond
        :param coupon: float - annual coupon of the bond
        :param call_price: float - call price of the bond
        :param face_value: float - face value of the bond
        :param num_coupon_periods: float - the number of coupon payments per year
        :param years_to_maturity: float - number of years remaining until the bond reaches maturity
        :return:
        """

        self.years_to_maturity = years_to_maturity
        self.n = int(years_to_maturity*num_coupon_periods)
        self.t = t
        self.price = price
        self.coupon = coupon
        self.call_price = call_price
        self.face_value = face_value
        self.num_coupon_periods = num_coupon_periods

    def calculate_bond_analytics(self):
        """
        Function to calculate the the Yield to Maturity (YTM), Yield to Call (YTC), Yield to Worst (YTW), Macaulay Duration
        and Modified Duration of a bond.

        :return: float, float, float, float, float
        """

        def f_ytm(ytm):
            """
            Define the bond pricing formula with YTM as the independent variable, with price moved to the right-hand side,
            setting the equation to zero. This allows us to find the root - the YTM.

            :param ytm: float
            :return: float
            """

            c = (self.coupon * 0.01 * self.face_value)/self.num_coupon_periods

            return ((c * (1 - (1 / ((1 + ytm) ** self.n)))) / ytm) + (self.face_value / ((1 + ytm) ** self.n)) - self.price

        def f_ytc(ytc):
            """
            Define the bond pricing formula with YTC as the independent variable, with price moved to the right-hand side,
            setting the equation to zero. This allows us to find the root - the YTC.

            :param ytc: float
            :return: float
            """

            c = self.coupon*0.01*self.face_value

            return (c / 2)*((1 - (1 + ytc/2) ** (-(2 * self.t)))/(ytc/2)) + self.call_price/((1 + ytc/2)**(2 * self.t)) - self.price

        def calculate_yields():
            """
            Function that approximates the root of the function f_YTM(YTM) and f_YTC(YTC), in order to approximate the
            Annual YTM, YTC and YTW of the bond

            :return: float, float, float
            """

            # define the range and increment over which we can find the bond yields (0 to 1)
            x1 = arange(0.0001, 1.0, 0.0001)

            # calculate the functions based on each YTM and YTC value - we then need to find the value closest to 0
            # (that is, where the price calculated from the yield = the actual bond price)
            y1 = f_ytm(x1)
            y2 = f_ytc(x1)

            # find the value in x1 that is closest to 0 - start at the index: initial = 0
            initial_ytm = 0
            initial_ytc = 0

            # search through the values of f_YTM(x), f_YTC(x) - the smallest absolute value (closest to zero) is the
            # best approximation for the actual YTM and YTC
            for i in range(0, len(y1)):
                # check if the current YTM value is a better approximation
                if abs(y1[initial_ytm]) > abs(y1[i]):
                    initial_ytm = i
                # check if the current YTC value is a better approximation
                if abs(y2[initial_ytc]) > abs(y2[i]):
                    initial_ytc = i

            # calculate the YTM and YTC values based on the approximations above
            ytm = self.num_coupon_periods*x1[initial_ytm]
            ytc = x1[initial_ytc]

            # calculate the YTW value - the minimum of the YTM and YTC
            ytw = min(ytm, ytc)

            # return the YTM, YTC and YTW of the bond
            return ytm, ytc, ytw

        def calculate_durations(ytm):
            """
            Function that calculates the Macaulay Duration and Modified Duration of the bond, based on the Annual YTM
            of the bond

            :param ytm: float
            :return: float, float
            """

            # calculate the coupon of the bond
            i = self.coupon * 0.01
            # calculate the periodic coupon payment of the bond
            c = self.coupon * 0.01 * self.face_value

            # calculate the present value of the bond
            pv = 0
            for t in range(1, self.n + 1, 1):
                pv += ((t * c) / ((1 + i) ** t))

            # calculate the Macaulay Duration of the bond
            macaulay_duration = (pv + ((self.n * self.face_value) / ((1 + i) ** t))) / self.price

            # calculate the Modified Duration of the bond
            modified_duration = macaulay_duration/(1 + ytm/self.num_coupon_periods)

            # return the Macaulay Duration and Modified Duration of the bond
            return macaulay_duration, modified_duration

        # calculate the YTM, YTC and YTW of the bond
        ytm, ytc, ytw = calculate_yields()

        # calculate the Macaulay Duration and Modified Duration of the bond
        macaulay_duration, modified_duration = calculate_durations(ytm)

        # print the values of the YTM, YTC, YTW, Macaulay Duration and Modified Duration of the bond
        print('Yield to Maturity (Annual) = {0:.2f}%'.format((ytm * 100)))
        print('Yield to Call (Annual) = {0:.2f}%'.format((ytc * 100)))
        print('Yield to Worst (Annual) = {0:.2f}%'.format((ytw * 100)))
        print('Macaulay Duration = {0:.2f} years'.format(macaulay_duration))
        print('Modified Duration = {0:.2f} years'.format(modified_duration))

        # return the values of the YTM, YTC, YTW, Macaulay Duration and Modified Duration of the bond
        return ytm, ytc, ytw, macaulay_duration, modified_duration

# instantiate a bond with market price = $1000, annual coupon = 5%, call price = $1100, face value = $1000, years left until
# the call date = 5, years remaining until maturity = 5 and number of coupon payments per year = 2
bond = Bond(1000, 5, 1100, 1000, 5, 5, 1)

# run analytics on the bond
bond.calculate_bond_analytics()
