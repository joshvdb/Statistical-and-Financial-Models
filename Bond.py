# class to define and run analytics on a bond
# import the required Python packages
from numpy import arange

# declare the Bond class
class Bond:
    def __init__(self, price, coupon, call_price, face_value, t, years_to_maturity, num_coupon_periods):
        """
        Function to initialize the bond object by setting the basic bond variables:
        t = number of years left until the call date of the bond
        price = market price of the bond
        coupon = annual coupon of the bond
        call_price = call price of the bond
        face_value = face value of the bond
        num_coupon_periods = the number of coupon payments per year
        n = number of payments remaining until the bond reaches maturity
        years_to_maturity = number of years remaining until the bond reaches maturity
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
        """

        def f_YTM(YTM):
            """
            Define the bond pricing formula with YTM as the independent variable, with price moved to the right-hand side,
            setting the equation to zero.
            This allows us to find the root - the YTM.
            """
            C = (self.coupon * 0.01 * self.face_value)/self.num_coupon_periods
            return ((C * (1 - (1 / ((1 + YTM) ** self.n)))) / YTM) + (self.face_value / ((1 + YTM) ** self.n)) - self.price;

        def f_YTC(YTC):
            """
            Define the bond pricing formula with YTC as the independent variable, with price moved to the right-hand side,
            setting the equation to zero.
            This allows us to find the root - the YTC.
            """
            C = self.coupon*0.01*self.face_value
            return (C / 2)*((1 - (1 + YTC/2) ** (-(2 * self.t)))/(YTC/2)) + self.call_price/((1 + YTC/2)**(2 * self.t)) - self.price;

        def calculate_yields():
            """
            Function that approximates the root of the function f_YTM(YTM) and f_YTC(YTC), in order to approximate the
            Annual YTM, YTC and YTW of the bond
            """

            # define the range and increment over which we can find the bond yields (0 to 1)
            x1 = arange(0.0001, 1.0, 0.0001)

            # calculate the functions based on each YTM and YTC value - we then need to find the value closest to 0
            # (that is, where the price calculated from the yield = the actual bond price)
            y1 = f_YTM(x1)
            y2 = f_YTC(x1)

            # find the value in x1 that is closest to 0 - start at the index: initial = 0
            initial_YTM = 0
            initial_YTC = 0

            # search through the values of f_YTM(x), f_YTC(x) - the smallest absolute value (closest to zero) is the
            # best approximation for the actual YTM and YTC
            for i in range(0, len(y1)):
                # check if the current YTM value is a better approximation
                if abs(y1[initial_YTM]) > abs(y1[i]):
                    initial_YTM = i
                # check if the current YTC value is a better approximation
                if abs(y2[initial_YTC]) > abs(y2[i]):
                    initial_YTC = i

            # calculate the YTM and YTC values based on the approximations above
            YTM = self.num_coupon_periods*x1[initial_YTM]
            YTC = x1[initial_YTC]

            # calculate the YTW value - the minimum of the YTM and YTC
            YTW = min(YTM, YTC)

            # return the YTM, YTC and YTW of the bond
            return YTM, YTC, YTW;

        def calculate_durations(ytm):
            """
            Function that calculates the Macaulay Duration and Modified Duration of the bond, based on the Annual YTM
            of the bond
            """

            # calculate the coupon of the bond
            i = self.coupon * 0.01
            # calculate the periodic coupon payment of the bond
            C = self.coupon * 0.01 * self.face_value

            # calculate the present value of the bond
            PV = 0
            for t in range(1, self.n + 1, 1):
                PV += ((t * C) / ((1 + i) ** t))

            # calculate the Macaulay Duration of the bond
            macaulay_duration = (PV + ((self.n * self.face_value) / ((1 + i) ** t))) / self.price

            # calculate the Modified Duration of the bond
            modified_duration = macaulay_duration/(1 + ytm/self.num_coupon_periods)

            # return the Macaulay Duration and Modified Duration of the bond
            return macaulay_duration, modified_duration;

        # calculate the YTM, YTC and YTW of the bond
        YTM, YTC, YTW = calculate_yields()

        # calculate the Macaulay Duration and Modified Duration of the bond
        macaulay_duration, modified_duration = calculate_durations(YTM)

        # print the values of the YTM, YTC, YTW, Macaulay Duration and Modified Duration of the bond
        print ('Yield to Maturity (Annual) = {0:.2f}%'.format((YTM * 100)))
        print('Yield to Call (Annual) = {0:.2f}%'.format((YTC * 100)))
        print('Yield to Worst (Annual) = {0:.2f}%'.format((YTW * 100)))
        print('Macaulay Duration = {0:.2f} years'.format(macaulay_duration))
        print('Modified Duration = {0:.2f} years'.format(modified_duration))

        # return the values of the YTM, YTC, YTW, Macaulay Duration and Modified Duration of the bond
        return YTM, YTC, YTW, macaulay_duration, modified_duration;

# instantiate a bond with market price = $1000, annual coupon = 5%, call price = $1100, face value = $1000, years left until
# the call date = 5, years remaining until maturity = 5 and number of coupon payments per year = 2
bond = Bond(1000, 5, 1100, 1000, 5, 5, 1)

# run analytics on the bond
bond.calculate_bond_analytics()
