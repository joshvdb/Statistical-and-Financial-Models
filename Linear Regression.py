# produced based on: https://onlinecourses.science.psu.edu/stat501/node/252/
# import the required Python packages
import matplotlib.pyplot as plt

# declare the Linear Regression class
class Linear_Regression:
    def __init__(self, x_values, y_values):
        """
        Function to initialize the LG object, taking in two arrays - the x values and the y values
        """
        self.x_values = x_values
        self.y_values = y_values

    def regression(self):
        """
        Function to calculate and plot the line of best fit for the x and y datasets
        """

        def calculate_means():
            """
            Function to calculate the mean of the x and y datasets
            """
            sum_x = 0
            sum_y = 0

            for i in range(0, len(self.x_values)):
                sum_x = sum_x + self.x_values[i]
                sum_y = sum_y + self.y_values[i]

            x_mean = sum_x / len(self.x_values)
            y_mean = sum_y / len(self.y_values)

            return x_mean, y_mean;

        means = calculate_means()

        x_mean, y_mean = means

        # declare the numerator and denominator used to calculate the gradient of the line of best fit
        numerator = 0
        denominator = 0

        for i in range(0, len(self.x_values)):
            numerator = numerator + ((self.y_values[i] - y_mean) * (self.x_values[i] - x_mean))
            denominator = denominator + ((self.x_values[i] - x_mean) ** 2)

        # calculate the gradient of the line of best fit
        gradient = numerator / denominator

        # calculate the intercept of the line of best fit
        intercept = y_mean - gradient * x_mean

        # declare the array of the y-coordinates of the line of best fit
        best_fit = []

        for i in range(0, len(self.x_values)):
            best_fit.append(intercept + gradient * self.x_values[i])

        # create a 2D plot of the regression vs the actual data
        # see here for symbol documentation in matplotlib: https://matplotlib.org/api/markers_api.html
        plt.scatter(self.x_values, self.y_values, c='b', marker='^')
        plt.plot(self.x_values, best_fit, marker='o', linestyle='-', color='r')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Linear Regression vs Original Plot')

        plt.show()

        # return the values of the intercept, gradient and y-coordinates of the line of best fit
        return intercept, gradient, best_fit;

# declare x and y values in lists
x = [63, 64, 66, 69, 69, 71, 71, 72, 73, 75]
y = [127, 121, 142, 157, 162, 156, 169, 165, 181, 208]

# instantiate Linear Regression object
LR = Linear_Regression(x,y)

# calculate the line of best fit of the x and y coordinate data
regression = LR.regression()
