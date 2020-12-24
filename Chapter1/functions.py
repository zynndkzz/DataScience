import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math
import sys

def find_intercept_point(m, c, x0, y0):
    """ find an intercept point of the line model with
        a normal from point (x0,y0) to it
    :param m slope of the line model
    :param c y-intercept of the line model
    :param x0 point's x coordinate
    :param y0 point's y coordinate
    :return intercept point
    """

    # intersection point with the model
    x = (x0 + m * y0 - m * c) / (1 + m ** 2)
    y = (m * x0 + (m ** 2) * y0 - (m ** 2) * c) / (1 + m ** 2) + c

    return x, y
def find_line_model(points):
    """ find a line model for the given points
    :param points selected points for model fitting
    :return line model
    """

    # vertical and horizontal lines should be treated differently
    #           here we just add some noise to avoid division by zero

    # find a line model for these points
    m = (points[1, 1] - points[0, 1]) / (
                points[1, 0] - points[0, 0] + sys.float_info.epsilon)  # slope (gradient) of the line
    c = points[1, 1] - m * points[1, 0]  # y-intercept of the line

    return m, c
def findRansacSolution(x, y):
    """ find best line model for the given variables
    :param independent and dependent values selected for model fitting
    :return from best line model information slope and intercept
    """

    # Ransac parameters
    max_iterations = 20  # number of iterations
    ransac_threshold = 3  # threshold
    ransac_ratio = 0.6  # ratio of inliers required to assert
    # that a model fits well to data
    n_samples = 50  # number of input points
    outliers_ratio = 0.4

    ratio = 0.
    model_m = 0.
    model_c = 0.
    n_inputs = 1
    n_outputs = 1

    x = (np.reshape(x.values,(-1,1)))
    y = (np.reshape(y.values, (-1, 1)))

    # add a little gaussian noise
    x_noise = x + np.random.normal(size=x.shape)
    y_noise = y + np.random.normal(size=y.shape)

    data = np.hstack((x_noise, y_noise))


    # add some outliers to the point-set
    n_outliers = int(outliers_ratio * n_samples)
    indices = np.arange(x_noise.shape[0])
    np.random.shuffle(indices)


    outlier_indices = indices[:n_outliers]

    x_noise[outlier_indices] = 30 * np.random.random(size=(n_outliers, n_inputs))

    # gaussian outliers
    y_noise[outlier_indices] = 30 * np.random.normal(size=(n_outliers, n_outputs))

    # non-gaussian outliers (only on one side)
    # y_noise[outlier_indices] = 30*(np.random.normal(size=(n_outliers,n_outputs))**2)

    # perform RANSAC iterations
    for it in range(max_iterations):

        # pick up two random points
        n = 2

        all_indices = np.arange(x_noise.shape[0])
        np.random.shuffle(all_indices)


        indices_1 = all_indices[:n]
        indices_2 = all_indices[n:]



        maybe_points = data[indices_1, :]
        test_points = data[indices_2, :]


        # find a line model for these points
        m, c = find_line_model(maybe_points)


        x_list = []
        y_list = []
        num = 0

        # find orthogonal lines to the model for all testing points
        for ind in range(test_points.shape[0]):

            x0 = test_points[ind, 0]
            y0 = test_points[ind, 1]

            # find an intercept point of the model with a normal from point (x0,y0)
            x1, y1 = find_intercept_point(m, c, x0, y0)

            # distance from point to the model
            dist = math.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2)


            # check whether it's an inlier or not
            if dist < ransac_threshold:
                x_list.append(x0)
                y_list.append(y0)
                num += 1

        x_inliers = np.array(x_list)
        y_inliers = np.array(y_list)

        # in case a new model is better - cache it
        if num / float(n_samples) > ratio:
            ratio = num / float(n_samples)
            model_m = m
            model_c = c



        # plot the current step
        #plot_ransac(it, x_noise, y_noise, m, c, False, x_inliers, y_inliers, maybe_points)

        # we are done in case we have enough inliers
        if num > n_samples * ransac_ratio:
            break

    # plot the final model
    plot_ransac(0, x_noise, y_noise, model_m, model_c, True)

    return {"intercept": model_c, "slope": model_m}
def computeRmse(input, output, intercept, slope):
    """ find the root mean squared error for the given variables
    :param independent and dependent values (input,output ) and line information (intercept,slope)
    :return root mean squared error
    """
    prediction = input * slope + intercept

    yi = output

    var = ((yi - prediction)*(yi - prediction)).abs().sum()

    RMSE = var / len(yi)

    return math.sqrt(RMSE)

def plot_regression_line(m,x ,y,solverType):
    """ plot the regression line for leastSquares
    :param points picked up points for modeling
    :param x      samples x
    :param y      samples y
    :param m      the line model
    :param solverType
    """

    #print("model:",m)

    plt.scatter(x, y, color = "m",
            marker = "o", s = 30)

    # predicted response vector
    y_pred = m["intercept"] + m["slope"]*x
    #print("y_pred" , y_pred)

    plt.scatter(x, y_pred, color="r",
                marker="o", s=30)

    # plotting the regression line
    plt.plot(x, y_pred, color = "g",label = "Regression line")
    plt.legend()
    # putting labels
    plt.xlabel('input')
    plt.ylabel('output')

    plt.title(label="Regression with {}:".format(solverType),fontweight=10,pad='2.0')

    #plt.figtext(.7, .14, "RMSE: {}".format(str(rmse)[:5]))

    # function to show plot
    plt.show()

def plot_ransac(n, x, y, m, c, final=False, x_in=(), y_in=(), points=()):
    """ plot the current RANSAC step
    :param n      iteration
    :param points picked up points for modeling
    :param x      samples x
    :param y      samples y
    :param m      slope of the line model
    :param c      shift of the line model
    :param x_in   inliers x
    :param y_in   inliers y
    """
    #print(x)
    fname = "figure_" + str(n) + ".png"
    line_width = 1.
    line_color = '#0080ff'
    title = 'iteration ' + str(n)

    if final:
        fname = "final.png"
        line_width = 3.
        line_color = '#ff0000'
        title = 'final solution'

    plt.figure("Ransac", figsize=(15., 15.))
    plt.scatter(x, y, color="r",
                marker="o", s=30)
    # grid for the plot
    grid = [int(min(x)) - 10, int(max(x)) + 10, int(min(y)) - 20, int(max(y)) + 20]
    plt.axis(grid)
    #print(grid)

    # put grid on the plot
    plt.grid(b=True, which='major', color='0.75', linestyle='--')

    plt.xticks([i for i in range(int(min(x)) - 10, int(max(x)) + 10, 5)])
    plt.yticks([i for i in range(int(min(y)) - 20, int(max(y)) + 20, 10)])

    # plot input points
    plt.plot(x[:, 0], y[:, 0], marker='o', label='Input points', color='#00cc00', linestyle='None', alpha=0.4)

    # draw the current model
    plt.plot(x, m * x + c, 'r', label='Line model', color=line_color, linewidth=line_width)

    # draw inliers
    if not final:
        plt.plot(x_in, y_in, marker='o', label='Inliers', linestyle='None', color='#ff0000', alpha=0.6)

    # draw points picked up for the modeling
    if not final:
        #print(points)
        plt.plot(points[:, 0], points[:, 1], marker='o', label='Picked points', color='#0000cc', linestyle='None',
                 alpha=0.6)
        plt.show()
    #plt.show()
    plt.title(title)
    plt.legend()
    plt.savefig(fname)
    plt.close()

def findLss(x,y):
    """ find best line model for the given variables solving with leastSquared solution
    :param independent and dependent values selected for model fitting
    :return from best line model information slope and intercept
    """
    n = np.size(x)

    # mean of x and y vector
    m_x, m_y = np.mean(x), np.mean(y)

    # calculating cross-deviation and deviation about x
    SS_xy = np.sum(y * x) - n * m_y * m_x
    SS_xx = np.sum(x * x) - n * m_x * m_x

    # calculating regression coefficients slope
    b_1 = SS_xy / SS_xx
    # intercept
    b_0 = m_y - b_1 * m_x

    return {"intercept" :b_0,"slope" :b_1}


def performLinearRegression(x, y, solverType):
    """ main function which orients code according to solverType
    :param independent and dependent values selected for model fitting
    :return from best line model information slope and intercept,and final error
    """

    if solverType == 'LeastSquares':
        m = findLss(x,y)
        plot_regression_line(m,x, y, solverType)

    elif solverType == 'Ransac':
        #plotting Ransac model is done in the findRansacSolution function
        m = findRansacSolution(x,y)
    else:
        raise Exception("Solver Type is not specified")

    return {m["intercept"], m["slope"], computeRmse(x,y,m["intercept"],m["slope"])}


if __name__ == '__main__':

    #load Data
    diabetesData = pd.read_csv("diabetes_csv.csv")

    #little modification on data it is givig warning for that
    for i in range(0,442):
        value = diabetesData['class'][i]
        if(value == 'tested_positive'):
            diabetesData['class'][i] = 1
        else:
            diabetesData['class'][i] = 0


    #diabetesData.describe()
    # take independent and dependent variables from data
    Y = diabetesData["class"][:50]
    X = diabetesData["plas"][:50]


    firstInterncept, firstSlope, firstRMSE = performLinearRegression(X, Y, "LeastSquares")
    print(firstRMSE)
    secondInterncept, secondSlope, secondRMSE  = performLinearRegression(X,Y, "Ransac")
    print(secondRMSE)


