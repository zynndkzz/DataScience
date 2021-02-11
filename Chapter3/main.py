from math import pi, cos, sin
from matplotlib.colors import ListedColormap
from sklearn import svm
import matplotlib.cm as cm
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import itertools
from mlxtend.plotting import plot_decision_regions
import numpy as np



def generateStandardEllipse(center,Xaxis,Yaxis,npoints):
    u = center[0]  # x-position of the center
    v = center[1]  # y-position of the center
    a = Xaxis  # radius on the x-axis
    b = Yaxis  # radius on the y-axis

    t = np.linspace(0, 2 * pi, npoints)
    X = u + a * np.cos(t)
    Y = v + b * np.sin(t)
    return X,Y

def rotatePoints(point, origin, degrees):
    radians = np.deg2rad(degrees)
    x,y = point
    offset_x, offset_y = origin
    adjusted_x = (x - offset_x)
    adjusted_y = (y - offset_y)
    cos_rad = np.cos(radians)
    sin_rad = np.sin(radians)
    qx = offset_x + cos_rad * adjusted_x + sin_rad * adjusted_y
    qy = offset_y + -sin_rad * adjusted_x + cos_rad * adjusted_y
    return qx, qy

def generateEllipse(center,Xaxis,Yaxis,npoints,angle):
    X, Y = generateStandardEllipse(center, Xaxis, Yaxis, npoints)
    X_prime, Y_prime = rotatePoints((X, Y), center, angle)

    return X_prime,Y_prime


def main():

    colors = [(1.0, 0.6, 0.0), "white", "blue"]
    cm.register_cmap(cmap=mpl.colors.LinearSegmentedColormap.from_list("owb", colors).reversed())
    cm.register_cmap(cmap=mpl.colors.LinearSegmentedColormap.from_list("owb", colors))

    #X_train,Y_train = generateStandardEllipse((0,0),20,1,100)
    #X_test,Y_test = generateEllipse((0,0),20,1,100,90)


    center = (0,-2)
    npoints = 100
    rotAngle = 0

    X_train, Y_train = generateStandardEllipse(center, 20, 12, npoints)
    X_test, Y_test = generateEllipse(center, 2, 1, npoints, rotAngle)



    X_raw = np.column_stack((X_train,Y_train))
    X_rot = np.column_stack((X_test,Y_test))

    noise = np.random.normal(0, .1, X_rot.shape)
    X_rot = X_rot + noise

    X = np.concatenate((X_raw, X_rot), axis=0)

    y=[]
    for i in range(npoints):
        y.append(1)
    for i in range(npoints):
        y.append(0)

    y = np.array(y)


    svr_rbf = svm.SVC(kernel='rbf', C=100)
    svr_lin = svm.SVC(kernel='linear', C=100)
    svr_poly = svm.SVC(kernel='poly', C=100)
    svr_sigmoid = svm.SVC(kernel='sigmoid')



    svrs = [svr_rbf, svr_lin, svr_poly,svr_sigmoid]
    kernel_label = ['RBF', 'Linear', 'Polynomial','Sigmoid']

    gs = gridspec.GridSpec(2, 2)

    fig = plt.figure(figsize=(10, 8))
    fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(15, 10), sharey=True)
    for clf, lab, grd in zip(svrs,
                             kernel_label,
                             itertools.product([0, 1], repeat=2)):
        clf.fit(X, y)
        ax = plt.subplot(gs[grd[0], grd[1]])
        fig = plot_decision_regions(X=X, y=y, clf=clf, legend=2)
        plt.title(lab)

    plt.show()

    svr_poly1 = svm.SVC(kernel='poly', C=0.01)
    svr_poly2 = svm.SVC(kernel='poly', C=0.1)
    svr_poly3 = svm.SVC(kernel='poly', C=1)
    svr_poly4 = svm.SVC(kernel='poly', C=10)

    svrs = [svr_poly1, svr_poly2, svr_poly3, svr_poly4]
    kernel_label = ['C=0.01', '0.1', '1', '10']

    gs = gridspec.GridSpec(2, 2)

    fig = plt.figure(figsize=(10, 8))
    fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(15, 10), sharey=True)
    for clf, lab, grd in zip(svrs,
                             kernel_label,
                             itertools.product([0, 1], repeat=2)):
        clf.fit(X, y)
        ax = plt.subplot(gs[grd[0], grd[1]])
        fig = plot_decision_regions(X=X, y=y, clf=clf, legend=2)
        plt.title(lab)

    plt.show()



if __name__ == '__main__':
    main()










