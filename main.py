
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures


def plotModel(degree,coeff,interval,dpi,color):
    coeff = np.flip(coeff)
    sample = (interval[1]-interval[0])*dpi
    vectorX = np.linspace(interval[0], interval[1], num=sample, endpoint=interval[1])
    vectorX = np.transpose(vectorX)
    poly = PolynomialFeatures(degree)
    temp =poly.fit_transform(vectorX.reshape([sample,1]))
    vectorY = np.dot(temp,coeff)
    plt.plot(vectorX,vectorY, c=color, linestyle='-')



def generateSample(degree,coeff,interval,noSample,stdDev):
    vectorX = np.random.uniform(interval[0],interval[1],noSample)
    poly = PolynomialFeatures(degree)
    temp = poly.fit_transform(vectorX.reshape([noSample, 1]))
    vectorY = np.dot(temp, coeff)
    vectorY += np.random.normal(0,stdDev,noSample)
    return vectorX,vectorY



def main():

    #plotModel(1,[3,5],(0,10),10,'k')
    #plt.show()

    K = [10,100,1000]
    stdDev=[1,5,20]
    degrees = [0,1,2,3]
    colors = ['g','r','b','m']
    coefficents = [[5],[3,5],[1,3,5],[1,1,3,5]]


    for i in range(0,len(degrees)):
        plotModel(degrees[i], coefficents[i], (0, 10), 10, colors[i])
    plt.title(label='Experiment 1:', fontweight=10, pad='2.0')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()


    for j in range(0,len(stdDev)):
        for i in range(0, len(degrees)):
            trainX, trainY = generateSample(degrees[i], coefficents[i], (0, 10), K[j], stdDev[j])
            testX, testY = generateSample(degrees[i], coefficents[i], (15, 20), K[j], stdDev[j])
            mymodel = np.poly1d(np.polyfit(trainX, trainY, degrees[i]))
            myTestmodel = np.poly1d(np.polyfit(testX, testY, degrees[i]))
            plotModel(degrees[i], mymodel.coefficients, (0, 20), 10, colors[i])
            plotModel(degrees[i], myTestmodel.coefficients, (0, 20), 10, colors[i])
    plt.title(label='Experiment 2:', fontweight=10, pad='2.0')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()





    #plt.scatter(trainX, trainY, c='k', marker='x')
    #plt.scatter(testX, testY, c='k', marker='^')

    #plt.show()

if __name__ == '__main__':

    main()
