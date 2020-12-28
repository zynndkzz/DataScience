# Importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from math import exp


# Creating the logistic regression model

# Helper function to normalize data
def normalize(X):
    return X - X.mean()


# Method to make predictions
def predict(X, b0, b1):
    return np.array([1 / (1 + exp(-1 * b0 + -1 * b1 * x)) for x in X])


def plotting(X,Y,x_label,y_label):
    plt.plot(X, Y, color="m", label=y_label)
    plt.legend()
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.plot()
    plt.show()

# Method to train the model
def GradientDescent(X, Y):
    X = normalize(X)

    # Initializing variables
    b0 = 0
    b1 = 0
    L = 0.0001
    epochs = 300

    cost = []
    derivatives = []

    for epoch in range(epochs):
        y_pred = predict(X, b0, b1)
        D_b0 = -2 * sum((Y - y_pred) * y_pred * (1 - y_pred))  # Derivative of loss wrt b0
        D_b1 = -2 * sum(X * (Y - y_pred) * y_pred * (1 - y_pred))  # Derivative of loss wrt b1
        # Update b0 and b1
        b0 = b0 - L * D_b0
        b1 = b1 - L * D_b1

        derivatives.append(D_b0)

        e = sum((y_pred- Y) ** 2)
        cost.append(e)


    iterations = [j for j in range(0,epoch+1)]

    plotting(iterations,cost,"Iterations","Cost Magnitude")
    plotting(iterations,derivatives, "Iterations", "Gradient Magnitude")

    return b0, b1



if __name__ == '__main__':
    plt.rcParams["figure.figsize"] = (10, 6)

    # Load the data
    data = pd.read_csv("diabetes_csv.csv")


    for i in range(0, 442):
        value = data['class'][i]
        if (value == 'tested_positive'):
            data['class'][i] = 1
        else:
            data['class'][i] = 0

    # Visualizing the dataset
    plt.scatter(data['plas'], data['class'])
    plt.legend()
    plt.xlabel("independent Variable")
    plt.ylabel("Class")
    plt.title(label="Dataset", fontweight=10, pad='2.0')
    plt.show()

    # Divide the data to training set and test set
    X_train, X_test, y_train, y_test = train_test_split(data['plas'], data['class'], test_size=0.20)



    #Training the model
    b0, b1 = GradientDescent(X_train, y_train)

    # Making predictions
    X_test_norm = normalize(X_test)
    y_pred = predict(X_test_norm, b0, b1)
    y_pred = [1 if p >= 0.5 else 0 for p in y_pred]

    plt.clf()
    plt.scatter(X_test, y_test)
    plt.scatter(X_test, y_pred, c="red",label = "Red dots: Test results")
    plt.legend()
    plt.xlabel("independent Variable")
    plt.ylabel("Class")
    plt.title(label="Classification Results", fontweight=10, pad='2.0')
    plt.show()

    # The accuracy
    accuracy = 0
    for i in range(len(y_pred)):
        if y_pred[i] == y_test.iloc[i]:
            accuracy += 1
    print(f"Accuracy = {accuracy / len(y_pred)}")