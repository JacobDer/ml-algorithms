import pandas as pd
import numpy as np
import math

import matplotlib.pyplot as plt


def logistic_function(x):
    return (1 + math.exp(-x))**(-1)


def cross_entropy(y_pred, y_train):
    total_loss = 0
    for i in range(0, len(y_pred)):
        loss = -y_train[i] * math.log(y_pred[i]) - (1 - y_train[i]) * math.log(1 - y_pred[i])
        total_loss += loss

    return total_loss


def gradient_descent(x_train, y_train, learning_rate, tol):
    '''Do gradient descent on the cross entropy loss function.

    The gradient of the cross entropy loss function is a function in the
    paramaters beta, defined by X_t(Y_hat - Y) where
        - X_t is the transpose of the training data matrix
        - Y_hat is a vector of outputs of the logistic function, using the
          current beta coefficients
        - Y is the vector of true labels from the training data
    This functions locally minimizes the cross entropy loss function and
    returns the optimal beta.

    Args:
        x_train (numpy array): independent training data
        y_train (numpy array): dependent training data
        learning_rate (float): learning rate (step size)
        tol (float):           threshold for concluding whether or not the
                               algorithm has converged
    '''

    x_train_transpose = np.transpose(x_train)

    beta = np.zeros(shape=(x_train.shape[1], 1))
    previous_loss = 0

    converged = False
    while not converged:
        y_pred_list = []
        for row in x_train:
            pred = logistic_function(np.matmul(row, beta))
            y_pred_list.append(pred)
        y_pred = np.array(y_pred_list).reshape(len(y_pred_list), -1)

        gradient_eval = np.matmul(x_train_transpose, y_pred - y_train)
        current_loss = cross_entropy(y_pred, y_train)
        if abs(previous_loss - current_loss) < tol:
            converged = True
            return beta
        elif abs(previous_loss - current_loss) >= tol:
            beta = beta - learning_rate * gradient_eval
            previous_loss = current_loss


class LogisticRegressor():
    def __init__(self):
        self.beta = None

    def fit(self, x_train, y_train, learning_rate, tol):
        self.beta = gradient_descent(x_train, y_train, learning_rate, tol)

    def predict(self, x_test):
        y_pred_list = []
        for row in x_test:
            pred = logistic_function(np.matmul(row, self.beta))
            y_pred_list.append(pred)
        y_pred = np.array(y_pred_list).reshape(len(y_pred_list), -1)

        return y_pred


if __name__ == '__main__':
    # --- testing --- #

    # data
    data = [
        (1, 0, 0),
        (1, 0.1, 0),
        (1, 0.2, 0),
        (1, 0.5, 0),
        (1, 0.87, 0),
        (1, 1, 0),
        (1, 1.2, 1),
        (1, 1.3, 0),
        (1, 3.4, 1),
        (1, 4.5, 1),
        (1, 4.7, 1),
        (1, 4.9, 1),
        (1, 5, 1)
    ]
    df = pd.DataFrame(data, columns=['1', 'x', 'y'])
    x_train = df[['1', 'x']].to_numpy().reshape(len(df), -1)
    y_train = df['y'].to_numpy().reshape(len(df), -1)

    # training and visualizing
    log_reg = LogisticRegressor()
    log_reg.fit(x_train, y_train, 0.01, 0.0001)

    x = np.linspace(0, 5, 200).reshape(200, -1)
    ones = np.ones(shape=(200, 1))
    cat = np.concatenate((ones, x), axis=1)
    y = log_reg.predict(cat)

    plt.scatter(df['x'], df['y'])
    plt.plot(x, y, 'orange')
    plt.show()
