import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

X, y = make_blobs(n_samples=100, centers=2, n_features=2, random_state=0)
y = y.reshape((y.shape[0], -1))

def initalisation(X):
    W = np.random.randn(X.shape[1], 1)
    b = np.random.randn(1)

    return W, b

W, b = initalisation(X)

def model(X, W, b):
    Z = X.dot(W) + b
    A = 1 / (1 + np.exp(-Z))
    return A

A = model(X, W, b)

def log_loss(A, y):
    return 1/len(y)*np.sum(-y*np.log(A) - (1-y)*np.log(1-A))

def gradient(A, X, y):
    dW = 1/len(y)*np.dot(X.T, (A-y))
    db = 1/len(y)*np.sum(A-y)
    return dW, db

dW, db = gradient(A, X, y)


def update(dW, db, W, b, learning_rate):
    W = W - learning_rate*dW
    b = b - learning_rate*db
    return (W, b)

def artifical_neuron(X, y, learning_rate=0.1, n_iter = 100):
    # Initalisation
    W, b = initalisation(X)

    Loss = []

    for i in range(n_iter):
        A = model(X,W, b)
        Loss.append(log_loss(A, y))
        dW, db = gradient(A, X, y)
        W, b = update(dW, db, W, b, learning_rate)

    plt.plot(Loss)
    plt.show()

artifical_neuron(X, y)