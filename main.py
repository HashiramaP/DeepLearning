import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import log_loss  # Importing log_loss from sklearn
from sklearn.metrics import accuracy_score

def initalisation(X):
    W = np.random.randn(X.shape[1], 1)
    b = np.random.randn(1)

    return W, b

def model(X, W, b):
    Z = X.dot(W) + b
    A = 1 / (1 + np.exp(-Z))
    return A

def log_losss(A, y):
    return 1/len(y)*np.sum(-y*np.log(A) - (1-y)*np.log(1-A))

def gradient(A, X, y):
    dW = 1/len(y)*np.dot(X.T, (A-y))
    db = 1/len(y)*np.sum(A-y)
    return dW, db

def update(dW, db, W, b, learning_rate):
    W = W - learning_rate*dW
    b = b - learning_rate*db
    return (W, b)

def predict(X, W, b):
    A = model(X, W, b)
    return A >= 0.5

def artificial_neuron(X, y, learning_rate=0.1, n_iter = 100):
    # Initalisation
    W, b = initalisation(X)

    Loss = []

    for i in range(n_iter):
        A = model(X,W, b)
        loss = log_loss(y, A)
        Loss.append(loss)
        dW, db = gradient(A, X, y)
        W, b = update(dW, db, W, b, learning_rate)

    y_pred = predict(X, W, b)
    accuracy = accuracy_score(y, y_pred)
    print(f'Accuracy: {accuracy}')

    # plotting the loss
    # plt.plot(Loss)
    # plt.xlabel('Iteration')
    # plt.ylabel('Log Loss')
    # plt.title('Log Loss vs Iteration')
    # plt.show()

    return (W, b)