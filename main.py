import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import log_loss  # Importing log_loss from sklearn
from sklearn.metrics import accuracy_score
from tqdm import tqdm

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

def artificial_neuron(x_train, y_train, x_test, y_test, learning_rate=0.1, n_iter = 100):
    # Initalisation
    W, b = initalisation(x_train)

    train_loss = []
    train_acc = []
    test_loss = []
    test_acc = []

    for i in tqdm(range(n_iter)):

        A = model(x_train,W, b)

        if n_iter % 10 == 0:
            
            # Training Loss and Accuracy
            train_loss.append(log_loss(y_train, A))

            y_pred = predict(x_train, W, b)
            train_acc.append(accuracy_score(y_train, y_pred))

            # Testing Loss and Accuracy
            A_test = model(x_test, W, b)
            test_loss.append(log_loss(y_test, A_test))
            y_pred_test = predict(x_test, W, b)
            test_acc.append(accuracy_score(y_test, y_pred_test))



        dW, db = gradient(A, x_train, y_train)
        W, b = update(dW, db, W, b, learning_rate)

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_loss, label='train')
    plt.plot(test_loss, label='test')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(train_acc, label='train')
    plt.plot(test_acc, label='test')
    plt.legend()
    plt.show()


    return (W, b)