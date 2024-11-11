import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import log_loss  # Importing log_loss from sklearn
from sklearn.metrics import accuracy_score
from tqdm import tqdm

def initalisation(n0, n1, n2):
    W1 = np.random.randn(n1, n0)
    b1 = np.random.randn(n1, 1)
    W2 = np.random.randn(n2, n1)
    b2 = np.random.randn(n2, 1)

    params = {
        'W1': W1,
        'b1': b1,
        'W2': W2,
        'b2': b2
    } 

    return params

def forward_propagation(X, params):

    W1 = params['W1']
    b1 = params['b1']
    W2 = params['W2']
    b2 = params['b2']

    #first layer
    Z1 = W1.dot(X) + b1
    A1 = 1 / (1 + np.exp(-Z1))

    #second layer
    Z2 = W2.dot(A1) + b2
    A2 = 1 / (1 + np.exp(-Z2))

    activations = {
        'A1': A1,
        'A2': A2
    }

    return activations

def log_losss(A, y):
    return 1/len(y)*np.sum(-y*np.log(A) - (1-y)*np.log(1-A))

def back_propagation(X, y, activations, params):

    A1 = activations['A1']
    A2 = activations['A2']
    W2 = params['W2']

    m = y.shape[1]
    dZ2 = A2 - y
    dW2 = 1/m*dZ2.dot(A1.T)
    db2 = 1/m*np.sum(dZ2, axis=1, keepdims=True)

    dZ1 = W2.T.dot(dZ2)*A1*(1-A1)
    dW1 = 1/m*dZ1.dot(X.T)
    db1 = 1/m*np.sum(dZ1, axis=1, keepdims=True)

    grads = {
        'dW1': dW1,
        'db1': db1,
        'dW2': dW2,
        'db2': db2
    }

    return grads

def update(grads, params, learning_rate):
    W1 = W1 - learning_rate*dW1
    b1 = b1 - learning_rate*db1
    W2 = W2 - learning_rate*dW2
    b2 = b2 - learning_rate*db2

    params = {
        'W1': W1,
        'b1': b1,
        'W2': W2,
        'b2': b2
    }

    return params

def predict(X, params):
    activations = forward_propagation(X, params)
    A2 = activations['A2'] 
    return A2 >= 0.5

def artificial_neuron(x_train, y_train, n1, learning_rate=0.1, n_iter = 100):
    # Initalisation

    n0 = x_train.shape[0]
    n2 = y_train.shape[0]

    params = initalisation(n0, n1, n2)

    train_loss = []
    train_acc = []

    for i in tqdm(range(n_iter)):

        activations = forward_propagation(x_train, params)
        gradients = back_propagation(x_train, y_train, activations, params)
        params = update(gradients, params, learning_rate)


        if n_iter % 10 == 0:
            
            # Training Loss and Accuracy
            train_loss.append(log_loss(y_train, activations['A2']))
            y_pred = predict(x_train, params)
            train_acc.append(accuracy_score(y_train.flatten(), y_pred.flatten()))

    return params