import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import log_loss, accuracy_score
from tqdm import tqdm

# Initialization function
def initialization(layer_dims):
    params = {}
    L = len(layer_dims)  # Total number of layers

    for l in range(1, L):
        params[f'W{l}'] = np.random.randn(layer_dims[l], layer_dims[l-1])
        params[f'b{l}'] = np.random.randn(layer_dims[l], 1)
    return params

# Forward Propagation
def forward_propagation(X, params):
    L = len(params) // 2
    activations = {'A0': X}

    for l in range(1, L + 1):
        Z = params[f'W{l}'].dot(activations[f'A{l-1}']) + params[f'b{l}']
        A = 1 / (1 + np.exp(-Z))  # Sigmoid activation
        activations[f'A{l}'] = A

    return activations

# Loss Function
def log_losss(A, y):
    return 1 / len(y) * np.sum(-y * np.log(A) - (1 - y) * np.log(1 - A))

# Back Propagation
def back_propagation(X, y, activations, params):
    L = len(params) // 2
    m = y.shape[1]
    grads = {}

    # Gradient for the output layer
    dZ = activations[f'A{L}'] - y
    grads[f'dW{L}'] = 1/m * dZ.dot(activations[f'A{L-1}'].T)
    grads[f'db{L}'] = 1/m * np.sum(dZ, axis=1, keepdims=True)

    # Backpropagate through hidden layers
    for l in range(L-1, 0, -1):
        dA = params[f'W{l+1}'].T.dot(dZ)
        dZ = dA * activations[f'A{l}'] * (1 - activations[f'A{l}'])
        grads[f'dW{l}'] = 1/m * dZ.dot(activations[f'A{l-1}'].T) if l > 1 else 1/m * dZ.dot(X.T)
        grads[f'db{l}'] = 1/m * np.sum(dZ, axis=1, keepdims=True)

    return grads

# Update Parameters
def update(grads, params, learning_rate):
    L = len(params) // 2
    for l in range(1, L + 1):
        params[f'W{l}'] -= learning_rate * grads[f'dW{l}']
        params[f'b{l}'] -= learning_rate * grads[f'db{l}']
    return params

# Prediction Function
def predict(X, params):
    activations = forward_propagation(X, params)
    A_final = activations[f'A{len(params) // 2}']
    return A_final >= 0.5

# Neural Network Training Function
def artificial_neuron(x_train, y_train, layer_dims, learning_rate=0.1, n_iter=100):
    params = initialization(layer_dims)
    train_loss = []
    train_acc = []

    for i in tqdm(range(n_iter)):
        # Forward Propagation
        activations = forward_propagation(x_train, params)
        
        # Back Propagation
        gradients = back_propagation(x_train, y_train, activations, params)
        
        # Update Parameters
        params = update(gradients, params, learning_rate)

        if i % 10 == 0:
            # Training Loss and Accuracy
            A_final = activations[f'A{len(layer_dims) - 1}']
            train_loss.append(log_loss(y_train, A_final))
            y_pred = predict(x_train, params)
            train_acc.append(accuracy_score(y_train.flatten(), y_pred.flatten()))

    return params, train_loss, train_acc
