import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import log_loss  # Importing log_loss from sklearn
from sklearn.metrics import accuracy_score
from tqdm import tqdm
from sklearn.datasets import make_blobs
from sklearn.metrics import accuracy_score

X, y = make_blobs(n_samples=3000, centers=2, n_features=2, random_state=0)
y = y.reshape((y.shape[0], -1))

def initialisation(X):
    W1 = np.random.randn(X.shape[1], 3)  # Weight matrix for 5 neurons
    b1 = np.random.randn(1, 3)  # Bias for 5 neurons
    W2 = np.random.randn(3, 1)  # Weight matrix to single output neuron
    b2 = np.random.randn(1, 1)  # Bias for output neuron
    return W1, b1, W2, b2


def model(X, W1, b1, W2, b2):
    Z1 = X.dot(W1) + b1  # First layer transformation
    A1 = 1 / (1 + np.exp(-Z1))  # Sigmoid activation for 5 neurons

    Z2 = A1.dot(W2) + b2  # Output layer transformation
    A2 = 1 / (1 + np.exp(-Z2))  # Sigmoid activation for final output (binary probability)
    return A1, A2  # Return both for potential debugging, but primarily use A2

# Custom log_loss function
def log_loss(A, y, epsilon=1e-8):
    A = np.clip(A, epsilon, 1 - epsilon)  # Clip A to avoid log(0)
    return -np.mean(y * np.log(A) + (1 - y) * np.log(1 - A))  # Log loss calculation

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
    W, b = initialisation(x_train)

    train_loss = []
    train_acc = []
    test_loss = []
    test_acc = []

    for i in tqdm(range(n_iter)):

        A = model(x_train,W, b)

        if n_iter % 10 == 0:
            
            # Training Loss and Accuracy
            train_loss.append(log_loss(y_train, A))

            y_pred = [predict(x_train, W, b)]
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


def neural_network(x_train, y_train, learning_rate=0.01, n_iter=1000):
    W1, b1, W2, b2 = initialisation(x_train)

    train_loss = []
    train_acc = []

    for i in tqdm(range(n_iter)):
        # Forward pass
        A1, A2 = model(x_train, W1, b1, W2, b2)

        # Calculate loss and accuracy every 10 iterations
        if i % 10 == 0:
            train_loss.append(log_loss(A2, y_train))
            y_pred = A2 >= 0.5  # Convert probabilities to binary predictions
            train_acc.append(accuracy_score(y_train, y_pred))

        # Backpropagation for both layers
        dW2, db2 = gradient(A2, A1, y_train)  # Gradients for output layer
        dW1, db1 = gradient(A1, x_train, y_train)  # Gradients for 5-neuron layer

        # Update weights and biases
        W2, b2 = update(dW2, db2, W2, b2, learning_rate)
        W1, b1 = update(dW1, db1, W1, b1, learning_rate)

    # Plotting
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_loss, label='loss')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(train_acc, label='accuracy')
    plt.legend()
    plt.show()
    
    return W1, b1, W2, b2



# Train the model
W, b, *others = neural_network(X, y, learning_rate=0.01, n_iter=3000)
