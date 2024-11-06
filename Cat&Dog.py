from utilities import *
import matplotlib.pyplot as plt
from main import *


x_train, y_train, x_test, y_test = load_data()

# Normalizing the data
# Each pixel is between 0 and 255, we will divide by 255 to get a value between 0 and 1
x_train = x_train/255
x_test = x_test/255

# Flatten the images, X is now a 2D array m x n
# m is the number of samples and n is the number of pixels
x_train = x_train.reshape(x_train.shape[0], -1)

# Train the model
W, b = artificial_neuron(x_train, y_train, learning_rate=0.1, n_iter=100)

# Test the model
x_test = x_test.reshape(x_test.shape[0], -1)
y_pred = predict(x_test, W, b)
accuracy = accuracy_score(y_test, y_pred)
 