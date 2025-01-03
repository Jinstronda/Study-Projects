import numpy as np
# Activation functions and their derivatives
def relu(x):
    return np.maximum(0, x)  # Uses and returns the Rectified Linear Value

def relu_derivative(z):
    return np.where(z > 0, 1, 0)  # Calculates RELU Derivative

# Loss functions and their derivatives
def mse(y_hat, y):
    loss = np.sum((y_hat - y) ** 2) / len(y_hat)  # Calculates MSE Loss Function
    return loss

def msederivative(y_hat, y):
    m = len(y_hat)  # Size of the samples
    diff = (y_hat - y) * 2  # Calculates MSE derivative
    return diff

def crossentropy_derivative(y_hat, y):
    epsilon = 1e-12  # Para evitar divisão por zero
    y_hat = np.clip(y_hat, epsilon, 1. - epsilon)
    return (y_hat - y) / y.shape[0]

def crossentropy_with_regularization(y_hat, y, weights, lambda_):
    # Crossentropy Loss with Regularization to graph it
    m = y.shape[0]
    epsilon = 1e-12
    y_hat = np.clip(y_hat, epsilon, 1. - epsilon)
    cross_entropy_loss = -np.sum(y * np.log(y_hat)) / m
    regularization_loss = (lambda_ / (2 * m)) * sum(np.sum(w ** 2) for w in weights)
    total_loss = cross_entropy_loss + regularization_loss
    return total_loss

# Utility functions
def one_hot_encode(labels, num_classes):
    return np.eye(num_classes)[labels]  # One Hot Encode the variables for Softmax