import numpy as np
from neural_network.utils.functions import msederivative, crossentropy_derivative, relu, relu_derivative, crossentropy_with_regularization
from neural_network.config import batch_size
class NeuralNetwork:
    def __init__(self, layers: list):
        self.layers = layers

    def forward(self, X):
        activation = X
        for i in self.layers:
            activation = i.forward(activation)
        return activation

    def backpropagation(self, X, y_train, learning_rate, lambda_):
        # Performs backpropagation to adjust the parameters
        y_hat = self.forward(X)
        # Starts the loss function depending on the output layer
        if self.layers[-1].type == "linear":
            dA = msederivative(y_hat, y_train)
        elif self.layers[-1].type == "softmax":
            dA = crossentropy_derivative(y_hat, y_train)  # Calculates activation derivative
        # Starts backpropagation from output layer
        for layer in reversed(self.layers):
            dA = layer.backward(dA, learning_rate, lambda_)

    def training(self):
        for i in self.layers:
            i.training = True

    def testing(self):
        for i in self.layers:
            i.training = False

# Define the Layer class
class Layer:
    def __init__(self, type, n_neurons, n_input, training):
        # n_neurons = Number of neurons, n_input = Number of inputs
        self.size = n_neurons
        self.type = type
        self.weights = np.zeros((n_neurons, n_input))
        self.bias = np.full((1, n_neurons), 0.01)
        # Initialize random weights
        for row_idx, row in enumerate(self.weights):
            for col_idx, element in enumerate(row):
                stdev = np.sqrt(2 / n_input)
                self.weights[row_idx, col_idx] = np.random.normal(0, stdev)
        # Momentum terms initialization
        self.V_dW = np.zeros_like(self.weights)  # Same shape as weights
        self.V_dB = np.zeros_like(self.bias)     # Same shape as bias
        self.S_dW = np.zeros_like(self.weights)
        self.S_dB = np.zeros_like(self.bias)
        self.t = 0  # Time for Adam optimizer
        # Initialize parameters for batch normalization
        self.beta = np.zeros_like(self.bias)     # Shift parameter (beta)
        self.gamma = np.ones_like(self.bias)     # Scale parameter (gamma)
        self.V_dbeta = np.zeros_like(self.bias)
        self.V_dgamma = np.zeros_like(self.bias)
        self.S_dbeta = np.zeros_like(self.bias)
        self.S_dgamma = np.zeros_like(self.bias)
        # Initialize mean, variance, and normalized inputs
        self.batch_mean = None   # Mean of the batch
        self.batch_var = None    # Variance of the batch
        self.normalized_inputs = None  # Normalized inputs (z_hat)
        self.z_tilde = None      # Scaled and shifted normalized inputs
        self.running_mean = np.zeros((1, n_neurons)) # For Testing
        self.running_var = np.ones((1, n_neurons)) # For Testing
        self.training = training

    def forward(self, x, momentum=0.9):
        self.inputs = x  # Save the inputs of the layer
        # Linear transformation
        self.z = np.dot(x, self.weights.T) + self.bias
        epsilon = 1e-8  # Small constant to prevent division by zero

        if self.training:
            # Compute batch statistics
            self.batch_mean = np.mean(self.z, axis=0, keepdims=True)
            self.batch_var = np.var(self.z, axis=0, keepdims=True)

            # Update running estimates
            self.running_mean = momentum * self.running_mean + (1 - momentum) * self.batch_mean
            self.running_var = momentum * self.running_var + (1 - momentum) * self.batch_var

            # Normalize
            self.normalized_inputs = (self.z - self.batch_mean) / np.sqrt(self.batch_var + epsilon)
        else:
            # Use running estimates
            self.normalized_inputs = (self.z - self.running_mean) / np.sqrt(self.running_var + epsilon)

        # Scale and shift
        self.z_tilde = self.gamma * self.normalized_inputs + self.beta

        # Apply activation function
        if self.type == "relu":
            self.activation = relu(self.z_tilde)
        elif self.type == "linear":
            self.activation = self.z_tilde
        elif self.type == "softmax":
            z_max = np.max(self.z_tilde, axis=1, keepdims=True)
            exp_z = np.exp(self.z_tilde - z_max)
            sum_exp_z = np.sum(exp_z, axis=1, keepdims=True)
            self.activation = exp_z / sum_exp_z
        return self.activation

    def backward(self, dA, l, lambda_, beta=0.9, beta_2=0.999, epsilon=1e-8):
        m = self.inputs.shape[0]  # Number of examples in the batch

        # Increment time step for Adam optimizer
        self.t += 1

        # Calculate dZ_tilde based on activation type
        if self.type == "linear":
            dZ_tilde = dA
        elif self.type == "relu":
            aD = relu_derivative(self.z_tilde)  # Activation derivative
            dZ_tilde = aD * dA  # Element-wise multiplication
        elif self.type == "softmax":
            dZ_tilde = dA

        # Gradients w.r.t gamma and beta
        self.dgamma = np.sum(dZ_tilde * self.normalized_inputs, axis=0, keepdims=True)
        self.dbeta = np.sum(dZ_tilde, axis=0, keepdims=True)

        # Backpropagate through batch normalization
        dZ_hat = dZ_tilde * self.gamma
        batch_var_eps = self.batch_var + epsilon
        sqrt_batch_var_eps = np.sqrt(batch_var_eps)
        inv_sqrt_batch_var_eps = 1 / sqrt_batch_var_eps

        # Compute gradients w.r.t variance and mean
        dvar = np.sum(dZ_hat * (self.z - self.batch_mean) * -0.5 * (batch_var_eps) ** (-1.5), axis=0)
        dmean = np.sum(dZ_hat * -inv_sqrt_batch_var_eps, axis=0) + dvar * np.mean(-2 * (self.z - self.batch_mean), axis=0)

        # Gradient w.r.t z (before batch normalization)
        dZ = dZ_hat * inv_sqrt_batch_var_eps + dvar * 2 * (self.z - self.batch_mean) / m + dmean / m

        # Compute gradients w.r.t weights and biases
        self.dW = np.dot(dZ.T, self.inputs) / m  # Gradient w.r.t weights
        self.dB = np.sum(dZ, axis=0, keepdims=True) / m  # Gradient w.r.t biases

        # Add regularization to gradients
        regularization = (lambda_ / m) * self.weights  # L2 regularization term
        self.dW += regularization

        # Update moving averages of the gradients (Momentum terms)
        self.V_dW = beta * self.V_dW + (1 - beta) * self.dW
        self.V_dB = beta * self.V_dB + (1 - beta) * self.dB
        self.V_dgamma = beta * self.V_dgamma + (1 - beta) * self.dgamma
        self.V_dbeta = beta * self.V_dbeta + (1 - beta) * self.dbeta

        # Compute bias-corrected first moment estimates
        V_dW_corrected = self.V_dW / (1 - beta ** self.t)
        V_dB_corrected = self.V_dB / (1 - beta ** self.t)
        V_dgamma_corrected = self.V_dgamma / (1 - beta ** self.t)
        V_dbeta_corrected = self.V_dbeta / (1 - beta ** self.t)

        # Update moving averages of the squared gradients (RMSprop terms)
        self.S_dW = beta_2 * self.S_dW + (1 - beta_2) * (self.dW ** 2)
        self.S_dB = beta_2 * self.S_dB + (1 - beta_2) * (self.dB ** 2)
        self.S_dgamma = beta_2 * self.S_dgamma + (1 - beta_2) * (self.dgamma ** 2)
        self.S_dbeta = beta_2 * self.S_dbeta + (1 - beta_2) * (self.dbeta ** 2)

        # Compute bias-corrected second moment estimates
        S_dW_corrected = self.S_dW / (1 - beta_2 ** self.t)
        S_dB_corrected = self.S_dB / (1 - beta_2 ** self.t)
        S_dgamma_corrected = self.S_dgamma / (1 - beta_2 ** self.t)
        S_dbeta_corrected = self.S_dbeta / (1 - beta_2 ** self.t)

        # Update parameters using Adam optimizer
        self.weights -= l * (V_dW_corrected / (np.sqrt(S_dW_corrected) + epsilon))
        self.bias -= l * (V_dB_corrected / (np.sqrt(S_dB_corrected) + epsilon))
        self.gamma -= l * (V_dgamma_corrected / (np.sqrt(S_dgamma_corrected) + epsilon))
        self.beta -= l * (V_dbeta_corrected / (np.sqrt(S_dbeta_corrected) + epsilon))

        # Compute dA_prev to pass to the previous layer
        dA_prev = np.dot(dZ, self.weights)  # Gradient w.r.t inputs

        return dA_prev  # Returns the derivative for the next layer to use
def forwardtest():
    # Testing if forwarding is done correctly
    X = np.array([[1, 2],
                  [2, 3],
                  [3, 4],
                  [4, 5]])
    layer1 = Layer("relu", 3, 2)
    layer2 = Layer("linear", 1, 3)
    layer1.weights = np.array([[0.1, 0.2],  # Neuron 1
                               [0.3, 0.4],  # Neuron 2
                               [0.5, 0.6]])
    layer1.bias = np.array([[0.1, 0.2, 0.3]])
    layer2.weights = np.array([[0.7, 0.8, 0.9]])
    layer2.bias = np.array([[0.1]])
    nn = NeuralNetwork([layer1, layer2])
    return nn.forward(X)  # Expected Values: 3.36,5.12,6.88,8.64

def training(nn: NeuralNetwork, X, y_train, learning_rate, epochs, lambda_, batch_size = batch_size, decayrate=0.01,training=True):
    # Trains and tests the neural network
    nn.training()
    num_samples = X.shape[0]  # Number of samples is the number of rows of X
    losses = []
    iterations = []
    alpha0 = learning_rate
    for epoch in range(epochs):
        permutation = np.random.permutation(num_samples)
        X_shuffled = X[permutation]       # Shuffles the data
        y_shuffled = y_train[permutation]  # Shuffles the data

        for i in range(0, num_samples, batch_size):  # Goes over the number of batch (the step size) in the examples
            X_batch = X_shuffled[i:i + batch_size]
            y_batch = y_shuffled[i:i + batch_size]
            nn.backpropagation(X_batch, y_batch, learning_rate, lambda_)
        if epoch % 5 == 0:
            # Collect weights from all layers
            weights_list = [layer.weights for layer in nn.layers]

            # Compute predictions on the training set
            y_pred = nn.forward(X)

            # Calculate total loss with regularization over all layers
            total_loss = crossentropy_with_regularization(y_pred, y_train, weights_list, lambda_)
            losses.append(total_loss)
            iterations.append(epoch)
            print(f"Epoch {epoch}, Loss: {total_loss}")
        learning_rate = alpha0 / (1 + decayrate * epoch)