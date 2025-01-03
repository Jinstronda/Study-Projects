# main.py
from neural_network.config import *
from neural_network.models.nn_initialization import *
from neural_network.utils.functions import *
from neural_network.utils.drawing import *
from sklearn.metrics import precision_score, recall_score, f1_score  # Just for metrics

def main():
    # Load the MNIST dataset
    mnist = MNIST("MNIST_dataset",gz=True)

    # Load the data (returns tuples)
    X_train, y_train = mnist.load_training()
    X_test, y_test = mnist.load_testing()

    # Convert the data to NumPy arrays
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_test = np.array(X_test)
    y_test = np.array(y_test)

    # Normalize the data to the range [0, 1]
    X_train = X_train / 255.0
    X_test = X_test / 255.0

    X_train = X_train.reshape(X_train.shape[0], -1)
    X_test = X_test.reshape(X_test.shape[0], -1)

    y_train = one_hot_encode(y_train, 10)
    y_test = one_hot_encode(y_test, 10)

    # Initialize the neural network layers
    layer1 = Layer("relu", 64, 784,True)
    layer2 = Layer("relu", 64, 64,True)
    layer3 = Layer("softmax", 10, 64, True)
    neuralnetwork = NeuralNetwork([layer1, layer2, layer3])

    # Testing Different Learning Rates and saving them, using a log scale
    # r = np.random.uniform(np.log10(0.0001), np.log10(0.01))
    # alpha = 10 ** r
    # Train the neural network
    training(neuralnetwork, X_train, y_train, alpha, 20, lambda_)
    neuralnetwork.testing()

    # Evaluate the neural network on test data
    y_test_labels = np.argmax(y_test, axis=1)
    y_train_labels = np.argmax(y_train, axis=1)
    softmax_output = neuralnetwork.forward(X_test)
    predictions = np.argmax(softmax_output, axis=1)  # Get the index of the highest probability for each sample
    accuracy = np.mean(predictions == y_test_labels)   # Calculate the proportion of correct predictions
    precision = precision_score(y_test_labels, predictions, average='macro')
    recall = recall_score(y_test_labels, predictions, average='macro')
    f1 = f1_score(y_test_labels, predictions, average='macro')

    print(f'Precision: {precision * 100:.2f}%')
    print(f'Recall: {recall * 100:.2f}%')
    print(f'F1 Score: {f1 * 100:.2f}%')
    print(f"Learning Rate:{alpha}")
    print(f"Regularization:{lambda_}")
    print(f"Batch Size:{batch_size}")
    with open("results.txt", "a") as f:
        f.write(f"Trial Number {trial}\n")
        f.write(f"Learning Rate:{alpha}\n")
        f.write(f"Regularization:{lambda_}\n")
        f.write(f"Precision: {precision * 100:.2f}%\n")
        f.write(f"Recall: {recall * 100:.2f}%\n")
        f.write(f"F1 Score: {f1 * 100:.2f}%\n")
        f.write(f"Accuracy: {accuracy * 100:.2f}%\n")
        f.write(f"Batch Size: {batch_size}\n")
        f.write(f"Epochs: {epochs}\n")
        f.write("\n")

    # Initialize and run the drawing application
    app = DrawingApp(neuralnetwork)
    app.run()

if __name__ == "__main__":
    main()