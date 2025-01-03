# Neural Networks from Scratch

### Overview
This project demonstrates the creation of a simple neural network from scratch using Python. The intent behind the project is to gain a deeper and more intrinsic understanding of neural networks without relying heavily on pre-built frameworks. 

### Table of Contents
- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)


### Introduction
Hello! I am a college student with a strong passion for artificial intelligence and machine learning. This project is my endeavor to understand neural networks better by building one from scratch. It focuses on leveraging fundamental concepts, rather than abstracted high-level APIs, to provide a more granular insight into how neural networks operate.

### Features
- **Custom Neural Network Implementation**: A neural network built from the ground up using NumPy for array operations.
- **Fully Connected Layers**: Custom implementation of dense layers.
- **Activation Functions**: Includes ReLU and Softmax activation functions.
- **Backpropagation**: Manually coded backpropagation algorithm.
- **Batch Normalization**: Integration of batch normalization in the layers.
- **MNIST Dataset Loader**: Custom loader for loading and preprocessing the MNIST dataset.
- **GUI for Digit Drawing and Prediction**: A simple Tkinter GUI to draw digits and see the neural network's predictions in real-time.

### Installation
1. **Clone the repository**
    ```sh
    git clone https://github.com/jinstronda/neural-networks-from-scratch.git
    ```
2. **Change to the project directory**
    ```sh
    cd neural-networks-from-scratch
    ```
3. **Create a virtual environment** (optional but recommended)
    ```sh
    python -m venv venv
    ```
4. **Activate the virtual environment**
    - On Windows
        ```sh
        venv\Scripts\activate
        ```
    - On macOS/Linux
        ```sh
        source venv/bin/activate
        ```
5. **Install the dependencies**
    ```sh
    pip install -r requirements.txt
    ```

### Usage
- **Training the Neural Network**
    ```sh
    python main.py
    ```
  This will execute the training process on the MNIST dataset and evaluate the neural network.

- **Drawing Application**
    Run the `main.py` script to open a Tkinter window where you can draw digits, and the network will predict the drawn digit.

### Project Structure
```plaintext
neural_network/
│
├── data/
│   └── mnist_loader.py
│
├── models/
│   ├── layer.py
│   └── neural_network.py
│
├── utils/
│   ├── activation_functions.py
│   ├── helpers.py
│   ├── loss_functions.py
│
├── gui/
│   └── drawing_app.py
│
├── main.py
├── README.md
└── requirements.txt
```
