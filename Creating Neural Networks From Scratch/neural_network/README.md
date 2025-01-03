
---

# **Neural Network from Scratch in Python**

This repository contains a custom implementation of a feedforward neural network built entirely from scratch in Python. The project is structured to provide a deeper understanding of neural networks by implementing core functionalities manually, without relying on high-level libraries like TensorFlow or PyTorch.

---

## **Recent Updates**

- **16/10/2024**: Created the initial neural network structure with support for ReLU activation and a linear output layer. This foundational work significantly enhanced understanding of neural network mechanics.
- **14/11/2024**: Implemented the Adam optimization algorithm, which resolved previous training issues. This improvement boosted the model's accuracy on the MNIST dataset from 92% to 98% with fewer epochs and without data augmentation.
- **22/11/2024**: Restructured the project into a modular format with a clear directory hierarchy:
  - **`models/`**: Contains core components of the neural network, including initialization and architecture setup.
  - **`utils/`**: Includes utility functions for preprocessing, visualization, and configuration management.
  - **`mnist_data/`**: Stores the MNIST dataset files for training and testing.
  - **`main.py`**: Serves as the central script for running the project.

---

## **Project Structure**

```
neural_network/
├── mnist_data/
│   ├── t10k-images-idx3-ubyte.gz
│   ├── t10k-labels-idx1-ubyte.gz
│   ├── train-images-idx3-ubyte.gz
│   └── train-labels-idx1-ubyte.gz
├── models/
│   ├── __init__.py
│   └── nn_initialization.py
├── utils/
│   ├── __init__.py
│   ├── drawing.py
│   ├── functions.py
│   └── config.py
├── main.py
└── README.md
```

---

## **Features**

### **Customizable Neural Network Architecture**
- Easily define network layers and activation functions to experiment with different architectures.

### **Supported Activation Functions**
- **ReLU (Rectified Linear Unit)**
- **Softmax** (for multi-class classification)

### **Loss Functions**
- **Mean Squared Error (MSE)**
- **Cross-Entropy Loss** (with support for regularization)

### **Optimization Algorithms**
- **Adam Optimizer**: Incorporates adaptive learning rates and momentum, leading to faster convergence and improved performance.

### **Backpropagation Implementation**
- Manually coded backpropagation algorithm for training the network using gradient descent optimization.

### **Regularization Techniques**
- **L2 Regularization**: Helps prevent overfitting by penalizing large weights.

### **Batch Training**
- Supports mini-batch gradient descent with adjustable batch sizes.

### **Evaluation Metrics**
- **Accuracy Calculation**
- **Loss Tracking and Visualization**

---

## **Current Status**

- **Accuracy**: Achieved 98% on the MNIST dataset using Adam optimization.
- **Structure**: The project is fully modularized for better scalability and maintainability.
- **Interface**: Includes a GUI built with `Tkinter` for real-time digit drawing and prediction.

---

## **Next Steps**

- Extend the project to include Convolutional Neural Networks (CNNs).
- Optimize performance further with additional regularization techniques.

---

## **How to Use**

1. **Install Requirements**:
   Ensure the required packages (e.g., `numpy`, `python-mnist`, `Pillow`, `Tkinter`) are installed.

   ```bash
   pip install numpy python-mnist pillow
   ```

2. **Run the Main Script**:
   ```bash
   python main.py
   ```

3. **Explore the Code**:
   - Modify the `config.py` file to adjust hyperparameters like learning rate, epochs, and batch size.
   - Check the `models/` folder to understand the architecture and initialization process.
   - Use the `utils/drawing.py` for real-time digit drawing and prediction.

---
