# neural-networks

# Neural Network Classifier (NumPy Implementation)

## Overview

This project implements a **fully connected neural network** using only **NumPy**, without relying on deep learning frameworks like TensorFlow or PyTorch. The model is trained on a classification dataset using **forward propagation, backpropagation, and gradient descent**. The training process includes performance visualization with real-time accuracy tracking and class prediction distribution plots.

## Features

- **Custom Neural Network Implementation**: Built using NumPy from scratch
- **Activation Functions**: Sigmoid and Softmax
- **Forward and Backward Propagation**
- **Gradient Descent Optimization**
- **Configurable Hyperparameters** (Hidden layers, Learning rate, Epochs)
- **Training and Testing Accuracy Tracking**
- **Real-Time Class Prediction Visualization**
- **Confusion Matrix Evaluation**

## Requirements

Make sure you have the following installed before running the code:

```
pip install numpy matplotlib seaborn
```

## Model Architecture

The model consists of:

- **Input Layer**: Size depends on the dataset features
- **Hidden Layer**: Configurable (default: 64 neurons, Sigmoid activation)
- **Output Layer**: Uses Softmax for multi-class classification

## Code Implementation

### 1️⃣ Initialize Model Parameters

```
def initialize_parameters(input_size, hidden_size, output_size):
    W1 = np.random.randn(hidden_size, input_size) * 0.01
    b1 = np.zeros((hidden_size, 1))
    W2 = np.random.randn(output_size, hidden_size) * 0.01
    b2 = np.zeros((output_size, 1))
    return {"W1": W1, "b1": b1, "W2": W2, "b2": b2}
```

### 2️⃣ Forward Propagation

```
def forward_propagation(X, parameters):
    W1, b1, W2, b2 = parameters["W1"], parameters["b1"], parameters["W2"], parameters["b2"]
    Z1 = np.dot(W1, X) + b1
    A1 = sigmoid(Z1)
    Z2 = np.dot(W2, A1) + b2
    A2 = softmax(Z2)
    return A2, {"Z1": Z1, "A1": A1, "Z2": Z2, "A2": A2}
```

### 3️⃣ Compute Cost

```
def compute_cost(A2, Y):
    m = Y.shape[1]
    return -np.sum(Y * np.log(A2 + 1e-8)) / m
```

### 4️⃣ Backpropagation & Parameter Updates

```
def backward_propagation(X, Y, parameters, cache, learning_rate):
    m = X.shape[1]
    W1, W2 = parameters["W1"], parameters["W2"]
    A1, A2 = cache["A1"], cache["A2"]

    dZ2 = A2 - Y
    dW2 = np.dot(dZ2, A1.T) / m
    db2 = np.sum(dZ2, axis=1, keepdims=True) / m
    dZ1 = np.dot(W2.T, dZ2) * (A1 * (1 - A1))
    dW1 = np.dot(dZ1, X.T) / m
    db1 = np.sum(dZ1, axis=1, keepdims=True) / m

    parameters["W1"] -= learning_rate * dW1
    parameters["b1"] -= learning_rate * db1
    parameters["W2"] -= learning_rate * dW2
    parameters["b2"] -= learning_rate * db2
    return parameters
```

### 5️⃣ Training the Neural Network

```
def train_neural_network(X_train, y_train, X_test, y_test, hidden_size=64, learning_rate=0.1, epochs=1000):
    input_size = X_train.shape[0]
    output_size = y_train.shape[0]
    parameters = initialize_parameters(input_size, hidden_size, output_size)

    for epoch in range(epochs):
        A2, cache = forward_propagation(X_train, parameters)
        cost = compute_cost(A2, y_train)
        parameters = backward_propagation(X_train, y_train, parameters, cache, learning_rate)

        if epoch % 100 == 0:
            predictions = predict(X_train, parameters)
            accuracy = np.mean(predictions == np.argmax(y_train, axis=0)) * 100
            print(f"Epoch {epoch}: Cost {cost:.4f}, Training Accuracy: {accuracy:.2f}%")

    return parameters
```

### 6️⃣ Predictions & Evaluation

```
def predict(X, parameters):
    A2, _ = forward_propagation(X, parameters)
    return np.argmax(A2, axis=0)
```

## Enhancements to Improve Accuracy

To improve model accuracy, the following enhancements were made:

1. **Increased Hidden Layers & Neurons**: More layers help the network learn complex patterns.
2. **Tuning Hyperparameters**: Optimized learning rate and epochs for better convergence.
3. **Data Normalization**: Scaling inputs to improve numerical stability.
4. **Visualization of Class Predictions**: Tracking class prediction counts over time.
5. **Confusion Matrix Analysis**: Identifying misclassified classes and improving model robustness.
