# Neural Network Implementation from Scratch

This repository contains a comprehensive implementation of neural networks from scratch, focusing on both traditional neural networks and ResNet architectures. The implementation is done purely in Python using NumPy, demonstrating deep understanding of neural network fundamentals and advanced architectures.

## Author
Nissim Brami

## Overview

This project implements two main neural network architectures:

1. **Traditional Neural Network**
   - Fully-connected layers with configurable dimensions
   - Support for different activation functions (ReLU, tanh)
   - L2 regularization
   - Comprehensive gradient checking functionality
   - Cross-entropy loss for classification tasks
   - Tested and validated on MNIST dataset
   - Forward and backward propagation implementations

2. **ResNet (Residual Neural Network)**
   - Residual connections for better gradient flow
   - Batch normalization implementation
   - Dropout layers for regularization
   - Support for different activation functions (ReLU, Leaky ReLU)
   - Tested and validated on CIFAR-10 dataset
   - Training and testing modes

## Key Features

### Neural Network Class
- Flexible architecture with customizable layer dimensions
- Gradient checking with detailed diagnostics
- Numerically stable softmax implementation
- Support for regularization to prevent overfitting

### ResNet Implementation
- Skip connections for handling vanishing gradients
- Advanced normalization techniques
- Configurable dropout rates
- Complete forward and backward propagation pipeline

## Usage

### Traditional Neural Network
```python
# Initialize with custom parameters
nn = NeuralNetwork(
    layer_dims=[784, 512, 256, 10],  # Example for MNIST
    activation='relu',     # Options: 'relu', 'tanh'
    regularization='l2',   # Options: 'l2', None
    reg_lambda=0.01       # Regularization strength
)

# Train on MNIST
nn.train_step(X_train, y_train)
predictions = nn.predict(X_test)
```

### ResNet
```python
# Initialize with custom parameters
resnet = ResNet(
    layer_dims=[3072, 1024, 512, 10],  # Example for CIFAR-10
    activation='relu',      # Options: 'relu', 'leaky_relu'
    regularization='l2',    # Options: 'l2', None
    reg_lambda=0.01,       # Regularization strength
    use_dropout=True,      # Enable/disable dropout
    dropout_rate=0.1       # Dropout rate
)

# Train on CIFAR-10
resnet.train_step(X_train, y_train)
predictions = resnet.predict(X_test)
```

## Customizable Parameters

### Neural Network
- `layer_dims`: List of integers defining the network architecture
- `activation`: Choice of activation function ('relu' or 'tanh')
- `regularization`: Type of regularization ('l2' or None)
- `reg_lambda`: Regularization strength parameter

### ResNet
- `layer_dims`: List of integers defining the network architecture
- `activation`: Choice of activation function ('relu' or 'leaky_relu')
- `regularization`: Type of regularization ('l2' or None)
- `reg_lambda`: Regularization strength parameter
- `use_dropout`: Boolean to enable/disable dropout
- `dropout_rate`: Dropout probability (0 to 1)
- `batch_norm`: Parameters for batch normalization can be tuned

## Datasets

### MNIST
- Used for testing the traditional Neural Network
- 28x28 grayscale images
- 10 classes (digits 0-9)
- 60,000 training examples
- 10,000 test examples

### CIFAR-10
- Used for testing the ResNet implementation
- 32x32 color images
- 10 classes
- 50,000 training images
- 10,000 test images

## Dependencies
- NumPy

## Future Improvements
- Implementation of additional optimization algorithms
- Support for different learning rate schedules
- Extended dataset compatibility
- Performance optimizations

## License

MIT License

Copyright (c) 2024 Nissim Brami

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
