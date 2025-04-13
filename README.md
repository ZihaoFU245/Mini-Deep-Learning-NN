# MDLNN (Mini Deep Learning Neural Network)

A lightweight, modular deep learning framework implemented in pure NumPy. This framework provides a Keras-like API for building and training neural networks.

## Features

- Sequential model architecture
- Various layer types (Dense, Dropout, Input, Flatten)
- Multiple activation functions (ReLU, Sigmoid, Tanh, Softmax)
- Different weight initializers (Xavier/Glorot, He initialization)
- Loss functions (Binary Cross-Entropy, MSE, MAE)
- Adam optimizer with bias correction
- Training with mini-batch support and progress bars
- Transfer learning support with trainable/non-trainable layers
- Model evaluation and prediction capabilities

## Installation

```bash
pip install numpy tqdm
```

## Quick Start

```python
from MDLNN.models import Sequential
from MDLNN.layers import Input, Dense
from MDLNN.utils import Initializers

# Create a simple binary classification model
model = Sequential([
    Input(input_shape=(2,)),
    Dense(4, activation="tanh", initializer=Initializers.xavier_uniform),
    Dense(1, activation="sigmoid", initializer=Initializers.xavier_uniform)
])

# Compile model with custom optimizer parameters
model.compile(
    loss="binary_cross_entropy",
    optimizer_params={'learning_rate': 0.01}
)

# Train the model with mini-batches and progress bar
model.fit(X, y, epochs=100, batch_size=32, verbose=True, shuffle=True)

# Make predictions
predictions = model.predict(X_test)
```

## Components

### Layers

- **Dense**: Fully connected layer with configurable activation
- **Dropout**: Regularization layer to prevent overfitting
- **Input**: Input layer that validates data shape
- **Flatten**: Reshapes input for transition between conv and dense layers
- **Conv2D**: (Placeholder for future implementation)

### Activation Functions

- ReLU
- Sigmoid
- Tanh
- Softmax

### Weight Initializers

- Zeros
- Ones
- Random Normal
- Random Uniform
- Xavier/Glorot (Uniform and Normal)
- He (Uniform and Normal)

### Loss Functions

- Binary Cross-Entropy
- Mean Squared Error (MSE)
- Mean Absolute Error (MAE)

### Optimizers

- Adam (with momentum and RMSprop-style moving averages)
  - Configurable learning rate
  - Customizable beta1, beta2, and epsilon parameters

## Advanced Features

### Transfer Learning

Layers can be frozen for transfer learning:

```python
model = Sequential([
    Input(input_shape=(784,)),
    Dense(256, activation="relu", initializer=Initializers.xavier_uniform, trainable=False),  # Frozen layer
    Dense(10, activation="softmax", initializer=Initializers.xavier_uniform)  # Trainable layer
])
```

### Model Evaluation

```python
# Evaluate model performance
loss, accuracy = model.evaluate(X_test, y_test)
```

### Model Summary

```python
model.summary()
```

Example output:
```
Model Summary:
--------------------------------------------------
Layer 1: Input (2,)
Layer 2: Dense (2 -> 4) | Params: 12
Layer 3: Dense (4 -> 1) | Params: 5
--------------------------------------------------
Total trainable parameters: 17
```

## Training Features

- Mini-batch training with progress bars
- Shuffle option for each epoch
- Training/evaluation mode switching
- Batch size optimization for large datasets
- Gradient computation for trainable parameters only
- Automatic weight initialization during compilation

## Optimizer Configuration

You can customize the Adam optimizer during model compilation:

```python
model.compile(
    loss="binary_cross_entropy",
    optimizer_params={
        'learning_rate': 0.001,  # Default: 0.001
        'beta1': 0.9,           # Default: 0.9
        'beta2': 0.999,         # Default: 0.999
        'epsilon': 1e-8         # Default: 1e-8
    }
)
```

## Requirements

- NumPy
- tqdm (for progress bars)

## Version

Current version: 0.1.0

## License

MIT License