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

# Compile model with custom optimizer choice and parameters
model.compile(
    loss="binary_cross_entropy",
    optimizer="adam",  # or "sgd"
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

The framework supports multiple optimizers:

- **Adam**: Advanced optimizer with adaptive learning rates
  ```python
  model.compile(
      loss="binary_cross_entropy",
      optimizer="adam",
      optimizer_params={
          'learning_rate': 0.001,  # Default: 0.001
          'beta1': 0.9,           # Default: 0.9
          'beta2': 0.999,         # Default: 0.999
          'epsilon': 1e-8         # Default: 1e-8
      }
  )
  ```

- **SGD**: Stochastic Gradient Descent with momentum support
  ```python
  model.compile(
      loss="binary_cross_entropy",
      optimizer="sgd",
      optimizer_params={
          'learning_rate': 0.01,  # Default: 0.01
          'momentum': 0.9         # Default: 0.0
      }
  )
  ```

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

### Model Saving and Loading

The framework provides two ways to save your models:

1. **Complete Model Saving (Recommended)**
   Save and load the entire model including architecture, weights, and training configuration:

```python
# Save complete model
model.save('my_model.h5')

# Load complete model
loaded_model = Sequential.load('my_model.h5')

# Use the loaded model directly
predictions = loaded_model.predict(X_test)
```

2. **Weights-Only Saving**
   Save and load just the model weights (requires maintaining model architecture):

```python
# Save model weights
model.save_weights('model_weights.npz')

# To load weights, first recreate the model architecture
new_model = Sequential([
    Input(input_shape=(784,)),
    Dense(512, activation="relu"),
    Dense(10, activation="softmax")
])
new_model.compile(loss="cross_entropy", optimizer="adam")

# Then load the weights
new_model.load_weights('model_weights.npz')
```

The complete model saving (HDF5 format) is recommended as it:
- Saves the full model architecture
- Preserves layer configurations
- Stores optimizer settings
- Maintains loss function configuration
- Requires less code to load and use
- Is safer and more efficient for large models

Example workflow:
```python
# Train your model
model.fit(X_train, y_train, epochs=10)

# Save the trained weights
model.save_weights('my_model.npz')

# Later, create a new model with the same architecture
new_model = Sequential([
    Input(input_shape=(784,)),
    Dense(512, activation="relu"),
    Dense(10, activation="softmax")
])
new_model.compile(loss="cross_entropy", optimizer="adam")

# Load the saved weights
new_model.load_weights('my_model.npz')
```

The weights are saved in NumPy's .npz format, which is efficient for storing multiple arrays.

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

## Real-World Examples

### Example 1: XOR Problem
The XOR problem is a classic non-linearly separable problem that demonstrates the basic capabilities of neural networks:

```python
from MDLNN.models import Sequential
from MDLNN.layers import Input, Dense
from MDLNN.utils import Initializers
import numpy as np

# XOR dataset
X = np.array([[0, 0],
              [0, 1],
              [1, 0],
              [1, 1]])
y = np.array([[0],
              [1],
              [1],
              [0]])

# Build the model
model = Sequential([
    Input(input_shape=(2,)),
    Dense(4, activation="tanh", initializer=Initializers.xavier_uniform),
    Dense(1, activation="sigmoid", initializer=Initializers.xavier_uniform)
])

# Compile and train
model.compile(
    loss="binary_cross_entropy",
    optimizer="adam",
    optimizer_params={'learning_rate': 0.01}
)

model.fit(X, y, epochs=200, verbose=True)

# Make predictions
predictions = model.predict(X)
print("\nPredictions (rounded):")
print(np.round(predictions))
```

Expected output:
```
Predictions (rounded):
[[0.]
 [1.]
 [1.]
 [0.]]
```

### Example 2: MNIST Classification
This example shows how to build a deeper network for classifying handwritten digits from the MNIST dataset:

```python
from MDLNN.models import Sequential
from MDLNN.layers import Dense, Input, Dropout
from MDLNN.utils import Initializers
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

# Load and preprocess MNIST data
X, y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False)
X = X.astype('float32') / 255.0  # Normalize pixel values

# Convert labels to one-hot encoding
y_onehot = np.zeros((y.shape[0], 10))
y = y.astype(int)
y_onehot[np.arange(y.shape[0]), y] = 1

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y_onehot, test_size=0.2, random_state=42
)

# Build a deep model with dropout
model = Sequential([
    Input(input_shape=(784,)),
    Dense(512, activation="relu", initializer=Initializers.He_uniform),
    Dropout(keep_p=0.5),
    Dense(256, activation="relu", initializer=Initializers.He_uniform),
    Dropout(keep_p=0.3),
    Dense(10, activation="softmax", initializer=Initializers.xavier_uniform)
])

# Compile with appropriate loss for multi-class classification
model.compile(
    loss="cross_entropy",
    optimizer="adam",
    optimizer_params={
        'learning_rate': 0.001,
        'beta1': 0.9,
        'beta2': 0.999,
        'epsilon': 1e-8
    }
)

# Train with mini-batches
model.fit(
    X_train, 
    y_train,
    epochs=10,
    batch_size=128,
    verbose=True,
    shuffle=True
)

# Evaluate
loss, accuracy = model.evaluate(X_test, y_test)
```

This MNIST example demonstrates several advanced features:
- Data preprocessing and normalization
- One-hot encoding for multi-class classification
- Using Dropout layers for regularization
- He initialization for ReLU activation layers
- Mini-batch training with progress bars
- Model evaluation with accuracy metrics

## Requirements

- NumPy
- tqdm (for progress bars)
- h5py (for complete model saving)

## Version

Current version: 0.1.0

## License

MIT License