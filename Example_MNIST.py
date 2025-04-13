from MDLNN.models import Sequential
from MDLNN.layers import Dense, Input, Dropout
from MDLNN.utils import Initializers
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

# Load MNIST dataset
print("Loading MNIST dataset...")
X, y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False)
print("Dataset loaded!")

# Preprocess the data
# Normalize pixel values
X = X.astype('float32') / 255.0

# Convert labels to one-hot encoding
y_onehot = np.zeros((y.shape[0], 10))
y = y.astype(int)
y_onehot[np.arange(y.shape[0]), y] = 1

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y_onehot, test_size=0.2, random_state=42
)

# Build the model
model = Sequential([
    Input(input_shape=(784,)),
    Dense(512, activation="relu", initializer=Initializers.He_uniform),
    Dropout(keep_p=0.5),
    Dense(256, activation="relu", initializer=Initializers.He_uniform),
    Dropout(keep_p=0.3),
    Dense(10, activation="softmax", initializer=Initializers.xavier_uniform)
])

# Print model summary
model.summary()

# Compile the model
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

# Train the model
print("\nTraining the model...")
model.fit(
    X_train, 
    y_train,
    epochs=10,
    batch_size=128,
    verbose=True,
    shuffle=True
)

# Evaluate the model
print("\nEvaluating the model...")
loss, accuracy = model.evaluate(X_test, y_test)

# Make some predictions
print("\nMaking predictions on test set...")
predictions = model.predict(X_test[:10])
print("\nPredicted classes for first 10 test samples:")
print(np.argmax(predictions, axis=1))
print("Actual classes:")
print(np.argmax(y_test[:10], axis=1))

