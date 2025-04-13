from MDLNN.models import Sequential
from MDLNN.layers import Dense, Input, Dropout
from MDLNN.utils import Initializers
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Load and preprocess MNIST data
print("Loading MNIST dataset...")
X, y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False)
X = X.astype('float32') / 255.0  # Normalize pixel values

# Store original images (unnormalized) for visualization
X_orig = X * 255.0

# Convert labels to integers and one-hot encoding
y = y.astype(int)
y_onehot = np.zeros((y.shape[0], 10))
y_onehot[np.arange(y.shape[0]), y] = 1

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y_onehot, test_size=0.2, random_state=42
)

# Keep original images for test set visualization
_, X_test_orig, _, y_test_orig = train_test_split(
    X_orig, y, test_size=0.2, random_state=42
)

# Build and train model
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

# Compile and train
model.compile(
    loss="cross_entropy",
    optimizer="adam",
    optimizer_params={'learning_rate': 0.001}
)

print("\nTraining the model...")
model.fit(X_train, y_train, epochs=10, batch_size=128, verbose=True)

# Make predictions
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true_classes = np.argmax(y_test, axis=1)

# Plot confusion matrix
plt.figure(figsize=(10, 8))
conf_matrix = confusion_matrix(y_true_classes, y_pred_classes)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

# Show some misclassified examples
misclassified_idx = np.where(y_pred_classes != y_true_classes)[0][:5]
if len(misclassified_idx) > 0:
    plt.figure(figsize=(15, 3))
    for i, idx in enumerate(misclassified_idx):
        plt.subplot(1, len(misclassified_idx), i + 1)
        plt.imshow(X_test_orig[idx].reshape(28, 28), cmap='gray')
        plt.title(f'True: {y_true_classes[idx]}\nPred: {y_pred_classes[idx]}')
        plt.axis('off')
    plt.tight_layout()
    plt.show()

# Print final accuracy
accuracy = np.mean(y_pred_classes == y_true_classes)
print(f"\nFinal Test Accuracy: {accuracy:.4f}")

