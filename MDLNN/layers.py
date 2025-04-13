import numpy as np
from .utils import Initializers , Activations

class Dense:
    """
    A fully connected (dense) layer in a neural network.

    Attributes:
        num (int): Number of neurons in the layer.
        activation_str (str): Name of the activation function.
        activation (callable): Activation function from Activations class.
        activation_d (callable): Derivative of the activation function.
        initializer (callable): Initialization function for weights.
        W (np.ndarray): Weight matrix of the layer. Shape (input_dim, num).
        b (np.ndarray): Bias vector of the layer. Shape (num,).
        X (np.ndarray): Input received during the forward pass.
        Z (np.ndarray): Linear output (pre-activation) during the forward pass.
        A (np.ndarray): Activated output during the forward pass.
        dW (np.ndarray): Gradient of the loss with respect to the weights.
        db (np.ndarray): Gradient of the loss with respect to the bias.
        trainable (bool): Whether the layer's parameters should be updated during training.
        training (bool): Whether the layer is in training mode.
    """
    def __init__(self, n_neurons: int, activation: str, initializer: Initializers, trainable: bool = True):
        """
        Initializes the Dense layer.

        Parameters:
            n_neurons (int): The number of neurons in this layer.
            activation (str): The name of the activation function to use (e.g., 'relu', 'sigmoid').
            initializer (callable): The function to use for initializing weights (e.g., Initializers.xavier_uniform).
            trainable (bool): Whether the layer's parameters should be updated during training.
        """
        self.num = n_neurons
        self.activation_str = activation
        self.activation = getattr(Activations, activation)
        self.activation_d = getattr(Activations, activation + "_d", None)
        if self.activation_d is None:
            print(f"Warning: Derivative for activation '{activation}' not found.")
        self.initializer = initializer
        self.trainable = trainable
        self.training = True  # Default to training mode

        self.W = None
        self.b = None

        self.dW = None
        self.db = None

    def forward(self , X : np.ndarray):
        """
        Performs the forward pass through the dense layer.

        Parameters:
            X (np.ndarray): Input data or activations from the previous layer. Shape (batch_size, input_dim).

        Returns:
            np.ndarray: The activated output of the layer. Shape (batch_size, num).
        """
        self.X = X
        # Dynamically initialize weights using input dimension if uninitialized
        if self.W is None:
            input_dim = X.shape[1]
            # Initialize weights using the specified initializer
            self.W = self.initializer((input_dim, self.num))
            # Initialize bias specifically with zeros
            self.b = Initializers.zeros((self.num,))

        self.Z = self.X @ self.W + self.b
        self.A = self.activation(self.Z)
        return self.A
    
    def backward(self , dA):
        """
        Performs the backward pass through the dense layer.

        Calculates the gradients of the loss with respect to the weights (dW),
        bias (db), and the input of this layer (dX).

        Parameters:
            dA (np.ndarray): Gradient of the loss with respect to the output activation (A) of this layer.
                             Shape (batch_size, num).

        Returns:
            np.ndarray: Gradient of the loss with respect to the input (X) of this layer.
                        Shape (batch_size, input_dim).
        """
        # Ensure activation derivative exists
        if self.activation_d is None:
            raise NotImplementedError(f"Backward pass requires derivative for activation '{self.activation_str}', but it's not defined.")
            
        dZ = dA * self.activation_d(self.Z)
        m = self.X.shape[0] # Number of examples in batch
        
        # Only compute parameter gradients if the layer is trainable
        if self.trainable and self.training:
            self.dW = np.dot(self.X.T, dZ) / m
            self.db = np.sum(dZ, axis=0) / m
        else:
            self.dW = None
            self.db = None
            
        dX = np.dot(dZ, self.W.T)
        return dX
    
    ### Helper Functions ###

    def get_param(self):
        """
        Returns the current weights and bias of the layer.

        Returns:
            tuple: A tuple containing (W, b).
        """
        return self.W , self.b
    
    def get_gradients(self):
        """
        Returns the calculated gradients for weights and bias.

        Returns:
            tuple: A tuple containing (dW, db).
        """
        return self.dW , self.db
    
    def set_params(self , W , b):
        """
        Sets the weights and bias of the layer.

        Parameters:
            W (np.ndarray): New weight matrix.
            b (np.ndarray): New bias vector.
        """
        self.W = W
        self.b = b


    
class Dropout:
    """
    Dropout regularization layer.

    Randomly sets a fraction of input units to 0 at each update during training
    time, which helps prevent overfitting. Inputs are scaled up by 1/keep_p
    during training to compensate for the dropped units. During evaluation,
    this layer does nothing.

    Attributes:
        keep_p (float): Probability of keeping a unit active. 1 - dropout rate.
        mask (np.ndarray): Boolean mask indicating which units were kept/dropped.
        training (bool): Flag indicating whether the layer is in training mode.
    """
    def __init__(self , keep_p=0.5, trainable=True):
        """
        Initializes the Dropout layer.

        Parameters:
            keep_p (float): The probability that each element is kept. Defaults to 0.5.
            trainable (bool): Whether the layer's parameters should be updated during training.
        """
        if not 0 < keep_p <= 1:
            raise ValueError("keep_p must be in the range (0, 1]")
        self.keep_p = keep_p
        self.mask = None
        self.training = True  # Default to training mode
        self.trainable = trainable  # Dropout doesn't have parameters but includes trainable for consistency

    def forward(self, X):
        """
        Performs the forward pass for the Dropout layer.

        Applies dropout during training, otherwise passes input through unchanged.

        Parameters:
            X (np.ndarray): Input data or activations from the previous layer.

        Returns:
            np.ndarray: Output after applying dropout (if training) or the input itself (if not training).
        """
        if self.training:
            self.mask = (np.random.rand(*X.shape) < self.keep_p)
            # Scale the activations during training
            return (X * self.mask) / self.keep_p
        else:
            # During evaluation/testing, dropout is turned off
            return X
        
    def backward(self , dA):
        """
        Performs the backward pass for the Dropout layer.

        Applies the same dropout mask used during the forward pass to the gradients.

        Parameters:
            dA (np.ndarray): Gradient of the loss with respect to the output of this layer.

        Returns:
            np.ndarray: Gradient of the loss with respect to the input of this layer.
        """
        if self.training:
            # Apply the mask and scale gradients
            return (dA * self.mask) / self.keep_p
        else:
            # During evaluation/testing, pass gradients through unchanged
            return dA
        

class Input:
    def __init__(self , input_shape, trainable=True):
        """
        Initializes the Input layer.

        Parameters:
            input_shape (tuple): The expected shape of the input data (excluding batch size).
            trainable (bool): Whether the layer's parameters should be updated during training.
        """
        self.input_shape = input_shape
        self.training = True  # Default to training mode
        self.trainable = trainable  # Input layer doesn't have parameters but includes trainable for consistency

    def forward(self , X):
        """
        Performs the forward pass for the Input layer.

        Parameters:
            X (np.ndarray): The input data.

        Returns:
            np.ndarray: The input data, unchanged.

        Raises:
            ValueError: If the shape of the input data does not match the expected input shape.
        """
        if X.shape[1:] != self.input_shape:  # Check shape, excluding batch size
            raise ValueError(f"Input shape {X.shape[1:]} does not match expected input shape {self.input_shape}")
        return X

    def backward(self , dA):
        """
        Performs the backward pass for the Input layer.  Since there are no weights, simply returns the derivative.

        Parameters:
            dA (np.ndarray): The derivative of the loss with respect to the output of this layer.

        Returns:
            np.ndarray: The derivative of the loss with respect to the input of this layer (dA unchanged).
        """
        return dA

class Flatten:
    """
    Flattens the input while preserving the batch size.

    This layer reshapes the input tensor from shape (batch_size, d1, d2, ...)
    to (batch_size, d1 * d2 * ...). It is commonly used to transition from
    convolutional layers to fully connected layers.

    Attributes:
        input_shape (tuple): The shape of the input tensor (excluding batch size)
                             as received during the forward pass. This is used
                             to reshape the gradient in the backward pass.
    """
    def __init__(self, trainable=True):
        """
        Initializes the Flatten layer.

        Parameters:
            trainable (bool): Whether the layer's parameters should be updated during training.
        """
        self.training = True  # Default to training mode
        self.trainable = trainable  # Flatten doesn't have parameters but includes trainable for consistency

    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Performs the forward pass by flattening the input tensor.

        Parameters:
            X (np.ndarray): Input tensor of shape (batch_size, d1, d2, ...).

        Returns:
            np.ndarray: Flattened tensor of shape (batch_size, d1 * d2 * ...).
        """
        self.input_shape = X.shape[1:]  # Store the input shape for backward pass
        batch_size = X.shape[0]
        return X.reshape(batch_size, -1)  # Use -1 to infer the flattened dimension

    def backward(self, dA: np.ndarray) -> np.ndarray:
        """
        Performs the backward pass by reshaping the gradient to the original input shape.

        Parameters:
            dA (np.ndarray): Gradient of the loss with respect to the output of this layer,
                             shape (batch_size, d1 * d2 * ...).

        Returns:
            np.ndarray: Gradient of the loss with respect to the input of this layer,
                         shape (batch_size, d1, d2, ...).
        """
        batch_size = dA.shape[0]
        return dA.reshape(batch_size, *self.input_shape)

class Conv2D:
    pass


