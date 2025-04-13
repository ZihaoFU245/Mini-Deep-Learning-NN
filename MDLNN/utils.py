import numpy as np

class Initializers:

    @staticmethod
    def zeros(shape):
        """
        Create an array of zeros.

        Parameters:
            shape (tuple): Shape of the output array.
        
        Returns:
            np.ndarray: Array of zeros with given shape.
        """
        return np.zeros(shape)
    
    @staticmethod
    def ones(shape):
        """
        Create an array of ones.

        Parameters:
            shape (tuple): Shape of the output array.
        
        Returns:
            np.ndarray: Array of ones with given shape.
        """
        return np.ones(shape)
    
    @staticmethod
    def RandN(shape, mean=0.0, std=1.0):
        """
        Create an array with random numbers from a normal distribution.

        Parameters:
            shape (tuple): Shape of the output array.
            mean (float): Mean of the distribution.
            std (float): Standard deviation of the distribution.
        
        Returns:
            np.ndarray: Array of normally distributed random numbers.
        """
        return np.random.normal(loc=mean, scale=std, size=shape)
    
    @staticmethod
    def RandUniform(shape, min_val=-0.05, max_val=0.05):
        """
        Create an array with random numbers from a uniform distribution.

        Parameters:
            shape (tuple): Shape of the output array.
            min_val (float): Lower bound of the distribution.
            max_val (float): Upper bound of the distribution.
        
        Returns:
            np.ndarray: Array of uniformly distributed random numbers.
        """
        return np.random.uniform(low=min_val, high=max_val, size=shape)
    
    @staticmethod
    def xavier_uniform(shape):
        """
        Xavier uniform initialization for weights.

        Parameters:
            shape (tuple): Expected shape (fan_in, fan_out) of the weights.
        
        Returns:
            np.ndarray: Array initialized with Xavier uniform strategy.
        """
        fan_in, fan_out = shape[0], shape[1]
        limit = np.sqrt(6 / (fan_in + fan_out))
        return np.random.uniform(-limit, limit, size=shape)
    
    @staticmethod
    def xavier_normal(shape):
        """
        Xavier normal initialization for weights.

        Parameters:
            shape (tuple): Expected shape (fan_in, fan_out) of the weights.
        
        Returns:
            np.ndarray: Array initialized with Xavier normal strategy.
        """
        fan_in, fan_out = shape[0], shape[1]
        stddev = np.sqrt(2 / (fan_in + fan_out))
        return np.random.normal(0, stddev, size=shape)
    
    @staticmethod
    def He_uniform(shape):
        """
        He uniform initialization for weights.

        Parameters:
            shape (tuple): Expected shape where shape[0] is fan_in.
        
        Returns:
            np.ndarray: Array initialized with He uniform strategy.
        """
        fan_in = shape[0]
        limit = np.sqrt(6 / fan_in)
        return np.random.uniform(-limit, limit, size=shape)
    
    @staticmethod
    def He_normal(shape):
        """
        He normal initialization for weights.

        Parameters:
            shape (tuple): Expected shape where shape[0] is fan_in.
        
        Returns:
            np.ndarray: Array initialized with He normal strategy.
        """
        fan_in = shape[0]
        stddev = np.sqrt(2 / fan_in)
        return np.random.normal(0, stddev, size=shape)


class Activations:

    @staticmethod
    def relu(x):
        """
        Rectified Linear Unit (ReLU) activation function.

        Parameters:
            x (np.ndarray): Input array.
        
        Returns:
            np.ndarray: Output array with ReLU applied elementwise.
        """
        return np.maximum(0, x)
    
    @staticmethod
    def relu_d(x):
        """
        Derivative of the ReLU function.

        Parameters:
            x (np.ndarray): Input array.
        
        Returns:
            np.ndarray: Derivative of ReLU, same shape as x.
        """
        return (x > 0).astype(int)
    
    @staticmethod
    def sigmoid(x):
        """
        Sigmoid activation function.

        Parameters:
            x (np.ndarray): Input array.
        
        Returns:
            np.ndarray: Output array with sigmoid applied elementwise.
        """
        return 1 / (1 + np.exp(-x))
    
    @staticmethod
    def sigmoid_d(x):
        """
        Derivative of the sigmoid function.

        Parameters:
            x (np.ndarray): Input array.
        
        Returns:
            np.ndarray: Derivative of sigmoid, same shape as x.
        """
        s = Activations.sigmoid(x)
        return s * (1 - s)
    
    @staticmethod
    def softmax(x):
        """
        Softmax activation function.

        Parameters:
            x (np.ndarray): Input array, typically a 2D array.
        
        Returns:
            np.ndarray: Softmax-transformed array, same shape as x.
        """
        exps = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exps / (np.sum(exps, axis=-1, keepdims=True))
    
    @staticmethod
    def tanh(x):
        """
        Hyperbolic tangent activation function.

        Parameters:
            x (np.ndarray): Input array.
        
        Returns:
            np.ndarray: tanh applied elementwise.
        """
        return np.tanh(x)

    @staticmethod
    def tanh_d(x):
        """
        Derivative of the tanh function.

        Parameters:
            x (np.ndarray): Input array.
        
        Returns:
            np.ndarray: Derivative of tanh, same shape as x.
        """
        return 1 - np.tanh(x)**2

    @staticmethod
    def softmax_d(x):
        """
        Simplified element-wise derivative of the softmax function.

        Parameters:
            x (np.ndarray): Input array.
        
        Returns:
            np.ndarray: Element-wise derivative, same shape as x.
        """
        s = Activations.softmax(x)
        return s * (1 - s)  

class Loss:

    @staticmethod
    def cross_entropy(y_pred, y_true):
        """
        Calculates the binary cross-entropy loss.

        Parameters:
            y_pred (np.ndarray): Predicted probabilities, shape (m, 1) or (m,).
            y_true (np.ndarray): True labels (0 or 1), same shape as y_pred.
        
        Returns:
            float: The binary cross-entropy loss.
        """
        m = y_pred.shape[0]
        eps = 1e-12  # Epsilon to prevent log(0)
        # Clip predictions to avoid log(0) or log(1) issues if predictions are exactly 0 or 1
        y_pred = np.clip(y_pred, eps, 1 - eps)
        loss = -np.sum(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)) / m
        return loss
    
    @staticmethod
    def binary_cross_entropy(y_pred, y_true):
        """
        Calculates the binary cross entropy loss.

        Parameters:
            y_pred (np.ndarray): Predicted probabilities, shape (m, 1) or (m,).
            y_true (np.ndarray): True labels (0 or 1), same shape as y_pred.
        
        Returns:
            float: The binary cross entropy loss.
        """
        return Loss.cross_entropy(y_pred, y_true)

    @staticmethod
    def binary_cross_entropy_d(y_pred, y_true):
        """
        Calculates the derivative of binary cross-entropy loss w.r.t. y_pred.

        Parameters:
            y_pred (np.ndarray): Predicted probabilities, shape (m, 1) or (m,).
            y_true (np.ndarray): True labels (0 or 1), same shape as y_pred.
        
        Returns:
            np.ndarray: The derivative of BCE loss w.r.t y_pred, same shape as y_pred.
        """
        eps = 1e-12 # Epsilon to prevent division by zero
        # Clip predictions slightly to avoid division by zero if y_pred is exactly 0 or 1
        y_pred = np.clip(y_pred, eps, 1 - eps)
        # Derivative: - (y_true / y_pred - (1 - y_true) / (1 - y_pred))
        return - (y_true / y_pred - (1 - y_true) / (1 - y_pred))

    @staticmethod
    def mean_squared_error(y_pred, y_true):
        """
        Calculates the Mean Squared Error (MSE) loss.

        Parameters:
            y_pred (np.ndarray): Predicted values.
            y_true (np.ndarray): True values, same shape as y_pred.
        
        Returns:
            float: The MSE loss.
        """
        m = y_pred.shape[0]
        loss = np.sum((y_pred - y_true)**2) / m
        return loss

    @staticmethod
    def mean_squared_error_d(y_pred, y_true):
        """
        Calculates the derivative of MSE loss w.r.t. y_pred.

        Parameters:
            y_pred (np.ndarray): Predicted values.
            y_true (np.ndarray): True values, same shape as y_pred.
        
        Returns:
            np.ndarray: The derivative of MSE loss w.r.t y_pred, same shape as y_pred.
        """
        m = y_pred.shape[0]
        # The derivative is 2 * (y_pred - y_true) / m, but often the 2/m is scaled into the learning rate
        # Returning (y_pred - y_true) is common practice for gradient calculation.
        return y_pred - y_true 

    @staticmethod
    def mean_absolute_error(y_pred, y_true):
        """
        Calculates the Mean Absolute Error (MAE) loss.

        Parameters:
            y_pred (np.ndarray): Predicted values.
            y_true (np.ndarray): True values, same shape as y_pred.
        
        Returns:
            float: The MAE loss.
        """
        m = y_pred.shape[0]
        loss = np.sum(np.abs(y_pred - y_true)) / m
        return loss

    @staticmethod
    def mean_absolute_error_d(y_pred, y_true):
        """
        Calculates the derivative of MAE loss w.r.t. y_pred.

        Parameters:
            y_pred (np.ndarray): Predicted values.
            y_true (np.ndarray): True values, same shape as y_pred.
        
        Returns:
            np.ndarray: The derivative of MAE loss w.r.t y_pred (sign), same shape as y_pred.
        """
        m = y_pred.shape[0]
        # Derivative is sign(y_pred - y_true) / m. Again, 1/m often absorbed.
        return np.sign(y_pred - y_true)

if __name__ == "__main__":
    arr = np.array([1, 2, 3, 4, -3, 0, -4])
    print(Activations.relu(arr))
