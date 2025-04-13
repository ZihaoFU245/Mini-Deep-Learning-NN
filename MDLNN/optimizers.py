import numpy as np
from abc import ABC, abstractmethod

class Optimizer(ABC):
    """
    Base class for all optimizers.
    """
    def __init__(self, parameters):
        """
        Initialize the optimizer.

        Parameters:
            parameters (list of tuples): List of tuples (W, b) for each layer.
        """
        self.parameters = parameters

    @abstractmethod
    def step(self, layers):
        """
        Perform an optimization step.

        Parameters:
            layers (list): List of layers having get_param(), get_gradients(), and set_params() methods.
        """
        pass

class Adam(Optimizer):
    """
    Adam optimizer implementation.
    Reference: https://arxiv.org/abs/1412.6980
    """
    def __init__(self, parameters, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        """
        Initialize the Adam optimizer.

        Parameters:
            parameters (list of tuples): List of tuples (W, b) for each layer.
            learning_rate (float): Learning rate.
            beta1 (float): Exponential decay rate for first moment estimates.
            beta2 (float): Exponential decay rate for second moment estimates.
            epsilon (float): Small constant to avoid division by zero.
        """
        super().__init__(parameters)
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.t = 0

        self.m = {}
        self.v = {}

        for i, (W, b) in enumerate(parameters):
            self.m[i] = {'W': np.zeros_like(W), 'b': np.zeros_like(b)}
            self.v[i] = {'W': np.zeros_like(W), 'b': np.zeros_like(b)}

    def step(self, layers):
        """
        Perform an Adam update step.
        """
        self.t += 1
        param_layer_idx = 0

        for i, layer in enumerate(layers):
            if not hasattr(layer, 'get_param'): 
                continue

            W, b = layer.get_param() 
            if W is None or b is None:
                param_layer_idx += 1
                continue
                
            dW, db = layer.get_gradients()
            if dW is None or db is None:
                param_layer_idx += 1
                continue

            if param_layer_idx not in self.m or param_layer_idx not in self.v:
                self.m[param_layer_idx] = {'W': np.zeros_like(W), 'b': np.zeros_like(b)}
                self.v[param_layer_idx] = {'W': np.zeros_like(W), 'b': np.zeros_like(b)}

            # First moment update
            self.m[param_layer_idx]['W'] = self.beta1 * self.m[param_layer_idx]['W'] + (1 - self.beta1) * dW
            self.m[param_layer_idx]['b'] = self.beta1 * self.m[param_layer_idx]['b'] + (1 - self.beta1) * db

            # Second moment update
            self.v[param_layer_idx]['W'] = self.beta2 * self.v[param_layer_idx]['W'] + (1 - self.beta2) * (dW ** 2)
            self.v[param_layer_idx]['b'] = self.beta2 * self.v[param_layer_idx]['b'] + (1 - self.beta2) * (db ** 2)

            # Bias correction
            m_W_hat = self.m[param_layer_idx]['W'] / (1 - self.beta1 ** self.t + self.epsilon) 
            m_b_hat = self.m[param_layer_idx]['b'] / (1 - self.beta1 ** self.t + self.epsilon)
            v_W_hat = self.v[param_layer_idx]['W'] / (1 - self.beta2 ** self.t + self.epsilon)
            v_b_hat = self.v[param_layer_idx]['b'] / (1 - self.beta2 ** self.t + self.epsilon)

            # Calculate update values
            update_W = self.learning_rate * m_W_hat / (np.sqrt(v_W_hat) + self.epsilon)
            update_b = self.learning_rate * m_b_hat / (np.sqrt(v_b_hat) + self.epsilon)

            # Update parameters IN-PLACE
            W -= update_W
            b -= update_b
            
            param_layer_idx += 1

class SGD(Optimizer):
    """
    Stochastic Gradient Descent optimizer with momentum.
    """
    def __init__(self, parameters, learning_rate=0.01, momentum=0.0):
        """
        Initialize SGD optimizer.

        Parameters:
            parameters (list of tuples): List of tuples (W, b) for each layer.
            learning_rate (float): Learning rate.
            momentum (float): Momentum factor (default: 0.0).
        """
        super().__init__(parameters)
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.v = {}

        for i, (W, b) in enumerate(parameters):
            self.v[i] = {'W': np.zeros_like(W), 'b': np.zeros_like(b)}

    def step(self, layers):
        """
        Perform an SGD update step.
        """
        param_layer_idx = 0

        for i, layer in enumerate(layers):
            if not hasattr(layer, 'get_param'):
                continue

            W, b = layer.get_param()
            if W is None or b is None:
                param_layer_idx += 1
                continue

            dW, db = layer.get_gradients()
            if dW is None or db is None:
                param_layer_idx += 1
                continue

            if param_layer_idx not in self.v:
                self.v[param_layer_idx] = {'W': np.zeros_like(W), 'b': np.zeros_like(b)}

            # Update with momentum
            self.v[param_layer_idx]['W'] = self.momentum * self.v[param_layer_idx]['W'] - self.learning_rate * dW
            self.v[param_layer_idx]['b'] = self.momentum * self.v[param_layer_idx]['b'] - self.learning_rate * db

            # Update parameters IN-PLACE
            W += self.v[param_layer_idx]['W']
            b += self.v[param_layer_idx]['b']

            param_layer_idx += 1
