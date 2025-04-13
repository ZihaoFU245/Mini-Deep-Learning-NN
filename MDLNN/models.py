from .optimizers import Adam
from .utils import Activations, Loss, Initializers
from .layers import Dense, Dropout, Input, Flatten
from typing import Tuple, Union, List
import numpy as np
from tqdm import tqdm

class Sequential:
    """
    A simple sequential model that chains layers together.

    This model performs forward and backward passes through a list of layers, and
    supports training via an optimizer. Each layer must implement forward() and backward().
    """

    def __init__(self, layers: List[Union[Dense, Dropout, Input, Flatten]] = []):
        self.layers = layers
        self.loss = None
        self.loss_d = None
        self.optimizer = None

    def add(self, layer: Union[Dense, Dropout, Input, Flatten]):
        self.layers.append(layer)
    
    def compile(self, loss: str, optimizer: str = "adam", optimizer_params: dict = None):
        """
        Configures the model for training.

        Parameters:
            loss (str): Name of the loss function to use (e.g., "binary_cross_entropy").
            optimizer (str): Name of the optimizer to use (e.g., "adam", "sgd").
            optimizer_params (dict): Optional parameters for the optimizer.
                                   For Adam: {'learning_rate': 0.001, 'beta1': 0.9, 'beta2': 0.999, 'epsilon': 1e-8}
                                   For SGD: {'learning_rate': 0.01, 'momentum': 0.0}
        """
        # Set up loss functions
        self.loss = getattr(Loss, loss)
        self.loss_d = getattr(Loss, f"{loss}_d", None)
        if self.loss_d is None:
            raise ValueError(f"Derivative of the loss function '{loss}' not found.")

        # Initialize weights by doing a forward pass with a small batch
        dummy_batch = np.zeros((1, *self.layers[0].input_shape))
        _ = self.forward(dummy_batch)

        # Collect parameters from trainable layers
        params = [layer.get_param() for layer in self.layers if hasattr(layer, 'get_param')]
        
        # Set up optimizer with default or custom parameters
        if optimizer_params is None:
            optimizer_params = {}

        # Initialize the appropriate optimizer
        optimizer = optimizer.lower()
        if optimizer == "adam":
            from .optimizers import Adam
            self.optimizer = Adam(parameters=params, **optimizer_params)
        elif optimizer == "sgd":
            from .optimizers import SGD
            self.optimizer = SGD(parameters=params, **optimizer_params)
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer}. Choose from: 'adam', 'sgd'")

    def forward(self , X):
        for layer in self.layers:
            X = layer.forward(X)

        return X
    
    def backward(self , y_pred , y_true):
        """
        Performs a backward pass through all layers.

        Parameters:
            y_pred: Predictions from the forward pass.
            y_true: True labels.
        """
        dA = self.loss_d(y_pred , y_true)
        for layer in reversed(self.layers):
            dA = layer.backward(dA)
    
    def train(self):
        """
        Sets all layers to training mode.
        """
        for layer in self.layers:
            if hasattr(layer, 'training'):
                layer.training = True

    def eval(self):
        """
        Sets all layers to evaluation mode.
        """
        for layer in self.layers:
            if hasattr(layer, 'training'):
                layer.training = False

    def summary(self):
        """
        Prints a summary of the model architecture including layer details and parameter counts.
        """
        print("Model Summary:")
        print("-" * 50)
        total_params = 0
        
        # Initialize weights by doing a forward pass with a small batch if not already done
        if any(isinstance(layer, Dense) and layer.W is None for layer in self.layers):
            dummy_batch = np.zeros((1, *self.layers[0].input_shape))
            _ = self.forward(dummy_batch)
            
        for i, layer in enumerate(self.layers):
            if isinstance(layer, Dense):
                if layer.W is None:
                    print(f"Warning: Layer {i+1} weights not initialized")
                    continue
                    
                W_shape = layer.W.shape
                params = np.prod(W_shape) + layer.num  # weights + biases
                total_params += params
                print(f"Layer {i+1}: Dense ({W_shape[0]} -> {W_shape[1]}) | activation: {layer.activation_str} | Params: {params}")
            elif isinstance(layer, Dropout):
                print(f"Layer {i+1}: Dropout (keep_prob={layer.keep_p})")
            elif isinstance(layer, Input):
                print(f"Layer {i+1}: Input {layer.input_shape}")
            elif isinstance(layer, Flatten):
                print(f"Layer {i+1}: Flatten")
        print("-" * 50)
        print(f"Total trainable parameters: {total_params}")

    def evaluate(self, X, y):
        """
        Evaluates the model on test data.

        Parameters:
            X: Input data
            y: True labels

        Returns:
            tuple: (loss, accuracy)
        """
        self.eval()  # Set to evaluation mode
        y_pred = self.predict(X)
        loss = self.loss(y_pred, y)
        
        # Handle both one-hot encoded and single-label targets
        if len(y.shape) > 1 and y.shape[1] > 1:  # One-hot encoded
            accuracy = np.mean(np.argmax(y_pred, axis=1) == np.argmax(y, axis=1))
        else:  # Single-label
            y_pred_classes = (y_pred > 0.5).astype(int)
            accuracy = np.mean(y_pred_classes == y)
            
        print(f"Loss: {loss:.4f} - Accuracy: {accuracy * 100:.2f}%")
        self.train()  # Set back to training mode
        return loss, accuracy

    def fit(self, X, y, epochs=10, batch_size=None, verbose=True, shuffle=True):
        """
        Trains the model on the provided data.

        Parameters:
            X: Input data.
            y: True labels.
            epochs (int): Number of training epochs.
            batch_size (int): Mini-batch size. If None, use full batch.
            verbose (bool): If True, prints training progress.
            shuffle (bool): Whether to shuffle the data at each epoch.
        """
        self.train()  # Ensure model is in training mode
        n_samples = X.shape[0]
        
        for epoch in range(epochs):
            if verbose:
                print(f"Epoch {epoch+1}/{epochs}")
            epoch_loss = 0
            
            # Create shuffled indices for this epoch if shuffle is True
            indices = np.arange(n_samples)
            if shuffle:
                np.random.shuffle(indices)
            
            if batch_size is None:
                # Full batch training
                y_pred = self.forward(X)
                loss_val = self.loss(y_pred, y)
                self.backward(y_pred, y)
                self.optimizer.step(self.layers)
                epoch_loss = loss_val
                
                if verbose:
                    print(f"Loss: {loss_val:.4f}")
            else:
                # Mini-batch training
                n_batches = (n_samples + batch_size - 1) // batch_size
                batch_losses = []
                
                # Create progress bar for batches
                pbar = tqdm(range(n_batches), desc="Training",
                          bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} '
                                   '[{elapsed}<{remaining}, {rate_fmt}] {postfix}')
                
                for batch in pbar:
                    # Calculate batch indices
                    start_idx = batch * batch_size
                    end_idx = min((batch + 1) * batch_size, n_samples)
                    
                    # Get the indices for this batch
                    batch_indices = indices[start_idx:end_idx]
                    
                    # Get batch data using the shuffled indices
                    X_batch = X[batch_indices]
                    y_batch = y[batch_indices]
                    
                    # Forward pass
                    y_pred = self.forward(X_batch)
                    loss_val = self.loss(y_pred, y_batch)
                    
                    # Backward pass and optimization
                    self.backward(y_pred, y_batch)
                    self.optimizer.step(self.layers)
                    
                    batch_losses.append(loss_val)
                    epoch_loss = np.mean(batch_losses)
                    
                    # Update progress bar with current loss
                    pbar.set_postfix({'loss': f'{epoch_loss:.4f}'})

                if verbose:
                    print(f"Average Loss: {epoch_loss:.4f}\n")

    def predict(self , X):
        """
        Generates predictions for the input data.

        Parameters:
            X: Input data.

        Returns:
            Predictions after passing through the model.
        """
        for layer in self.layers:
            if hasattr(layer , "training"):
                layer.training = False

        return self.forward(X)

    def save_weights(self, filepath):
        """
        Save model weights to a file.

        Parameters:
            filepath (str): Path where to save the weights. Should end with .npz
        """
        weights = {}
        for i, layer in enumerate(self.layers):
            if isinstance(layer, Dense):
                weights[f'layer_{i}_W'] = layer.W
                weights[f'layer_{i}_b'] = layer.b
        np.savez(filepath, **weights)
        print(f"Model weights saved to {filepath}")

    def load_weights(self, filepath):
        """
        Load model weights from a file.

        Parameters:
            filepath (str): Path to the weights file (.npz)
        """
        weights = np.load(filepath)
        weight_idx = 0
        for i, layer in enumerate(self.layers):
            if isinstance(layer, Dense):
                layer.W = weights[f'layer_{i}_W']
                layer.b = weights[f'layer_{i}_b']
                weight_idx += 1
        print(f"Model weights loaded from {filepath}")

    def save(self, filepath):
        """
        Save the complete model (architecture and weights) to an HDF5 file.
        
        Parameters:
            filepath (str): Path where to save the model. Should end with .h5
        """
        import h5py
        import json
        
        with h5py.File(filepath, 'w') as f:
            # Create model architecture group
            arch_group = f.create_group('architecture')
            
            # Save layer configurations
            layer_configs = []
            for i, layer in enumerate(self.layers):
                layer_group = arch_group.create_group(f'layer_{i}')
                
                # Save layer type and configuration
                config = {
                    'type': layer.__class__.__name__,
                    'config': {
                        'activation': getattr(layer, 'activation_str', None),
                        'units': getattr(layer, 'num', None),
                        'keep_p': getattr(layer, 'keep_p', None),
                        'input_shape': getattr(layer, 'input_shape', None)
                    }
                }
                layer_group.attrs['config'] = json.dumps(config)
                
                # Save weights and biases for Dense layers
                if isinstance(layer, Dense):
                    weights_group = layer_group.create_group('weights')
                    weights_group.create_dataset('W', data=layer.W)
                    weights_group.create_dataset('b', data=layer.b)
            
            # Save optimizer configuration
            if self.optimizer:
                opt_group = f.create_group('optimizer')
                opt_config = {
                    'type': type(self.optimizer).__name__,
                    'config': {k: v for k, v in self.optimizer.__dict__.items() 
                             if k not in ['parameters', 't', 'm', 'v']}
                }
                opt_group.attrs['config'] = json.dumps(opt_config)
            
            # Save loss function name
            if self.loss:
                f.attrs['loss'] = self.loss.__name__
        
        print(f"Model saved to {filepath}")

    @classmethod
    def load(cls, filepath):
        """
        Load a complete model from an HDF5 file.
        
        Parameters:
            filepath (str): Path to the model file (.h5)
            
        Returns:
            Sequential: A new Sequential model instance with loaded architecture and weights
        """
        import h5py
        import json
        
        with h5py.File(filepath, 'r') as f:
            # Create layers from saved configurations
            layers = []
            arch_group = f['architecture']
            
            # Sort layer keys to maintain order
            layer_keys = sorted([k for k in arch_group.keys()], 
                              key=lambda x: int(x.split('_')[1]))
            
            for layer_key in layer_keys:
                layer_group = arch_group[layer_key]
                config = json.loads(layer_group.attrs['config'])
                
                # Create appropriate layer based on type
                if config['type'] == 'Input':
                    layer = Input(input_shape=config['config']['input_shape'])
                elif config['type'] == 'Dense':
                    layer = Dense(
                        n_neurons=config['config']['units'],
                        activation=config['config']['activation'],
                        initializer=None  # Will be overwritten by weights
                    )
                elif config['type'] == 'Dropout':
                    layer = Dropout(keep_p=config['config']['keep_p'])
                elif config['type'] == 'Flatten':
                    layer = Flatten()
                
                # Load weights for Dense layers
                if config['type'] == 'Dense':
                    weights_group = layer_group['weights']
                    layer.W = weights_group['W'][:]
                    layer.b = weights_group['b'][:]
                
                layers.append(layer)
            
            # Create model
            model = cls(layers)
            
            # Load and set optimizer if it exists
            if 'optimizer' in f:
                opt_config = json.loads(f['optimizer'].attrs['config'])
                
                # Compile model with saved configuration
                if 'loss' in f.attrs:
                    model.compile(
                        loss=f.attrs['loss'],
                        optimizer=opt_config['type'].lower(),
                        optimizer_params=opt_config['config']
                    )
            
            return model



