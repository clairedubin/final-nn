# Imports
import numpy as np
from typing import List, Dict, Tuple, Union
from numpy.typing import ArrayLike
import warnings

class NeuralNetwork:
    """
    This is a class that generates a fully-connected neural network.

    Parameters:
        nn_arch: List[Dict[str, float]]
            A list of dictionaries describing the layers of the neural network.
            e.g. [{'input_dim': 64, 'output_dim': 32, 'activation': 'relu'}, {'input_dim': 32, 'output_dim': 8, 'activation:': 'sigmoid'}]
            will generate a two-layer deep network with an input dimension of 64, a 32 dimension hidden layer, and an 8 dimensional output.
        lr: float
            Learning rate (alpha).
        seed: int
            Random seed to ensure reproducibility.
        batch_size: int
            Size of mini-batches used for training.
        epochs: int
            Max number of epochs for training.
        loss_function: str
            Name of loss function.

    Attributes:
        arch: list of dicts
            (see nn_arch above)
    """

    def __init__(
        self,
        #note: changed from Union(int, str) to Union[int, str]
        nn_arch: List[Dict[str, Union[int, str]]],
        lr: float,
        seed: int,
        batch_size: int,
        epochs: int,
        loss_function: str
    ):

        # Save architecture
        self.arch = nn_arch

        # Save hyperparameters
        self._lr = lr
        self._seed = seed
        self._epochs = epochs
        self._loss_func = loss_function
        self._batch_size = batch_size

        # Initialize the parameter dictionary for use in training
        self._param_dict = self._init_params()

        #parameter for whether or not fit() has been run
        self._model_fit = False

    def _init_params(self) -> Dict[str, ArrayLike]:
        """
        DO NOT MODIFY THIS METHOD! IT IS ALREADY COMPLETE!

        This method generates the parameter matrices for all layers of
        the neural network. This function returns the param_dict after
        initialization.

        Returns:
            param_dict: Dict[str, ArrayLike]
                Dictionary of parameters in neural network.
        """

        # Seed NumPy
        np.random.seed(self._seed)

        # Define parameter dictionary
        param_dict = {}

        # Initialize each layer's weight matrices (W) and bias matrices (b)
        for idx, layer in enumerate(self.arch):
            layer_idx = idx + 1
            input_dim = layer['input_dim']
            output_dim = layer['output_dim']
            param_dict['W' + str(layer_idx)] = np.random.randn(output_dim, input_dim) * 0.1
            param_dict['b' + str(layer_idx)] = np.random.randn(output_dim, 1) * 0.1

        return param_dict

    def _single_forward(
        self,
        W_curr: ArrayLike,
        b_curr: ArrayLike,
        A_prev: ArrayLike,
        activation: str
    ) -> Tuple[ArrayLike, ArrayLike]:
        """
        This method is used for a single forward pass on a single layer.

        Args:
            W_curr: ArrayLike
                Current layer weight matrix.
            b_curr: ArrayLike
                Current layer bias matrix.
            A_prev: ArrayLike
                Previous layer activation matrix.
            activation: str
                Name of activation function for current layer.

        Returns:
            A_curr: ArrayLike
                Current layer activation matrix.
            Z_curr: ArrayLike
                Current layer linear transformed matrix.
        """

        assert activation in ['sigmoid', 'relu'], "activation function {} not supported".format(activation)

        Z_curr = np.dot(A_prev, W_curr.T) + b_curr.T

        if activation == 'sigmoid':
            A_curr = self._sigmoid(Z_curr)
        elif activation == 'relu':
            A_curr = self._relu(Z_curr)

        return A_curr, Z_curr

    def forward(self, X: ArrayLike) -> Tuple[ArrayLike, Dict[str, ArrayLike]]:
        """
        This method is responsible for one forward pass of the entire neural network.

        Args:
            X: ArrayLike
                Input matrix with shape [batch_size, features].

        Returns:
            output: ArrayLike
                Output of forward pass.
            cache: Dict[str, ArrayLike]:
                Dictionary storing Z and A matrices from `_single_forward` for use in backprop.
        """
        cache = {}
        A_prev = X
        cache[0] = {'A_curr':A_prev}

        for i,layer in enumerate(self.arch):

            idx = i + 1
            activation = layer['activation']
            W_curr = self._param_dict['W' + str(idx)]
            b_curr = self._param_dict['b' + str(idx)]
        
            A_curr, Z_curr = self._single_forward(W_curr, b_curr, A_prev, activation)
            cache[idx] = {'A_curr': A_curr, 'Z_curr': Z_curr}

            # print('forward pass for layer', idx)
            # print(layer)
            # print('A curr: {}'.format(A_curr.shape))
            # print('Z curr: {}'.format(Z_curr.shape))
            # print('W curr: {}'.format(W_curr.shape))
            # print('b_curr: {}'.format(b_curr.shape))
            # print()
            A_prev = A_curr

        return A_curr, cache

    def _single_backprop(
        self,
        W_curr: ArrayLike,
        b_curr: ArrayLike,
        Z_curr: ArrayLike,
        A_prev: ArrayLike,
        dA_curr: ArrayLike,
        activation_curr: str
    ) -> Tuple[ArrayLike, ArrayLike, ArrayLike]:
        """
        This method is used for a single backprop pass on a single layer.

        Args:
            W_curr: ArrayLike
                Current layer weight matrix.
            b_curr: ArrayLike
                Current layer bias matrix.
            Z_curr: ArrayLike
                Current layer linear transform matrix.
            A_prev: ArrayLike
                Previous layer activation matrix.
            dA_curr: ArrayLike
                Partial derivative of loss function with respect to current layer activation matrix.
            activation_curr: str
                Name of activation function of layer.

        Returns:
            dA_prev: ArrayLike
                Partial derivative of loss function with respect to previous layer activation matrix.
            dW_curr: ArrayLike
                Partial derivative of loss function with respect to current layer weight matrix.
            db_curr: ArrayLike
                Partial derivative of loss function with respect to current layer bias matrix.
        """
        assert activation_curr in ['sigmoid', 'relu'], "activation function {} not supported".format(activation_curr)

        if activation_curr == 'sigmoid':
            dZ = self._sigmoid_backprop(dA_curr, Z_curr)
        elif activation_curr == 'relu':
            dZ = self._relu_backprop(dA_curr, Z_curr)

        # print('dZ:', dZ.shape)

        dA_prev = np.dot(dZ, W_curr)
        dW_curr = np.dot(dZ.T, A_prev) 
        db_curr = np.sum(dZ, axis=0) [:, None] #dimension issue

        return dA_prev, dW_curr, db_curr


    def backprop(self, y: ArrayLike, y_hat: ArrayLike, cache: Dict[str, ArrayLike]):
        """
        This method is responsible for the backprop of the whole fully connected neural network.

        Args:
            y (array-like):
                Ground truth labels.
            y_hat: ArrayLike
                Predicted output values.
            cache: Dict[str, ArrayLike]
                Dictionary containing the information about the
                most recent forward pass, specifically A and Z matrices.

        Returns:
            grad_dict: Dict[str, ArrayLike]
                Dictionary containing the gradient information from this pass of backprop.
        """
        grad_dict = {}

        #dA curr for last layer is measured by loss backprop
        if self._loss_func == 'mse':
            dA_curr = self._mean_squared_error_backprop(y, y_hat)
        elif self._loss_func == 'log loss' or self._loss_func == 'binary_cross_entropy':
            dA_curr = self._binary_cross_entropy_backprop(y, y_hat)

        for layer in range(1, len(self.arch)+1)[::-1]:
            
            A_prev, Z_curr = cache[layer-1]['A_curr'], cache[layer]['Z_curr']
            activation = self.arch[layer-1]['activation']
            W_curr = self._param_dict['W' + str(layer)]
            b_curr =  self._param_dict['b' + str(layer)]
            # print('backprop for layer', layer)
            # print('activation: {}'.format(activation))
            # print('A prev: {}'.format(A_prev.shape))
            # print('Z curr: {}'.format(Z_curr.shape))
            # print('W curr: {}'.format(W_curr.shape))
            # print('b_curr: {}'.format(b_curr.shape))
            # print('dA_curr: {}'.format(dA_curr.shape))

            dA_curr, dW_curr, db_curr = self._single_backprop(W_curr, b_curr, Z_curr, A_prev, dA_curr, activation)
            
            # print('dW_curr (just calculated): {}'.format( dW_curr.shape))
            # print('db_curr (just calculated): {}'.format( db_curr.shape))


            grad_dict[layer] = {'dA_prev': dA_curr, 'dW_curr': dW_curr, 'db_curr': db_curr}
            # print()

        self._model_fit = True

        return grad_dict

    def _update_params(self, grad_dict: Dict[str, ArrayLike]):
        """
        This function updates the parameters in the neural network after backprop. This function
        only modifies internal attributes and does not return anything

        Args:
            grad_dict: Dict[str, ArrayLike]
                Dictionary containing the gradient information from most recent round of backprop.
        """
        for layer in range(len(self.arch)):

            idx = str(layer + 1)
            info = grad_dict[int(idx)]
            # print('updating params for layer: {}'.format(idx))

            self._param_dict['W'+idx] = self._param_dict['W'+idx] - self._lr * info['dW_curr']
            self._param_dict['b'+idx] = self._param_dict['b'+idx] - self._lr * info['db_curr']

        return

    def fit(
        self,
        X_train: ArrayLike,
        y_train: ArrayLike,
        X_val: ArrayLike,
        y_val: ArrayLike
    ) -> Tuple[List[float], List[float]]:
        """
        This function trains the neural network by backpropagation for the number of epochs defined at
        the initialization of this class instance.

        Args:
            X_train: ArrayLike
                Input features of training set.
            y_train: ArrayLike
                Labels for training set.
            X_val: ArrayLike
                Input features of validation set.
            y_val: ArrayLike
                Labels for validation set.

        Returns:
            per_epoch_loss_train: List[float]
                List of per epoch loss for training set.
            per_epoch_loss_val: List[float]
                List of per epoch loss for validation set.
        """

        per_epoch_loss_train, per_epoch_loss_val = [], []

        #add extra (empty) axis if y is 1-D
        if len(y_train.shape) == 1:
            y_train = y_train[:, np.newaxis]
            y_val = y_val[:, np.newaxis]

        for epoch in range(self._epochs):

            # print('EPOCH {}'.format(epoch+1))
            ##TRAINING 
            batch_count =  np.ceil(X_train.shape[0] / self._batch_size)

            #shuffle and batch inputs
            order = np.arange(X_train.shape[0])
            np.random.shuffle(order)
            X_train_batched = np.array_split(X_train[order],batch_count)
            y_train_batched = np.array_split(y_train[order],batch_count)

            train_loss = []

            for X_train_batch, y_train_batch in zip(X_train_batched, y_train_batched):
                y_hat, cache = self.forward(X_train_batch)

                if self._loss_func == 'mse':
                    train_loss += [self._mean_squared_error(y_train_batch, y_hat)]
                elif self._loss_func == 'log loss' or self._loss_func == 'binary_cross_entropy':
                    train_loss += [self._binary_cross_entropy(y_train_batch, y_hat)]

                grad_dict = self.backprop(y_train_batch, y_hat, cache)
                self._update_params(grad_dict)

            per_epoch_loss_train += [np.mean(train_loss)]

            ##VALIDATION
            y_hat = self.predict(X_val)
            if self._loss_func == 'mse':
                val_loss = self._mean_squared_error(y_val, y_hat)
            elif self._loss_func == 'log loss' or self._loss_func == 'binary_cross_entropy':
                val_loss = self._binary_cross_entropy(y_val, y_hat)

            per_epoch_loss_val += [val_loss]

        return per_epoch_loss_train, per_epoch_loss_val

    def predict(self, X: ArrayLike) -> ArrayLike:
        """
        This function returns the prediction of the neural network.

        Args:
            X: ArrayLike
                Input data for prediction.

        Returns:
            y_hat: ArrayLike
                Prediction from the model.
        """

        if not self._model_fit:
            warnings.warn('Warning: Model has not been fit! Run .fit to fit model before predicting.')

        y_hat, cache = self.forward(X)
        return y_hat

    def _sigmoid(self, Z: ArrayLike) -> ArrayLike:
        """
        Sigmoid activation function.

        Args:
            Z: ArrayLike
                Output of layer linear transform.

        Returns:
            nl_transform: ArrayLike
                Activation function output.
        """
        #force float array to work with exp function
        Z = Z.astype(float)

        nl_transform = 1 / (1 + np.exp(-Z))
        return nl_transform

    def _sigmoid_backprop(self, dA: ArrayLike, Z: ArrayLike):
        """
        Sigmoid derivative for backprop.

        Args:
            dA: ArrayLike
                Partial derivative of previous layer activation matrix.
            Z: ArrayLike
                Output of layer linear transform.

        Returns:
            dZ: ArrayLike
                Partial derivative of current layer Z matrix.
        """
        #sigmoid derivative is given by f′(z)=f(z)(1−f(z)) where f(z) is sigmoid function
        #source: http://ufldl.stanford.edu/tutorial/supervised/MultiLayerNeuralNetworks/

        sigmoid_Z = self._sigmoid(Z)
        dZ = sigmoid_Z * (1 - sigmoid_Z)
        return dZ*dA

    def _relu(self, Z: ArrayLike) -> ArrayLike:
        """
        ReLU activation function.

        Args:
            Z: ArrayLike
                Output of layer linear transform.

        Returns:
            nl_transform: ArrayLike
                Activation function output.
        """

        nl_transform = np.maximum(0, Z)
        return nl_transform 

    def _relu_backprop(self, dA: ArrayLike, Z: ArrayLike) -> ArrayLike:
        """
        ReLU derivative for backprop.

        Args:
            dA: ArrayLike
                Partial derivative of previous layer activation matrix.
            Z: ArrayLike
                Output of layer linear transform.

        Returns:
            dZ: ArrayLike
                Partial derivative of current layer Z matrix.
        """

        #The rectified linear function has gradient 0 when z≤0 and 1 otherwise.
        #source: http://ufldl.stanford.edu/tutorial/supervised/MultiLayerNeuralNetworks/

        dZ =  np.where(Z<=0, 0, 1)
        return dZ*dA

    def _binary_cross_entropy(self, y: ArrayLike, y_hat: ArrayLike) -> float:
        """
        Binary cross entropy loss function.

        Args:
            y_hat: ArrayLike
                Predicted output.
            y: ArrayLike
                Ground truth output.

        Returns:
            loss: float
                Average loss over mini-batch.
        """
        #slightly adjust 0 or 1 values in y_hat to prevent undefined values
        y_hat[y_hat == 0] = 0.000000001
        y_hat[y_hat == 1] = 1 - 0.000000001

        loss = -np.mean(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))
        return loss


    def _binary_cross_entropy_backprop(self, y: ArrayLike, y_hat: ArrayLike) -> ArrayLike:
        """
        Binary cross entropy loss function derivative for backprop.

        Args:
            y_hat: ArrayLike
                Predicted output.
            y: ArrayLike
                Ground truth output.

        Returns:
            dA: ArrayLike
                partial derivative of loss with respect to A matrix.
        """
        #slightly adjust 0 or 1 values in y and y_hat to prevent undefined values
        y_hat[y_hat == 0] = 0.000000001
        y_hat[y_hat == 1] = 1 - 0.000000001

        #calculus! :(
        dA = ((1 - y)/(1 - y_hat) - (y/y_hat)) / y.shape[0]
        return dA

    def _mean_squared_error(self, y: ArrayLike, y_hat: ArrayLike) -> float:
        """
        Mean squared error loss.

        Args:
            y: ArrayLike
                Ground truth output.
            y_hat: ArrayLike
                Predicted output.

        Returns:
            loss: float
                Average loss of mini-batch.
        """
        loss = np.mean((y - y_hat)**2)
        return loss

    def _mean_squared_error_backprop(self, y: ArrayLike, y_hat: ArrayLike) -> ArrayLike:
        """
        Mean square error loss derivative for backprop.

        Args:
            y_hat: ArrayLike
                Predicted output.
            y: ArrayLike
                Ground truth output.

        Returns:
            dA: ArrayLike
                partial derivative of loss with respect to A matrix.
        """

        dA = 2 * (y_hat-y) /  y.shape[0]
        return dA