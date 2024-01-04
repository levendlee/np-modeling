# Activations.

import numpy as np

from layers import layer


class Activation(layer.Layer):
    pass


class ReLU(Activation):
    def forward(self, x: np.ndarray) -> np.ndarray:
        self._x = x
        return np.maximum(x, 0.0)

    def backward(self, dy: np.ndarray) -> np.ndarray:
        assert dy.shape == self._x.shape, f'{dy.shape} vs {self._x.shape}'
        return np.where(self._x >= 0.0, dy, 0.0)


class Softmax(Activation):
    def forward(self, x: np.ndarray) -> np.ndarray:
        self._x = x

        x_max = np.max(x, axis=-1, keepdims=True)
        exp_x = np.exp(x - x_max)
        exp_x_sum = np.sum(exp_x, axis=-1, keepdims=True)
        self._y = exp_x / exp_x_sum

        return self._y

    def backward(self, dy: np.ndarray, *args, **kwargs) -> np.ndarray:
        rank = len(self._y.shape)
        batch = self._y.shape[:-1]
        n = self._y.shape[-1]

        # https://towardsdatascience.com/derivative-of-the-softmax-function-and-the-categorical-cross-entropy-loss-ffceefc081d1
        # Jacobian
        # dy_i/dx_j = y_i(1{i=j} - y_j)
        # Broadcasting batch dimensions and last 2 dimensions as Jacobian.
        j = np.expand_dims(np.eye(n), axis=tuple(range(rank - 1)))
        j = j - np.expand_dims(self._y, axis=rank - 1)
        j = j * np.expand_dims(self._y, axis=rank)
        return np.einsum('...a,...ba->...b', dy, j)
