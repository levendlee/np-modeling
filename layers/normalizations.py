# Normalization.

import numpy as np

import optimizer
from layers import layer


class DropOut(layer.Layer):
    def __init__(self, drop_prob: float, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._drop_prob = drop_prob

    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        if training and self._drop_prob != 0.0:
            keep_prob = 1 - self._drop_prob
            # Use the n=1 case of binominal distribution as bernoulli distribution.
            # https://en.wikipedia.org/wiki/Bernoulli_distribution
            # https://en.wikipedia.org/wiki/Binomial_distribution
            self._mask = np.random.binomial(n=1, p=keep_prob,
                                            size=x.size).reshape(x.shape)
            return np.where(self._mask, x / keep_prob, 0.0)
        return x

    def backward(self, dl_dy: np.ndarray, *args, **kwargs) -> np.ndarray:
        if self._drop_prob != 0.0:
            # Backward pass only run in training.
            keep_prob = 1 - self._drop_prob
            return np.where(self._mask, dl_dy / keep_prob, 0.0)
        return dl_dy


class LayerNormalization(layer.StatefulLayer):
    def __init__(self, epsilon: float = 1e-3, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._epsilon = epsilon

    def initialize(self, x: np.ndarray):
        self._col = x.shape[-1]
        self._gamma = self._initializer([self._col])
        self._beta = self._initializer([self._col])

    def forward(self, x: np.ndarray) -> np.ndarray:
        self._x = x
        self._mean = np.mean(x, axis=-1, keepdims=True)
        self._var = np.var(x, axis=-1, keepdims=True)
        self._y = (x - self._mean) / np.sqrt(self._var + self._epsilon)
        return self._gamma * self._y + self._beta

    def backward(self, dl_dz: np.ndarray,
                 optimizer_: optimizer.Optimizer) -> np.ndarray:
        rank = len(self._x.shape)
        batch_dims = tuple(range(rank - 1))

        dl_dbeta = np.sum(dl_dz, axis=batch_dims)
        dl_dgamma = np.sum(dl_dz * self._y, axis=batch_dims)
        dl_dy = dl_dz * self._gamma

        # Jacobian on last two dimensions, row as dx, col as dy.
        dmean_dx = np.array(1.0 / self._col)
        dvar_dx = 2.0 * (self._x - self._mean) / self._col

        f = self._x - self._mean
        g = self._var + self._epsilon
        # dy/dx = (\sigma + \epsilon) ^ {-1/2}(I - 1/N)(\sigma + \epsilon) -
        #         (\sigma + \epsilon) ^ {-3/2}((x_i - \mu) / N)(x_j - \mu)
        dy_dx = (np.expand_dims(g, rank)**-0.5 *
                 np.expand_dims(np.eye(self._col) - dmean_dx, batch_dims) -
                 0.5 * np.expand_dims(g, rank)**-1.5 *
                 np.expand_dims(dvar_dx, rank) * np.expand_dims(f, rank - 1))
        dl_dx = np.einsum('...a,...ab->...b', dl_dy, dy_dx)

        optimizer_.update(self, '_gamma', dl_dgamma)
        optimizer_.update(self, '_beta', dl_dbeta)
        return dl_dx
