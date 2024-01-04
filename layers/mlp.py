# Dense layer.

from typing import Optional

import numpy as np

import optimizer
from layers import activations, layer


class Linear(layer.StatefulLayer):
    def __init__(self, units: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._output_units = units

    def initialize(self, x: np.ndarray) -> None:
        self._input_units = x.shape[-1]
        self._w = self._initializer([self._input_units, self._output_units])
        self._b = self._initializer([self._output_units])

    def forward(self, x: np.ndarray) -> np.ndarray:
        self._x = x
        y = np.matmul(self._x, self._w)
        y += self._b
        return y

    def backward(self, dy: np.ndarray,
                 optimizer_: optimizer.Optimizer) -> np.ndarray:
        # dy: [m, n]
        # b/db: [n]
        # w/dw: [k, n]
        # x/dx: [m, k]
        assert dy.shape == (self._x.shape[0], self._w.shape[1])
        db = np.sum(dy, axis=0)
        dw = np.matmul(np.transpose(self._x), dy)
        dx = np.matmul(dy, np.transpose(self._w))
        assert dx.shape == self._x.shape
        optimizer_.update(self, '_w', dw)
        optimizer_.update(self, '_b', db)
        return dx

    @property
    def w(self) -> np.ndarray:
        assert self._initialized
        return self._w

    @property
    def b(self) -> np.ndarray:
        assert self._initialized
        return self._b


class Dense(layer.StatefulLayer):
    """Dense w/ ReLU activation."""
    def __init__(self,
                 units: int,
                 activation: Optional[activations.Activation] = None,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self._linear = Linear(units=units)
        self._activation = activation or activations.ReLU()

    def initialize(self, x: np.ndarray) -> None:
        self._linear.initialize(x)
        self._linear._initialized = True
        self._activation.initialize()
        self._activation._initialized = True

    def forward(self, x: np.ndarray) -> np.ndarray:
        y = self._linear.forward(x)
        return self._activation.forward(y)

    def backward(self, dy: np.ndarray,
                 optimizer_: optimizer.Optimizer) -> np.ndarray:
        dy = self._activation.backward(dy)
        return self._linear.backward(dy, optimizer_)

    @property
    def linear(self) -> Linear:
        assert self._initialized
        return self._linear
