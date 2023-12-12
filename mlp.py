# Dense layer

import abc
import enum
from typing import Callable, Optional, Sequence, Type

import numpy as np

import layer


class Activation(layer.Layer):
    pass


class ReLU(Activation):
    def forward(self, x: np.ndarray) -> np.ndarray:
        self._x = x
        return np.maximum(x, 0.0)

    def backward(self, dy: np.ndarray) -> np.ndarray:
        assert dy.shape == self._x.shape, f'{dy.shape} vs {self._x.shape}'
        return np.where(self._x >= 0.0, dy, 0.0)


class Initializer(metaclass=abc.ABCMeta):
    def __call__(self, shape: Sequence[int]) -> np.ndarray:
        pass


class RandomInitializer(Initializer):
    def __call__(self, shape: Sequence[int]) -> np.ndarray:
        return np.random.normal(size=shape).astype(np.float32)


class Linear(layer.Layer):
    def __init__(self, units: int, initializer: Optional[Initializer] = None):
        super().__init__()
        self._output_units = units
        self._initializer = initializer or RandomeInitializer()

    def initialize(self, x: np.ndarray) -> None:
        self._input_units = x.shape[-1]
        self._w = self._initializer([self._input_units, self._output_units])
        self._b = self._initializer([self._output_units])
        self._initialized = True

    def forward(self, x: np.ndarray) -> np.ndarray:
        self._x = x
        y = np.matmul(self._x, self._w)
        y += self._b
        return y

    def backward(self, dy: np.ndarray, learning_rate: float) -> np.ndarray:
        # dy: [m, n]
        # b/db: [n]
        # w/dw: [k, n]
        # x/dx: [m, k]
        assert dy.shape == (self._x.shape[0], self._w.shape[1])
        db = np.sum(dy, axis=0)
        dw = np.matmul(np.transpose(self._x), dy)
        dx = np.matmul(dy, np.transpose(self._w))
        assert dx.shape == self._x.shape
        self._b -= learning_rate * db
        self._w -= learning_rate * dw
        return dx

    @property
    def w(self) -> np.ndarray:
        assert self._initialized
        return self._w

    @property
    def b(self) -> np.ndarray:
        assert self._initialized
        return self._b


class Dense(layer.Layer):
    """Dense w/ ReLU activation."""
    def __init__(self,
                 units: int,
                 activation: Optional[Activation] = None,
                 initializer: Optional[Initializer] = None,
                 name: str = ''):
        super().__init__(name)
        self._linear = Linear(units=units,
                              initializer=initializer or RandomInitializer())
        self._activation = activation or ReLU()

    def initialize(self, x: np.ndarray) -> None:
        self._linear.initialize(x)
        self._activation.initialize()
        self._initialized = True

    def forward(self, x: np.ndarray) -> np.ndarray:
        y = self._linear.forward(x)
        return self._activation.forward(y)

    def backward(self, dy: np.ndarray, learning_rate: float) -> np.ndarray:
        dy = self._activation.backward(dy)
        return self._linear.backward(dy, learning_rate)

    @property
    def linear(self) -> Linear:
        assert self._initialized
        return self._linear
