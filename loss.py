# Loss

import abc

import numpy as np

from layers import layer


class Loss(layer.Layer):
    @abc.abstractmethod
    def forward(self, *args, **kwargs) -> float:
        pass

    @abc.abstractmethod
    def backward(self, *args, **kwargs) -> np.ndarray:
        pass


class MSELoss(Loss):
    def forward(self, y: np.ndarray, targets: np.ndarray) -> float:
        self._y = y
        self._targets = targets
        diff = y - targets
        return np.sum(diff**2) / y.size

    def backward(self, *args, **kwargs) -> np.ndarray:
        diff = self._y - self._targets
        return 2 * diff / self._y.size


class CrossEntropyLoss(Loss):
    def forward(self, y: np.ndarray, targets: np.ndarray) -> float:
        self._y = y
        self._targets = targets
        return -np.sum(targets * np.log(y))

    def backward(self, *args, **kwargs) -> np.ndarray:
        return -self._targets / self._y
