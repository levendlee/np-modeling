# Layer.

import abc
from typing import Mapping, Union

import numpy as np


class Layer(metaclass=abc.ABCMeta):
    def __init__(self, *args, **kwargs):
        self._initialized = False

    def initialize(self, *args, **kwargs) -> None:
        self._initialized = True

    @abc.abstractmethod
    def forward(self, *args, **kwargs) -> np.ndarray:
        pass

    @abc.abstractmethod
    def backward(self, *args, **kwargs) -> np.ndarray:
        pass

    def __call__(self, *args, backprop: bool = False, **kwargs) -> np.ndarray:
        if not self._initialized:
            self.initialize(*args, **kwargs)

        if backprop:
            return self.backward(*args, **kwargs)
        else:
            return self.forward(*args, **kwargs)
