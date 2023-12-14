# Layer.

import abc
from typing import Mapping, Optional, Sequence, Union

import numpy as np


class Layer(metaclass=abc.ABCMeta):
    def __init__(self, name: str = '', *args, **kwargs):
        self._name = name
        self._initialized = False

    def initialize(self, *args, **kwargs) -> None:
        pass

    @abc.abstractmethod
    def forward(self, *args, **kwargs) -> np.ndarray:
        pass

    @abc.abstractmethod
    def backward(self, *args, **kwargs) -> np.ndarray:
        pass

    def __call__(self, *args, backprop: bool = False, **kwargs) -> np.ndarray:
        if not self._initialized:
            self.initialize(*args, **kwargs)
            self._initialized = True

        if backprop:
            return self.backward(*args, **kwargs)
        else:
            return self.forward(*args, **kwargs)

    @property
    def name(self):
        return self._name


class Initializer(metaclass=abc.ABCMeta):
    def __call__(self, shape: Sequence[int]) -> np.ndarray:
        pass


class RandomInitializer(Initializer):
    def __call__(self, shape: Sequence[int]) -> np.ndarray:
        data = np.random.normal(size=shape).astype(np.float32)
        return np.minimum(np.maximum(data, -1.0), 1.0)


class StatefulLayer(Layer):
    def __init__(self,
                 initializer: Optional[Initializer] = None,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self._initializer = initializer or RandomInitializer()
