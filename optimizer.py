# Optimizer.

# Uses operator overloading to have optimizers act as drop-in replacement of
# learning rate scalars.

import abc
from typing import Mapping, Optional, Sequence, Union

import numpy as np


class Optimizer(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def update(self, obj: object, attribute: str,
               gradient: np.ndarray) -> np.ndarray:
        pass


class DefaultOptimizer(Optimizer):
    def __init__(self, learning_rate: float) -> float:
        self._learning_rate = learning_rate

    def update(self, obj: object, attribute: str,
               gradient: np.ndarray) -> np.ndarray:
        variable = getattr(obj, attribute)
        variable -= self._learning_rate * gradient
        setattr(obj, attribute, variable)


class AdamOptimizer(Optimizer):
    pass
