# Optimizer.

# Uses operator overloading to have optimizers act as drop-in replacement of
# learning rate scalars.

import abc
import dataclasses
from typing import Mapping, Optional, Sequence, Union

import numpy as np


class Optimizer(metaclass=abc.ABCMeta):
    def update(self, obj: object, attribute: str,
               gradient: np.ndarray) -> None:
        identifier = f'{id(obj)}.{attribute}'
        variable = getattr(obj, attribute)
        variable = self.update_variable(identifier, variable, gradient)
        setattr(obj, attribute, variable)

    @abc.abstractmethod
    def update_variable(self, identifier: str, variable: np.ndarray,
                        gradient: np.ndarray) -> np.ndarray:
        pass


class DefaultOptimizer(Optimizer):
    def __init__(self, learning_rate: float) -> float:
        self._learning_rate = learning_rate

    def update_variable(self, identifier: str, variable: np.ndarray,
                        gradient: np.ndarray) -> np.ndarray:
        variable -= self._learning_rate * gradient
        return variable


@dataclasses.dataclass
class AdamOptimizerConfig:
    learning_rate: float
    beta1: float = 0.9
    beta2: float = 0.999
    epsilon: float = 1e-7

    def __post_init__(self, *args, **kwargs):
        self._steps = {}
        self._momentums = {}
        self._velocities = {}


class AdamOptimizer(AdamOptimizerConfig, Optimizer):
    def update_variable(self, identifier: str, variable: np.ndarray,
                        gradient: np.ndarray) -> None:

        t = self._steps.get(identifier, 1)
        # print(f'Updating {identifier} at step {t}')
        m = self._momentums.get(identifier, np.zeros(gradient.shape))
        v = self._velocities.get(identifier, np.zeros(gradient.shape))

        new_m = self.beta1 * m + (1 - self.beta1) * gradient
        new_n = self.beta2 * v + (1 - self.beta2) * gradient**2
        corrected_m = new_m / (1 - self.beta1**t)
        corrected_n = new_n / (1 - self.beta2**t)
        variable -= self.learning_rate * (corrected_m /
                                          np.sqrt(corrected_n + self.epsilon))

        self._steps[identifier] = t + 1
        self._momentums[identifier] = new_m
        self._velocities[identifier] = new_n

        return variable
