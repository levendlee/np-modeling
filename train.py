# Train

import logging
from typing import Optional, Sequence

import numpy as np

import loss
import optimizer
from layers import layer


class Trainer:
    def __init__(self,
                 layers: Sequence[layer.Layer],
                 loss_: Optional[loss.Loss] = None):
        self._layers = layers
        self._loss = loss_ or loss.MSELoss()

    def train(self, inputs: np.ndarray, targets: np.ndarray, steps: int,
              optimizer_: optimizer.Optimizer) -> None:

        for i in range(steps):
            print('Step: ', i)

            logging.info('Running forward pass')
            y = inputs
            for layer_ in self._layers:
                logging.info('Running Layer ', layer_.name)
                y = layer_(y)
            l = self._loss(y, targets)
            print('Loss: ', l)

            logging.info('Running backward pass')
            dy = self._loss(backprop=True)
            for layer_ in reversed(self._layers):
                logging.info('Running Layer ', layer_.name)
                dy = layer_(dy, backprop=True, optimizer_=optimizer_)
                logging.info(dy.shape)

    def eval(self, inputs: np.ndarray, targets: np.ndarray) -> None:
        y = inputs
        for layer_ in self._layers:
            y = layer_(y)
        l = self._loss(y, targets)
        print('Loss: ', l)
