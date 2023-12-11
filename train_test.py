# Trainer Test

import unittest

import jax
from jax import numpy as jnp
import numpy as np

import mlp
import train


def _rand(shape) -> np.ndarray:
    return np.random.normal(size=shape).astype(np.float32)


class TrainTest(unittest.TestCase):
    def test_forward_and_backward_execute(self):
        np.random.seed(0)

        batch_size = 128
        input_features = 16

        features = [16, 32, 64, 32, 16]

        layers = []
        for i, f in enumerate(features):
            layers.append(mlp.Dense(units=f, name=f'layer_{i}'))

        x = np.random.uniform(0.0, 1.0,
                              size=[batch_size,
                                    input_features]).astype(np.float32)
        targets = np.random.uniform(0.0, 1.0,
                                    size=[batch_size,
                                          features[-1]]).astype(np.float32)

        trainer = train.Trainer(layers)
        print('Training:')
        # Dataset is so small that learning_rate=1e-4 will converge in one step
        trainer.train(inputs=x, targets=targets, steps=10, learning_rate=1e-5)
        print('Eval:')
        trainer.eval(inputs=x, targets=targets)
        # Additional eval run won't change loss
        trainer.eval(inputs=x, targets=targets)
