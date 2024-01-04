# Trainer Test

import unittest

import numpy as np
from parameterized import parameterized

import optimizer
import train
from layers import conv, mlp


class TrainTest(unittest.TestCase):
    @parameterized.expand([['None'], ['Adam']])
    def test_train_mlp(self, optimizer_name: str):
        optimizer_cls = (optimizer.AdamOptimizer if optimizer_name == 'Adam'
                         else optimizer.SGDOptimizer)

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
        print(f'\nTraining MLP with Optimizer {optimizer_name}:')
        trainer.train(inputs=x,
                      targets=targets,
                      steps=10,
                      optimizer_=optimizer_cls(1e-4))
        print('Eval:')
        trainer.eval(inputs=x, targets=targets)
        # Additional eval run won't change loss
        trainer.eval(inputs=x, targets=targets)

    def test_train_conv(self):
        np.random.seed(0)

        batch_size = 16
        height = 32
        width = 32
        input_features = 16

        kernel_sizes = [1, 3, 5, 3, 1]
        channels = [16, 32, 64, 32, 16]

        layers = []
        for i, (c, k) in enumerate(zip(channels, kernel_sizes)):
            layers.append(
                conv.Conv2D(channels=c, kernel_size=k, name=f'layer_{i}'))

        x = np.random.uniform(-1.0,
                              1.0,
                              size=[batch_size, height, width,
                                    input_features]).astype(np.float32)
        targets = np.random.uniform(
            0.0, 1.0, size=[batch_size, height, width,
                            channels[-1]]).astype(np.float32)

        trainer = train.Trainer(layers)
        print('\nTraining Conv:')
        trainer.train(inputs=x,
                      targets=targets,
                      steps=10,
                      optimizer_=optimizer.SGDOptimizer(1e-6))
        print('Eval:')
        trainer.eval(inputs=x, targets=targets)
        # Additional eval run won't change loss
        trainer.eval(inputs=x, targets=targets)
