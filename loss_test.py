# Normalization test.

import flax
import jax
import numpy as np
import optax
from jax import numpy as jnp

import loss
from layers import Softmax, utils

np.random.seed(2024)


class MSELossTest(np.testing.TestCase):
    def test_forward_and_backward(self):
        shape = [128, 32]
        y = utils.rand(shape=shape)
        targets = utils.rand(shape=shape)

        def _mse(y, targets):
            return jnp.sum((y - targets)**2) / y.size

        mse_loss = loss.MSELoss()
        np.testing.assert_allclose(_mse(y, targets),
                                   mse_loss(y, targets),
                                   atol=1e-6)
        np.testing.assert_allclose(
            jax.grad(_mse)(y, targets), mse_loss(y, targets, backprop=True))


class CrossEntropyLossTest(np.testing.TestCase):
    def test_forward_and_backward(self):
        shape = [128, 32]
        y = flax.linen.softmax(utils.rand(shape=shape))
        targets = flax.linen.softmax(utils.rand(shape=shape))

        def _cross_entropy(y, targets):
            return -jnp.sum(targets * jnp.log(y))

        ce_loss = loss.CrossEntropyLoss()
        np.testing.assert_allclose(_cross_entropy(y, targets),
                                   ce_loss(y, targets))
        np.testing.assert_allclose(
            jax.grad(_cross_entropy)(y, targets),
            ce_loss(y, targets, backprop=True))


class SoftmaxCrossEntropyLossTest(np.testing.TestCase):
    def test_forward_and_backward(self):
        shape = [128, 32]
        y = utils.rand(shape=shape)
        targets = flax.linen.softmax(utils.rand(shape=shape))

        def _softmax_cross_entropy(y, targets):
            return jnp.sum(optax.softmax_cross_entropy(y, targets))

        ce_loss = loss.CrossEntropyLoss()
        softmax = Softmax()
        np.testing.assert_allclose(_softmax_cross_entropy(y, targets),
                                   ce_loss(softmax(y), targets))
        np.testing.assert_allclose(jax.grad(_softmax_cross_entropy)(y,
                                                                    targets),
                                   softmax(ce_loss(y, targets, backprop=True),
                                           backprop=True),
                                   atol=1e-6)
