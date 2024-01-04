# Activation layers tests.

import flax
import jax
import numpy as np

from layers import activations, utils


class SotmaxTest(utils.NNTestCase):
    def test_forward_and_backward(self):
        np.random.seed(0)

        shape = [128, 128]
        x = utils.rand(shape)
        targets = utils.rand(shape)

        @jax.jit
        def flax_forward(x, targets):
            y = flax.linen.softmax(x)
            return utils.mse_loss(y, targets)

        flax_grad = jax.jit(jax.grad(flax_forward))
        # print(flax_grad.lower(x, targets).as_text())
        flax_dx = flax_grad(x, targets)

        softmax = activations.Softmax()
        y = softmax(x)
        dy = jax.grad(utils.mse_loss)(y, targets)
        dx = softmax(dy, backprop=True)

        self.assert_allclose(flax_dx, dx, rtol=1e-5, atol=1e-5)