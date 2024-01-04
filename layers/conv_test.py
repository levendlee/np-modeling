# Conv layer test.

import unittest

import jax
import numpy as np
from jax import numpy as jnp

from layers import conv, utils


class ConvTest(unittest.TestCase):
    def assert_allclose(self, lhs, rhs, *, rtol=1e-6, atol=1e-6):
        return np.testing.assert_allclose(lhs, rhs, rtol=rtol, atol=atol)

    def test_forward_and_backward_execute(self):
        batch_size = 64
        h = 32
        w = 16
        kernel_size = 3
        input_features = 32
        output_features = 16

        layer = conv.Conv2D(channels=output_features, kernel_size=kernel_size)
        x = utils.rand([batch_size, h, w, input_features])

        # 1.1 Forward
        y = layer(x)
        self.assertEqual(y.shape, (batch_size, h, w, output_features))

        # 3.1 Backward
        learning_rate = 0.01
        dy = utils.rand([batch_size, h, w, output_features])
        dx = layer(dy, backprop=True, learning_rate=learning_rate)
        self.assertEqual(dx.shape, (batch_size, h, w, input_features))

    def test_forward_and_backward_numerics(self):
        np.random.seed(0)

        batch_size = 64
        h = 32
        w = 16
        kernel_size = 3
        input_features = 32
        output_features = 16

        layer = conv.Conv2D(channels=output_features, kernel_size=kernel_size)
        x = utils.rand([batch_size, h, w, input_features])
        targets = utils.rand([batch_size, h, w, output_features])

        # 1.1 Forward
        # 1.2 To test
        y = layer(x)
        self.assertEqual(y.shape, (batch_size, h, w, output_features))

        # 1.3 JAX baseline
        filters = layer.w
        bias = layer.b

        @jax.jit
        def _jax_forward(x, filters, b):
            # Default for jax.lax.conv: NCHW @ OIHW -> NCHW
            y = jax.lax.conv(lhs=jnp.transpose(x, [0, 3, 1, 2]),
                             rhs=jnp.transpose(filters, [3, 2, 0, 1]),
                             window_strides=(1, 1),
                             padding='SAME')
            y = jnp.transpose(y, [0, 2, 3, 1])
            y += b
            return jnp.maximum(y, 0.0)

        self.assert_allclose(y, _jax_forward(x, filters, bias), atol=3e-5)

        # 2. Loss
        @jax.jit
        def _mse_loss(y, targets):
            diff = y - targets
            return jnp.sum(diff**2) / y.size

        # 3.1 Backward
        learning_rate = 0.01

        # 3.2 JAX baseline
        loss_grad_fn = jax.jit(jax.grad(_mse_loss))
        # print(loss_grad_fn.lower(y, targets).as_text())
        dy = loss_grad_fn(y, targets)

        @jax.jit
        def _jax_loss(x, filters, bias, targets):
            y = _jax_forward(x, filters, bias)
            return _mse_loss(y, targets)

        grad_fn = jax.jit(jax.grad(_jax_loss, argnums=(0, 1, 2)))
        # print(grad_fn.lower(x, filters, bias, targets).as_text())
        grads = grad_fn(x, filters, bias, targets)

        jax_dx, jax_dfilters, jax_dbias = grads
        jax_filters = filters - learning_rate * jax_dfilters
        jax_bias = bias - learning_rate * jax_dbias

        # 3.3 To test
        # Updates `w` and `b` in place.
        dx = layer(dy, backprop=True, learning_rate=learning_rate)
        self.assertEqual(dx.shape, (batch_size, h, w, input_features))

        self.assert_allclose(dx, jax_dx)
        self.assert_allclose(filters, jax_filters)
        self.assert_allclose(bias, jax_bias)
