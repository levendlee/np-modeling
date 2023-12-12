# MLP layer test

import unittest

import jax
from jax import numpy as jnp
import numpy as np

import mlp


def _rand(shape) -> np.ndarray:
    return np.random.normal(size=shape).astype(np.float32)


class MLPTest(unittest.TestCase):
    def assert_allclose(self, lhs, rhs, rtol=1e-6, atol=1e-6):
        return np.testing.assert_allclose(lhs, rhs, rtol=rtol, atol=atol)

    def test_forward_and_backward_execute(self):
        batch_size = 64
        input_features = 32
        output_features = 16

        layer = mlp.Dense(units=output_features)
        x = _rand([batch_size, input_features])
        targets = _rand([batch_size, output_features])

        # 1.1 Forward
        y = layer(x)
        self.assertEqual(y.shape, (batch_size, output_features))

        # 3.1 Backward
        learning_rate = 0.01
        dy = _rand([batch_size, output_features])
        dx = layer(dy, backprop=True, learning_rate=learning_rate)
        self.assertEqual(dx.shape, (batch_size, input_features))

    def test_forward_and_backward_numerics(self):
        batch_size = 64
        input_features = 32
        output_features = 16

        layer = mlp.Dense(units=output_features)
        x = _rand([batch_size, input_features])
        targets = _rand([batch_size, output_features])

        # 1.1 Forward
        # 1.2 To test
        y = layer(x)
        self.assertEqual(y.shape, (batch_size, output_features))

        # 1.3 JAX baseline
        w = layer.linear.w
        b = layer.linear.b

        @jax.jit
        def _jax_forward(x, w, b):
            return jnp.maximum(jnp.dot(x, w) + b, 0.0)

        self.assert_allclose(y, _jax_forward(x, w, b))

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
        def _jax_loss(x, w, b, targets):
            y = _jax_forward(x, w, b)
            return _mse_loss(y, targets)

        grad_fn = jax.grad(_jax_loss, argnums=(0, 1, 2))
        # print(grad_fn.lower(x, w, b, targets).as_text())
        grads = grad_fn(x, w, b, targets)

        jax_dx, jax_dw, jax_db = grads
        jax_w = w - learning_rate * jax_dw
        jax_b = b - learning_rate * jax_db

        # 3.3 To test
        # Updates `w` and `b` in place.
        dy = np.broadcast_to(dy, [batch_size, output_features])
        dx = layer(dy, backprop=True, learning_rate=learning_rate)
        self.assertEqual(dx.shape, (batch_size, input_features))

        # TODO: This number looks not reliable
        self.assert_allclose(dx, jax_dx, atol=1e-6, rtol=1e-6)
        self.assert_allclose(w, jax_w, atol=1e-6, rtol=1e-6)
        self.assert_allclose(b, jax_b, atol=1e-6, rtol=1e-6)
