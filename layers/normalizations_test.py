# Normalization test.

import flax.linen as nn
import jax
import numpy as np
from jax import numpy as jnp

from layers import normalizations, utils


class DropOutTest(np.testing.TestCase):
    def assert_allclose(self, lhs, rhs, *, rtol=1e-6, atol=1e-6):
        return np.testing.assert_allclose(lhs, rhs, rtol=rtol, atol=atol)

    def test_forward_and_backward(self):
        shape = [128, 32]
        x = utils.rand(shape)

        drop_rate = 0.5
        keep_rate = 0.5
        dropout = normalizations.DropOut(drop_rate)
        y = dropout(x, training=True)

        def _dropout(x, mask):
            return jnp.where(mask, x / keep_rate, 0.0)

        dy = utils.rand(shape)
        jacobian = jax.jacobian(_dropout)(x, dropout._mask)
        self.assert_allclose(np.einsum('ab,abcd->cd', dy, jacobian),
                             dropout(dy, backprop=True))


class LayerNormTest(np.testing.TestCase):
    def assert_allclose(self, lhs, rhs, *, rtol=1e-6, atol=1e-6):
        return np.testing.assert_allclose(lhs, rhs, rtol=rtol, atol=atol)

    def test_forward_and_backward(self):
        np.random.seed(0)
        rng = jax.random.PRNGKey(0)

        batch = 32
        features = 128

        x = utils.rand([batch, features])
        targets = utils.rand([batch, features])

        # Flax baseline
        flax_norm = nn.LayerNorm()
        variables = flax_norm.init(rng, x)
        bound_flax_norm = flax_norm.bind(variables)

        norm = normalizations.LayerNormalization()
        norm(x)
        utils.bind_layer_norm(norm, bound_flax_norm)

        flax_z = bound_flax_norm(x)
        z = norm(x)
        self.assert_allclose(z, flax_z)

        @jax.jit
        def _jax_forward(variables, x, targets):
            z = flax_norm.apply(variables, x)
            return utils.mse_loss(z, targets)

        learning_rate = 0.001
        flax_grad_fn = jax.jit(jax.grad(_jax_forward, argnums=(0, 1)))
        # print(flax_grad_fn.lower(variables, x, targets).as_text())

        flax_grad = flax_grad_fn(variables, x, targets)
        dl_dz = jax.grad(utils.mse_loss)(z, targets)
        grad = norm(dl_dz, backprop=True, learning_rate=learning_rate)

        self.assert_allclose(
            norm._gamma, variables['params']['scale'] -
            learning_rate * flax_grad[0]['params']['scale'])
        self.assert_allclose(
            norm._beta, variables['params']['bias'] -
            learning_rate * flax_grad[0]['params']['bias'])

        mean_grad = jax.grad(
            jax.jit(lambda x: jnp.sum(jnp.mean(x, axis=-1, keepdims=True))))(x)
        var_grad = jax.grad(
            jax.jit(lambda x: jnp.sum(jnp.var(x, axis=-1, keepdims=True))))(x)

        self.assert_allclose(mean_grad, 1.0 / features)
        self.assert_allclose(
            var_grad,
            2.0 * (x - np.mean(x, axis=-1, keepdims=True)) / features)

        self.assert_allclose(grad, flax_grad[1])
