# Transformer Test.

import collections.abc
import copy
import unittest

import flax.linen as nn
import jax
from jax import numpy as jnp
import numpy as np
from parameterized import parameterized

import transformer
import utils


# It is kind of testing my own implementation against my own implementation. :)
class _TransformerEncoder(nn.Module):
    qkv_features: int
    num_heads: int
    hidden_dim: int
    norm_first: bool

    def setup(self):
        self.self_attention = nn.attention.MultiHeadDotProductAttention(
            num_heads=self.num_heads, qkv_features=self.qkv_features)
        self.dense1 = nn.Dense(features=self.hidden_dim)
        self.dense2 = nn.Dense(features=self.qkv_features)
        self.norm1 = nn.LayerNorm()
        self.norm2 = nn.LayerNorm()

    def __call__(self, qkv: jax.Array) -> jax.Array:
        skip = qkv
        if self.norm_first:
            qkv = self.norm1(qkv)
        out = self.self_attention(qkv)
        out += skip
        if not self.norm_first:
            out = self.norm1(out)

        skip = out
        if self.norm_first:
            out = self.norm2(out)
        out = self.dense1(out)
        out = nn.relu(out)
        out = self.dense2(out)
        out += skip
        if not self.norm_first:
            out = self.norm2(out)

        return out


class _TransformerDecoder(nn.Module):
    qkv_features: int
    num_heads: int
    hidden_dim: int
    norm_first: bool

    def setup(self):
        self.self_attention = nn.attention.MultiHeadDotProductAttention(
            num_heads=self.num_heads, qkv_features=self.qkv_features)
        self.cross_attention = nn.attention.MultiHeadDotProductAttention(
            num_heads=self.num_heads, qkv_features=self.qkv_features)
        self.dense1 = nn.Dense(features=self.hidden_dim)
        self.dense2 = nn.Dense(features=self.qkv_features)
        self.norm1 = nn.LayerNorm()
        self.norm2 = nn.LayerNorm()
        self.norm3 = nn.LayerNorm()

    def __call__(self, q: jax.Array, kv: jax.Array) -> jax.Array:
        skip = q

        if self.norm_first:
            q = self.norm1(q)
        out = self.self_attention(q)
        out += skip
        if not self.norm_first:
            out = self.norm1(out)

        skip = out

        if self.norm_first:
            out = self.norm2(out)
        out = self.cross_attention(out, kv)
        out += skip
        if not self.norm_first:
            out = self.norm2(out)

        skip = out

        if self.norm_first:
            out = self.norm3(out)
        out = self.dense1(out)
        out = nn.relu(out)
        out = self.dense2(out)
        out += skip
        if not self.norm_first:
            out = self.norm3(out)

        return out


def _bind_layer_norm(ln: transformer.LayerNormalization, bound_flax_ln):
    assert hasattr(ln, '_gamma')
    ln._gamma = bound_flax_ln.get_variable('params', 'scale')
    assert hasattr(ln, '_beta')
    ln._beta = bound_flax_ln.get_variable('params', 'bias')
    assert hasattr(ln, '_epsilon')
    ln._epsilon = bound_flax_ln.epsilon


def _bind_encoder(encoder: transformer.TransformerEncoder, bound_flax_encoder):
    utils.bind_attention_variables_to_layer(
        encoder._self_attention,
        *utils.read_attention_variables_from_flax(
            bound_flax_encoder.self_attention.variables))

    assert hasattr(encoder._dense1._linear, '_w')
    encoder._dense1._linear._w = bound_flax_encoder.dense1.get_variable(
        'params', 'kernel')
    assert hasattr(encoder._dense1._linear, '_b')
    encoder._dense1._linear._b = bound_flax_encoder.dense1.get_variable(
        'params', 'bias')
    assert hasattr(encoder._dense2, '_w')
    encoder._dense2._w = bound_flax_encoder.dense2.get_variable(
        'params', 'kernel')
    assert hasattr(encoder._dense2, '_b')
    encoder._dense2._b = bound_flax_encoder.dense2.get_variable(
        'params', 'bias')
    _bind_layer_norm(encoder._norm1, bound_flax_encoder.norm1)
    _bind_layer_norm(encoder._norm2, bound_flax_encoder.norm2)


def _bind_decoder(decoder: transformer.TransformerDecoder, bound_flax_decoder):
    _bind_encoder(decoder, bound_flax_decoder)

    utils.bind_attention_variables_to_layer(
        decoder._cross_attention,
        *utils.read_attention_variables_from_flax(
            bound_flax_decoder.cross_attention.variables))

    _bind_layer_norm(decoder._norm3, bound_flax_decoder.norm3)


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

        norm = transformer.LayerNormalization()
        norm(x)
        _bind_layer_norm(norm, bound_flax_norm)

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


class TransformerEncoderTest(np.testing.TestCase):
    def assert_allclose(self, lhs, rhs, *, rtol=1e-5, atol=1e-5):
        return np.testing.assert_allclose(lhs, rhs, rtol=rtol, atol=atol)

    @parameterized.expand([['norm_first', True], ['norm_last', False]])
    def test_forward_and_backward(self, name, norm_first):
        np.random.seed(0)
        rng = jax.random.PRNGKey(0)

        batch = 16
        seq_len_q = 32
        seq_len_kv = 128
        num_heads = 8
        qkv_features = 128
        hidden_dim = 256

        # Flax baseline
        flax_encoder = _TransformerEncoder(qkv_features=qkv_features,
                                           num_heads=num_heads,
                                           hidden_dim=hidden_dim,
                                           norm_first=norm_first)
        qkv = utils.rand([batch, seq_len_q, qkv_features])
        targets = utils.rand([batch, seq_len_q, qkv_features])
        variables = flax_encoder.init(rng, qkv)
        bound_flax_encoder = flax_encoder.bind(variables)

        # print(jax.tree_map(lambda x: x.shape, variables))
        # print(
        #    jax.tree_map(
        #        lambda x: x.shape,
        #        bound_flax_encoder.self_attention.variables))

        flax_output = bound_flax_encoder(qkv)
        self.assertEqual(flax_output.shape, (batch, seq_len_q, qkv_features))

        # Initialize
        encoder = transformer.TransformerEncoder(num_heads=num_heads,
                                                 hidden_units=hidden_dim,
                                                 norm_first=norm_first)
        output = encoder(qkv)
        self.assertEqual(output.shape, (batch, seq_len_q, qkv_features))

        _bind_encoder(encoder, bound_flax_encoder)

        output = encoder(qkv)
        self.assert_allclose(output, flax_output)

        @jax.jit
        def _jax_forward(variables, x, targets):
            z = flax_encoder.apply(variables, x)
            return utils.mse_loss(z, targets)

        flax_grad_fn = jax.jit(jax.grad(_jax_forward, argnums=(0, 1)))
        flax_grad = flax_grad_fn(variables, qkv, targets)

        learning_rate = 0.001
        dl_dz = jax.grad(utils.mse_loss)(output, targets)
        grad = encoder(dl_dz, backprop=True, learning_rate=learning_rate)
        self.assert_allclose(grad, flax_grad[1])


class TransformerDecoderTest(np.testing.TestCase):
    def assert_allclose(self, lhs, rhs, *, rtol=1e-5, atol=1e-5):
        return np.testing.assert_allclose(lhs, rhs, rtol=rtol, atol=atol)

    @parameterized.expand([['norm_first', True], ['norm_last', False]])
    def test_forward_and_backward(self, name, norm_first):
        np.random.seed(0)
        rng = jax.random.PRNGKey(0)

        batch = 16
        seq_len_q = 32
        seq_len_kv = 128
        num_heads = 8
        qkv_features = 128
        hidden_dim = 256

        # Flax baseline
        flax_decoder = _TransformerDecoder(qkv_features=qkv_features,
                                           num_heads=num_heads,
                                           hidden_dim=hidden_dim,
                                           norm_first=norm_first)
        q = utils.rand([batch, seq_len_q, qkv_features])
        kv = utils.rand([batch, seq_len_kv, qkv_features])
        targets = utils.rand([batch, seq_len_q, qkv_features])
        variables = flax_decoder.init(rng, q, kv)
        bound_flax_decoder = flax_decoder.bind(variables)

        # print(jax.tree_map(lambda x: x.shape, variables))
        # print(
        #    jax.tree_map(
        #        lambda x: x.shape,
        #        bound_flax_decoder.self_attention.variables))

        flax_output = bound_flax_decoder(q, kv)
        self.assertEqual(flax_output.shape, (batch, seq_len_q, qkv_features))

        # Initialize
        decoder = transformer.TransformerDecoder(num_heads=num_heads,
                                                 hidden_units=hidden_dim,
                                                 norm_first=norm_first)
        output = decoder(q, kv)
        self.assertEqual(output.shape, (batch, seq_len_q, qkv_features))

        _bind_decoder(decoder, bound_flax_decoder)

        output = decoder(q, kv)
        self.assert_allclose(output, flax_output)

        @jax.jit
        def _jax_forward(variables, q, kv, targets):
            z = flax_decoder.apply(variables, q, kv)
            return utils.mse_loss(z, targets)

        flax_grad_fn = jax.jit(jax.grad(_jax_forward, argnums=(0, 1, 2)))
        flax_grad = flax_grad_fn(variables, q, kv, targets)

        learning_rate = 0.001
        dl_dz = jax.grad(utils.mse_loss)(output, targets)
        grad = decoder(dl_dz, backprop=True, learning_rate=learning_rate)
        self.assert_allclose(grad[0], flax_grad[1])
        self.assert_allclose(grad[1], flax_grad[2])
