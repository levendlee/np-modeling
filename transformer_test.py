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
    assert hasattr(encoder._norm1, '_gamma')
    encoder._norm1._gamma = bound_flax_encoder.norm1.get_variable(
        'params', 'scale')
    assert hasattr(encoder._norm1, '_beta')
    encoder._norm1._beta = bound_flax_encoder.norm1.get_variable(
        'params', 'bias')
    assert hasattr(encoder._norm2, '_gamma')
    encoder._norm2._gamma = bound_flax_encoder.norm2.get_variable(
        'params', 'scale')
    assert hasattr(encoder._norm2, '_beta')
    encoder._norm2._beta = bound_flax_encoder.norm2.get_variable(
        'params', 'bias')


def _bind_decoder(decoder: transformer.TransformerDecoder, bound_flax_decoder):
    _bind_encoder(decoder, bound_flax_decoder)

    utils.bind_attention_variables_to_layer(
        decoder._cross_attention,
        *utils.read_attention_variables_from_flax(
            bound_flax_decoder.cross_attention.variables))

    assert hasattr(decoder._norm3, '_gamma')
    decoder._norm3._gamma = bound_flax_decoder.norm3.get_variable(
        'params', 'scale')
    assert hasattr(decoder._norm3, '_beta')
    decoder._norm3._beta = bound_flax_decoder.norm3.get_variable(
        'params', 'bias')


class TransformerEncoderTest(np.testing.TestCase):
    def assert_allclose(self, lhs, rhs, *, rtol=1e-3, atol=1e-3):
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
        self.assert_allclose(output,
                             flax_output,
                             atol=4e-3 if norm_first else 1e-3)


class TransformerDecoderTest(np.testing.TestCase):
    def assert_allclose(self, lhs, rhs, *, rtol=1e-3, atol=1e-3):
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
        self.assert_allclose(output,
                             flax_output,
                             atol=4e-3 if norm_first else 1e-3)
