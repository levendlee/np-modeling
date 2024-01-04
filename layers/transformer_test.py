# Transformer tests.

import flax.linen as nn
import jax
import numpy as np
from parameterized import parameterized

from layers import transformer, utils


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

        utils.bind_encoder(encoder, bound_flax_encoder)

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

        utils.bind_decoder(decoder, bound_flax_decoder)

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
