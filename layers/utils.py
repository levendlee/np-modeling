# Whatever

import unittest

import jax
import numpy as np
from jax import numpy as jnp

from layers import normalizations, transformer


class NNTestCase(unittest.TestCase):
    def assert_allclose(self, lhs, rhs, *, rtol=1e-5, atol=2e-5):
        return np.testing.assert_allclose(lhs, rhs, rtol=rtol, atol=atol)


def rand(shape) -> np.ndarray:
    return np.random.normal(size=shape).astype(np.float32)


@jax.jit
def mse_loss(y, targets):
    diff = y - targets
    return jnp.sum(diff**2) / y.size


def read_attention_variables_from_flax(variables):
    wq = np.transpose(variables['params']['query']['kernel'], [1, 2, 0])
    wk = np.transpose(variables['params']['key']['kernel'], [1, 2, 0])
    wv = np.transpose(variables['params']['value']['kernel'], [1, 2, 0])
    wo = np.transpose(variables['params']['out']['kernel'], [2, 0, 1])

    bq = variables['params']['query']['bias']
    bk = variables['params']['key']['bias']
    bv = variables['params']['value']['bias']
    bo = variables['params']['out']['bias']

    return wq, wk, wv, wo, bq, bk, bv, bo


def bind_attention_variables_to_layer(layer, wq, wk, wv, wo, bq, bk, bv, bo):
    assert layer._wq.shape == wq.shape
    assert layer._wk.shape == wk.shape
    assert layer._wv.shape == wv.shape
    assert layer._wo.shape == wo.shape

    assert layer._bq.shape == bq.shape
    assert layer._bk.shape == bk.shape
    assert layer._bv.shape == bv.shape
    assert layer._bo.shape == bo.shape

    layer._wq = wq
    layer._wk = wk
    layer._wv = wv
    layer._wo = wo
    layer._bq = bq
    layer._bk = bk
    layer._bv = bv
    layer._bo = bo


def bind_layer_norm(ln: normalizations.LayerNormalization, bound_flax_ln):
    assert hasattr(ln, '_gamma')
    ln._gamma = bound_flax_ln.get_variable('params', 'scale')
    assert hasattr(ln, '_beta')
    ln._beta = bound_flax_ln.get_variable('params', 'bias')
    assert hasattr(ln, '_epsilon')
    ln._epsilon = bound_flax_ln.epsilon


def bind_encoder(encoder: transformer.TransformerEncoder, bound_flax_encoder):
    bind_attention_variables_to_layer(
        encoder._self_attention,
        *read_attention_variables_from_flax(
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
    bind_layer_norm(encoder._norm1, bound_flax_encoder.norm1)
    bind_layer_norm(encoder._norm2, bound_flax_encoder.norm2)


def bind_decoder(decoder: transformer.TransformerDecoder, bound_flax_decoder):
    bind_encoder(decoder, bound_flax_decoder)

    bind_attention_variables_to_layer(
        decoder._cross_attention,
        *read_attention_variables_from_flax(
            bound_flax_decoder.cross_attention.variables))

    bind_layer_norm(decoder._norm3, bound_flax_decoder.norm3)
