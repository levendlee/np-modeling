# Attention layer test

import unittest

import collections.abc
import flax
import jax
from jax import numpy as jnp
import numpy as np

import attention


def _rand(shape) -> np.ndarray:
    return np.random.normal(size=shape).astype(np.float32)


def _nested_printer(nested):
    if isinstance(nested, jnp.ndarray):
        return f'shape:{nested.shape}'
    if isinstance(nested, collections.abc.Mapping):
        return '{' + ', '.join(f'{k}:{_nested_printer(v)}'
                               for k, v in nested.items()) + '}'
    if isinstance(nested, collections.abc.Iterable):
        return '[' + ', '.join(_nested_printer(v) for v in nested) + ']'
    return f'{nested}'


class AttentionTest(unittest.TestCase):
    def assert_allclose(self, lhs, rhs, *, rtol=1e-6, atol=1e-6):
        return np.testing.assert_allclose(lhs, rhs, rtol=rtol, atol=atol)

    def test_forward(self):
        np.random.seed(0)
        
        batch = 16
        seq_len_q = 32
        seq_len_kv = 128
        num_heads = 8
        qkv_features = 128

        # Flax baseline
        flax_attention = flax.linen.attention.MultiHeadDotProductAttention(
            num_heads=num_heads, qkv_features=qkv_features)

        query = _rand([batch, seq_len_q, qkv_features])
        variables = flax_attention.init(jax.random.key(0), query)
        # {params:{
        #   query:{kernel:shape:(128, 8, 16), bias:shape:(8, 16)},
        #   key:{kernel:shape:(128, 8, 16), bias:shape:(8, 16)},
        #   value:{kernel:shape:(128, 8, 16), bias:shape:(8, 16)},
        #   out:{kernel:shape:(8, 16, 128), bias:shape:(128,)}}}
        # print(_nested_printer(variables))

        layer = attention.MultiHeadAttention(num_heads=num_heads)
        # Initialize layer
        output = layer(query=query)

        wq = np.transpose(variables['params']['query']['kernel'], [1, 2, 0])
        wk = np.transpose(variables['params']['key']['kernel'], [1, 2, 0])
        wv = np.transpose(variables['params']['value']['kernel'], [1, 2, 0])
        wo = np.transpose(variables['params']['out']['kernel'], [2, 0, 1]) 
        
        bq = variables['params']['query']['bias']
        bk = variables['params']['key']['bias']
        bv = variables['params']['value']['bias']
        bo = variables['params']['out']['bias']

        self.assertEqual(layer._wq.shape, wq.shape)
        self.assertEqual(layer._wk.shape, wk.shape)
        self.assertEqual(layer._wv.shape, wv.shape)
        self.assertEqual(layer._wo.shape, wo.shape)
        
        self.assertEqual(layer._bq.shape, bq.shape)
        self.assertEqual(layer._bk.shape, bk.shape)
        self.assertEqual(layer._bv.shape, bv.shape)
        self.assertEqual(layer._bo.shape, bo.shape)
        
        layer._wq = wq
        layer._wk = wk
        layer._wv = wv
        layer._wo = wo
        layer._bq = bq
        layer._bk = bk
        layer._bv = bv
        layer._bo = bo

        output = layer(query)
        flax_output = flax_attention.apply(variables, query)
       
        self.assert_allclose(output, flax_output)

