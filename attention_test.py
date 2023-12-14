# Attention layer test

import copy
import unittest

import collections.abc
import flax
import jax
from jax import numpy as jnp
import numpy as np

import attention


def _rand(shape) -> np.ndarray:
    return np.random.normal(size=shape).astype(np.float32)


@jax.jit
def _mse_loss(y, targets):
    diff = y - targets
    return jnp.sum(diff**2) / y.size


def _nested_printer(nested):
    if isinstance(nested, jnp.ndarray):
        return f'shape:{nested.shape}'
    if isinstance(nested, collections.abc.Mapping):
        return '{' + ', '.join(f'{k}:{_nested_printer(v)}'
                               for k, v in nested.items()) + '}'
    if isinstance(nested, collections.abc.Iterable):
        return '[' + ', '.join(_nested_printer(v) for v in nested) + ']'
    return f'{nested}'


class NNTestCase(unittest.TestCase):
    def assert_allclose(self, lhs, rhs, *, rtol=1e-5, atol=2e-5):
        return np.testing.assert_allclose(lhs, rhs, rtol=rtol, atol=atol)


class SotmaxTest(NNTestCase):
    def test_forward_and_backward(self):
        np.random.seed(0)

        shape = [128, 128]
        x = _rand(shape)
        targets = _rand(shape)

        @jax.jit
        def flax_forward(x, targets):
            y = flax.linen.softmax(x)
            return _mse_loss(y, targets)

        flax_grad = jax.jit(jax.grad(flax_forward))
        # print(flax_grad.lower(x, targets).as_text())
        flax_dx = flax_grad(x, targets)

        softmax = attention.Softmax()
        y = softmax.forward(x)
        dy = jax.grad(_mse_loss)(y, targets)
        dx = softmax.backward(dy)

        self.assert_allclose(flax_dx, dx, rtol=1e-5, atol=1e-5)


class AttentionTest(NNTestCase):
    def test_forward_and_backward(self):
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
        # print(jax.tree_util.tree_map(lambda x : x.shape, variables))
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

        targets = _rand([batch, seq_len_q, qkv_features])

        @jax.jit
        def flax_forward(variables, query, targets):
            y = flax_attention.apply(variables, query)
            return _mse_loss(y, targets)

        flax_backward = jax.jit(jax.grad(flax_forward, argnums=(0, 1)))
        # print(flax_backward.lower(variables, query, targets).as_text())

        flax_grad = flax_backward(variables, query, targets)
        # print(jax.tree_util.tree_map(lambda x : x.shape, flax_output))

        flax_dwq = np.transpose(flax_grad[0]['params']['query']['kernel'],
                                [1, 2, 0])
        flax_dwk = np.transpose(flax_grad[0]['params']['key']['kernel'],
                                [1, 2, 0])
        flax_dwv = np.transpose(flax_grad[0]['params']['value']['kernel'],
                                [1, 2, 0])
        flax_dwo = np.transpose(flax_grad[0]['params']['out']['kernel'],
                                [2, 0, 1])

        flax_dbq = flax_grad[0]['params']['query']['bias']
        flax_dbk = flax_grad[0]['params']['key']['bias']
        flax_dbv = flax_grad[0]['params']['value']['bias']
        flax_dbo = flax_grad[0]['params']['out']['bias']

        flax_dquery = flax_grad[1]

        learning_rate = 0.01

        dy = jax.grad(_mse_loss)(output, targets)

        # Makes a deepcopy as the backward path is going to update weights.
        layer = copy.deepcopy(layer)
        dquery, dkey, dvalue = layer.backward(dy, learning_rate)

        self.assert_allclose(flax_dquery, dquery + dkey + dvalue)
        self.assert_allclose(layer._wo, -learning_rate * flax_dwo + wo)
        self.assert_allclose(layer._bo, -learning_rate * flax_dbo + bo)
        self.assert_allclose(layer._wv, -learning_rate * flax_dwv + wv)
        self.assert_allclose(layer._bv, -learning_rate * flax_dbv + bv)
        self.assert_allclose(layer._wk, -learning_rate * flax_dwk + wk)
        self.assert_allclose(layer._bk, -learning_rate * flax_dbk + bk)
        self.assert_allclose(layer._wq, -learning_rate * flax_dwq + wq)
        self.assert_allclose(layer._bq, -learning_rate * flax_dbq + bq)
