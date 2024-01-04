# Attention layer test.

import copy

import flax
import jax
import numpy as np

from layers import attentions, utils


class AttentionTest(utils.NNTestCase):
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

        query = utils.rand([batch, seq_len_q, qkv_features])
        variables = flax_attention.init(jax.random.key(0), query)
        # {params:{
        #   query:{kernel:shape:(128, 8, 16), bias:shape:(8, 16)},
        #   key:{kernel:shape:(128, 8, 16), bias:shape:(8, 16)},
        #   value:{kernel:shape:(128, 8, 16), bias:shape:(8, 16)},
        #   out:{kernel:shape:(8, 16, 128), bias:shape:(128,)}}}
        # print(_nested_printer(variables))
        # print(jax.tree_util.tree_map(lambda x : x.shape, variables))
        layer = attentions.MultiHeadAttention(num_heads=num_heads)
        # Initialize layer
        output = layer(query=query)

        wq, wk, wv, wo, bq, bk, bv, bo = utils.read_attention_variables_from_flax(
            variables)

        utils.bind_attention_variables_to_layer(layer, wq, wk, wv, wo, bq, bk,
                                                bv, bo)

        output = layer(query)
        flax_output = flax_attention.apply(variables, query)

        self.assert_allclose(output, flax_output)

        targets = utils.rand([batch, seq_len_q, qkv_features])

        @jax.jit
        def flax_forward(variables, query, targets):
            y = flax_attention.apply(variables, query)
            return utils.mse_loss(y, targets)

        flax_backward = jax.jit(jax.grad(flax_forward, argnums=(0, 1)))
        # print(flax_backward.lower(variables, query, targets).as_text())

        flax_grad = flax_backward(variables, query, targets)
        # print(jax.tree_util.tree_map(lambda x : x.shape, flax_output))

        flax_dwq, flax_dwk, flax_dwv, flax_dwo, flax_dbq, flax_dbk, flax_dbv, flax_dbo = utils.read_attention_variables_from_flax(
            flax_grad[0])
        flax_dquery = flax_grad[1]

        learning_rate = 0.01

        dy = jax.grad(utils.mse_loss)(output, targets)

        # Makes a deepcopy as the backward path is going to update weights.
        layer = copy.deepcopy(layer)
        dquery, dkey, dvalue = layer(dy,
                                     backprop=True,
                                     learning_rate=learning_rate)

        self.assert_allclose(flax_dquery, dquery + dkey + dvalue)
        self.assert_allclose(layer._wo, -learning_rate * flax_dwo + wo)
        self.assert_allclose(layer._bo, -learning_rate * flax_dbo + bo)
        self.assert_allclose(layer._wv, -learning_rate * flax_dwv + wv)
        self.assert_allclose(layer._bv, -learning_rate * flax_dbv + bv)
        self.assert_allclose(layer._wk, -learning_rate * flax_dwk + wk)
        self.assert_allclose(layer._bk, -learning_rate * flax_dbk + bk)
        self.assert_allclose(layer._wq, -learning_rate * flax_dwq + wq)
        self.assert_allclose(layer._bq, -learning_rate * flax_dbq + bq)
