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

    def test_mha_forward(self):
        np.random.seed(0)

        batch = 16
        seq_len_q = 32
        seq_len_kv = 128
        num_heads = 8
        qkv_features = 128

        key_dim = qkv_features // num_heads
        value_dim = qkv_features // num_heads

        def mha_fwd(query, key, value, mask):
            # query: [batch, seq_len_q, features]
            # key:   [batch, seq_len_kv, features]
            # value: [batch, seq_len_kv, features]

            # wq: [num_heads, key_dim, features]
            # wk: [num_heads, key_dim, features]
            # wv: [num_heads, value_dim, features]

            # bq: [num_heads, key_dim]
            # bk: [num_heads, key_dim]
            # bq: [num_heads, value_dim]

            q = np.einsum('...ab,cdb->...acd', query, wq)
            q += bq
            k = np.einsum('...ab,cdb->...acd', key, wk)
            k += bk
            v = np.einsum('...ab,cdb->...acd', value, wv)
            v += bv

            # q: [batch, seq_len_q, num_heads, key_dim]
            # k: [batch, seq_len_kv, num_heads, key_dim]
            # v: [batch, seq_len_kv, num_heads, value_dim]

            # p: [batch, num_heads, seq_len_q, seq_len_kv]
            attention = np.einsum('...abc,...dbc->...bad', q, k)
            attention *= (1 / np.sqrt(key_dim))

            if mask:
                attention = np.where(mask, attention, float('-inf'))

            def softmax(x):
                column_maximum = np.max(x, axis=-1, keepdims=True)
                expontential = np.exp(x - column_maximum)
                expontential_sum = np.sum(expontential, axis=-1, keepdims=True)
                return expontential / expontential_sum

            attention_scores = softmax(attention)
            # values: [batch, seq_len_q, num_heads, value_dim]
            values = np.einsum('...abc,...cad->...bad', attention_scores, v)
            o = np.einsum('...abc,...dbc->...ad', values, wo)
            o += bo
            return o

        # Flax baseline
        flax_attention = flax.linen.attention.MultiHeadDotProductAttention(
            num_heads=num_heads, qkv_features=qkv_features)

        query = utils.rand([batch, seq_len_q, qkv_features])
        variables = flax_attention.init(jax.random.key(0), query)

        wq, wk, wv, wo, bq, bk, bv, bo = utils.read_attention_variables_from_flax(
            variables)

        output = mha_fwd(query=query, key=query, value=query, mask=None)
        flax_output = flax_attention.apply(variables, query)

        self.assert_allclose(output, flax_output)

    def test_flash_attention_forward(self):
        np.random.seed(0)

        batch = 16
        seq_len_qkv = 128
        num_heads = 8
        qkv_features = 128

        key_dim = qkv_features // num_heads
        value_dim = qkv_features // num_heads

        def mha_fwd(query, key, value, mask):
            # query: [batch, seq_len_q, features]
            # key:   [batch, seq_len_kv, features]
            # value: [batch, seq_len_kv, features]

            # wq: [num_heads, key_dim, features]
            # wk: [num_heads, key_dim, features]
            # wv: [num_heads, value_dim, features]

            # bq: [num_heads, key_dim]
            # bk: [num_heads, key_dim]
            # bq: [num_heads, value_dim]

            q = np.einsum('...ab,cdb->...acd', query, wq)
            q += bq
            k = np.einsum('...ab,cdb->...acd', key, wk)
            k += bk
            v = np.einsum('...ab,cdb->...acd', value, wv)
            v += bv

            # q: [batch, seq_len_q, num_heads, key_dim]
            # k: [batch, seq_len_kv, num_heads, key_dim]
            # v: [batch, seq_len_kv, num_heads, value_dim]

            # p: [batch, num_heads, seq_len_q, seq_len_kv]
            q_block = 32
            kv_block = 32

            values = np.zeros([batch, seq_len_qkv, num_heads,
                               value_dim]).astype(np.float32)
            for q_start in range(0, seq_len_qkv, q_block):
                # tq: [batch, q_block, num_heads, key_dim]
                tq = q[:, q_start:q_start + q_block, :, :]

                # Tracking the maximum element used for shifting.
                m_i = np.zeros([batch, num_heads, q_block]).astype(
                    np.float32) + float('-inf')
                # Tracking the sum of shifted exponential.
                l_i = np.zeros([batch, num_heads, q_block]).astype(np.float32)
                acc = np.zeros([batch, q_block, num_heads,
                                value_dim]).astype(np.float32)
                for kv_start in range(0, seq_len_qkv, kv_block):
                    # tk: [batch, kv_block, num_heads, key_dim]
                    tk = k[:, kv_start:kv_start + kv_block, :, :]
                    # tv: [batch, kv_block, num_heads, value_dim]
                    tv = v[:, kv_start:kv_start + kv_block, :, :]

                    # p_ij: [batch, num_heads, q_block, kv_block]
                    p_ij = np.einsum('...abc,...dbc->...bad', tq, tk)
                    p_ij *= (1.0 / np.sqrt(key_dim))

                    m_i_new = np.maximum(np.max(p_ij, axis=3), m_i)
                    l_i *= np.exp(m_i - m_i_new)
                    p_ij = np.exp(p_ij - np.expand_dims(m_i_new, axis=3))
                    l_i_new = np.sum(p_ij, axis=3) + l_i
                    p_ij *= 1.0 / np.expand_dims(l_i_new, axis=3)
                    acc *= np.transpose(np.expand_dims(l_i / l_i_new, axis=3),
                                        [0, 2, 1, 3])
                    '''
                    # Tracking the column maximum.
                    m_i_new = np.maximum(np.max(p_ij, axis=3), m_i)
                    # Rescaling denominator.
                    l_i *= np.exp(m_i - m_i_new)
                    # Shifted exponential
                    p_ij = np.exp(p_ij - np.expand_dims(m_i_new, axis=3))
                    # Adjusting denominator of exponential sum.
                    l_i_new = np.sum(p_ij, axis=3) + l_i
                    # "Local" softmax with updated denominator.
                    p_ij *= np.expand_dims(1.0 / l_i_new, axis=3)
                    acc *= np.transpose(np.expand_dims(l_i / l_i_new, axis=3),
                                        [0, 2, 1, 3])
                    '''
                    m_i = m_i_new
                    l_i = l_i_new

                    acc += np.einsum('...abc,...cad->...bad', p_ij, tv)

                values[:, q_start:q_start + q_block, :, :] = acc

            o = np.einsum('...abc,...dbc->...ad', values, wo)
            o += bo
            return o

        # Flax baseline
        flax_attention = flax.linen.attention.MultiHeadDotProductAttention(
            num_heads=num_heads, qkv_features=qkv_features)

        query = utils.rand([batch, seq_len_qkv, qkv_features])
        variables = flax_attention.init(jax.random.key(0), query)

        wq, wk, wv, wo, bq, bk, bv, bo = utils.read_attention_variables_from_flax(
            variables)

        output = mha_fwd(query=query, key=query, value=query, mask=None)
        flax_output = flax_attention.apply(variables, query)

        self.assert_allclose(output, flax_output)

    def test_gqa_forward(self):
        np.random.seed(0)

        batch = 16
        seq_len_q = 128
        seq_len_kv = 128
        num_q = 8
        num_kv = 4
        qkv_features = 128

        key_dim = qkv_features // num_q
        value_dim = key_dim

        def gqa_fwd(query, key, value, mask):
            # query: [batch, seq_len_q, features]
            # key:   [batch, seq_len_kv, features]
            # value: [batch, seq_len_kv, features]

            # wq: [num_q, key_dim, features]
            # wk: [num_kv, key_dim, features]
            # wv: [num_kv, value_dim, features]

            # bq: [num_q, key_dim]
            # bk: [num_kv, key_dim]
            # bq: [num_kv, value_dim]

            q = np.einsum('...ab,cdb->...acd', query, wq)
            q += bq
            k = np.einsum('...ab,cdb->...acd', key, wk)
            k += bk
            v = np.einsum('...ab,cdb->...acd', value, wv)
            v += bv

            # q: [batch, seq_len_q, num_q, key_dim]
            # k: [batch, seq_len_kv, num_kv, key_dim]
            # v: [batch, seq_len_kv, num_kv, value_dim]

            # p: [batch, num_q, seq_len_q, seq_len_kv]
            q = np.reshape(
                q, [batch, seq_len_q, num_q // num_kv, num_kv, key_dim])
            attention = np.einsum('...abcd,...ecd->...bcae', q, k)
            attention = np.reshape(attention,
                                   [batch, num_q, seq_len_q, seq_len_kv])
            attention *= (1 / np.sqrt(key_dim))

            if mask:
                attention = np.where(mask, attention, float('-inf'))

            def softmax(x):
                column_maximum = np.max(x, axis=-1, keepdims=True)
                expontential = np.exp(x - column_maximum)
                expontential_sum = np.sum(expontential, axis=-1, keepdims=True)
                return expontential / expontential_sum

            attention_scores = softmax(attention)
            # values: [batch, seq_len_q, num_q, value_dim]
            attention_scores = np.reshape(
                attention_scores, [batch, num_q // num_kv, num_kv, seq_len_q, seq_len_kv])
            values = np.einsum('...abcd,...dbe->...cabe', attention_scores, v)
            values = np.reshape(values, [batch, seq_len_q, num_q, value_dim])
            o = np.einsum('...abc,...dbc->...ad', values, wo)
            o += bo
            return o

        # Flax baseline
        flax_attention = flax.linen.attention.MultiHeadDotProductAttention(
            num_heads=num_q, qkv_features=qkv_features)

        query = utils.rand([batch, seq_len_q, qkv_features])
        variables = flax_attention.init(jax.random.key(0), query)
        for name in ('key', 'value'):
            for kv_head in range(0, num_q, num_kv):
                variables['params'][name]['kernel'] = np.repeat(
                    variables['params'][name]['kernel'][:, 0:num_kv, :],
                    num_q // num_kv,
                    axis=1)
                variables['params'][name]['bias'] = np.repeat(
                    variables['params'][name]['bias'][0:num_kv, :],
                    num_q // num_kv,
                    axis=0)

        wq, wk, wv, wo, bq, bk, bv, bo = utils.read_attention_variables_from_flax(
            variables)
        wk = wk[0:num_kv, :, :]
        wv = wv[0:num_kv, :, :]
        bk = bk[0:num_kv, :]
        bv = bv[0:num_kv, :]

        output = gqa_fwd(query=query, key=query, value=query, mask=None)
        flax_output = flax_attention.apply(variables, query)

        self.assert_allclose(output, flax_output)
