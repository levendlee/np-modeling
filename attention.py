# Multi-Head Attention

from typing import Callable, Optional, Sequence, Type

import numpy as np

import layer
import mlp


class Softmax(mlp.Activation):
    def forward(self, x: np.ndarray) -> np.ndarray:
        self._x = x

        x_max = np.max(x, axis=-1, keepdims=True)
        exp_x = np.exp(x - x_max)
        exp_x_sum = np.sum(exp_x, axis=-1, keepdims=True)
        self._y = exp_x / exp_x_sum

        return self._y

    def backward(self, dy: np.ndarray) -> np.ndarray:
        rank = len(self._y.shape)
        batch = self._y.shape[:-1]
        n = self._y.shape[-1]
        
        # https://towardsdatascience.com/derivative-of-the-softmax-function-and-the-categorical-cross-entropy-loss-ffceefc081d1
        # Jacobian
        # dy_i/dx_j = y_i(1{i=j} - y_j)
        # Broadcasting batch dimensions and last 2 dimensions as Jacobian. 
        j = np.expand_dims(np.eye(n), axis=tuple(range(rank - 1)))
        j = j - np.expand_dims(self._y, axis=rank-1)
        j = j * np.expand_dims(self._y, axis=rank)
        return np.einsum('...a,...ba->...b', dy, j)

class MultiHeadAttention(layer.StatefulLayer):
    def __init__(self, num_heads: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._num_heads = num_heads
        self._softmax = Softmax()

    def initialize(self,
                   query: np.ndarray,
                   key: Optional[np.ndarray] = None,
                   value: Optional[np.ndarray] = None,
                   *args,
                   **kwargs) -> None:
        # query:      [batch, seq_len_q,  num_heads * key_dim   ]
        # key:        [batch, seq_len_kv, num_heads * key_dim   ]
        # value:      [batch, seq_len_kv, num_heads * value_dim ]

        if key is None:
            key = query
        if value is None:
            value = key

        assert query.shape[0] == key.shape[0]
        assert query.shape[2] == key.shape[2]
        assert query.shape[0] == value.shape[0]
        assert key.shape[1] == value.shape[1]

        self._seq_len_q = query.shape[1]
        self._seq_len_kv = key.shape[1]

        assert key.shape[2] % self._num_heads == 0
        self._key_dim = key.shape[2] // self._num_heads

        assert value.shape[2] % self._num_heads == 0
        self._value_dim = value.shape[2] // self._num_heads

        # wq: [num_heads, key_dim, num_heads * key_dim]
        self._wq = self._initializer(
            [self._num_heads, self._key_dim, self._num_heads * self._key_dim])
        # wk: [num_heads, key_dim, num_heads * key_dim]
        self._wk = self._initializer(
            [self._num_heads, self._key_dim, self._num_heads * self._key_dim])
        # wv: [num_heads, key_dim, num_heads * value_dim]
        self._wv = self._initializer([
            self._num_heads, self._value_dim, self._num_heads * self._value_dim
        ])
        # wo: [num_heads * key_dim, num_heads, value_dim]
        self._wo = self._initializer([
            self._num_heads * self._key_dim, self._num_heads, self._value_dim
        ])

        # bq: [num_heads, key_dim]
        self._bq = self._initializer([self._num_heads, self._key_dim])
        self._bk = self._initializer([self._num_heads, self._key_dim])
        self._bv = self._initializer([self._num_heads, self._value_dim])
        self._bo = self._initializer([self._num_heads * self._key_dim])

    def forward(self,
                query: np.ndarray,
                key: Optional[np.ndarray] = None,
                value: Optional[np.ndarray] = None,
                mask: Optional[np.ndarray] = None) -> np.ndarray:

        if key is None:
            key = query
        if value is None:
            value = key

        self._query = query
        self._key = key
        self._value = value
        self._mask = mask

        batch = query.shape[0]
        if mask:
            assert mask.shape == (batch, self._num_heads, self._seq_len_q,
                                  self._seq_len_kv)

        input_projection_equation = '...ab,cdb->...acd'
        # [batch, seq_len_q, num_heads, key_dim]
        q = np.einsum(input_projection_equation, query, self._wq)
        q += self._bq
        self._q = q
        # [batch, seq_len_kv, num_heads, key_dim]
        k = np.einsum(input_projection_equation, key, self._wk)
        k += self._bk
        self._k = k
        # [batch, seq_len_kv, num_heads, value_dim]
        v = np.einsum(input_projection_equation, value, self._wv)
        v += self._bv
        self._v = v

        # [batch, num_heads, seq_len_q, seq_len_kv]
        attention = np.einsum('...abc,...dbc->...bad', q, k)
        scaled_attention = (1.0 / np.sqrt(self._key_dim)) * attention
        self._mask = mask
        if mask:
            scaled_attention = np.where(mask, scaled_attention, float('-inf'))
        attention_scores = self._softmax(scaled_attention)

        # values: [batch, num_heads, seq_len_q, value_dim]
        self._attention_scores = attention_scores
        values = np.einsum('...abc,...cad->...abd', attention_scores, v)

        self._attention_values = values
        # wo: [num_heads * key_dim, num_heads, value_dim]
        o = np.einsum('...abc,...dac->...bd', values, self._wo)
        o += self._bo

        # [batch, seq_len_q, num_heads * key_dim]
        return o

    def backward(
            self, dy: np.ndarray,
            learning_rate: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:

        batch = dy.shape[0]

        # Output projection
        dbo = np.sum(dy, axis=(0, 1))
        # values: [batch, num_heads, seq_len_q, value_dim]
        # w:      [num_heads * key_dim, num_heads, value_dim]
        # y:      [batch, seq_len_q, num_heads * key_dim]
        dwo = np.sum(np.einsum('...abc,...bd->...dac', self._attention_values,
                               dy),
                     axis=0)
        dvalues = np.einsum('...ab,bcd->...cad', dy, self._wo)

        # Softmax@V
        # attention_scores: [batch, num_heads, seq_len_q, seq_len_kv]
        # v:                [batch, seq_len_kv, num_heads, value_dim]
        # values:           [batch, num_heads, seq_len_q, value_dim]
        assert dvalues.shape == (batch, self._num_heads, self._seq_len_q,
                                 self._value_dim)
        assert self._v.shape == (batch, self._seq_len_kv, self._num_heads,
                                 self._value_dim)
        dscores = np.einsum('...abc,...dac->...abd', dvalues, self._v)
        dv = np.einsum('...abc,...abd->...cad', self._attention_scores,
                       dvalues)

        dscaled_attention = self._softmax.backward(dscores)

        if self._mask:
            raise NotImplementedError

        dattention = dscaled_attention / np.sqrt(self._key_dim)

        # Q^T@K
        # attention: [batch, num_heads, seq_len_q, seq_len_kv]
        # q:         [batch, seq_len_q, num_heads, key_dim]
        # k:         [batch, seq_len_kv, num_heads, key_dim]
        dq = np.einsum('...abc,...cad->...bad', dattention, self._k)
        dk = np.einsum('...abc,...bad->...dbc', self._q, dattention)

        # query: [batch, seq_len_q, num_heads * key_dim]
        # wq:    [num_heads, key_dim, num_heads * key_dim]
        # q:     [batch, seq_len_q, num_heads, key_dim]
        input_projection_grad_w_equation = '...ab,...acd->...cdb'
        input_projection_grad_x_equation = '...abc,bcd->...ad'
        dwq = np.sum(np.einsum(input_projection_grad_w_equation, self._query,
                               dq),
                     axis=0)
        dquery = np.einsum(input_projection_grad_x_equation, dq, self._wq)
        dwk = np.sum(np.einsum(input_projection_grad_w_equation, self._key,
                               dk),
                     axis=0)
        dkey = np.einsum(input_projection_grad_x_equation, dk, self._wk)
        assert self._value.shape == (batch, self._seq_len_kv,
                                     self._num_heads * self._value_dim)
        assert dv.shape == (batch, self._seq_len_kv, self._num_heads,
                            self._value_dim)
        dwv = np.sum(np.einsum(input_projection_grad_w_equation, self._value,
                               dv),
                     axis=0)
        dvalue = np.einsum(input_projection_grad_x_equation, dv, self._wv)

        dbq = np.sum(dq, axis=(0, 1))
        dbk = np.sum(dk, axis=(0, 1))
        dbv = np.sum(dv, axis=(0, 1))

        self._wq -= learning_rate * dwq
        self._bq -= learning_rate * dbq
        self._wk -= learning_rate * dwk
        self._bk -= learning_rate * dbk
        self._wv -= learning_rate * dwv
        self._bv -= learning_rate * dbv
        self._wo -= learning_rate * dwo
        self._bo -= learning_rate * dbo

        return dquery, dkey, dvalue
