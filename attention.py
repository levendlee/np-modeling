# Multi-Head Attention

from typing import Callable, Optional, Sequence, Type

import numpy as np

import layer
import mlp


class Softmax(mlp.Activation):
    def forward(self, x: np.ndarray) -> np.ndarray:

        x_max = np.max(x, axis=-1, keepdims=True)
        exp_x = np.exp(x - x_max)
        exp_x_sum = np.sum(exp_x, axis=-1, keepdims=True)
        return exp_x / exp_x_sum

    def backward(self, *args, **kwargs) -> np.ndarray:
        raise NotImplementedError


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

        batch = query.shape[0]
        if mask:
            assert mask.shape == (batch, self._num_heads, self._seq_len_q,
                                  self._seq_len_kv)

        input_projection_equation = '...ab,cdb->...acd'
        # [batch, seq_len_q, num_heads, key_dim]
        q = np.einsum(input_projection_equation, query, self._wq)
        q += self._bq
        # [batch, seq_len_kv, num_heads, key_dim]
        k = np.einsum(input_projection_equation, key, self._wk)
        k += self._bk
        # [batch, seq_len_kv, num_heads, value_dim]
        v = np.einsum(input_projection_equation, value, self._wv)
        v += self._bv

        # [batch, num_heads, seq_len_q, seq_len_kv]
        attention = np.einsum('...abc,...dbc->...bad', q, k)
        scaled_attention = (1.0 / np.sqrt(self._key_dim)) * attention
        if mask:
            scaled_attention = np.where(mask, scaled_attention, float('-inf'))
        attention_scores = self._softmax(scaled_attention)

        # [batch, num_heads, seq_len_q, value_dim]
        values = np.einsum('...abc,...cad->...abd', attention_scores, v)

        o = np.einsum('...abc,...dac->...bd', values, self._wo)
        o += self._bo

        return o

    def backward(self, *args, **kwargs) -> np.ndarray:
        raise NotImplementedError
