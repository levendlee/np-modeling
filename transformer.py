# Transformer.

from typing import Callable, Optional, Sequence, Type

import numpy as np

import attention
import layer
import mlp


class LayerNormalization(layer.StatefulLayer):
    def __init__(self, epsilon: float = 1e-3, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._epsilon = epsilon

    def initialize(self, x: np.ndarray):
        self._col = x.shape[-1]
        self._gamma = self._initializer([self._col])
        self._beta = self._initializer([self._col])

    def forward(self, x: np.ndarray) -> np.ndarray:
        self._x = x
        self._mean = np.mean(x, axis=-1, keepdims=True)
        self._var = np.var(x, axis=-1, keepdims=True)
        self._y = (x - self._mean) / np.sqrt(self._var + self._epsilon)
        return self._gamma * self._y + self._beta

    def backward(self, dl_dz: np.ndarray, learning_rate: float) -> np.ndarray:
        rank = len(self._x.shape)
        batch_dims = tuple(range(rank - 1))

        dl_dbeta = np.sum(dl_dz, axis=batch_dims)
        dl_dgamma = np.sum(dl_dz * self._y, axis=batch_dims)
        dl_dy = dl_dz * self._gamma

        # Jacobian on last two dimensions, row as dx, col as dy.
        dmean_dx = np.array(1.0 / self._col)
        dvar_dx = 2.0 * (self._x - self._mean) / self._col

        f = self._x - self._mean
        g = self._var + self._epsilon
        f_dg = f * (-0.5 * g**-1.5 * dvar_dx)
        df_g = (np.expand_dims((np.eye(self._col) - dmean_dx), batch_dims) *
                np.expand_dims(g**-0.5, rank))
        dy_dx = np.expand_dims(f_dg, rank) + df_g
        dl_dx = np.einsum('...a,...ab->...b', dl_dy, dy_dx)

        self._gamma -= learning_rate * dl_dgamma
        self._beta -= learning_rate * dl_dbeta
        return dl_dx


class TransformerEncoder(layer.Layer):
    def __init__(self, num_heads: int, hidden_units: int, norm_first: bool,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._self_attention = attention.MultiHeadAttention(num_heads)
        self._dense1 = mlp.Dense(units=hidden_units)
        self._norm1 = LayerNormalization()
        self._norm2 = LayerNormalization()
        self._norm_first = norm_first

    def initialize(self, qkv: np.ndarray):
        features = qkv.shape[-1]
        self._dense2 = mlp.Linear(units=features)  # No activation

    def forward(self, qkv: np.ndarray) -> np.ndarray:
        # TODO: add mask support.
        # TODO: add dropout support.

        skip = qkv
        if self._norm_first:
            qkv = self._norm1(qkv)
        out = self._self_attention(qkv)
        out += skip
        if not self._norm_first:
            out = self._norm1(out)

        skip = out

        if self._norm_first:
            out = self._norm2(out)
        out = self._dense1(out)
        out = self._dense2(out)
        out += skip
        if not self._norm_first:
            out = self._norm2(out)

        return out

    def backward(self, *args, **kwargs):
        raise NotImplementedError


class TransformerDecoder(layer.Layer):
    def __init__(self, num_heads: int, hidden_units: int, norm_first: bool,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._self_attention = attention.MultiHeadAttention(num_heads)
        self._cross_attention = attention.MultiHeadAttention(num_heads)
        self._dense1 = mlp.Dense(units=hidden_units)
        self._norm1 = LayerNormalization()
        self._norm2 = LayerNormalization()
        self._norm3 = LayerNormalization()
        self._norm_first = norm_first

    def initialize(self, q: np.ndarray, kv: np.ndarray):
        features = q.shape[-1]
        self._dense2 = mlp.Linear(units=features)  # No activation

    def forward(self, q: np.ndarray, kv: np.ndarray) -> np.ndarray:
        # TODO: support cache

        skip = q
        if self._norm_first:
            q = self._norm1(q)
        out = self._self_attention(q)
        out += skip
        if not self._norm_first:
            out = self._norm1(out)

        skip = out

        if self._norm_first:
            out = self._norm2(out)
        out = self._cross_attention(out, kv)
        out += skip
        if not self._norm_first:
            out = self._norm2(out)

        skip = out

        if self._norm_first:
            out = self._norm3(out)
        out = self._dense1(out)
        out = self._dense2(out)
        out += skip
        if not self._norm_first:
            out = self._norm3(out)

        return out

    def backward(self, *args, **kwargs):
        raise NotImplementedError
