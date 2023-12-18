# Transformer.

from typing import Callable, Optional, Sequence, Type

import numpy as np

import attention
import layer
import mlp
import optimizer


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

    def backward(self, dl_dz: np.ndarray,
                 optimizer_: optimizer.Optimizer) -> np.ndarray:
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
        # dy/dx = (\sigma + \epsilon) ^ {-1/2}(I - 1/N)(\sigma + \epsilon) -
        #         (\sigma + \epsilon) ^ {-3/2}((x_i - \mu) / N)(x_j - \mu)
        dy_dx = (np.expand_dims(g, rank)**-0.5 *
                 np.expand_dims(np.eye(self._col) - dmean_dx, batch_dims) -
                 0.5 * np.expand_dims(g, rank)**-1.5 *
                 np.expand_dims(dvar_dx, rank) * np.expand_dims(f, rank - 1))
        dl_dx = np.einsum('...a,...ab->...b', dl_dy, dy_dx)

        optimizer_.update(self, '_gamma', dl_dgamma)
        optimizer_.update(self, '_beta', dl_dbeta)
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

        batch, seq_len_q, features = qkv.shape

        skip = qkv
        if self._norm_first:
            qkv = self._norm1(qkv)
        out = self._self_attention(qkv)
        out += skip
        if not self._norm_first:
            out = self._norm1(out)

        # Our dense layer doesn't support more than 1 batch dimension. :)
        out = np.reshape(out, [-1, features])
        skip = out

        if self._norm_first:
            out = self._norm2(out)
        out = self._dense1(out)
        out = self._dense2(out)
        out += skip
        if not self._norm_first:
            out = self._norm2(out)

        out = np.reshape(out, [batch, seq_len_q, features])
        return out

    def backward(self, dy: np.ndarray, optimizer_: float) -> np.ndarray:
        # Uses dy to represent dl/dy and so on.

        batch, seq_len_q, features = dy.shape

        dy = np.reshape(dy, [-1, features])
        if not self._norm_first:
            dy = self._norm2.backward(dy, optimizer_)
        dskip = dy
        dy = self._dense2.backward(dy, optimizer_)
        dy = self._dense1.backward(dy, optimizer_)
        if self._norm_first:
            dy = self._norm2.backward(dy, optimizer_)

        dy += dskip
        dy = np.reshape(dy, [batch, seq_len_q, features])

        if not self._norm_first:
            dy = self._norm1.backward(dy, optimizer_)
        dskip = dy
        dy = self._self_attention.backward(dy, optimizer_)
        dy = np.sum(dy, axis=0)
        if self._norm_first:
            dy = self._norm1.backward(dy, optimizer_)

        dy += dskip

        return dy


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

        batch, seq_len_q, features = q.shape

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

        # Our dense layer doesn't support more than 1 batch dimension. :)
        out = np.reshape(out, [-1, features])
        skip = out

        if self._norm_first:
            out = self._norm3(out)
        out = self._dense1(out)
        out = self._dense2(out)
        out += skip
        if not self._norm_first:
            out = self._norm3(out)

        out = np.reshape(out, [batch, seq_len_q, features])
        return out

    def backward(self, dy: np.ndarray, optimizer_: float) -> np.ndarray:
        batch, seq_len_q, features = dy.shape

        dy = np.reshape(dy, [-1, features])
        if not self._norm_first:
            dy = self._norm3.backward(dy, optimizer_)
        dskip = dy
        dy = self._dense2.backward(dy, optimizer_)
        dy = self._dense1.backward(dy, optimizer_)
        if self._norm_first:
            dy = self._norm3.backward(dy, optimizer_)

        dy += dskip
        dy = np.reshape(dy, [batch, seq_len_q, features])

        if not self._norm_first:
            dy = self._norm2.backward(dy, optimizer_)
        dskip = dy
        dy = self._cross_attention.backward(dy, optimizer_)
        dkv = np.sum(dy[1:3], axis=0)
        dy = dy[0]
        if self._norm_first:
            dy = self._norm2.backward(dy, optimizer_)

        dy += dskip
        if not self._norm_first:
            dy = self._norm1.backward(dy, optimizer_)
        dskip = dy
        dy = self._self_attention.backward(dy, optimizer_)
        dy = np.sum(dy, axis=0)
        if self._norm_first:
            dy = self._norm1.backward(dy, optimizer_)

        dy += dskip

        return dy, dkv
