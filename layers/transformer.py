# Transformer.

import numpy as np

from layers import attentions, layer, mlp, normalizations


class TransformerEncoder(layer.Layer):
    def __init__(self, num_heads: int, hidden_units: int, norm_first: bool,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._self_attention = attentions.MultiHeadAttention(num_heads)
        self._dense1 = mlp.Dense(units=hidden_units)
        self._norm1 = normalizations.LayerNormalization()
        self._norm2 = normalizations.LayerNormalization()
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
        self._self_attention = attentions.MultiHeadAttention(num_heads)
        self._cross_attention = attentions.MultiHeadAttention(num_heads)
        self._dense1 = mlp.Dense(units=hidden_units)
        self._norm1 = normalizations.LayerNormalization()
        self._norm2 = normalizations.LayerNormalization()
        self._norm3 = normalizations.LayerNormalization()
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
