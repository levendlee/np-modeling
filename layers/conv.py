# Conv layer.

from typing import Optional, Sequence

import numpy as np

import optimizer
from layers import activations, layer


class Conv2D(layer.StatefulLayer):
    """Conv2D w/ ReLU activation.

    Assumes:
      - Padding as 'SAME'.
      - Strides as (1, 1).
    """
    def __init__(self,
                 channels: int,
                 kernel_size: int,
                 padding: str = 'SAME',
                 strides: Sequence[int] = (1, 1),
                 activation: Optional[activations.Activation] = None,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        assert padding == 'SAME'
        # TODO: Support dialted convolution.
        assert strides == (1, 1)
        self._output_channels = channels
        self._kernel_size = kernel_size
        self._activation = activation or activations.ReLU()

    def initialize(self, x: np.ndarray) -> None:
        # Assums x in NHWC format. filters in HWIO format.
        self._input_channels = x.shape[-1]
        self._w = self._initializer([
            self._kernel_size, self._kernel_size, self._input_channels,
            self._output_channels
        ])
        self._b = self._initializer([self._output_channels])
        self._activation.initialize()

    def forward(self, x: np.ndarray) -> np.ndarray:
        self._x = x
        y = _conv2d(x, self._w)
        y += self._b
        return self._activation.forward(y)

    def backward(self, dy: np.ndarray,
                 optimizer_: optimizer.Optimizer) -> np.ndarray:
        assert dy.shape[:3] == self._x.shape[:3]
        assert dy.shape[3] == self._output_channels
        dy = self._activation.backward(dy)
        db = np.sum(dy, axis=(0, 1, 2))
        dw = _conv2d_grad_w(dy, self._x, self._kernel_size)
        dx = _conv2d_grad_x(dy, self._w)
        assert dx.shape == self._x.shape
        optimizer_.update(self, '_w', dw)
        optimizer_.update(self, '_b', db)
        return dx

    @property
    def w(self) -> np.ndarray:
        assert self._initialized
        return self._w

    @property
    def b(self) -> np.ndarray:
        assert self._initialized
        return self._b


def _conv2d(x: np.ndarray, filters: np.ndarray) -> np.ndarray:
    """Conv2D.
    
    Assumes:
      - Padding as 'SAME'.
      - Strides as (1, 1).

    Args:
      x: [N, H, W, C]
      filters: [H, W, I, O]

    Returns
      y: [N, H, W, C]
    """

    n, h, w, c0 = x.shape
    assert filters.shape[0] == filters.shape[1]
    assert filters.shape[2] == c0
    k, _, _, c1 = filters.shape

    assert k % 2
    p = k // 2

    padded_x = np.zeros([n, h + k - 1, w + k - 1, c0])
    padded_x[:, p:h + p, p:w + p, :] = x

    result = np.zeros([n, h, w, c1])
    for i in range(k):
        for j in range(k):
            rhs = filters[i, j, :, :].reshape([c0, c1])
            lhs = padded_x[:, i:h + i, j:w + j, :].reshape([n * h * w, c0])
            result += np.matmul(lhs, rhs).reshape([n, h, w, c1])

    return result


def _conv2d_transpose(y: np.ndarray, filters: np.ndarray) -> np.ndarray:
    """Conv2D tranpose.
    
    Assumes:
      - Padding as 'SAME'.
      - Strides as (1, 1).

    Args:
      y: [N, H, W, C]
      filters: [H, W, I, O]

    Returns
      x: [N, H, W, C]
    """

    n, h, w, c1 = y.shape
    assert filters.shape[0] == filters.shape[1]
    assert filters.shape[3] == c1
    k, _, c0, _ = filters.shape

    return _conv2d(y, np.transpose(filters[::-1, ::-1, :, :], [0, 1, 3, 2]))


def _conv2d_grad_x(dy: np.ndarray, filters: np.ndarray) -> np.ndarray:
    """Gradient of input in Conv2D backprop.

    Note: Following uses lower case to represent indicies, upper case to
    represent dimensions.

    In forward pass, element at x ([n, h, w, c0]) is going to accumulated
    at C1 * K * K elements in y. So the gradient is a sum of these C1 * K * K
    elements in dy multiplied by filter, in 
    ( [n, h + p - k + 1, w + p - k + 1, :],
      ...,
      [n, h + p, w + p, :] ).

    Bear in mind, for k = 2 * p + 1, so it becomes:
    ( [n, h - p, w - p, :],
      ...,
      [n, h + p, w + p, :]).

    It is simply transposed conv2d, as the filter mapping are flipped.
    """
    return _conv2d_transpose(dy, filters)


def _conv2d_grad_w(dy: np.ndarray, x: np.ndarray,
                   filter_size: int) -> np.ndarray:
    """Gradient of filters in Conv2D backprop.
  
    In forward pass, element as w ([i, j, c0, c1]) is going to accumulated
    at N * H * W elements in y [:, :, :, c1], meanwhile, it going to
    be muliplied by N * H * W elements in x [:, :, :, c0]. So,
      dw = x^T @ dy
    For each filter index. The x and y should have a shift in spatial
    dimensions, for example, x[n, h + i - p, w + j - p, :] maps to
    y[n, h, w, :].
    
    Alternatively, we can also think it as totally K * K of 1x1 convolutions
    accumulated together, which makes the first second of above description
    easier to understand.

    Essentially, it should be padded X convoluted with y (as filters) without
    additional padding.
    """

    assert dy.shape[:3] == x.shape[:3]

    n, h, w, c1 = dy.shape
    n, h, w, c0 = x.shape
    k = filter_size

    assert k % 2
    p = k // 2

    padded_x = np.zeros([n, h + k - 1, w + k - 1, c0])
    padded_x[:, p:h + p, p:w + p, :] = x

    result = np.zeros([k, k, c0, c1])
    for i in range(k):
        for j in range(k):
            lhs = padded_x[:, i:h + i, j:w + j, :].reshape([n * h * w, c0]).T
            rhs = dy.reshape([n * h * w, c1])
            result[i, j, :, :] += np.matmul(lhs, rhs)
    return result
