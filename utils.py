# Whatever utils.

import flax
import jax
from jax import numpy as jnp
import numpy as np


def rand(shape) -> np.ndarray:
    return np.random.normal(size=shape).astype(np.float32)


@jax.jit
def mse_loss(y, targets):
    diff = y - targets
    return jnp.sum(diff**2) / y.size


def read_attention_variables_from_flax(variables):
    wq = np.transpose(variables['params']['query']['kernel'], [1, 2, 0])
    wk = np.transpose(variables['params']['key']['kernel'], [1, 2, 0])
    wv = np.transpose(variables['params']['value']['kernel'], [1, 2, 0])
    wo = np.transpose(variables['params']['out']['kernel'], [2, 0, 1])

    bq = variables['params']['query']['bias']
    bk = variables['params']['key']['bias']
    bv = variables['params']['value']['bias']
    bo = variables['params']['out']['bias']

    return wq, wk, wv, wo, bq, bk, bv, bo


def bind_attention_variables_to_layer(layer, wq, wk, wv, wo, bq, bk, bv, bo):
    assert layer._wq.shape == wq.shape
    assert layer._wk.shape == wk.shape
    assert layer._wv.shape == wv.shape
    assert layer._wo.shape == wo.shape

    assert layer._bq.shape == bq.shape
    assert layer._bk.shape == bk.shape
    assert layer._bv.shape == bv.shape
    assert layer._bo.shape == bo.shape

    layer._wq = wq
    layer._wk = wk
    layer._wv = wv
    layer._wo = wo
    layer._bq = bq
    layer._bk = bk
    layer._bv = bv
    layer._bo = bo
