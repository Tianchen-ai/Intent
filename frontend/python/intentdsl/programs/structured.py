"""Ordered and neighborhood computation author programs (pure Python/NumPy)."""

import numpy as np

from . import inputs
from .constants import SIZE
from .specs import KernelSpec


def causal_convolution(context):
    kernel = context.compile(KernelSpec("causal_convolution"), constexprs={"SILU": True})
    x = inputs.normal((2, 16, 256), "f16")
    weight = inputs.normal((16, 4), "f16")
    bias = inputs.normal((16,), "f16")
    return context.call(kernel, x, weight, bias)


def cholesky(context):
    kernel = context.compile(KernelSpec("cholesky", mutable_inputs=(0,)))
    basis = inputs.normal((16, SIZE, SIZE), "f32").data
    matrices = inputs.array(basis @ np.swapaxes(basis, -1, -2) +
                            np.eye(SIZE, dtype=np.float32), "f32")
    context.call(kernel, matrices)
    return matrices
