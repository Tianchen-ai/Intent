"""Host input preparation for ordered and neighborhood computations."""

import numpy as np

from . import inputs

from kernels.convolution.direct import causal_depthwise_conv1d
from kernels.factorization.cholesky import SIZE, batched_cholesky_lower


def causal_convolution(context):
    kernel = context.compile(causal_depthwise_conv1d, constexprs={"SILU": True})
    x = inputs.normal((2, 16, 256), "f16")
    weight = inputs.normal((16, 4), "f16")
    bias = inputs.normal((16,), "f16")
    return context.call(kernel, x, weight, bias)


def cholesky(context):
    kernel = context.compile(batched_cholesky_lower)
    basis = inputs.normal((16, SIZE, SIZE), "f32").data
    matrices = inputs.array(basis @ np.swapaxes(basis, -1, -2) +
                            np.eye(SIZE, dtype=np.float32), "f32")
    context.call(kernel, matrices)
    return matrices
