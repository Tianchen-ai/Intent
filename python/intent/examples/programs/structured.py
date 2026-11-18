"""Host input preparation for ordered and neighborhood computations."""

from dataclasses import dataclass

import numpy as np

from . import inputs

from intent.examples.kernels.convolution.direct import causal_depthwise_conv1d
from intent.examples.kernels.convolution.varlen import (
    varlen_aligned_causal_depthwise_conv1d, varlen_causal_conv1d_final_state,
)
from intent.examples.kernels.factorization.cholesky import SIZE, batched_cholesky_lower
from intent.runtime import PreparedSequence


@dataclass
class VarlenCausalConvolution:
    forward: object
    final_state: object

    @classmethod
    def from_jit(cls, jit):
        return cls(jit(varlen_aligned_causal_depthwise_conv1d),
                   jit(varlen_causal_conv1d_final_state))

    def prepare(self, x, offsets, chunks, weight, bias):
        forward = self.forward.prepare(x, offsets, chunks, weight, bias)
        state = self.final_state.prepare(x, offsets)
        return PreparedSequence((forward, state), outputs=(forward.result(), state.result()))

    def run(self, *arguments):
        call = self.prepare(*arguments)
        call.launch()
        return call.result()


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
