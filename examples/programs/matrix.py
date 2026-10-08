"""Contraction inputs, accumulators and storage formats are explicit."""

import numpy as np

from . import inputs

from kernels.contraction.batched_gemm import batched_gemm_nn
from kernels.contraction.block_scaled import block_scaled_matmul
from kernels.contraction.gemm import Activation, gemm as gemm_definition, gemm_i8
from kernels.quantization.quantized_projection import quantized_projection
from kernels.ragged.grouped_gemm import ragged_grouped_gemm


def gemm(context):
    kernel = context.compile(gemm_definition, constexprs={"ACTIVATION": Activation.NONE})
    a = inputs.normal((128, 256), "f16")
    b = inputs.normal((256, 192), "f16")
    return context.call(kernel, a, b)


def batched_gemm(context):
    kernel = context.compile(batched_gemm_nn)
    a = inputs.normal((4, 128, 256), "bf16")
    b = inputs.normal((4, 256, 192), "bf16")
    return context.call(kernel, a, b)


def ragged_gemm(context):
    kernel = context.compile(ragged_grouped_gemm)
    lengths = np.array([17, 31, 48, 32], dtype=np.int32)
    offsets = inputs.array(np.concatenate((np.zeros(1, dtype=np.int32),
                                           np.cumsum(lengths, dtype=np.int32))), "i32")
    x = inputs.normal((128, 256), "f16")
    weight = inputs.normal((4, 256, 192), "f16")
    return context.call(kernel, x, offsets, weight)


def int8_gemm(context):
    kernel = context.compile(gemm_i8)
    a = inputs.integers(-16, 16, (128, 256), "i8")
    b = inputs.integers(-16, 16, (256, 192), "i8")
    bias = inputs.arange(192, dtype="i32")
    return context.call(kernel, a, b, bias)


def q4_projection(context):
    kernel = context.compile(quantized_projection)
    # A zero-filled 144-byte Q4_K record has zero scales and represents zero.
    # Real callers supply their model's existing Q4_K records in this layout.
    weights = inputs.zeros((128, 4, 144), "u8")
    activation = inputs.normal((4, 256), "f32")
    return context.call(kernel, weights, activation)


def fp8_gemm(context):
    kernel = context.compile(block_scaled_matmul)
    lhs = inputs.normal((128, 8, 32), "f8e4m3fn")
    rhs = inputs.normal((8, 32, 128), "f8e4m3fn")
    lhs_scale = inputs.full((128, 8), 127, "u8")
    rhs_scale = inputs.full((8, 128), 127, "u8")
    return context.call(kernel, lhs, lhs_scale, rhs, rhs_scale)
