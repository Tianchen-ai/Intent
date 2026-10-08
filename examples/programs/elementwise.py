"""Elementwise and mutable-state examples; target is selected by the host."""

import numpy as np

from . import inputs

from kernels.activation.pointwise import relu_forward
from kernels.activation.swiglu import swiglu_forward
from kernels.layout.transpose import matrix_transpose
from kernels.optimization.adamw import adamw_update
from kernels.position.rope import HEAD_DIMENSION, ROPE_QUERY_HEADS, ROPE_KEY_HEADS, rotary_qk_inplace
from kernels.regularization.dropout import xor_shift_dropout


def relu(context):
    kernel = context.compile(relu_forward)
    x = inputs.normal((64, 1024), "f16")
    return context.call(kernel, x)


def swiglu(context):
    kernel = context.compile(swiglu_forward)
    gate = inputs.normal((64, 1024), "bf16")
    up = inputs.normal(gate.shape, gate.dtype)
    return context.call(kernel, gate, up)


def rope(context):
    kernel = context.compile(rotary_qk_inplace)
    query = inputs.normal((2, ROPE_QUERY_HEADS, 128, HEAD_DIMENSION), "f16")
    key = inputs.normal((2, ROPE_KEY_HEADS, 128, HEAD_DIMENSION), "f16")
    phase = inputs.normal((1, 128, HEAD_DIMENSION), "f16").to_numpy("f32")
    context.call(kernel, query, key, inputs.array(np.cos(phase), "f16"),
                 inputs.array(np.sin(phase), "f16"))
    return query, key


def transpose(context):
    kernel = context.compile(matrix_transpose)
    x = inputs.normal((127, 257), "f16")
    return context.call(kernel, x)


def dropout(context):
    kernel = context.compile(xor_shift_dropout)
    x = inputs.normal((64, 1024), "f16")
    probability = 0.1
    return context.call(kernel, x, 12345, probability, 1.0 / (1.0 - probability))


def adamw(context):
    kernel = context.compile(adamw_update)
    gradient = inputs.normal((65536,), "f32")
    parameter = inputs.normal(gradient.shape, gradient.dtype)
    first_moment = inputs.zeros(gradient.shape, gradient.dtype)
    second_moment = inputs.zeros(gradient.shape, gradient.dtype)
    beta1, beta2 = 0.9, 0.999
    context.call(kernel, gradient, parameter, first_moment, second_moment,
                 1.0e-3, beta1, beta2, 1.0 - beta1, 1.0 - beta2, 1.0e-8, 0.01)
    return parameter, first_moment, second_moment
