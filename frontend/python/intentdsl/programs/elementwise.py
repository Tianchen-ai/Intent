"""Elementwise and mutable-state author programs (pure Python/NumPy)."""

import numpy as np

from . import inputs
from .constants import HEAD_DIMENSION, ROPE_KEY_HEADS, ROPE_QUERY_HEADS
from .specs import KernelSpec


def relu(context):
    kernel = context.compile(KernelSpec("relu"))
    x = inputs.normal((64, 1024), "f16")
    return context.call(kernel, x)


def swiglu(context):
    kernel = context.compile(KernelSpec("swiglu"))
    gate = inputs.normal((64, 1024), "bf16")
    up = inputs.normal(gate.shape, gate.dtype)
    return context.call(kernel, gate, up)


def rope(context):
    kernel = context.compile(KernelSpec("rope", mutable_inputs=(0, 1)))
    query = inputs.normal((2, ROPE_QUERY_HEADS, 128, HEAD_DIMENSION), "f16")
    key = inputs.normal((2, ROPE_KEY_HEADS, 128, HEAD_DIMENSION), "f16")
    phase = inputs.normal((1, 128, HEAD_DIMENSION), "f16").to_numpy("f32")
    context.call(kernel, query, key, inputs.array(np.cos(phase), "f16"),
                 inputs.array(np.sin(phase), "f16"))
    return query, key


def transpose(context):
    kernel = context.compile(KernelSpec("transpose"))
    x = inputs.normal((127, 257), "f16")
    return context.call(kernel, x)


def dropout(context):
    kernel = context.compile(KernelSpec("dropout"))
    x = inputs.normal((64, 1024), "f16")
    probability = 0.1
    return context.call(kernel, x, 12345, probability, 1.0 / (1.0 - probability))


def adamw(context):
    kernel = context.compile(KernelSpec("adamw", mutable_inputs=(1, 2, 3)))
    gradient = inputs.normal((65536,), "f32")
    parameter = inputs.normal(gradient.shape, gradient.dtype)
    first_moment = inputs.zeros(gradient.shape, gradient.dtype)
    second_moment = inputs.zeros(gradient.shape, gradient.dtype)
    beta1, beta2 = 0.9, 0.999
    context.call(kernel, gradient, parameter, first_moment, second_moment,
                 1.0e-3, beta1, beta2, 1.0 - beta1, 1.0 - beta2, 1.0e-8, 0.01)
    return parameter, first_moment, second_moment
