"""Statistics, prefixes and structured state use the original algorithms."""

import numpy as np

from . import inputs

from kernels.backward.group_norm import (
    BATCH, CHANNELS, SPATIAL, GROUPS, CHANNELS_PER_GROUP,
    group_norm_backward_dx, group_norm_backward_weight_bias,
)
from kernels.normalization.batch_norm import batch_norm_training
from kernels.normalization.layer_norm import weighted_layer_norm
from kernels.normalization.softmax import stable_softmax_f16
from kernels.statistics.histogram import histogram_256
from kernels.streaming.attention_f32 import causal_linear_attention_f32
from kernels.streaming.ordered_prefix import row_cumsum_f32
from .composition import GroupNormBackward


def softmax(context):
    kernel = context.compile(stable_softmax_f16)
    x = inputs.normal((64, 1024), "f16")
    return context.call(kernel, x)


def layer_norm(context):
    kernel = context.compile(weighted_layer_norm)
    x = inputs.normal((64, 1024), "f32")
    weight = inputs.normal((1024,), "f32")
    bias = inputs.normal(weight.shape, weight.dtype)
    return context.call(kernel, x, weight, bias, 1.0 / x.shape[1], 1.0e-5)


def batch_norm(context):
    kernel = context.compile(batch_norm_training)
    x = inputs.normal((4, 16, 128), "f16")
    weight = inputs.ones((16,), "f32")
    bias = inputs.zeros(weight.shape, weight.dtype)
    running_mean = inputs.zeros(weight.shape, weight.dtype)
    running_variance = inputs.ones(weight.shape, weight.dtype)
    output, saved_mean, saved_rstd = context.call(
        kernel, x, weight, bias, running_mean, running_variance, 1.0e-5, 0.1)
    return output, saved_mean, saved_rstd, running_mean, running_variance


def group_norm_backward(context):
    dx = context.compile(group_norm_backward_dx)
    affine = context.compile(group_norm_backward_weight_bias)
    x = inputs.normal((BATCH, CHANNELS, SPATIAL), "f16", scale=0.5)
    grad_y = inputs.normal(x.shape, "f16", scale=0.05)
    weight = inputs.normal((CHANNELS,), "f16")
    grouped = x.to_numpy("f32").reshape(BATCH, GROUPS, CHANNELS_PER_GROUP, SPATIAL)
    mean = inputs.array(grouped.mean(axis=(2, 3), dtype=np.float32), "f16")
    centered = grouped - mean.to_numpy("f32")[:, :, None, None]
    variance = np.square(centered).mean(axis=(2, 3), dtype=np.float32)
    rstd = inputs.array(np.reciprocal(np.sqrt(variance + np.float32(1.0e-5))), "f16")
    program = GroupNormBackward(dx, affine)
    return context.call(program, x, grad_y, weight, mean, rstd,
                        1.0 / (CHANNELS_PER_GROUP * SPATIAL))


def cumsum(context):
    kernel = context.compile(row_cumsum_f32)
    x = inputs.normal((64, 1024), "f32")
    return context.call(kernel, x)


def causal_linear_attention(context):
    kernel = context.compile(causal_linear_attention_f32)
    q = inputs.normal((2, 256, 32), "f32", scale=0.1)
    k = inputs.normal(q.shape, "f32", scale=0.1)
    v = inputs.normal((2, 256, 64), "f32")
    return context.call(kernel, q, k, v)


def histogram(context):
    kernel = context.compile(histogram_256)
    samples = inputs.array(inputs.integers(-16, 272, (65536,), "i64").data, "f32")
    return context.call(kernel, samples)
