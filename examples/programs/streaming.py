"""Stateful kernels and author-owned partial/merge orchestration."""

import numpy as np

from . import inputs

from kernels.streaming.attention import flash_attention_bf16_fwd
from kernels.streaming.gated_delta import recurrent_gated_delta_fwd
from kernels.streaming.mamba import mamba_chunk_state_bf16_fwd
from kernels.streaming.paged_attention import paged_gqa_decode_partials
from kernels.streaming.splitk_reduce import splitk_attention_f32_to_f16_reduce
from .composition import PagedDecode


def attention(context):
    kernel = context.compile(flash_attention_bf16_fwd,
                             constexprs={"HEAD_GROUP": 4, "CAUSAL": True})
    q = inputs.normal((2, 8, 256, 64), "bf16")
    k = inputs.normal((2, 2, 256, 64), "bf16")
    v = inputs.normal(k.shape, k.dtype)
    return context.call(kernel, q, k, v, 64**-0.5)


def paged_decode(context):
    batch, query_heads, kv_heads, dimension = 2, 8, 2, 64
    sequence, page_size, splits = 512, 16, 8
    pages_per_sequence = sequence // page_size
    pages = batch * pages_per_sequence
    partials = context.compile(paged_gqa_decode_partials, constexprs={
        "PAGE_SIZE": page_size, "HEAD_GROUP": query_heads // kv_heads, "SPLITS": splits})
    merge = context.compile(splitk_attention_f32_to_f16_reduce)
    q = inputs.normal((batch, query_heads, dimension), "f16")
    key_cache = inputs.normal((pages, page_size, kv_heads, dimension), "f16")
    value_cache = inputs.normal(key_cache.shape, key_cache.dtype)
    page_indices = inputs.arange(pages, dtype="i32")
    page_offsets = inputs.arange(0, pages + 1, pages_per_sequence, dtype="i32")
    split_offsets = inputs.arange(0, pages + 1, pages_per_sequence // splits, dtype="i32")
    lengths = inputs.full((batch,), sequence, "i32")
    program = PagedDecode(partials, merge)
    return context.call(program, q, key_cache, value_cache, page_offsets, page_indices,
                        lengths, split_offsets, dimension**-0.5)


def mamba_chunk_state(context):
    kernel = context.compile(mamba_chunk_state_bf16_fwd, constexprs={"HEAD_GROUP": 4})
    batch, chunks, chunk_size, heads, groups = 2, 4, 64, 8, 2
    dimension, state_dimension = 64, 32
    basis = inputs.normal((batch, chunks * chunk_size, groups, state_dimension), "bf16")
    x = inputs.normal((batch, chunks * chunk_size, heads, dimension), "bf16")
    dt = inputs.uniform((batch, heads, chunks, chunk_size), "f32")
    decay_steps = -inputs.uniform(dt.shape, "f32").data * np.float32(0.01)
    cumulative_decay = inputs.array(np.cumsum(decay_steps, axis=-1, dtype=np.float32), "f32")
    return context.call(kernel, basis, x, dt, cumulative_decay)


def gated_delta(context):
    kernel = context.compile(recurrent_gated_delta_fwd, constexprs={"HEAD_GROUP": 2})
    q = inputs.normal((2, 128, 2, 32), "bf16", scale=0.05)
    k = inputs.normal(q.shape, "bf16", scale=0.05)
    v = inputs.normal((2, 128, 4, 32), "bf16")
    gate = inputs.array(-inputs.uniform((2, 128, 4), "f32").data * np.float32(0.01), "bf16")
    beta = inputs.uniform((2, 128, 4), "bf16")
    return context.call(kernel, q, k, v, gate, beta, 32**-0.5)
