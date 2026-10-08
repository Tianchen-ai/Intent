import math

import intent
import intent.language as I

from kernels.streaming.attention import empty_attention_summary
from kernels.streaming.attention import merge_attention_summaries
from kernels.streaming.attention import normalize_attention_summary
from kernels.streaming.attention import summarize_attention_chunk_f16
from kernels.streaming.attention import summarize_masked_attention_chunk


BATCH = 8
QUERY_HEADS = 32
KV_HEADS = 8
HEAD_GROUP = QUERY_HEADS // KV_HEADS
PAGE_SIZE = 64
HEAD_DIMENSION = 128
SEQUENCE_LENGTHS = (8191, 7937, 7683, 7429, 7175, 6921, 6667, 6413)
SCALE = 1.0 / math.sqrt(HEAD_DIMENSION)
SPLITS = 16


@intent.kernel
def paged_gqa_decode_attention(
    q: I.In[I.f16, ("B", "HQ", "D")],
    key_cache: I.In[I.f16, ("P", "PS", "HK", "D")],
    value_cache: I.In[I.f16, ("P", "PS", "HK", "DV")],
    page_offsets: I.In[I.i32, ("B_PLUS_1",)],
    page_indices: I.In[I.i32, ("S",)],
    sequence_lengths: I.In[I.i32, ("B",)],
    output: I.Out[I.f16, ("B", "HQ", "DV")],
    scale: I.f32,
    PAGE_SIZE: I.Constexpr[int],
    HEAD_GROUP: I.Constexpr[int],
):
    B, HQ, D = q.shape
    DV = value_cache.shape[3]
    for batch in I.parallel(I.domain(0, B)):
        sequence_length = I.cast(sequence_lengths[batch], I.index)
        logical_tokens = I.domain(0, sequence_length)
        token_coordinates = I.indices(logical_tokens)
        page_slot = I.cast(page_offsets[batch], I.index) + (
            token_coordinates // PAGE_SIZE
        )
        I.assume_in_bounds(page_slot, page_indices, axis=0)
        physical_page = I.cast(page_indices[page_slot], I.index)
        I.assume_in_bounds(physical_page, key_cache, axis=0)
        I.assume_in_bounds(physical_page, value_cache, axis=0)
        page_offset = token_coordinates % PAGE_SIZE
        for query_head in I.parallel(I.domain(0, HQ)):
            key_head = query_head // HEAD_GROUP
            query = I.reshape(q[batch, query_head, :], (1, D))
            summary = I.region_fold(
                source=(
                    key_cache[physical_page, page_offset, key_head, :],
                    value_cache[physical_page, page_offset, key_head, :],
                    token_coordinates,
                ),
                axis=0,
                summarize=summarize_attention_chunk_f16,
                combine=merge_attention_summaries,
                identity=empty_attention_summary(1, DV),
                operands=(
                    query,
                    I.full((1,), fill=0, dtype=I.index),
                    scale,
                    False,
                ),
            )
            output[batch, query_head, :] = I.reshape(
                I.cast(normalize_attention_summary(summary), I.f16),
                (DV,),
            )


@intent.kernel
def paged_gqa_decode_partials(
    q: I.In[I.f16, ("B", "HQ", "D")],
    key_cache: I.In[I.f16, ("P", "PS", "HK", "D")],
    value_cache: I.In[I.f16, ("P", "PS", "HK", "DV")],
    page_offsets: I.In[I.i32, ("B_PLUS_1",)],
    page_indices: I.In[I.i32, ("S",)],
    sequence_lengths: I.In[I.i32, ("B",)],
    split_offsets: I.In[I.i32, ("SP_PLUS_1",)],
    partial_lse: I.Out[I.f32, ("B", "HQ", "SPLITS")],
    partial_output: I.Out[I.f32, ("B", "HQ", "SPLITS", "DV")],
    scale: I.f32,
    PAGE_SIZE: I.Constexpr[int],
    HEAD_GROUP: I.Constexpr[int],
    SPLITS: I.Constexpr[int],
):
    B, HQ, D = q.shape
    PS = key_cache.shape[1]
    HK = key_cache.shape[2]
    DV = value_cache.shape[3]
    jobs = I.domain(0, B * SPLITS)
    local_query_heads = I.domain(0, HEAD_GROUP)
    for job in I.parallel(jobs):
        batch = job // SPLITS
        split = job % SPLITS
        page_begin = I.cast(page_offsets[batch], I.index)
        sequence_length = I.cast(sequence_lengths[batch], I.index)
        selected_begin = I.cast(split_offsets[job], I.index)
        page_count = I.cast(
            split_offsets[job + 1] - split_offsets[job], I.index
        )
        token_count = page_count * PS
        flat_tokens = I.domain(0, token_count)
        flat_token = I.indices(flat_tokens)
        page_ordinal = flat_token // PS
        token_offset = flat_token % PS
        page_position = selected_begin + page_ordinal
        I.assume_in_bounds(page_position, page_indices, axis=0)
        physical_page = I.cast(page_indices[page_position], I.index)
        I.assume_in_bounds(physical_page, key_cache, axis=0)
        I.assume_in_bounds(physical_page, value_cache, axis=0)
        logical_token = (
            (page_position - page_begin) * PAGE_SIZE + token_offset
        )
        active = logical_token < sequence_length
        for key_head in I.parallel(I.domain(0, HK)):
            query_heads = key_head * HEAD_GROUP + I.indices(local_query_heads)
            query = I.gather(q, index=(batch, query_heads, slice(None)))
            summary = I.region_fold(
                source=(
                    key_cache[physical_page, token_offset, key_head, :],
                    value_cache[physical_page, token_offset, key_head, :],
                    logical_token,
                    active,
                ),
                axis=0,
                summarize=summarize_masked_attention_chunk,
                combine=merge_attention_summaries,
                identity=empty_attention_summary(local_query_heads, DV),
                operands=(
                    query,
                    I.full((HEAD_GROUP,), fill=0, dtype=I.index),
                    scale,
                ),
            )
            safe_denominator = I.select(
                summary.valid,
                summary.denominator,
                1.0,
            )
            inverse_denominator = 1.0 / safe_denominator
            I.scatter_unique(
                partial_lse,
                index=(batch, query_heads, split),
                value=I.select(
                    summary.valid,
                    summary.maximum + I.log(safe_denominator) * I.LOG2E,
                    -I.inf,
                ),
            )
            I.scatter_unique(
                partial_output,
                index=(batch, query_heads, split, slice(None)),
                value=I.select(
                    summary.valid[:, None],
                    summary.accumulator * inverse_denominator[:, None],
                    0.0,
                ),
            )


@intent.kernel
def splitk_paged_gqa_decode_partials(
    q: I.In[I.f16, ("B", "HQ", "D")],
    key_cache: I.In[I.f16, ("P", "PS", "HK", "D")],
    value_cache: I.In[I.f16, ("P", "PS", "HK", "DV")],
    page_offsets: I.In[I.i32, ("B_PLUS_1",)],
    page_indices: I.In[I.i32, ("S",)],
    sequence_lengths: I.In[I.i32, ("B",)],
    split_offsets: I.In[I.i32, ("SP_PLUS_1",)],
    partial_lse: I.Out[I.f32, ("B", "HQ", "SPLITS")],
    partial_output: I.Out[I.bf16, ("B", "HQ", "SPLITS", "DV")],
    scale: I.f32,
    PAGE_SIZE: I.Constexpr[int],
    HEAD_GROUP: I.Constexpr[int],
    SPLITS: I.Constexpr[int],
):
    B, HQ, D = q.shape
    PS = key_cache.shape[1]
    HK = key_cache.shape[2]
    DV = value_cache.shape[3]
    jobs = I.domain(0, B * SPLITS)
    local_query_heads = I.domain(0, HEAD_GROUP)
    for job in I.parallel(jobs):
        batch = job // SPLITS
        split = job % SPLITS
        page_begin = I.cast(page_offsets[batch], I.index)
        sequence_length = I.cast(sequence_lengths[batch], I.index)
        selected_begin = I.cast(split_offsets[job], I.index)
        page_count = I.cast(
            split_offsets[job + 1] - split_offsets[job], I.index
        )
        token_count = page_count * PS
        flat_tokens = I.domain(0, token_count)
        flat_token = I.indices(flat_tokens)
        page_ordinal = flat_token // PS
        token_offset = flat_token % PS
        page_position = selected_begin + page_ordinal
        I.assume_in_bounds(page_position, page_indices, axis=0)
        physical_page = I.cast(page_indices[page_position], I.index)
        I.assume_in_bounds(physical_page, key_cache, axis=0)
        I.assume_in_bounds(physical_page, value_cache, axis=0)
        logical_token = (
            (page_position - page_begin) * PAGE_SIZE + token_offset
        )
        active = logical_token < sequence_length
        for key_head in I.parallel(I.domain(0, HK)):
            query_heads = key_head * HEAD_GROUP + I.indices(local_query_heads)
            query = I.gather(
                q,
                index=(batch, query_heads, slice(None)),
            )
            keys = key_cache[
                physical_page,
                token_offset,
                key_head,
                :,
            ]
            values = value_cache[
                physical_page,
                token_offset,
                key_head,
                :,
            ]
            summary = I.region_fold(
                source=(
                    keys,
                    values,
                    logical_token,
                    active,
                ),
                axis=0,
                summarize=summarize_masked_attention_chunk,
                combine=merge_attention_summaries,
                identity=empty_attention_summary(local_query_heads, DV),
                operands=(
                    query,
                    I.full((HEAD_GROUP,), fill=0, dtype=I.index),
                    scale,
                ),
            )
            safe_denominator = I.select(
                summary.valid,
                summary.denominator,
                1.0,
            )
            I.scatter_unique(
                partial_lse,
                index=(batch, query_heads, split),
                value=I.select(
                    summary.valid,
                    summary.maximum + I.log(safe_denominator) * I.LOG2E,
                    -I.inf,
                ),
            )
            I.scatter_unique(
                partial_output,
                index=(batch, query_heads, split, slice(None)),
                value=I.cast(normalize_attention_summary(summary), I.bf16),
            )
