import math

import intent
import intent.language as I

from kernels.streaming.attention import empty_attention_summary
from kernels.streaming.attention import merge_attention_summaries
from kernels.streaming.attention import summarize_masked_attention_chunk


BATCH = 8
QUERY_HEADS = 32
KV_HEADS = 8
HEAD_GROUP = QUERY_HEADS // KV_HEADS
SEQUENCE = 4096
HEAD_DIMENSION = 128
BLOCK_SIZE = 64
SELECTED_BLOCKS = 32
SPLITS = 4
SCALE = 1.0 / math.sqrt(HEAD_DIMENSION)


@intent.kernel
def block_sparse_gqa_decode_partials(
    q: I.In[I.f16, ("B", "HQ", "D")],
    k: I.In[I.f16, ("B", "K", "HK", "D")],
    v: I.In[I.f16, ("B", "K", "HK", "D")],
    block_indices: I.In[I.i32, ("B", "HK", "S")],
    cache_lengths: I.In[I.i32, ("B",)],
    split_offsets: I.In[I.i32, ("P_PLUS_1",)],
    partial_lse: I.Out[I.f32, ("B", "HQ", "SPLITS")],
    partial_output: I.Out[I.f32, ("B", "HQ", "SPLITS", "D")],
    scale: I.f32,
    HEAD_GROUP: I.Constexpr[int],
    BLOCK_SIZE: I.Constexpr[int],
    SPLITS: I.Constexpr[int],
):
    B, HQ, D = q.shape
    KV_HEADS = k.shape[2]
    K = k.shape[1]
    key_members = I.domain(0, K)
    local_query_head_axis = I.domain(0, HEAD_GROUP)
    for batch in I.parallel(I.domain(0, B)):
        for key_head in I.parallel(I.domain(0, KV_HEADS)):
            I.assume_in_bounds(key_head, block_indices, axis=1)
            query_head_indices = (
                key_head * HEAD_GROUP + I.indices(local_query_head_axis)
            )
            I.assume_in_bounds(query_head_indices, q, axis=1)
            query = I.gather(
                q,
                index=(batch, query_head_indices, slice(None)),
            )
            for split in I.parallel(I.domain(0, SPLITS)):
                selected_begin = I.cast(split_offsets[split], I.index)
                selected_count = I.cast(
                    split_offsets[split + 1] - split_offsets[split], I.index
                )
                summary = empty_attention_summary(local_query_head_axis, D)
                for selected_ordinal in I.domain(0, selected_count):
                    selected_index = selected_begin + selected_ordinal
                    I.assume_in_bounds(selected_index, block_indices, axis=2)
                    selected_block = I.cast(
                        block_indices[batch, key_head, selected_index],
                        I.index,
                    )
                    block_valid = selected_block >= 0
                    safe_block = I.select(
                        block_valid,
                        selected_block,
                        I.cast(0, I.index),
                    )
                    selected_block_begin = safe_block * BLOCK_SIZE
                    selected_block_end = selected_block_begin + BLOCK_SIZE
                    I.assume_in_bounds(selected_block_end - 1, k, axis=1)
                    I.assume_in_bounds(selected_block_end - 1, v, axis=1)
                    key_region = key_members[
                        selected_block_begin:selected_block_end
                    ]
                    token_indices = I.indices(key_region)
                    active = block_valid & (
                        token_indices < I.cast(cache_lengths[batch], I.index)
                    )
                    block_summary = summarize_masked_attention_chunk(
                        k[batch, key_region, key_head, :],
                        v[batch, key_region, key_head, :],
                        token_indices,
                        active,
                        query,
                        query_head_indices,
                        scale,
                    )
                    summary = merge_attention_summaries(summary, block_summary)
                safe_denominator = I.select(
                    summary.valid,
                    summary.denominator,
                    1.0,
                )
                inverse_denominator = 1.0 / safe_denominator
                I.scatter_unique(
                    partial_lse,
                    index=(batch, query_head_indices, split),
                    value=I.select(
                        summary.valid,
                        summary.maximum + I.log(safe_denominator) * I.LOG2E,
                        -I.inf,
                    ),
                )
                I.scatter_unique(
                    partial_output,
                    index=(
                        batch,
                        query_head_indices,
                        split,
                        slice(None),
                    ),
                    value=I.select(
                        summary.valid[:, None],
                        summary.accumulator * inverse_denominator[:, None],
                        0.0,
                    ),
                )


@intent.kernel
def block_sparse_gqa_decode_combine(
    partial_lse: I.In[I.f32, ("B", "HQ", "SPLITS")],
    partial_output: I.In[I.f32, ("B", "HQ", "SPLITS", "D")],
    output: I.Out[I.f16, ("B", "HQ", "D")],
    SPLITS: I.Constexpr[int],
):
    B, HQ, P, D = partial_output.shape
    splits = I.domain(0, P)
    dimensions = I.domain(0, D)
    for batch in I.parallel(I.domain(0, B)):
        for query_head in I.parallel(I.domain(0, HQ)):
            lse = partial_lse[batch, query_head, splits]
            maximum = I.reduce.max(lse, axis=0)
            safe_maximum = I.mask(
                maximum,
                valid=maximum != -I.inf,
                fill=0.0,
            )
            weights = I.exp2(lse - safe_maximum)
            denominator = I.reduce.sum(
                weights, axis=0
            )
            safe_denominator = I.mask(
                denominator,
                valid=denominator > 0.0,
                fill=1.0,
            )
            numerator = I.reshape(
                I.reduce.sum(
                    weights[:, None]
                    * partial_output[
                        batch, query_head, splits, dimensions
                    ],
                    axis=0,
                ),
                (D,),
            )
            output[batch, query_head, dimensions] = I.cast(
                numerator * (1.0 / safe_denominator), I.f16
            )
