import math

import intent
import intent.language as I

from kernels.streaming.attention import empty_attention_summary
from kernels.streaming.attention import merge_attention_summaries
from kernels.streaming.attention import normalize_attention_summary
from kernels.streaming.attention import reduce_score_maximum


ABSORBED_MLA_BATCH = 1
ABSORBED_MLA_SEQUENCE = 512
ABSORBED_MLA_HEADS = 8
ABSORBED_MLA_NOPE_DIMENSION = 128
ABSORBED_MLA_ROPE_DIMENSION = 64
ABSORBED_MLA_LATENT_DIMENSION = 512
ABSORBED_MLA_VALUE_DIMENSION = 128
ABSORBED_MLA_SCALE = 1.0 / math.sqrt(
    ABSORBED_MLA_NOPE_DIMENSION + ABSORBED_MLA_ROPE_DIMENSION
)
SPARSE_MLA_QUERIES = 64
SPARSE_MLA_HEADS = 8
SPARSE_MLA_KEYS = 4096
SPARSE_MLA_SELECTED_KEYS = 256
SPARSE_MLA_LATENT_DIMENSION = 512
SPARSE_MLA_ROPE_DIMENSION = 64
SPARSE_MLA_SCALE = 1.0 / math.sqrt(
    SPARSE_MLA_LATENT_DIMENSION + SPARSE_MLA_ROPE_DIMENSION
)
PAGED_MLA_BATCH = 8
PAGED_MLA_QUERY_HEADS = 32
PAGED_MLA_KV_HEADS = 1
PAGED_MLA_HEAD_GROUP = PAGED_MLA_QUERY_HEADS // PAGED_MLA_KV_HEADS
PAGED_MLA_PAGE_SIZE = 64
PAGED_MLA_LATENT_DIMENSION = 128
PAGED_MLA_ROPE_DIMENSION = 64
PAGED_MLA_SEQUENCE_LENGTHS = (4096, 4093, 4087, 4081, 4079, 4073, 4069, 4063)
PAGED_MLA_SCALE = 1.0 / math.sqrt(
    PAGED_MLA_LATENT_DIMENSION + PAGED_MLA_ROPE_DIMENSION
)
MLA_DECODE_BATCH = 8
MLA_DECODE_HEADS = 64
MLA_DECODE_SEQUENCE = 8192
MLA_DECODE_LATENT_DIMENSION = 512
MLA_DECODE_ROPE_DIMENSION = 64
MLA_DECODE_SPLITS = 16


@intent.fn
def merge_natural_attention_summaries(lhs, rhs):
    valid = lhs.valid | rhs.valid
    maximum = I.select(lhs.valid, lhs.maximum, rhs.maximum)
    maximum = I.select(
        rhs.valid,
        I.maximum_num(maximum, rhs.maximum),
        maximum,
    )
    lhs_maximum = I.select(lhs.valid, lhs.maximum, maximum)
    rhs_maximum = I.select(rhs.valid, rhs.maximum, maximum)
    lhs_scale = I.select(
        lhs.valid,
        I.exp(lhs_maximum - maximum),
        0.0,
    )
    rhs_scale = I.select(
        rhs.valid,
        I.exp(rhs_maximum - maximum),
        0.0,
    )
    return I.record(
        valid=valid,
        maximum=maximum,
        denominator=(
            lhs_scale * lhs.denominator
            + rhs_scale * rhs.denominator
        ),
        accumulator=(
            lhs_scale[:, None] * lhs.accumulator
            + rhs_scale[:, None] * rhs.accumulator
        ),
    )


@intent.fn
def summarize_mla_chunk(
    latent_chunk,
    rope_chunk,
    value_chunk,
    key_coordinates,
    latent_query,
    rope_query,
    query_coordinates,
    scale,
    causal,
):
    scores = (
        I.matmul(
            latent_chunk,
            latent_query,
            transpose_rhs=True,
            acc_dtype=I.f32,
        )
        + I.matmul(
            rope_chunk,
            rope_query,
            transpose_rhs=True,
            acc_dtype=I.f32,
        )
    ) * (scale * I.LOG2E)
    valid = I.full(scores.shape, fill=True, dtype=I.bool)
    if causal:
        valid = query_coordinates[None, :] >= key_coordinates[:, None]
    scores = I.select(valid, scores, -I.inf)
    chunk_valid = I.reduce.any(valid, axis=0)
    maximum = I.select(
        chunk_valid,
        reduce_score_maximum(scores, axis=0),
        0.0,
    )
    probability = I.select(
        valid,
        I.exp2(scores - maximum[None, :], approximate=True, flush_to_zero=True),
        0.0,
    )
    return I.record(
        valid=chunk_valid,
        maximum=maximum,
        denominator=I.reduce.sum(probability, axis=0),
        accumulator=I.transpose(
            I.matmul(
                value_chunk,
                I.cast(probability, I.f16),
                transpose_lhs=True,
                acc_dtype=I.f32,
            ),
            (1, 0),
        ),
    )


@intent.fn
def summarize_masked_mla_chunk(
    latent_chunk,
    rope_chunk,
    value_chunk,
    key_coordinates,
    active,
    latent_query,
    rope_query,
    query_coordinates,
    scale,
):
    scores = (
        I.matmul(
            latent_chunk,
            latent_query,
            transpose_rhs=True,
            acc_dtype=I.f32,
        )
        + I.matmul(
            rope_chunk,
            rope_query,
            transpose_rhs=True,
            acc_dtype=I.f32,
        )
    ) * (scale * I.LOG2E)
    valid = I.full(scores.shape, fill=True, dtype=I.bool) & active[:, None]
    scores = I.select(valid, scores, -I.inf)
    chunk_valid = I.reduce.any(valid, axis=0)
    maximum = I.select(
        chunk_valid,
        reduce_score_maximum(scores, axis=0),
        0.0,
    )
    probability = I.select(
        valid,
        I.exp2(scores - maximum[None, :], approximate=True, flush_to_zero=True),
        0.0,
    )
    return I.record(
        valid=chunk_valid,
        maximum=maximum,
        denominator=I.reduce.sum(probability, axis=0),
        accumulator=I.transpose(
            I.matmul(
                value_chunk,
                I.cast(probability, I.f16),
                transpose_lhs=True,
                acc_dtype=I.f32,
            ),
            (1, 0),
        ),
    )


@intent.fn
def summarize_masked_mla_chunk_natural(
    latent_chunk,
    rope_chunk,
    value_chunk,
    key_coordinates,
    active,
    latent_query,
    rope_query,
    query_coordinates,
    scale,
):
    scores = (
        I.matmul(
            latent_query,
            latent_chunk,
            transpose_rhs=True,
            acc_dtype=I.f32,
        )
        + I.matmul(
            rope_query,
            rope_chunk,
            transpose_rhs=True,
            acc_dtype=I.f32,
        )
    ) * scale
    valid = I.full(scores.shape, fill=True, dtype=I.bool) & active[None, :]
    scores = I.select(valid, scores, -I.inf)
    chunk_valid = I.reduce.any(valid, axis=1)
    maximum = I.select(
        chunk_valid,
        reduce_score_maximum(scores, axis=1),
        0.0,
    )
    probability = I.select(
        valid,
        I.exp(scores - maximum[:, None]),
        0.0,
    )
    return I.record(
        valid=chunk_valid,
        maximum=maximum,
        denominator=I.reduce.sum(probability, axis=1),
        accumulator=I.matmul(
            I.cast(probability, I.f16),
            value_chunk,
            acc_dtype=I.f32,
        ),
    )


@intent.fn
def sparse_mla_summary(
    latent_query,
    rope_query,
    latent_keys,
    rope_keys,
    values,
    active,
    scale,
):
    scores = (
        I.matmul(
            latent_query,
            latent_keys,
            transpose_rhs=True,
            acc_dtype=I.f32,
        )
        + I.matmul(
            rope_query,
            rope_keys,
            transpose_rhs=True,
            acc_dtype=I.f32,
        )
    ) * (scale * I.LOG2E)
    valid = I.full(scores.shape, fill=True, dtype=I.bool) & active[:, None, :]
    scores = I.select(valid, scores, -I.inf)
    summary_valid = I.reduce.any(valid, axis=2)
    maximum = I.select(
        summary_valid,
        reduce_score_maximum(scores, axis=2),
        0.0,
    )
    probability = I.select(
        valid,
        I.exp2(scores - maximum[:, :, None], approximate=True, flush_to_zero=True),
        0.0,
    )
    return I.record(
        valid=summary_valid,
        maximum=maximum,
        denominator=I.reduce.sum(probability, axis=2),
        accumulator=I.matmul(
            I.cast(probability, I.bf16),
            values,
            acc_dtype=I.f32,
        ),
    )


@intent.kernel
def absorbed_mla_prefill(
    q_latent: I.In[I.f16, ("B", "Q", "H", "C")],
    q_rope: I.In[I.f16, ("B", "Q", "H", "DR")],
    latent_cache: I.In[I.f16, ("B", "K", "C")],
    rope_cache: I.In[I.f16, ("B", "K", "DR")],
    output: I.Out[I.f16, ("B", "Q", "H", "C")],
    scale: I.f32,
):
    B, Q, H, C = q_latent.shape
    K = latent_cache.shape[1]
    query_axis = I.domain(0, Q)
    key_axis = I.domain(0, K)
    for batch in I.parallel(I.domain(0, B)):
        for head in I.parallel(I.domain(0, H)):
            summary = I.region_fold(
                source=(
                    latent_cache[batch, key_axis, :],
                    rope_cache[batch, key_axis, :],
                    latent_cache[batch, key_axis, :],
                    I.indices(key_axis),
                ),
                axis=0,
                summarize=summarize_mla_chunk,
                combine=merge_attention_summaries,
                identity=empty_attention_summary(Q, C),
                operands=(
                    q_latent[batch, query_axis, head, :],
                    q_rope[batch, query_axis, head, :],
                    I.indices(query_axis),
                    scale,
                    True,
                ),
            )
            output[batch, query_axis, head, :] = I.cast(
                normalize_attention_summary(summary),
                I.f16,
            )


@intent.kernel
def token_sparse_mla_prefill(
    q_latent: I.In[I.bf16, ("Q", "H", "C")],
    q_rope: I.In[I.bf16, ("Q", "H", "DR")],
    latent_cache: I.In[I.bf16, ("K", "C")],
    rope_cache: I.In[I.bf16, ("K", "DR")],
    selected_tokens: I.In[I.i32, ("Q", "T")],
    output: I.Out[I.bf16, ("Q", "H", "C")],
    maximum_output: I.Out[I.f32, ("Q", "H")],
    lse_output: I.Out[I.f32, ("Q", "H")],
    scale: I.f32,
):
    Q, H, C = q_latent.shape
    K = latent_cache.shape[0]
    T = selected_tokens.shape[1]
    queries = I.domain(0, Q)
    selections = I.domain(0, T)
    token_index = I.cast(selected_tokens[queries, selections], I.index)
    active = (token_index >= 0) & (token_index < K)
    safe_token = I.select(active, token_index, 0)
    summary = sparse_mla_summary(
        q_latent[queries, :, :],
        q_rope[queries, :, :],
        I.gather(latent_cache, index=(safe_token, slice(None))),
        I.gather(rope_cache, index=(safe_token, slice(None))),
        I.gather(latent_cache, index=(safe_token, slice(None))),
        active,
        scale,
    )
    safe_denominator = I.select(summary.valid, summary.denominator, 1.0)
    inverse_denominator = I.fdiv(1.0, safe_denominator, approximate=True, flush_to_zero=True)
    output[queries, :, :] = I.cast(
        I.select(
            summary.valid[:, :, None],
            summary.accumulator * inverse_denominator[:, :, None],
            0.0,
        ),
        I.bf16,
    )
    maximum_output[queries, :] = I.select(summary.valid, summary.maximum, -I.inf)
    lse_output[queries, :] = I.select(
        summary.valid,
        summary.maximum + I.log(safe_denominator) * I.LOG2E,
        -I.inf,
    )


@intent.kernel
def token_sparse_mla_value_prefill(
    query_content: I.In[I.bf16, ("Q", "H", "D")],
    query_rope: I.In[I.bf16, ("Q", "H", "DR")],
    key_content: I.In[I.bf16, ("K", "D")],
    value_cache: I.In[I.bf16, ("K", "DV")],
    key_rope: I.In[I.bf16, ("K", "DR")],
    selected_tokens: I.In[I.i32, ("Q", "T")],
    output: I.Out[I.bf16, ("Q", "H", "DV")],
    scale: I.f32,
):
    Q, H, D = query_content.shape
    K = key_content.shape[0]
    T = selected_tokens.shape[1]
    queries = I.domain(0, Q)
    selections = I.domain(0, T)
    token_index = I.cast(selected_tokens[queries, selections], I.index)
    active = (token_index >= 0) & (token_index < K)
    safe_token = I.select(active, token_index, 0)
    summary = sparse_mla_summary(
        query_content[queries, :, :],
        query_rope[queries, :, :],
        I.gather(key_content, index=(safe_token, slice(None))),
        I.gather(key_rope, index=(safe_token, slice(None))),
        I.gather(value_cache, index=(safe_token, slice(None))),
        active,
        scale,
    )
    safe_denominator = I.select(summary.valid, summary.denominator, 1.0)
    inverse_denominator = I.fdiv(1.0, safe_denominator, approximate=True, flush_to_zero=True)
    output[queries, :, :] = I.cast(
        I.select(
            summary.valid[:, :, None],
            summary.accumulator * inverse_denominator[:, :, None],
            0.0,
        ),
        I.bf16,
    )


@intent.kernel
def paged_mla_decode(
    q_latent: I.In[I.f16, ("B", "HQ", "C")],
    q_rope: I.In[I.f16, ("B", "HQ", "DR")],
    latent_cache: I.In[I.f16, ("P", "PS", "HK", "C")],
    rope_cache: I.In[I.f16, ("P", "PS", "HK", "DR")],
    page_offsets: I.In[I.i32, ("B_PLUS_1",)],
    page_indices: I.In[I.i32, ("S",)],
    sequence_lengths: I.In[I.i32, ("B",)],
    output: I.Out[I.f16, ("B", "HQ", "C")],
    scale: I.f32,
    PAGE_SIZE: I.Constexpr[int],
    HEAD_GROUP: I.Constexpr[int],
):
    B, HQ, C = q_latent.shape
    DV = C
    for batch in I.parallel(I.domain(0, B)):
        sequence_length = I.cast(sequence_lengths[batch], I.index)
        logical_tokens = I.domain(0, sequence_length)
        coordinates = I.indices(logical_tokens)
        page_slot = I.cast(page_offsets[batch], I.index) + coordinates // PAGE_SIZE
        I.assume_in_bounds(page_slot, page_indices, axis=0)
        physical_page = I.cast(page_indices[page_slot], I.index)
        I.assume_in_bounds(physical_page, latent_cache, axis=0)
        I.assume_in_bounds(physical_page, rope_cache, axis=0)
        page_token = coordinates % PAGE_SIZE
        for query_head in I.parallel(I.domain(0, HQ)):
            key_head = query_head // HEAD_GROUP
            latent_values = latent_cache[physical_page, page_token, key_head, :]
            summary = I.region_fold(
                source=(
                    latent_values,
                    rope_cache[physical_page, page_token, key_head, :],
                    latent_values,
                    coordinates,
                ),
                axis=0,
                summarize=summarize_mla_chunk,
                combine=merge_attention_summaries,
                identity=empty_attention_summary(1, DV),
                operands=(
                    I.reshape(q_latent[batch, query_head, :], (1, C)),
                    I.reshape(q_rope[batch, query_head, :], (1, q_rope.shape[2])),
                    I.full((1,), fill=0, dtype=I.index),
                    scale,
                    False,
                ),
            )
            output[batch, query_head, :] = I.reshape(
                I.cast(normalize_attention_summary(summary), I.f16),
                (C,),
            )


@intent.kernel
def paged_mla_decode_partials(
    q_latent: I.In[I.f16, ("B", "HQ", "C")],
    q_rope: I.In[I.f16, ("B", "HQ", "DR")],
    latent_cache: I.In[I.f16, ("PT", "HK", "C")],
    rope_cache: I.In[I.f16, ("PT", "HK", "DR")],
    page_offsets: I.In[I.i32, ("B_PLUS_1",)],
    page_indices: I.In[I.i32, ("S",)],
    sequence_lengths: I.In[I.i32, ("B",)],
    partial_lse: I.Out[I.f32, ("B", "HQ", "SPLITS")],
    partial_output: I.Out[I.f32, ("B", "HQ", "SPLITS", "C")],
    scale: I.f32,
    PAGE_SIZE: I.Constexpr[int],
    HEAD_GROUP: I.Constexpr[int],
    SPLITS: I.Constexpr[int],
):
    B, HQ, C = q_latent.shape
    DR = q_rope.shape[2]
    HK = latent_cache.shape[1]
    for batch in I.parallel(I.domain(0, B)):
        page_begin = I.cast(page_offsets[batch], I.index)
        sequence_length = I.cast(sequence_lengths[batch], I.index)
        split_extent = (sequence_length + SPLITS - 1) // SPLITS
        for split in I.parallel(I.domain(0, SPLITS)):
            selected_begin = split * split_extent
            selected_end = I.minimum(selected_begin + split_extent, sequence_length)
            selected = I.domain(selected_begin, selected_end)
            logical_token = I.indices(selected)
            active = logical_token < sequence_length
            page_slot = page_begin + logical_token // PAGE_SIZE
            I.assume_in_bounds(page_slot, page_indices, axis=0)
            physical_page = I.cast(page_indices[page_slot], I.index)
            physical_token = physical_page * PAGE_SIZE + logical_token % PAGE_SIZE
            I.assume_in_bounds(physical_token, latent_cache, axis=0)
            I.assume_in_bounds(physical_token, rope_cache, axis=0)
            local_query_heads = I.domain(0, HEAD_GROUP)
            for key_head in I.parallel(I.domain(0, HK)):
                query_heads = key_head * HEAD_GROUP + I.indices(local_query_heads)
                latent_values = latent_cache[physical_token, key_head, :]
                summary = I.region_fold(
                    source=(
                        latent_values,
                        rope_cache[physical_token, key_head, :],
                        latent_values,
                        logical_token,
                        active,
                    ),
                    axis=0,
                    summarize=summarize_masked_mla_chunk_natural,
                    combine=merge_natural_attention_summaries,
                    identity=empty_attention_summary(local_query_heads, C),
                    operands=(
                        I.gather(q_latent, index=(batch, query_heads, slice(None))),
                        I.gather(q_rope, index=(batch, query_heads, slice(None))),
                        query_heads,
                        scale,
                    ),
                )
                safe_denominator = I.select(summary.valid, summary.denominator, 1.0)
                I.scatter_unique(
                    partial_lse,
                    index=(batch, query_heads, split),
                    value=I.select(
                        summary.valid,
                        summary.maximum + I.log(safe_denominator),
                        -I.inf,
                    ),
                )
                I.scatter_unique(
                    partial_output,
                    index=(batch, query_heads, split, slice(None)),
                    value=normalize_attention_summary(summary),
                )


@intent.kernel
def absorbed_mla_decode(
    q_latent: I.In[I.f16, ("B", "H", "C")],
    q_rope: I.In[I.f16, ("B", "H", "DR")],
    latent_cache: I.In[I.f16, ("B", "K", "C")],
    rope_cache: I.In[I.f16, ("B", "K", "DR")],
    output: I.Out[I.f16, ("B", "H", "C")],
    scale: I.f32,
):
    B, H, C = q_latent.shape
    K = latent_cache.shape[1]
    key_axis = I.domain(0, K)
    heads = I.domain(0, H)
    for batch in I.parallel(I.domain(0, B)):
        latent_values = latent_cache[batch, key_axis, :]
        summary = I.region_fold(
            source=(
                latent_values,
                rope_cache[batch, key_axis, :],
                latent_values,
                I.indices(key_axis),
            ),
            axis=0,
            summarize=summarize_mla_chunk,
            combine=merge_attention_summaries,
            identity=empty_attention_summary(H, C),
            operands=(
                q_latent[batch, heads, :],
                q_rope[batch, heads, :],
                I.full((H,), fill=0, dtype=I.index),
                scale,
                False,
            ),
        )
        output[batch, heads, :] = I.cast(normalize_attention_summary(summary), I.f16)


@intent.kernel
def splitk_mla_decode_partials(
    q_latent: I.In[I.f16, ("B", "H", "C")],
    q_rope: I.In[I.f16, ("B", "H", "DR")],
    latent_cache: I.In[I.f16, ("B", "K", "C")],
    rope_cache: I.In[I.f16, ("B", "K", "DR")],
    partial_lse: I.Out[I.f32, ("B", "H", "SPLITS")],
    partial_output: I.Out[I.f16, ("B", "H", "SPLITS", "C")],
    scale: I.f32,
    SPLITS: I.Constexpr[int],
    SPLIT_SIZE: I.Constexpr[int],
):
    B, H, C = q_latent.shape
    K = latent_cache.shape[1]
    heads = I.domain(0, H)
    split_slots = I.domain(0, SPLIT_SIZE)
    for batch in I.parallel(I.domain(0, B)):
        for split in I.parallel(I.domain(0, SPLITS)):
            key_coordinates = split * SPLIT_SIZE + I.indices(split_slots)
            active = (key_coordinates >= 0) & (key_coordinates < K)
            latent_values = I.gather(
                latent_cache, index=(batch, key_coordinates, slice(None)),
                valid=active[:, None], fill=0.0,
            )
            summary = I.region_fold(
                source=(
                    latent_values,
                    I.gather(
                        rope_cache, index=(batch, key_coordinates, slice(None)),
                        valid=active[:, None], fill=0.0,
                    ),
                    latent_values,
                    key_coordinates,
                    active,
                ),
                axis=0,
                summarize=summarize_masked_mla_chunk,
                combine=merge_attention_summaries,
                identity=empty_attention_summary(H, C),
                operands=(
                    q_latent[batch, heads, :],
                    q_rope[batch, heads, :],
                    I.full((H,), fill=0, dtype=I.index),
                    scale,
                ),
            )
            safe_denominator = I.select(summary.valid, summary.denominator, 1.0)
            partial_lse[batch, heads, split] = I.select(
                summary.valid,
                summary.maximum + I.log(safe_denominator) * I.LOG2E,
                -I.inf,
            )
            partial_output[batch, heads, split, :] = I.cast(
                normalize_attention_summary(summary), I.f16
            )
