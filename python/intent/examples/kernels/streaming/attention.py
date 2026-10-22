import math

import intent
import intent.language as I


BATCH = 4
HEADS = 32
SEQUENCE = 4096
HEAD_DIMENSION = 128
SCALE = 1.0 / math.sqrt(HEAD_DIMENSION)
VARLEN_BATCH = 8
VARLEN_TOTAL_TOKENS = 29114
VARLEN_GQA_QUERY_HEADS = 32
VARLEN_GQA_KV_HEADS = 8
VARLEN_GQA_HEAD_GROUP = VARLEN_GQA_QUERY_HEADS // VARLEN_GQA_KV_HEADS
VARLEN_GQA_TOTAL_TOKENS = 29184
VARLEN_GQA_DECODE_BATCH = 8
VARLEN_GQA_DECODE_QUERY_HEADS = 32
VARLEN_GQA_DECODE_KV_HEADS = 8
VARLEN_GQA_DECODE_HEAD_DIMENSION = 64
VARLEN_GQA_DECODE_BLOCK_SIZE = 64
VARLEN_GQA_DECODE_MAX_BLOCKS = 64
VARLEN_GQA_DECODE_HEAD_GROUP = (
    VARLEN_GQA_DECODE_QUERY_HEADS // VARLEN_GQA_DECODE_KV_HEADS
)
VARLEN_GQA_DECODE_SCALE = 1.0 / math.sqrt(VARLEN_GQA_DECODE_HEAD_DIMENSION)
GQA_DECODE_BATCH = 32
GQA_DECODE_QUERY_HEADS = 32
GQA_DECODE_KV_HEADS = 8
GQA_DECODE_SEQUENCE = 8192
GQA_DECODE_HEAD_DIMENSION = 128
GQA_DECODE_HEAD_GROUP = GQA_DECODE_QUERY_HEADS // GQA_DECODE_KV_HEADS
GQA_DECODE_SCALE = 1.0 / math.sqrt(GQA_DECODE_HEAD_DIMENSION)
MLA_PREFILL_BATCH = 1
MLA_PREFILL_QUERY_HEADS = 8
MLA_PREFILL_KV_HEADS = 2
MLA_PREFILL_SEQUENCE = 1024
MLA_PREFILL_CONTENT_DIMENSION = 128
MLA_PREFILL_POSITION_DIMENSION = 64
MLA_PREFILL_HEAD_GROUP = MLA_PREFILL_QUERY_HEADS // MLA_PREFILL_KV_HEADS
MLA_PREFILL_SCALE = 1.0 / math.sqrt(
    MLA_PREFILL_CONTENT_DIMENSION + MLA_PREFILL_POSITION_DIMENSION
)


@intent.fn
def maximum_score(lhs, rhs):
    return I.maximum_num(lhs, rhs)


@intent.fn
def reduce_score_maximum(scores, axis):
    return I.reduce(
        scores,
        axis=axis,
        identity=-I.inf,
        combine=maximum_score,
    )


@intent.fn
def empty_attention_summary(query_count, value_width):
    return I.record(
        valid=I.full((query_count,), fill=False, dtype=I.bool),
        maximum=I.full((query_count,), fill=0.0, dtype=I.f32),
        denominator=I.zeros((query_count,), dtype=I.f32),
        accumulator=I.zeros((query_count, value_width), dtype=I.f32),
    )


@intent.fn
def empty_scalar_attention_summary(value_width):
    return I.record(
        valid=False,
        maximum=I.cast(0.0, I.f32),
        denominator=I.cast(0.0, I.f32),
        accumulator=I.zeros((value_width,), dtype=I.f32),
    )


@intent.fn
def merge_attention_summaries(lhs, rhs):
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
        I.exp2(lhs_maximum - maximum, approximate=True, flush_to_zero=True),
        0.0,
    )
    rhs_scale = I.select(
        rhs.valid,
        I.exp2(rhs_maximum - maximum, approximate=True, flush_to_zero=True),
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
def merge_scalar_attention_summaries(lhs, rhs):
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
        I.exp2(lhs_maximum - maximum, approximate=True, flush_to_zero=True),
        0.0,
    )
    rhs_scale = I.select(
        rhs.valid,
        I.exp2(rhs_maximum - maximum, approximate=True, flush_to_zero=True),
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
            lhs_scale * lhs.accumulator
            + rhs_scale * rhs.accumulator
        ),
    )


@intent.fn
def summarize_attention_chunk_f16(
    key_chunk,
    value_chunk,
    key_coordinates,
    queries,
    query_coordinates,
    scale,
    causal,
):
    scores = I.matmul(
        queries,
        key_chunk,
        transpose_rhs=True,
        acc_dtype=I.f32,
    ) * (scale * I.LOG2E)
    valid = I.full(scores.shape, fill=True, dtype=I.bool)
    if causal:
        valid = query_coordinates[:, None] >= key_coordinates[None, :]
    masked_scores = I.select(valid, scores, -I.inf)
    chunk_valid = I.reduce.any(valid, axis=1)
    raw_maximum = reduce_score_maximum(masked_scores, axis=1)
    maximum = I.select(chunk_valid, raw_maximum, 0.0)
    probability = I.select(
        valid,
        I.exp2(masked_scores - maximum[:, None], approximate=True, flush_to_zero=True),
        0.0,
    )
    denominator = I.reduce.sum(probability, axis=1)
    accumulator = I.matmul(
        I.cast(probability, I.f16),
        value_chunk,
        acc_dtype=I.f32,
    )
    return I.record(
        valid=chunk_valid,
        maximum=maximum,
        denominator=denominator,
        accumulator=I.select(chunk_valid[:, None], accumulator, 0.0),
    )


@intent.fn
def summarize_attention_chunk_bf16(
    key_chunk,
    value_chunk,
    key_coordinates,
    queries,
    query_coordinates,
    scale,
    causal,
    active,
):
    scores = I.matmul(
        queries,
        key_chunk,
        transpose_rhs=True,
        acc_dtype=I.f32,
    ) * (scale * I.LOG2E)
    valid = I.full(scores.shape, fill=True, dtype=I.bool) & active
    if causal:
        valid = valid & (query_coordinates[:, None] >= key_coordinates[None, :])
    masked_scores = I.select(valid, scores, -I.inf)
    chunk_valid = I.reduce.any(valid, axis=1)
    raw_maximum = reduce_score_maximum(masked_scores, axis=1)
    maximum = I.select(chunk_valid, raw_maximum, 0.0)
    probability = I.select(
        valid,
        I.exp2(masked_scores - maximum[:, None], approximate=True, flush_to_zero=True),
        0.0,
    )
    return I.record(
        valid=chunk_valid,
        maximum=maximum,
        denominator=I.reduce.sum(probability, axis=1),
        accumulator=I.select(
            chunk_valid[:, None],
            I.matmul(
                I.cast(probability, I.bf16),
                value_chunk,
                acc_dtype=I.f32,
            ),
            0.0,
        ),
    )


@intent.fn
def summarize_masked_attention_chunk_bf16(
    key_chunk, value_chunk, key_coordinates, active,
    queries, query_coordinates, scale,
):
    return summarize_attention_chunk_bf16(
        key_chunk, value_chunk, key_coordinates,
        queries, query_coordinates, scale, False, active,
    )


@intent.fn
def summarize_masked_attention_chunk(
    key_chunk,
    value_chunk,
    key_coordinates,
    active,
    queries,
    query_coordinates,
    scale,
):
    scores = I.matmul(
        queries,
        key_chunk,
        transpose_rhs=True,
        acc_dtype=I.f32,
    ) * (scale * I.LOG2E)
    valid = I.full(scores.shape, fill=True, dtype=I.bool) & active[None, :]
    masked_scores = I.select(valid, scores, -I.inf)
    chunk_valid = I.reduce.any(valid, axis=1)
    raw_maximum = reduce_score_maximum(masked_scores, axis=1)
    maximum = I.select(chunk_valid, raw_maximum, 0.0)
    probability = I.select(
        valid,
        I.exp2(masked_scores - maximum[:, None], approximate=True, flush_to_zero=True),
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
def summarize_masked_scalar_query_chunk(
    key_chunk,
    value_chunk,
    active,
    query,
    scale,
):
    scores = I.reduce.sum(
        query[None, :] * key_chunk,
        axis=1,
    ) * (scale * I.LOG2E)
    valid = I.full(scores.shape, fill=True, dtype=I.bool) & active
    scores = I.select(valid, scores, -I.inf)
    chunk_valid = I.reduce.any(valid, axis=0)
    maximum = I.select(
        chunk_valid,
        reduce_score_maximum(scores, axis=0),
        0.0,
    )
    probability = I.select(
        valid,
        I.exp2(scores - maximum, approximate=True, flush_to_zero=True),
        0.0,
    )
    return I.record(
        valid=chunk_valid,
        maximum=maximum,
        denominator=I.reduce.sum(probability, axis=0),
        accumulator=I.reduce.sum(
            I.cast(
                I.cast(probability, I.f16)[:, None] * value_chunk,
                I.f32,
            ),
            axis=0,
        ),
    )


@intent.fn
def summarize_biased_attention_chunk(
    key_chunk,
    value_chunk,
    key_coordinates,
    bias_chunk,
    queries,
    query_coordinates,
    scale,
):
    scores = (
        I.matmul(
            queries,
            key_chunk,
            transpose_rhs=True,
            acc_dtype=I.f32,
        )
        * scale
        + bias_chunk[None, :]
    ) * I.LOG2E
    valid = I.full(scores.shape, fill=True, dtype=I.bool)
    chunk_valid = I.reduce.any(valid, axis=1)
    raw_maximum = reduce_score_maximum(scores, axis=1)
    maximum = I.select(chunk_valid, raw_maximum, 0.0)
    probability = I.select(
        valid,
        I.exp2(scores - maximum[:, None], approximate=True, flush_to_zero=True),
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
def summarize_mla_chunk(
    key_chunk,
    key_position_chunk,
    value_chunk,
    key_coordinates,
    queries,
    query_positions,
    query_coordinates,
    scale,
):
    scores = (
        I.matmul(queries, key_chunk, transpose_rhs=True, acc_dtype=I.f32)
        + I.matmul(
            query_positions,
            key_position_chunk,
            transpose_rhs=True,
            acc_dtype=I.f32,
        )
    ) * (scale * I.LOG2E)
    valid = query_coordinates[:, None] >= key_coordinates[None, :]
    scores = I.select(valid, scores, -I.inf)
    chunk_valid = I.reduce.any(valid, axis=1)
    maximum = I.select(
        chunk_valid,
        reduce_score_maximum(scores, axis=1),
        0.0,
    )
    probability = I.select(
        valid,
        I.exp2(scores - maximum[:, None], approximate=True, flush_to_zero=True),
        0.0,
    )
    return I.record(
        valid=chunk_valid,
        maximum=maximum,
        denominator=I.reduce.sum(probability, axis=1),
        accumulator=I.select(
            chunk_valid[:, None],
            I.matmul(
                I.cast(probability, I.f16),
                value_chunk,
                acc_dtype=I.f32,
            ),
            0.0,
        ),
    )


@intent.fn
def normalize_attention_summary(summary):
    safe_denominator = I.select(summary.valid, summary.denominator, 1.0)
    inverse_denominator = I.fdiv(1.0, safe_denominator, approximate=True, flush_to_zero=True)
    return I.select(
        summary.valid[:, None],
        summary.accumulator * inverse_denominator[:, None],
        0.0,
    )


@intent.kernel
def flash_attention_fwd(
    q: I.In[I.f16, ("B", "H", "Q", "D")],
    k: I.In[I.f16, ("B", "H", "K", "D")],
    v: I.In[I.f16, ("B", "H", "K", "DV")],
    output: I.Out[I.f16, ("B", "H", "Q", "DV")],
    scale: I.f32,
    CAUSAL: I.Constexpr[bool],
):
    B, H, Q, _ = q.shape
    K = k.shape[2]
    DV = v.shape[3]
    q_axis = I.domain(0, Q)
    k_axis = I.domain(0, K)
    q_coordinates = I.indices(q_axis)
    k_coordinates = I.indices(k_axis)
    for batch in I.parallel(I.domain(0, B)):
        for head in I.parallel(I.domain(0, H)):
            summary = I.region_fold(
                source=(
                    k[batch, head, k_axis, :],
                    v[batch, head, k_axis, :],
                    k_coordinates,
                ),
                axis=0,
                summarize=summarize_attention_chunk_f16,
                combine=merge_attention_summaries,
                identity=empty_attention_summary(Q, DV),
                operands=(
                    q[batch, head, q_axis, :],
                    q_coordinates,
                    scale,
                    CAUSAL,
                ),
            )
            output[batch, head, q_axis, :] = I.cast(
                normalize_attention_summary(summary),
                I.f16,
            )


@intent.kernel
def flash_gqa_attention_fwd(
    q: I.In[I.f16, ("B", "HQ", "Q", "D")],
    k: I.In[I.f16, ("B", "HK", "K", "D")],
    v: I.In[I.f16, ("B", "HK", "K", "DV")],
    output: I.Out[I.f16, ("B", "HQ", "Q", "DV")],
    scale: I.f32,
    HEAD_GROUP: I.Constexpr[int],
    CAUSAL: I.Constexpr[bool],
):
    B, HQ, Q, _ = q.shape
    K = k.shape[2]
    DV = v.shape[3]
    q_axis = I.domain(0, Q)
    k_axis = I.domain(0, K)
    q_coordinates = I.indices(q_axis)
    k_coordinates = I.indices(k_axis)
    for batch in I.parallel(I.domain(0, B)):
        for query_head in I.parallel(I.domain(0, HQ)):
            key_head = query_head // HEAD_GROUP
            summary = I.region_fold(
                source=(
                    k[batch, key_head, k_axis, :],
                    v[batch, key_head, k_axis, :],
                    k_coordinates,
                ),
                axis=0,
                summarize=summarize_attention_chunk_f16,
                combine=merge_attention_summaries,
                identity=empty_attention_summary(Q, DV),
                operands=(
                    q[batch, query_head, q_axis, :],
                    q_coordinates,
                    scale,
                    CAUSAL,
                ),
            )
            output[batch, query_head, q_axis, :] = I.cast(
                normalize_attention_summary(summary),
                I.f16,
            )


@intent.kernel
def flash_attention_bf16_fwd(
    q: I.In[I.bf16, ("B", "HQ", "Q", "D")],
    k: I.In[I.bf16, ("B", "HK", "K", "D")],
    v: I.In[I.bf16, ("B", "HK", "K", "DV")],
    output: I.Out[I.bf16, ("B", "HQ", "Q", "DV")],
    scale: I.f32,
    HEAD_GROUP: I.Constexpr[int],
    CAUSAL: I.Constexpr[bool],
):
    B, HQ, Q, _ = q.shape
    K = k.shape[2]
    DV = v.shape[3]
    q_axis = I.domain(0, Q)
    k_axis = I.domain(0, K)
    q_coordinates = I.indices(q_axis)
    k_coordinates = I.indices(k_axis)
    for batch in I.parallel(I.domain(0, B)):
        for query_head in I.parallel(I.domain(0, HQ)):
            key_head = query_head // HEAD_GROUP
            summary = I.region_fold(
                source=(
                    k[batch, key_head, k_axis, :],
                    v[batch, key_head, k_axis, :],
                    k_coordinates,
                ),
                axis=0,
                summarize=summarize_attention_chunk_bf16,
                combine=merge_attention_summaries,
                identity=empty_attention_summary(Q, DV),
                operands=(
                    q[batch, query_head, q_axis, :],
                    q_coordinates,
                    scale,
                    CAUSAL,
                    True,
                ),
            )
            output[batch, query_head, q_axis, :] = I.cast(
                normalize_attention_summary(summary),
                I.bf16,
            )


@intent.kernel
def grouped_flash_decode_partials(
    q: I.In[I.bf16, ("B", "HQ", 1, "D")],
    k: I.In[I.bf16, ("B", "HK", "K", "D")],
    v: I.In[I.bf16, ("B", "HK", "K", "DV")],
    partial_lse: I.Out[I.f32, ("B", "HQ", "P")],
    partial_output: I.Out[I.bf16, ("B", "HQ", "P", "DV")],
    scale: I.f32,
    HEAD_GROUP: I.Constexpr[int],
    P: I.Constexpr[int],
    SPLIT_SIZE: I.Constexpr[int],
):
    B, HQ, _, _ = q.shape
    HK = k.shape[1]
    K = k.shape[2]
    DV = v.shape[3]
    split_slots = I.domain(0, SPLIT_SIZE)
    parts = I.domain(0, P)
    local_query_heads = I.domain(0, HEAD_GROUP)
    for batch in I.parallel(I.domain(0, B)):
        for key_head in I.parallel(I.domain(0, HK)):
            query_heads = key_head * HEAD_GROUP + I.indices(local_query_heads)
            query = I.gather(q, index=(batch, query_heads, 0, slice(None)))
            for part in I.parallel(parts):
                key_coordinates = part * SPLIT_SIZE + I.indices(split_slots)
                active = (key_coordinates >= 0) & (key_coordinates < K)
                summary = I.region_fold(
                    source=(
                        I.gather(
                            k, index=(batch, key_head, key_coordinates, slice(None)),
                            valid=active[:, None], fill=0.0,
                        ),
                        I.gather(
                            v, index=(batch, key_head, key_coordinates, slice(None)),
                            valid=active[:, None], fill=0.0,
                        ),
                        key_coordinates,
                        active,
                    ),
                    axis=0,
                    summarize=summarize_masked_attention_chunk_bf16,
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
                    index=(batch, query_heads, part),
                    value=I.select(
                        summary.valid,
                        summary.maximum + I.log(safe_denominator) * I.LOG2E,
                        -I.inf,
                    ),
                )
                I.scatter_unique(
                    partial_output,
                    index=(batch, query_heads, part, slice(None)),
                    value=I.cast(
                        normalize_attention_summary(summary),
                        I.bf16,
                    ),
                )


@intent.kernel
def continuous_gqa_decode(
    q: I.In[I.f16, ("B", "HQ", "D")],
    k: I.In[I.f16, ("B", "K", "HK", "D")],
    v: I.In[I.f16, ("B", "K", "HK", "DV")],
    valid_mask: I.In[I.u8, ("B", "K", "HK")],
    output: I.Out[I.f16, ("B", "HQ", "DV")],
    scale: I.f32,
    HEAD_GROUP: I.Constexpr[int],
):
    B, HQ, D = q.shape
    K = k.shape[1]
    HK = k.shape[2]
    DV = v.shape[3]
    key_axis = I.domain(0, K)
    key_coordinates = I.indices(key_axis)
    for batch in I.parallel(I.domain(0, B)):
        for key_head in I.parallel(I.domain(0, HK)):
            for head_offset in I.parallel(I.domain(0, HEAD_GROUP)):
                query_head = key_head * HEAD_GROUP + head_offset
                query = I.reshape(q[batch, query_head, :], (1, D))
                summary = I.region_fold(
                    source=(
                        k[batch, key_axis, key_head, :],
                        v[batch, key_axis, key_head, :],
                        key_coordinates,
                        valid_mask[batch, key_axis, key_head] != 0,
                    ),
                    axis=0,
                    summarize=summarize_masked_attention_chunk,
                    combine=merge_attention_summaries,
                    identity=empty_attention_summary(1, DV),
                    operands=(
                        query,
                        I.full((1,), fill=0, dtype=I.index),
                        scale,
                    ),
                )
                output[batch, query_head, :] = I.reshape(
                    I.cast(normalize_attention_summary(summary), I.f16),
                    (DV,),
                )


@intent.kernel
def varlen_gqa_decode_with_sink_logits(
    q: I.In[I.f16, ("B", "HQ", "D")],
    k: I.In[I.f16, ("U", "HK", "D")],
    v: I.In[I.f16, ("U", "HK", "DV")],
    cu_seqlens: I.In[I.i32, ("B_PLUS_1",)],
    sink: I.In[I.f32, ("HQ",)],
    output: I.Out[I.f16, ("B", "HQ", "DV")],
    block_logits: I.Out[
        I.f32, ("B", "HQ", VARLEN_GQA_DECODE_MAX_BLOCKS)
    ],
    scale: I.f32,
    HEAD_GROUP: I.Constexpr[int],
    MAX_BLOCKS: I.Constexpr[int],
):
    B, HQ, D = q.shape
    DV = v.shape[2]
    members = I.domain(0, k.shape[0])
    blocks = I.domain(0, MAX_BLOCKS)
    for sequence in I.parallel(I.domain(0, B)):
        sequence_begin = I.cast(cu_seqlens[sequence], I.index)
        sequence_end = I.cast(cu_seqlens[sequence + 1], I.index)
        sequence_length = sequence_end - sequence_begin
        block_count = (
            sequence_length + VARLEN_GQA_DECODE_BLOCK_SIZE - 1
        ) // VARLEN_GQA_DECODE_BLOCK_SIZE
        for query_head in I.parallel(I.domain(0, HQ)):
            key_head = query_head // HEAD_GROUP
            query = I.reshape(q[sequence, query_head, :], (1, D))
            summary = empty_attention_summary(1, DV)
            block_maxima = I.buffer((MAX_BLOCKS,), I.f32)
            for block in blocks:
                begin = I.minimum(
                    sequence_begin + block * VARLEN_GQA_DECODE_BLOCK_SIZE,
                    sequence_end,
                )
                end = I.minimum(
                    begin + VARLEN_GQA_DECODE_BLOCK_SIZE,
                    sequence_end,
                )
                key_region = members[begin:end]
                if block < block_count:
                    block_summary = summarize_attention_chunk_f16(
                        k[key_region, key_head, :],
                        v[key_region, key_head, :],
                        I.indices(key_region),
                        query,
                        I.full((1,), fill=0, dtype=I.index),
                        scale,
                        False,
                    )
                    summary = merge_attention_summaries(summary, block_summary)
                    I.store(block_maxima, block, block_summary.maximum[0])
                else:
                    I.store(block_maxima, block, 0.0)
            sink_denominator = summary.denominator + sink[query_head]
            safe_denominator = I.select(
                summary.valid,
                sink_denominator,
                1.0,
            )
            inverse_denominator = I.fdiv(1.0, safe_denominator, approximate=True, flush_to_zero=True)
            output[sequence, query_head, :] = I.reshape(
                I.cast(
                    I.select(
                        summary.valid[:, None],
                        summary.accumulator * inverse_denominator[:, None],
                        0.0,
                    ),
                    I.f16,
                ),
                (DV,),
            )
            for block in blocks:
                block_maximum = I.mutable_load(block_maxima, block)
                probability = I.exp2(block_maximum - summary.maximum[0], approximate=True, flush_to_zero=True)
                block_logits[sequence, query_head, block] = I.select(
                    block < block_count,
                    probability * inverse_denominator[0],
                    0.0,
                )


@intent.kernel
def mla_prefill(
    q: I.In[I.f16, ("B", "H", "Q", "D")],
    qpe: I.In[I.f16, ("B", "H", "Q", "P")],
    k: I.In[I.f16, ("B", "HK", "K", "D")],
    kpe: I.In[I.f16, ("B", 1, "K", "P")],
    v: I.In[I.f16, ("B", "HK", "K", "D")],
    output: I.Out[I.f16, ("B", "H", "Q", "D")],
    scale: I.f32,
    HEAD_GROUP: I.Constexpr[int],
):
    B, H, Q, D = q.shape
    K = k.shape[2]
    query_axis = I.domain(0, Q)
    key_axis = I.domain(0, K)
    query_coordinates = I.indices(query_axis)
    key_coordinates = I.indices(key_axis)
    for batch in I.parallel(I.domain(0, B)):
        for query_head in I.parallel(I.domain(0, H)):
            key_head = query_head // HEAD_GROUP
            summary = I.region_fold(
                source=(
                    k[batch, key_head, key_axis, :],
                    kpe[batch, 0, key_axis, :],
                    v[batch, key_head, key_axis, :],
                    key_coordinates,
                ),
                axis=0,
                summarize=summarize_mla_chunk,
                combine=merge_attention_summaries,
                identity=empty_attention_summary(Q, D),
                operands=(
                    q[batch, query_head, query_axis, :],
                    qpe[batch, query_head, query_axis, :],
                    query_coordinates,
                    scale,
                ),
            )
            output[batch, query_head, query_axis, :] = I.cast(
                normalize_attention_summary(summary),
                I.f16,
            )


@intent.kernel
def flash_attention_bias_fwd(
    q: I.In[I.f16, ("B", "H", "Q", "D")],
    k: I.In[I.f16, ("B", "H", "K", "D")],
    v: I.In[I.f16, ("B", "H", "K", "DV")],
    bias: I.In[I.f32, ("B", "H", "K")],
    output: I.Out[I.f16, ("B", "H", "Q", "DV")],
    scale: I.f32,
):
    B, H, Q, _ = q.shape
    K = k.shape[2]
    DV = v.shape[3]
    q_axis = I.domain(0, Q)
    k_axis = I.domain(0, K)
    for batch in I.parallel(I.domain(0, B)):
        for head in I.parallel(I.domain(0, H)):
            summary = I.region_fold(
                source=(
                    k[batch, head, k_axis, :],
                    v[batch, head, k_axis, :],
                    I.indices(k_axis),
                    bias[batch, head, k_axis],
                ),
                axis=0,
                summarize=summarize_biased_attention_chunk,
                combine=merge_attention_summaries,
                identity=empty_attention_summary(Q, DV),
                operands=(
                    q[batch, head, q_axis, :],
                    I.indices(q_axis),
                    scale,
                ),
            )
            output[batch, head, q_axis, :] = I.cast(
                normalize_attention_summary(summary),
                I.f16,
            )


@intent.kernel
def flash_varlen_attention_fwd(
    q: I.In[I.f16, ("U", "D")],
    k: I.In[I.f16, ("U", "D")],
    v: I.In[I.f16, ("U", "DV")],
    sequence_lengths: I.In[I.i32, ("B",)],
    cu_seqlens: I.In[I.i32, ("B_PLUS_1",)],
    output: I.Out[I.f16, ("U", "DV")],
    scale: I.f32,
    CAUSAL: I.Constexpr[bool],
):
    U, _ = q.shape
    B = sequence_lengths.shape[0]
    DV = v.shape[1]
    sequences = I.ragged(
        outer=I.domain(0, B),
        members=I.domain(0, U),
        offsets=cu_seqlens,
    )
    for sequence in I.parallel(sequences.outer):
        positions = sequences[sequence]
        position_count = cu_seqlens[sequence + 1] - cu_seqlens[sequence]
        summary = I.region_fold(
            source=(k[positions, :], v[positions, :], I.indices(positions)),
            axis=0,
            summarize=summarize_attention_chunk_f16,
            combine=merge_attention_summaries,
            identity=empty_attention_summary(positions, DV),
            operands=(
                q[positions, :],
                I.indices(positions),
                scale,
                CAUSAL,
            ),
        )
        output[positions, :] = I.cast(
            normalize_attention_summary(summary),
            I.f16,
        )


@intent.kernel
def flash_varlen_gqa_prefill(
    q: I.In[I.f16, ("U", "HQ", "D")],
    k: I.In[I.f16, ("U", "HK", "D")],
    v: I.In[I.f16, ("U", "HK", "DV")],
    sequence_lengths: I.In[I.i32, ("B",)],
    cu_seqlens: I.In[I.i32, ("B_PLUS_1",)],
    output: I.Out[I.f16, ("U", "HQ", "DV")],
    scale: I.f32,
    HEAD_GROUP: I.Constexpr[int],
):
    U, HQ, _ = q.shape
    B = sequence_lengths.shape[0]
    DV = v.shape[2]
    sequences = I.ragged(
        outer=I.domain(0, B),
        members=I.domain(0, U),
        offsets=cu_seqlens,
    )
    for sequence in I.parallel(sequences.outer):
        positions = sequences[sequence]
        position_count = cu_seqlens[sequence + 1] - cu_seqlens[sequence]
        for query_head in I.parallel(I.domain(0, HQ)):
            key_head = query_head // HEAD_GROUP
            summary = I.region_fold(
                source=(
                    k[positions, key_head, :],
                    v[positions, key_head, :],
                    I.indices(positions),
                ),
                axis=0,
                summarize=summarize_attention_chunk_f16,
                combine=merge_attention_summaries,
                identity=empty_attention_summary(positions, DV),
                operands=(
                    q[positions, query_head, :],
                    I.indices(positions),
                    scale,
                    True,
                ),
            )
            output[positions, query_head, :] = I.cast(
                normalize_attention_summary(summary),
                I.f16,
            )
