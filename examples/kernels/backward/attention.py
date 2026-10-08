import math

import intent
import intent.language as I


BATCH = 2
QUERY_HEADS = 8
KV_HEADS = 2
HEAD_GROUP = QUERY_HEADS // KV_HEADS
SEQUENCE = 1024
HEAD_DIMENSION = 64
SCALE = 1.0 / math.sqrt(HEAD_DIMENSION)


@intent.kernel
def attention_backward_delta(
    output: I.In[I.f16, ("B", "HQ", "Q", "D")],
    grad_output: I.In[I.f16, ("B", "HQ", "Q", "D")],
    delta: I.Out[I.f32, ("B", "HQ", "Q")],
):
    B, HQ, Q, _ = output.shape
    query_axis = I.domain(0, Q)
    for batch in I.parallel(I.domain(0, B)):
        for query_head in I.parallel(I.domain(0, HQ)):
            output_block = I.cast(output[batch, query_head, query_axis, :], I.f32)
            grad_output_block = I.cast(
                grad_output[batch, query_head, query_axis, :], I.f32
            )
            delta[batch, query_head, query_axis] = I.reduce.sum(
                output_block * grad_output_block,
                axis=1,
            )


@intent.fn
def summarize_key_value_gradients(
    queries, grad_outputs, lse, delta, query_coordinates,
    keys, values, key_coordinates, scale, causal,
):
    scores = I.matmul(keys, queries, transpose_rhs=True, acc_dtype=I.f32)
    probability = I.exp2(
        scores * (scale * I.LOG2E) - lse[None, :] * I.LOG2E,
        approximate=True, flush_to_zero=True,
    )
    if causal:
        probability = I.mask(
            probability,
            valid=key_coordinates[:, None] <= query_coordinates[None, :],
            fill=0.0,
        )
    grad_values = I.matmul(
        I.cast(probability, I.f16), grad_outputs, acc_dtype=I.f32
    )
    grad_probability = I.matmul(
        values, grad_outputs, transpose_rhs=True, acc_dtype=I.f32
    )
    grad_scores = probability * (grad_probability - delta[None, :])
    grad_keys = I.matmul(
        I.cast(grad_scores, I.f16), queries, acc_dtype=I.f32
    )
    return I.record(grad_k=grad_keys, grad_v=grad_values)


@intent.fn
def merge_key_value_gradients(lhs, rhs):
    return I.record(grad_k=lhs.grad_k + rhs.grad_k, grad_v=lhs.grad_v + rhs.grad_v)


@intent.kernel
def attention_backward_dkdv(
    q: I.In[I.f16, ("B", "HQ", "Q", "D")],
    k: I.In[I.f16, ("B", "HK", "K", "D")],
    v: I.In[I.f16, ("B", "HK", "K", "D")],
    grad_output: I.In[I.f16, ("B", "HQ", "Q", "D")],
    lse: I.In[I.f32, ("B", "HQ", "Q")],
    delta: I.In[I.f32, ("B", "HQ", "Q")],
    grad_k: I.Out[I.f16, ("B", "HK", "K", "D")],
    grad_v: I.Out[I.f16, ("B", "HK", "K", "D")],
    scale: I.f32,
    HEAD_GROUP: I.Constexpr[int],
    CAUSAL: I.Constexpr[bool],
):
    B, HQ, Q, D = q.shape
    _, HK, K, _ = k.shape
    key_axis = I.domain(0, K)
    group_query_count = HEAD_GROUP * Q
    group_queries = I.domain(0, group_query_count)
    for batch in I.parallel(I.domain(0, B)):
        for key_head in I.parallel(I.domain(0, HK)):
            key_block = k[batch, key_head, key_axis, :]
            value_block = v[batch, key_head, key_axis, :]
            members = I.indices(group_queries)
            query_heads = key_head * HEAD_GROUP + members // Q
            q_index = members % Q
            query_block = q[batch, query_heads, q_index, :]
            grad_output_block = grad_output[batch, query_heads, q_index, :]
            query_lse = lse[batch, query_heads, q_index]
            query_delta = delta[batch, query_heads, q_index]
            k_index = I.indices(key_axis)
            summary = I.region_fold(
                source=(
                    query_block, grad_output_block,
                    query_lse, query_delta, q_index,
                ),
                axis=0,
                summarize=summarize_key_value_gradients,
                combine=merge_key_value_gradients,
                identity=I.record(
                    grad_k=I.zeros((K, D), dtype=I.f32),
                    grad_v=I.zeros((K, D), dtype=I.f32),
                ),
                operands=(key_block, value_block, k_index, scale, CAUSAL),
            )
            grad_k[batch, key_head, key_axis, :] = I.cast(
                summary.grad_k * scale, I.f16
            )
            grad_v[batch, key_head, key_axis, :] = I.cast(
                summary.grad_v, I.f16
            )


@intent.fn
def summarize_query_gradients(
    keys, values, key_coordinates,
    queries, grad_outputs, lse, delta, query_coordinates, scale, causal,
):
    scores = I.matmul(queries, keys, transpose_rhs=True, acc_dtype=I.f32)
    probability = I.exp2(
        scores * (scale * I.LOG2E) - lse[:, None] * I.LOG2E,
        approximate=True, flush_to_zero=True,
    )
    if causal:
        probability = I.mask(
            probability,
            valid=query_coordinates[:, None] >= key_coordinates[None, :],
            fill=0.0,
        )
    grad_probability = I.matmul(
        grad_outputs, values, transpose_rhs=True, acc_dtype=I.f32
    )
    grad_scores = probability * (grad_probability - delta[:, None])
    return I.matmul(I.cast(grad_scores, I.f16), keys, acc_dtype=I.f32)


@intent.fn
def merge_query_gradients(lhs, rhs):
    return lhs + rhs


@intent.kernel
def attention_backward_dq(
    q: I.In[I.f16, ("B", "HQ", "Q", "D")],
    k: I.In[I.f16, ("B", "HK", "K", "D")],
    v: I.In[I.f16, ("B", "HK", "K", "D")],
    grad_output: I.In[I.f16, ("B", "HQ", "Q", "D")],
    lse: I.In[I.f32, ("B", "HQ", "Q")],
    delta: I.In[I.f32, ("B", "HQ", "Q")],
    grad_q: I.Out[I.f16, ("B", "HQ", "Q", "D"), I.constraints(noalias=True)],
    scale: I.f32,
    HEAD_GROUP: I.Constexpr[int],
    CAUSAL: I.Constexpr[bool],
):
    B, HQ, Q, D = q.shape
    K = k.shape[2]
    query_axis = I.domain(0, Q)
    key_axis = I.domain(0, K)
    for batch in I.parallel(I.domain(0, B)):
        for query_head in I.parallel(I.domain(0, HQ)):
            key_head = query_head // HEAD_GROUP
            query_block = q[batch, query_head, query_axis, :]
            grad_output_block = grad_output[
                batch, query_head, query_axis, :
            ]
            query_lse = lse[batch, query_head, query_axis]
            query_delta = delta[batch, query_head, query_axis]
            key_block = k[batch, key_head, key_axis, :]
            value_block = v[batch, key_head, key_axis, :]
            grad_q_value = I.region_fold(
                source=(key_block, value_block, I.indices(key_axis)),
                axis=0,
                summarize=summarize_query_gradients,
                combine=merge_query_gradients,
                identity=I.zeros((Q, D), dtype=I.f32),
                operands=(
                    query_block, grad_output_block, query_lse, query_delta,
                    I.indices(query_axis), scale, CAUSAL,
                ),
            )
            grad_q[batch, query_head, query_axis, :] = I.cast(
                grad_q_value * scale, I.f16
            )
