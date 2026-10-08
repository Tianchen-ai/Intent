import intent
import intent.language as I

from .attention import empty_attention_summary


@intent.fn
def summarize_attention_f32(keys, values, key_coordinates, queries, query_coordinates, scale):
    scores = I.matmul(queries, keys, transpose_rhs=True, acc_dtype=I.f32) * scale
    valid = query_coordinates[:, None] >= key_coordinates[None, :]
    masked = I.select(valid, scores, -I.inf)
    present = I.reduce.any(valid, axis=1)
    maximum = I.select(present, I.reduce.max(masked, axis=1), 0.0)
    probabilities = I.select(valid, I.exp(masked - maximum[:, None]), 0.0)
    return I.record(
        valid=present,
        maximum=maximum,
        denominator=I.reduce.sum(probabilities, axis=1),
        accumulator=I.matmul(probabilities, values, acc_dtype=I.f32),
    )


@intent.fn
def merge_attention_f32(lhs, rhs):
    maximum = I.select(lhs.valid, lhs.maximum, rhs.maximum)
    maximum = I.select(rhs.valid, I.maximum(maximum, rhs.maximum), maximum)
    lhs_maximum = I.select(lhs.valid, lhs.maximum, maximum)
    rhs_maximum = I.select(rhs.valid, rhs.maximum, maximum)
    lhs_scale = I.select(lhs.valid, I.exp(lhs_maximum - maximum), 0.0)
    rhs_scale = I.select(rhs.valid, I.exp(rhs_maximum - maximum), 0.0)
    return I.record(
        valid=lhs.valid | rhs.valid,
        maximum=maximum,
        denominator=lhs_scale * lhs.denominator + rhs_scale * rhs.denominator,
        accumulator=lhs_scale[:, None] * lhs.accumulator + rhs_scale[:, None] * rhs.accumulator,
    )


@intent.kernel
def causal_attention_f32(
    q: I.In[I.f32, ("B", "Q", "D")],
    k: I.In[I.f32, ("B", "K", "D")],
    v: I.In[I.f32, ("B", "K", "DV")],
    output: I.Out[I.f32, ("B", "Q", "DV")],
    scale: I.f32,
):
    B, Q, D = q.shape
    K, DV = k.shape[1], v.shape[2]
    queries, keys = I.domain(0, Q), I.domain(0, K)
    query_coordinates, key_coordinates = I.indices(queries), I.indices(keys)
    for batch in I.parallel(I.domain(0, B)):
        summary = I.region_fold(
            source=(k[batch, keys, :], v[batch, keys, :], key_coordinates),
            axis=0,
            summarize=summarize_attention_f32,
            combine=merge_attention_f32,
            identity=empty_attention_summary(Q, DV),
            operands=(q[batch, queries, :], query_coordinates, scale),
        )
        denominator = I.select(summary.valid, summary.denominator, 1.0)
        output[batch, queries, :] = I.select(
            summary.valid[:, None], summary.accumulator / denominator[:, None], 0.0
        )


@intent.fn
def summarize_linear_f32(queries, keys, values, coordinates):
    return I.matmul(keys, values, transpose_lhs=True, acc_dtype=I.f32)


@intent.fn
def add_linear_f32(lhs, rhs):
    return lhs + rhs


@intent.fn
def emit_linear_f32(queries, keys, values, coordinates, state):
    scores = I.matmul(queries, keys, transpose_rhs=True, acc_dtype=I.f32)
    scores = I.mask(scores, valid=coordinates[:, None] >= coordinates[None, :], fill=0.0)
    return I.matmul(scores, values, acc_dtype=I.f32) + I.matmul(queries, state, acc_dtype=I.f32)


@intent.kernel
def causal_linear_attention_f32(
    q: I.In[I.f32, ("B", "S", "D")],
    k: I.In[I.f32, ("B", "S", "D")],
    v: I.In[I.f32, ("B", "S", "DV")],
    output: I.Out[I.f32, ("B", "S", "DV")],
    final_state: I.Out[I.f32, ("B", "D", "DV")],
):
    B, S, D = q.shape
    DV = v.shape[2]
    tokens = I.domain(0, S)
    coordinates = I.indices(tokens)
    for batch in I.parallel(I.domain(0, B)):
        initial = I.zeros((D, DV), dtype=I.f32)
        result, state = I.region_scan(
            source=(q[batch, tokens, :], k[batch, tokens, :], v[batch, tokens, :], coordinates),
            axis=0,
            summarize=summarize_linear_f32,
            combine=add_linear_f32,
            identity=initial,
            initial_state=initial,
            apply=add_linear_f32,
            emit=emit_linear_f32,
        )
        output[batch, tokens, :] = result
        final_state[batch, :, :] = state
