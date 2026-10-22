import intent
import intent.language as I


BATCH = 1
SEQUENCE = 2048
HEADS = 16
DIMENSION = 128


@intent.fn
def add_matrix_summary(lhs, rhs):
    return lhs + rhs


@intent.fn
def apply_matrix_summary(prefix, initial_state):
    return initial_state + prefix


@intent.fn
def summarize_linear_forward_slice(
    query_slice,
    key_slice,
    value_slice,
    token_coordinates,
):
    return I.matmul(
        key_slice,
        value_slice,
        transpose_lhs=True,
        acc_dtype=I.f32,
    )


@intent.fn
def emit_linear_forward_slice(
    query_slice,
    key_slice,
    value_slice,
    token_coordinates,
    incoming_state,
):
    scores = I.matmul(
        query_slice,
        key_slice,
        transpose_rhs=True,
        acc_dtype=I.f32,
    )
    scores = I.mask(
        scores,
        valid=token_coordinates[:, None] >= token_coordinates[None, :],
        fill=0.0,
    )
    intra = I.matmul(
        I.cast(scores, I.f16),
        value_slice,
        acc_dtype=I.f32,
    )
    inter = I.matmul(
        query_slice,
        I.cast(incoming_state, I.f16),
        acc_dtype=I.f32,
    )
    return intra + inter


@intent.fn
def summarize_linear_backward_q_slice(
    key_slice,
    value_slice,
    gradient_slice,
    token_coordinates,
):
    return I.matmul(
        value_slice,
        key_slice,
        transpose_lhs=True,
        acc_dtype=I.f32,
    )


@intent.fn
def emit_linear_backward_q_slice(
    key_slice,
    value_slice,
    gradient_slice,
    token_coordinates,
    incoming_state,
):
    score_gradient = I.matmul(
        gradient_slice,
        value_slice,
        transpose_rhs=True,
        acc_dtype=I.f32,
    )
    score_gradient = I.mask(
        score_gradient,
        valid=token_coordinates[:, None] >= token_coordinates[None, :],
        fill=0.0,
    )
    local = I.matmul(
        I.cast(score_gradient, I.f16),
        key_slice,
        acc_dtype=I.f32,
    )
    carried = I.matmul(
        gradient_slice,
        I.cast(incoming_state, I.f16),
        acc_dtype=I.f32,
    )
    return local + carried


@intent.fn
def summarize_linear_backward_kv_slice(
    query_slice,
    key_slice,
    value_slice,
    gradient_slice,
    token_coordinates,
):
    return I.matmul(
        query_slice,
        gradient_slice,
        transpose_lhs=True,
        acc_dtype=I.f32,
    )


@intent.fn
def emit_linear_backward_kv_slice(
    query_slice,
    key_slice,
    value_slice,
    gradient_slice,
    token_coordinates,
    incoming_state,
):
    future = token_coordinates[None, :] >= token_coordinates[:, None]
    score_gradient = I.matmul(
        value_slice,
        gradient_slice,
        transpose_rhs=True,
        acc_dtype=I.f32,
    )
    score_gradient = I.mask(score_gradient, valid=future, fill=0.0)
    local_grad_key = I.matmul(
        I.cast(score_gradient, I.f16),
        query_slice,
        acc_dtype=I.f32,
    )
    carried_grad_key = I.matmul(
        value_slice,
        I.cast(incoming_state, I.f16),
        transpose_rhs=True,
        acc_dtype=I.f32,
    )

    scores = I.matmul(
        key_slice,
        query_slice,
        transpose_rhs=True,
        acc_dtype=I.f32,
    )
    scores = I.mask(scores, valid=future, fill=0.0)
    local_grad_value = I.matmul(
        I.cast(scores, I.f16),
        gradient_slice,
        acc_dtype=I.f32,
    )
    carried_grad_value = I.matmul(
        key_slice,
        I.cast(incoming_state, I.f16),
        acc_dtype=I.f32,
    )
    return I.record(
        key=local_grad_key + carried_grad_key,
        value=local_grad_value + carried_grad_value,
    )


@intent.fn
def compose_retention_transition(lhs, rhs):
    return I.record(
        scale=rhs.scale * lhs.scale,
        matrix=rhs.scale * lhs.matrix + rhs.matrix,
    )


@intent.kernel
def fused_chunk_linear_attention_fwd(
    q: I.In[I.f16, ("B", "S", "H", "D")],
    k: I.In[I.f16, ("B", "S", "H", "D")],
    v: I.In[I.f16, ("B", "S", "H", "DV")],
    output: I.Out[I.f32, ("B", "S", "H", "DV")],
    final_state: I.Out[I.f32, ("B", "H", "D", "DV")],
    scale: I.f32,
):
    B, S, H, D = q.shape
    DV = v.shape[3]
    positions = I.domain(0, S)
    key_dimensions = I.domain(0, D)
    value_dimensions = I.domain(0, DV)
    token_coordinates = I.indices(positions)
    for batch in I.parallel(I.domain(0, B)):
        for head in I.parallel(I.domain(0, H)):
            query = I.cast(
                I.cast(q[batch, positions, head, key_dimensions], I.f32)
                * scale,
                I.f16,
            )
            key = k[batch, positions, head, key_dimensions]
            value = v[batch, positions, head, value_dimensions]
            identity = I.zeros((D, DV), dtype=I.f32)
            result, state = I.region_scan(
                source=(query, key, value, token_coordinates),
                axis=0,
                summarize=summarize_linear_forward_slice,
                combine=add_matrix_summary,
                identity=identity,
                initial_state=identity,
                apply=apply_matrix_summary,
                emit=emit_linear_forward_slice,
            )
            output[batch, positions, head, value_dimensions] = result
            final_state[batch, head, key_dimensions, value_dimensions] = state


@intent.kernel
def fused_chunk_linear_attention_bwd(
    q: I.In[I.f16, ("B", "S", "H", "D")],
    k: I.In[I.f16, ("B", "S", "H", "D")],
    v: I.In[I.f16, ("B", "S", "H", "DV")],
    grad_output: I.In[I.f16, ("B", "S", "H", "DV")],
    grad_q: I.Out[I.f32, ("B", "S", "H", "D")],
    grad_k: I.Out[I.f32, ("B", "S", "H", "D")],
    grad_v: I.Out[I.f32, ("B", "S", "H", "DV")],
    scale: I.f32,
):
    B, S, H, D = q.shape
    DV = v.shape[3]
    positions = I.domain(0, S)
    key_dimensions = I.domain(0, D)
    value_dimensions = I.domain(0, DV)
    token_coordinates = I.indices(positions)
    reverse_positions = S - 1 - token_coordinates
    for batch in I.parallel(I.domain(0, B)):
        for head in I.parallel(I.domain(0, H)):
            key = k[batch, positions, head, key_dimensions]
            value = v[batch, positions, head, value_dimensions]
            gradient = grad_output[batch, positions, head, value_dimensions]
            q_identity = I.zeros((DV, D), dtype=I.f32)
            query_gradient, _ = I.region_scan(
                source=(key, value, gradient, token_coordinates),
                axis=0,
                summarize=summarize_linear_backward_q_slice,
                combine=add_matrix_summary,
                identity=q_identity,
                initial_state=q_identity,
                apply=apply_matrix_summary,
                emit=emit_linear_backward_q_slice,
            )
            grad_q[batch, positions, head, key_dimensions] = query_gradient * scale

            reverse_query = I.cast(
                I.cast(
                    q[batch, reverse_positions, head, key_dimensions],
                    I.f32,
                )
                * scale,
                I.f16,
            )
            reverse_key = k[batch, reverse_positions, head, key_dimensions]
            reverse_value = v[batch, reverse_positions, head, value_dimensions]
            reverse_gradient = grad_output[
                batch, reverse_positions, head, value_dimensions
            ]
            kv_identity = I.zeros((D, DV), dtype=I.f32)
            gradients, _ = I.region_scan(
                source=(
                    reverse_query,
                    reverse_key,
                    reverse_value,
                    reverse_gradient,
                    reverse_positions,
                ),
                axis=0,
                summarize=summarize_linear_backward_kv_slice,
                combine=add_matrix_summary,
                identity=kv_identity,
                initial_state=kv_identity,
                apply=apply_matrix_summary,
                emit=emit_linear_backward_kv_slice,
            )
            I.scatter_unique(
                grad_k,
                index=(batch, reverse_positions, head, key_dimensions),
                value=gradients.key,
            )
            I.scatter_unique(
                grad_v,
                index=(batch, reverse_positions, head, value_dimensions),
                value=gradients.value,
            )


@intent.kernel
def chunk_retention_fwd(
    q: I.In[I.f16, ("B", "S", "H", "D")],
    k: I.In[I.f16, ("B", "S", "H", "D")],
    v: I.In[I.f16, ("B", "S", "H", "DV")],
    output: I.Out[I.f16, ("B", "S", "H", "DV")],
    scale: I.f32,
):
    B, S, H, D = q.shape
    DV = v.shape[3]
    positions = I.domain(0, S)
    key_dimensions = I.domain(0, D)
    value_dimensions = I.domain(0, DV)
    for batch in I.parallel(I.domain(0, B)):
        for head in I.parallel(I.domain(0, H)):
            log_decay = I.log(1.0 - I.exp(-5.0 - I.cast(head, I.f32)))
            decay = I.exp(log_decay)
            query = I.cast(
                I.cast(q[batch, positions, head, key_dimensions], I.f32)
                * scale,
                I.f16,
            )
            key = k[batch, positions, head, key_dimensions]
            value = v[batch, positions, head, value_dimensions]
            transitions = I.record(
                scale=I.full((S,), fill=decay, dtype=I.f32),
                matrix=I.cast(
                    key[:, :, None] * value[:, None, :],
                    I.f32,
                ),
            )
            states = I.scan(
                transitions,
                axis=0,
                identity=I.record(
                    scale=I.cast(1.0, I.f32),
                    matrix=I.zeros((D, DV), dtype=I.f32),
                ),
                combine=compose_retention_transition,
                inclusive=True,
                reverse=False,
            )
            result = I.reduce.sum(
                query[:, :, None] * I.cast(states.matrix, I.f16),
                axis=1,
            )
            output[batch, positions, head, value_dimensions] = I.cast(
                result,
                I.f16,
            )
