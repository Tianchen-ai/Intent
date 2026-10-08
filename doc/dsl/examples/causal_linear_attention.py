import intent
import intent.language as I


@intent.fn
def positive_feature_map(value):
    value = I.cast(value, I.f32)
    return I.maximum(value, 0.0) + 1.0


@intent.fn
def combine_linear_transitions(lhs, rhs):
    return I.record(
        matrix=lhs.matrix + rhs.matrix,
        key=lhs.key + rhs.key,
    )


@intent.fn
def summarize_linear_slice(
    key_slice,
    value_slice,
    token_coordinates,
    query_slice,
):
    # region_scan slices every source component at the same boundaries before
    # calling both helpers.  This summary does not use the coordinate or query
    # slices; emit consumes their corresponding slices directly.
    matrix = I.matmul(
        key_slice,
        value_slice,
        transpose_lhs=True,
        acc_dtype=I.f32,
    )
    key = I.reduce.sum(
        key_slice,
        axis=0,
    )
    return I.record(matrix=matrix, key=key)


@intent.fn
def apply_linear_transition(prefix, initial_state):
    return I.record(
        matrix=initial_state.matrix + prefix.matrix,
        key=initial_state.key + prefix.key,
    )


@intent.fn
def emit_linear_slice(
    key_slice,
    value_slice,
    token_coordinates,
    query_slice,
    incoming_state,
):
    # These coordinates still refer to the complete source token domain; they
    # are not ordinals local to the compiler-selected slice.  query_slice was
    # cut at the same source boundaries and needs no gather reconstruction.

    inter_numerator = I.matmul(
        query_slice,
        incoming_state.matrix,
        acc_dtype=I.f32,
    )
    inter_denominator = I.matvec(
        query_slice,
        incoming_state.key,
        acc_dtype=I.f32,
    )

    local_scores = I.matmul(
        query_slice,
        key_slice,
        transpose_rhs=True,
        acc_dtype=I.f32,
    )
    causal = (
        token_coordinates[:, None]
        >= token_coordinates[None, :]
    )
    local_scores = I.select(causal, local_scores, 0.0)
    intra_numerator = I.matmul(
        local_scores,
        value_slice,
        acc_dtype=I.f32,
    )
    intra_denominator = I.reduce.sum(
        local_scores,
        axis=1,
    )

    numerator = inter_numerator + intra_numerator
    denominator = inter_denominator + intra_denominator
    nonzero = denominator != 0.0
    safe_denominator = I.select(nonzero, denominator, 1.0)
    return I.select(
        nonzero[:, None],
        numerator / safe_denominator[:, None],
        0.0,
    )


@intent.kernel
def causal_linear_attention(
    q: I.In[I.f16, ("B", "H", "T", "D")],
    k: I.In[I.f16, ("B", "H", "T", "D")],
    v: I.In[I.f16, ("B", "H", "T", "DV")],
    output: I.Out[I.f16, ("B", "H", "T", "DV")],
    final_matrix: I.Out[I.f32, ("B", "H", "D", "DV")],
    final_key: I.Out[I.f32, ("B", "H", "D")],
):
    B, H, T, D = q.shape
    DV = v.shape[-1]
    token_axis = I.domain(0, T)
    token_coordinates = I.indices(token_axis)

    for batch in I.parallel(I.domain(0, B)):
        for head in I.parallel(I.domain(0, H)):
            queries = positive_feature_map(
                q[batch, head, token_axis, :]
            )
            keys = positive_feature_map(
                k[batch, head, token_axis, :]
            )
            values = I.cast(
                v[batch, head, token_axis, :],
                I.f32,
            )
            initial_state = I.record(
                matrix=I.zeros((D, DV), dtype=I.f32),
                key=I.zeros((D,), dtype=I.f32),
            )
            identity = I.record(
                matrix=I.zeros((D, DV), dtype=I.f32),
                key=I.zeros((D,), dtype=I.f32),
            )

            outputs, final_state = I.region_scan(
                source=(keys, values, token_coordinates, queries),
                axis=0,
                summarize=summarize_linear_slice,
                combine=combine_linear_transitions,
                identity=identity,
                initial_state=initial_state,
                apply=apply_linear_transition,
                emit=emit_linear_slice,
            )

            output[batch, head, token_axis, :] = I.cast(
                outputs,
                I.f16,
            )
            final_matrix[batch, head, :, :] = final_state.matrix
            final_key[batch, head, :] = final_state.key


# The token domain and output are algorithm-visible; region boundaries are not.
# summarize forms K^T V and sum(K), while emit forms the inter-slice and causal
# intra-slice contractions.  Changing the hidden segmentation therefore changes
# neither token outputs nor final state.
