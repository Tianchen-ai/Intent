import math

import intent
import intent.language as I


BATCH = 2
SEQUENCE = 2048
HEADS = 8
KEY_DIMENSION = 128
VALUE_DIMENSION = 128
SCALE = 1.0 / math.sqrt(KEY_DIMENSION)
CHUNK_SIZE = 64


@intent.kernel
def recurrent_gated_delta_fwd(
    query: I.In[I.bf16, ("B", "T", "H", "K")],
    key: I.In[I.bf16, ("B", "T", "H", "K")],
    value: I.In[I.bf16, ("B", "T", "HV", "V")],
    gate: I.In[I.bf16, ("B", "T", "HV")],
    beta: I.In[I.bf16, ("B", "T", "HV")],
    output: I.Out[I.bf16, ("B", "T", "HV", "V")],
    final_state: I.Out[I.f32, ("B", "HV", "K", "V")],
    scale: I.f32,
    HEAD_GROUP: I.Constexpr[int],
):
    B, T, _, K = query.shape
    HV = value.shape[2]
    V = value.shape[3]
    positions = I.domain(0, T)
    key_features = I.domain(0, K)
    value_features = I.domain(0, V)
    for batch in I.parallel(I.domain(0, B)):
        for value_head in I.parallel(I.domain(0, HV)):
            query_head = value_head // HEAD_GROUP
            I.assume_in_bounds(query_head, query, axis=2)
            I.assume_in_bounds(query_head, key, axis=2)
            state = I.zeros((K, V), dtype=I.f32)
            for position in positions:
                query_vector = I.cast(
                    query[batch, position, query_head, key_features],
                    I.f32,
                ) * scale
                key_vector = I.cast(
                    key[batch, position, query_head, key_features],
                    I.f32,
                )
                value_vector = I.cast(
                    value[batch, position, value_head, value_features],
                    I.f32,
                )
                decay = I.exp(I.cast(gate[batch, position, value_head], I.f32))
                decayed_state = state * decay
                remembered = I.reduce.sum(
                    decayed_state * key_vector[:, None],
                    axis=0,
                )
                update = (value_vector - remembered) * I.cast(
                    beta[batch, position, value_head],
                    I.f32,
                )
                state = decayed_state + key_vector[:, None] * update[None, :]
                output[batch, position, value_head, value_features] = I.cast(
                    I.reduce.sum(
                        state * query_vector[:, None],
                        axis=0,
                    ),
                    I.bf16,
                )
            final_state[
                batch,
                value_head,
                key_features,
                value_features,
            ] = state


@intent.kernel
def chunk_gated_delta_prepare(
    query: I.In[I.bf16, ("B", "T", "H", "K")],
    key: I.In[I.bf16, ("B", "T", "H", "K")],
    value: I.In[I.bf16, ("B", "T", "H", "V")],
    gate: I.In[I.bf16, ("B", "T", "H")],
    beta: I.In[I.bf16, ("B", "T", "H")],
    query_chunks: I.Out[I.bf16, ("B", "H", "T", "K")],
    key_chunks: I.Out[I.bf16, ("B", "H", "T", "K")],
    corrected_values: I.Out[I.bf16, ("B", "H", "T", "V")],
    cumulative_keys: I.Out[I.bf16, ("B", "H", "T", "K")],
    cumulative_gate: I.Out[I.f32, ("B", "H", "T")],
    scale: I.f32,
):
    B, T, H, K = query.shape
    C = (T + CHUNK_SIZE - 1) // CHUNK_SIZE
    V = value.shape[3]
    sequence = I.domain(0, T)
    key_dimensions = I.domain(0, K)
    value_dimensions = I.domain(0, V)
    for batch in I.parallel(I.domain(0, B)):
        for head in I.parallel(I.domain(0, H)):
            for chunk in I.parallel(I.domain(0, C)):
                chunk_begin = chunk * CHUNK_SIZE
                chunk_end = I.minimum(chunk_begin + CHUNK_SIZE, T)
                chunk_positions = sequence[chunk_begin:chunk_end]
                source_positions = I.indices(chunk_positions)
                local = source_positions - chunk_begin
                lower = local[:, None] >= local[None, :]
                strict_lower = local[:, None] > local[None, :]
                identity = I.cast(local[:, None] == local[None, :], I.f32)
                key_input = key[batch, source_positions, head, key_dimensions]
                key_values = I.cast(key_input, I.f32)
                beta_values = I.cast(beta[batch, source_positions, head], I.f32)
                gate_values = I.cast(gate[batch, source_positions, head], I.f32)
                gate_prefix = I.cumsum(
                    gate_values,
                    axis=0,
                )
                decay = I.mask(
                    I.exp(gate_prefix[:, None] - gate_prefix[None, :]),
                    valid=lower,
                    fill=0.0,
                )
                weighted_keys = key_values * beta_values[:, None]
                base = I.matmul(
                    I.cast(weighted_keys, I.bf16),
                    key_input,
                    transpose_rhs=True,
                    acc_dtype=I.f32,
                )
                triangular = I.mask(
                    -(base * decay),
                    valid=strict_lower,
                    fill=0.0,
                )
                row_norm = I.reduce.sum(
                    I.abs(triangular),
                    axis=1,
                )
                norm = I.reduce.max(row_norm, axis=0)
                inverse = triangular
                if norm < 1.0:
                    inverse = identity + triangular
                    power = triangular
                    for _ in range(1, 6):
                        power_input = I.cast(power, I.bf16)
                        power = I.matmul(
                            power_input,
                            power_input,
                            acc_dtype=I.f32,
                        )
                        inverse = I.matmul(
                            I.cast(inverse, I.bf16),
                            I.cast(identity + power, I.bf16),
                            acc_dtype=I.f32,
                        )
                else:
                    for row_index in I.domain(1, chunk_end - chunk_begin):
                        is_row = local == row_index
                        row = I.reduce.sum(
                            I.mask(
                                inverse,
                                valid=is_row[:, None],
                                fill=0.0,
                            ),
                            axis=0,
                        )
                        correction = I.reduce.sum(
                            row[:, None] * inverse,
                            axis=0,
                        )
                        inverse = inverse + I.mask(
                            correction[None, :],
                            valid=is_row[:, None],
                            fill=0.0,
                        )
                    inverse = inverse + identity

                value_values = I.cast(
                    value[batch, source_positions, head, value_dimensions],
                    I.f32,
                )
                inverse_input = I.cast(inverse, I.bf16)
                corrected = I.matmul(
                    inverse_input,
                    I.cast(value_values * beta_values[:, None], I.bf16),
                    acc_dtype=I.f32,
                )
                cumulative_key = I.matmul(
                    inverse_input,
                    I.cast(weighted_keys * I.exp(gate_prefix)[:, None], I.bf16),
                    acc_dtype=I.f32,
                )
                query_values = I.cast(
                    I.cast(
                        query[batch, source_positions, head, key_dimensions],
                        I.f32,
                    )
                    * scale,
                    I.bf16,
                )
                query_chunks[batch, head, source_positions, key_dimensions] = (
                    query_values
                )
                key_chunks[batch, head, source_positions, key_dimensions] = key_input
                corrected_values[
                    batch,
                    head,
                    source_positions,
                    value_dimensions,
                ] = I.cast(corrected, I.bf16)
                cumulative_keys[
                    batch,
                    head,
                    source_positions,
                    key_dimensions,
                ] = I.cast(cumulative_key, I.bf16)
                cumulative_gate[batch, head, source_positions] = gate_prefix


@intent.kernel
def chunk_gated_delta_recurrence(
    query_chunks: I.In[I.bf16, ("B", "H", "T", "K")],
    key_chunks: I.In[I.bf16, ("B", "H", "T", "K")],
    corrected_values: I.In[I.bf16, ("B", "H", "T", "V")],
    cumulative_keys: I.In[I.bf16, ("B", "H", "T", "K")],
    cumulative_gate: I.In[I.f32, ("B", "H", "T")],
    output: I.Out[I.bf16, ("B", "T", "H", "V")],
    final_state: I.Out[I.f32, ("B", "H", "K", "V")],
):
    B, H, T, K = query_chunks.shape
    C = (T + CHUNK_SIZE - 1) // CHUNK_SIZE
    V = corrected_values.shape[3]
    sequence = I.domain(0, T)
    key_dimensions = I.domain(0, K)
    value_dimensions = I.domain(0, V)
    for batch in I.parallel(I.domain(0, B)):
        for head in I.parallel(I.domain(0, H)):
            state = I.zeros((K, V), dtype=I.f32)
            for chunk in I.domain(0, C):
                chunk_begin = chunk * CHUNK_SIZE
                chunk_end = I.minimum(chunk_begin + CHUNK_SIZE, T)
                chunk_positions = sequence[chunk_begin:chunk_end]
                source_positions = I.indices(chunk_positions)
                local = source_positions - chunk_begin
                causal = local[:, None] >= local[None, :]
                query = query_chunks[
                    batch,
                    head,
                    source_positions,
                    key_dimensions,
                ]
                key = key_chunks[
                    batch,
                    head,
                    source_positions,
                    key_dimensions,
                ]
                corrected = I.cast(
                    corrected_values[
                        batch,
                        head,
                        source_positions,
                        value_dimensions,
                    ],
                    I.f32,
                )
                cumulative_key = cumulative_keys[
                    batch,
                    head,
                    source_positions,
                    key_dimensions,
                ]
                gate_prefix = cumulative_gate[batch, head, source_positions]
                # Keep the recurrent state in f32; round only the matmul inputs.
                state_input = I.cast(state, I.bf16)
                projected_state = I.matmul(
                    cumulative_key,
                    state_input,
                    acc_dtype=I.f32,
                )
                corrected = corrected - projected_state
                corrected_input = I.cast(corrected, I.bf16)
                weighted_query = I.cast(query, I.f32) * I.exp(gate_prefix)[:, None]
                inter = I.matmul(
                    I.cast(weighted_query, I.bf16),
                    state_input,
                    acc_dtype=I.f32,
                )
                scores = I.matmul(
                    query,
                    key,
                    transpose_rhs=True,
                    acc_dtype=I.f32,
                )
                decay = I.mask(
                    I.exp(gate_prefix[:, None] - gate_prefix[None, :]),
                    valid=causal,
                    fill=0.0,
                )
                intra = I.matmul(
                    I.cast(scores * decay, I.bf16),
                    corrected_input,
                    acc_dtype=I.f32,
                )
                output[
                    batch,
                    source_positions,
                    head,
                    value_dimensions,
                ] = I.cast(inter + intra, I.bf16)
                last_position = chunk_end - 1
                I.assume_in_bounds(last_position, cumulative_gate, axis=2)
                last_gate = cumulative_gate[batch, head, last_position]
                weighted_key = I.cast(key, I.f32) * I.exp(last_gate - gate_prefix)[:, None]
                update = I.matmul(
                    I.cast(weighted_key, I.bf16),
                    corrected_input,
                    transpose_lhs=True,
                    acc_dtype=I.f32,
                )
                state = state * I.exp(last_gate) + update
            final_state[batch, head, key_dimensions, value_dimensions] = state
