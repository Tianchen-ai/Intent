import intent
import intent.language as I


BATCH = 1
SEQUENCE = 2048
HEADS = 32
GROUPS = 8
HEAD_DIMENSION = 64
STATE_DIMENSION = 128
CHUNK_SIZE = 256
CHUNKS = SEQUENCE // CHUNK_SIZE
HEADS_PER_GROUP = HEADS // GROUPS
MAMBA3_QK_HEADS = 4
MAMBA3_HEADS = 16
MAMBA3_QK_DIMENSION = 32
MAMBA3_VALUE_DIMENSION = 64
MAMBA3_ANGLE_DIMENSION = 16
MAMBA3_CHUNK_SIZE = 64


@intent.kernel
def mamba_chunk_state_fwd(
    state_basis: I.In[I.f16, ("B", "L", "G", "N")],
    x: I.In[I.f16, ("B", "L", "H", "P")],
    dt: I.In[I.f16, ("B", "H", "C", "S")],
    cumulative_decay: I.In[I.f16, ("B", "H", "C", "S")],
    states: I.Out[I.f16, ("B", "C", "H", "P", "N")],
    HEAD_GROUP: I.Constexpr[int],
):
    B, L, H, P = x.shape
    C = dt.shape[2]
    S = dt.shape[3]
    N = state_basis.shape[3]
    chunk_axis = I.domain(0, S)
    state_axis = I.domain(0, N)
    dimensions = I.domain(0, P)
    for batch in I.parallel(I.domain(0, B)):
        for chunk in I.parallel(I.domain(0, C)):
            for head in I.parallel(I.domain(0, H)):
                group = head // HEAD_GROUP
                I.assume_in_bounds(group, state_basis, axis=2)
                last_position = S - 1
                I.assume_in_bounds(
                    last_position, cumulative_decay, axis=3
                )
                last_decay = I.cast(
                    cumulative_decay[batch, head, chunk, last_position], I.f32
                )
                scale = I.exp(
                    I.minimum(
                        last_decay
                        - I.cast(
                            cumulative_decay[batch, head, chunk, chunk_axis],
                            I.f32,
                        ),
                        0.0,
                    )
                ) * I.cast(dt[batch, head, chunk, chunk_axis], I.f32)
                lhs = I.cast(
                    x[
                        batch,
                        chunk * S + I.indices(chunk_axis),
                        head,
                        dimensions,
                    ],
                    I.f16,
                )
                rhs = I.cast(
                    state_basis[
                        batch,
                        chunk * S + I.indices(chunk_axis),
                        group,
                        state_axis,
                    ]
                    * I.cast(scale[:, None], I.f16),
                    I.f16,
                )
                result = I.matmul(
                    lhs,
                    rhs,
                    transpose_lhs=True,
                    acc_dtype=I.f32,
                )
                states[batch, chunk, head, dimensions, state_axis] = (
                    I.cast(result, I.f16)
                )


@intent.kernel
def mamba_chunk_state_bf16_fwd(
    state_basis: I.In[I.bf16, ("B", "L", "G", "N")],
    x: I.In[I.bf16, ("B", "L", "H", "P")],
    dt: I.In[I.f32, ("B", "H", "C", "S")],
    cumulative_decay: I.In[I.f32, ("B", "H", "C", "S")],
    states: I.Out[I.f32, ("B", "C", "H", "P", "N")],
    HEAD_GROUP: I.Constexpr[int],
):
    B, L, H, P = x.shape
    C = dt.shape[2]
    S = dt.shape[3]
    N = state_basis.shape[3]
    chunk_axis = I.domain(0, S)
    state_axis = I.domain(0, N)
    dimensions = I.domain(0, P)
    for batch in I.parallel(I.domain(0, B)):
        for chunk in I.parallel(I.domain(0, C)):
            for head in I.parallel(I.domain(0, H)):
                group = head // HEAD_GROUP
                I.assume_in_bounds(group, state_basis, axis=2)
                last_position = S - 1
                I.assume_in_bounds(
                    last_position, cumulative_decay, axis=3
                )
                last_decay = cumulative_decay[
                    batch, head, chunk, last_position
                ]
                scale = I.exp(
                    I.minimum(
                        last_decay
                        - cumulative_decay[batch, head, chunk, chunk_axis],
                        0.0,
                    )
                ) * dt[batch, head, chunk, chunk_axis]
                lhs = x[
                    batch,
                    chunk * S + I.indices(chunk_axis),
                    head,
                    dimensions,
                ]
                rhs = I.cast(
                    I.cast(
                        state_basis[
                            batch,
                            chunk * S + I.indices(chunk_axis),
                            group,
                            state_axis,
                        ],
                        I.f32,
                    )
                    * scale[:, None],
                    I.bf16,
                )
                states[
                    batch,
                    chunk,
                    head,
                    dimensions,
                    state_axis,
                ] = I.matmul(
                    lhs,
                    rhs,
                    transpose_lhs=True,
                    acc_dtype=I.f32,
                )


@intent.kernel
def mamba_state_passing_fwd(
    chunk_states: I.In[I.f32, ("B", "C", "H", "D")],
    chunk_decay: I.In[I.f32, ("B", "H", "C")],
    initial_states: I.In[I.f32, ("B", "H", "D")],
    states_before_chunk: I.Out[I.f32, ("B", "C", "H", "D")],
    final_states: I.Out[I.f32, ("B", "H", "D")],
):
    B, C, H, D = chunk_states.shape
    dimensions = I.domain(0, D)
    for batch in I.parallel(I.domain(0, B)):
        for head in I.parallel(I.domain(0, H)):
            state = I.cast(initial_states[batch, head, dimensions], I.f32)
            for chunk in I.domain(0, C):
                states_before_chunk[batch, chunk, head, dimensions] = state
                state = (
                    I.exp(chunk_decay[batch, head, chunk]) * state
                    + chunk_states[batch, chunk, head, dimensions]
                )
            final_states[batch, head, dimensions] = state


@intent.kernel
def mamba3_siso_step(
    query: I.In[I.bf16, ("B", MAMBA3_QK_HEADS, MAMBA3_QK_DIMENSION)],
    key: I.In[I.bf16, ("B", MAMBA3_QK_HEADS, MAMBA3_QK_DIMENSION)],
    value: I.In[I.bf16, ("B", MAMBA3_HEADS, MAMBA3_VALUE_DIMENSION)],
    adt: I.In[I.f32, ("B", MAMBA3_HEADS)],
    dt: I.In[I.f32, ("B", MAMBA3_HEADS)],
    trap: I.In[I.f32, ("B", MAMBA3_HEADS)],
    query_bias: I.In[I.bf16, (MAMBA3_HEADS, MAMBA3_QK_DIMENSION)],
    key_bias: I.In[I.bf16, (MAMBA3_HEADS, MAMBA3_QK_DIMENSION)],
    angles: I.In[I.f32, ("B", MAMBA3_HEADS, MAMBA3_ANGLE_DIMENSION)],
    residual_scale: I.In[I.f32, (MAMBA3_HEADS,)],
    gate: I.In[I.bf16, ("B", MAMBA3_HEADS, MAMBA3_VALUE_DIMENSION)],
    input_angle_state: I.In[
        I.f32, ("B", MAMBA3_HEADS, MAMBA3_ANGLE_DIMENSION)
    ],
    input_ssm_state: I.In[
        I.f32,
        ("B", MAMBA3_HEADS, MAMBA3_VALUE_DIMENSION, MAMBA3_QK_DIMENSION),
    ],
    input_key_state: I.In[
        I.f32, ("B", MAMBA3_HEADS, MAMBA3_QK_DIMENSION)
    ],
    input_value_state: I.In[
        I.f32, ("B", MAMBA3_HEADS, MAMBA3_VALUE_DIMENSION)
    ],
    output: I.Out[I.bf16, ("B", MAMBA3_HEADS, MAMBA3_VALUE_DIMENSION)],
    output_angle_state: I.Out[
        I.f32, ("B", MAMBA3_HEADS, MAMBA3_ANGLE_DIMENSION)
    ],
    output_ssm_state: I.Out[
        I.f32,
        ("B", MAMBA3_HEADS, MAMBA3_VALUE_DIMENSION, MAMBA3_QK_DIMENSION),
    ],
    output_key_state: I.Out[
        I.f32, ("B", MAMBA3_HEADS, MAMBA3_QK_DIMENSION)
    ],
    HEAD_GROUP: I.Constexpr[int],
):
    B = query.shape[0]
    value_dimensions = I.domain(0, MAMBA3_VALUE_DIMENSION)
    angle_dimensions = I.domain(0, MAMBA3_ANGLE_DIMENSION)
    qk_dimensions = I.domain(0, MAMBA3_QK_DIMENSION)
    for batch in I.parallel(I.domain(0, B)):
        for head in I.parallel(I.domain(0, MAMBA3_HEADS)):
            angle_region = angle_dimensions
            value_region = value_dimensions
            query_head = head // HEAD_GROUP
            query_block = I.cast(
                query[batch, query_head, qk_dimensions], I.f32
            ) + I.cast(query_bias[head, qk_dimensions], I.f32)
            key_block = I.cast(
                key[batch, query_head, qk_dimensions], I.f32
            ) + I.cast(key_bias[head, qk_dimensions], I.f32)
            query_pairs = I.reshape(query_block, (angle_region, 2))
            key_pairs = I.reshape(key_block, (angle_region, 2))
            query_first = query_pairs[:, 0]
            query_second = query_pairs[:, 1]
            key_first = key_pairs[:, 0]
            key_second = key_pairs[:, 1]

            angle_delta = (
                2.0
                * I.sigmoid(2.0 * angles[batch, head, angle_region])
                - 1.0
            ) * 3.141592653589793 * dt[batch, head]
            angle = (
                angle_delta
                + input_angle_state[batch, head, angle_region]
            )
            angle = angle - 6.283185307179586 * I.floor(
                angle / 6.283185307179586
            )
            output_angle_state[batch, head, angle_region] = angle
            cosine = I.cos(angle)
            sine = I.sin(angle)

            rotated_query_first = I.cast(
                query_first * cosine - query_second * sine, I.bf16
            )
            rotated_query_second = I.cast(
                query_first * sine + query_second * cosine, I.bf16
            )
            rotated_key_first = I.cast(
                key_first * cosine - key_second * sine, I.bf16
            )
            rotated_key_second = I.cast(
                key_first * sine + key_second * cosine, I.bf16
            )
            rotated_key = I.reshape(
                I.join(rotated_key_first, rotated_key_second),
                (qk_dimensions,),
            )
            output_key_state[batch, head, qk_dimensions] = I.cast(
                rotated_key, I.f32
            )

            alpha = I.exp2(adt[batch, head] * I.LOG2E)
            trap_value = I.sigmoid(trap[batch, head])
            beta = alpha * dt[batch, head] * (1.0 - trap_value)
            gamma = trap_value * dt[batch, head]
            previous_key = input_key_state[batch, head, qk_dimensions]
            previous_value = input_value_state[
                batch, head, value_region
            ]
            current_value = I.cast(
                value[batch, head, value_region], I.f32
            )
            state = (
                input_ssm_state[
                    batch, head, value_region, qk_dimensions
                ]
                * alpha
                + (beta * previous_value)[:, None]
                * previous_key[None, :]
                + (gamma * current_value)[:, None]
                * I.cast(rotated_key, I.f32)[None, :]
            )
            output_ssm_state[batch, head, value_region, qk_dimensions] = state
            rotated_query = I.reshape(
                I.join(rotated_query_first, rotated_query_second),
                (qk_dimensions,),
            )
            projected = I.matmul(
                I.cast(state, I.bf16),
                I.reshape(rotated_query, (qk_dimensions, 1)),
                acc_dtype=I.f32,
            )
            projected = I.reshape(projected, (value_region,))
            gate_value = I.cast(
                gate[batch, head, value_region], I.f32
            )
            output[batch, head, value_region] = I.cast(
                (projected + residual_scale[head] * current_value)
                * gate_value
                * I.sigmoid(gate_value),
                I.bf16,
            )


@intent.kernel
def mamba3_siso_forward(
    query: I.In[I.bf16, ("B", "S", MAMBA3_QK_HEADS, MAMBA3_QK_DIMENSION)],
    key: I.In[I.bf16, ("B", "S", MAMBA3_QK_HEADS, MAMBA3_QK_DIMENSION)],
    value: I.In[I.bf16, ("B", "S", MAMBA3_HEADS, MAMBA3_VALUE_DIMENSION)],
    adt: I.In[I.f32, ("B", MAMBA3_HEADS, "S")],
    dt: I.In[I.f32, ("B", MAMBA3_HEADS, "S")],
    trap: I.In[I.f32, ("B", MAMBA3_HEADS, "S")],
    query_bias: I.In[I.bf16, (MAMBA3_HEADS, MAMBA3_QK_DIMENSION)],
    key_bias: I.In[I.bf16, (MAMBA3_HEADS, MAMBA3_QK_DIMENSION)],
    angles: I.In[
        I.f32, ("B", "S", MAMBA3_HEADS, MAMBA3_ANGLE_DIMENSION)
    ],
    residual_scale: I.In[I.f32, (MAMBA3_HEADS,)],
    gate: I.In[I.bf16, ("B", "S", MAMBA3_HEADS, MAMBA3_VALUE_DIMENSION)],
    output: I.Out[I.bf16, ("B", "S", MAMBA3_HEADS, MAMBA3_VALUE_DIMENSION)],
    query_store: I.InOut[
        I.bf16, ("B", "S", MAMBA3_HEADS, MAMBA3_QK_DIMENSION)
    ],
    key_store: I.InOut[
        I.bf16, ("B", "S", MAMBA3_HEADS, MAMBA3_QK_DIMENSION)
    ],
    qk_store: I.InOut[I.f32, ("B", MAMBA3_HEADS, "S")],
    scale_store: I.InOut[I.f32, ("B", MAMBA3_HEADS, "S")],
    gamma_store: I.InOut[I.f32, ("B", MAMBA3_HEADS, "S")],
    HEAD_GROUP: I.Constexpr[int],
):
    B, S, _, _ = query.shape
    chunks = (S + MAMBA3_CHUNK_SIZE - 1) // MAMBA3_CHUNK_SIZE
    sequence = I.domain(0, S)
    pairs = I.domain(0, MAMBA3_QK_DIMENSION // 2)
    qk_dimensions = I.domain(0, MAMBA3_QK_DIMENSION)
    value_dimensions = I.domain(0, MAMBA3_VALUE_DIMENSION)
    qk_dimension_indices = I.indices(qk_dimensions)
    value_dimension_indices = I.indices(value_dimensions)
    pair = I.indices(pairs)
    even = pair * 2
    odd = even + 1
    for batch in I.parallel(I.domain(0, B)):
        for head in I.parallel(I.domain(0, MAMBA3_HEADS)):
            query_head = head // HEAD_GROUP
            for chunk in I.domain(0, chunks):
                chunk_begin = chunk * MAMBA3_CHUNK_SIZE
                chunk_end = I.minimum(chunk_begin + MAMBA3_CHUNK_SIZE, S)
                chunk_positions = sequence[chunk_begin:chunk_end]
                chunk_length = chunk_end - chunk_begin
                source_positions = I.indices(chunk_positions)
                source_grid = source_positions[:, None]
                query_block = I.cast(
                    query[
                        batch,
                        source_grid,
                        query_head,
                        qk_dimension_indices[None, :],
                    ],
                    I.f32,
                ) + I.cast(
                    query_bias[head, qk_dimension_indices], I.f32
                )[None, :]
                key_block = I.cast(
                    key[
                        batch,
                        source_grid,
                        query_head,
                        qk_dimension_indices[None, :],
                    ],
                    I.f32,
                ) + I.cast(
                    key_bias[head, qk_dimension_indices], I.f32
                )[None, :]
                query_pairs = I.reshape(
                    query_block,
                    (chunk_positions, pairs, 2),
                )
                key_pairs = I.reshape(
                    key_block,
                    (chunk_positions, pairs, 2),
                )
                query_first = query_pairs[:, :, 0]
                query_second = query_pairs[:, :, 1]
                key_first = key_pairs[:, :, 0]
                key_second = key_pairs[:, :, 1]
                current_dt = dt[batch, head, source_positions]
                shifted_positions = source_positions + 1
                shifted_valid = shifted_positions < S
                safe_shifted_positions = I.select(
                    shifted_valid,
                    shifted_positions,
                    source_positions,
                )
                shifted_dt = I.mask(
                    dt[batch, head, safe_shifted_positions],
                    valid=shifted_valid,
                    fill=0.0,
                )
                current_trap = I.sigmoid(trap[batch, head, source_positions])
                shifted_trap = I.sigmoid(
                    I.mask(
                        trap[batch, head, safe_shifted_positions],
                        valid=shifted_valid,
                        fill=0.0,
                    )
                )
                gamma = current_dt * current_trap
                shifted_gamma = shifted_dt * (1.0 - shifted_trap)
                transition_scale = gamma + shifted_gamma
                angle = angles[batch, source_grid, head, pair[None, :]]
                cosine = I.cos(angle)
                sine = I.sin(angle)
                rotated_query_first = I.cast(
                    query_first * cosine - query_second * sine, I.bf16
                )
                rotated_query_second = I.cast(
                    query_first * sine + query_second * cosine, I.bf16
                )
                rotated_key_first = I.cast(
                    key_first * cosine - key_second * sine, I.bf16
                )
                rotated_key_second = I.cast(
                    key_first * sine + key_second * cosine, I.bf16
                )
                rotated_query = I.reshape(
                    I.join(rotated_query_first, rotated_query_second),
                    (chunk_positions, qk_dimensions),
                )
                rotated_key = I.reshape(
                    I.join(rotated_key_first, rotated_key_second),
                    (chunk_positions, qk_dimensions),
                ) * I.cast(transition_scale[:, None], I.bf16)
                qk_dot = I.reshape(
                    I.matmul(
                        I.cast(
                            query_first * key_first
                            + query_second * key_second,
                            I.bf16,
                        ),
                        I.full(
                            (pairs, 1),
                            1.0,
                            dtype=I.bf16,
                        ),
                        acc_dtype=I.f32,
                    ),
                    (chunk_positions,),
                ) * gamma
                query_store[batch, source_grid, head, qk_dimension_indices[None, :]] = (
                    rotated_query
                )
                key_store[batch, source_grid, head, qk_dimension_indices[None, :]] = (
                    rotated_key
                )
                qk_store[batch, head, source_positions] = qk_dot
                scale_store[batch, head, source_positions] = transition_scale
                gamma_store[batch, head, source_positions] = gamma

            state = I.zeros(
                (value_dimensions, qk_dimensions),
                dtype=I.f32,
            )
            residual = residual_scale[head]
            for chunk in I.domain(0, chunks):
                chunk_begin = chunk * MAMBA3_CHUNK_SIZE
                chunk_end = I.minimum(chunk_begin + MAMBA3_CHUNK_SIZE, S)
                chunk_positions = sequence[chunk_begin:chunk_end]
                source_positions = I.indices(chunk_positions)
                local = source_positions - chunk_begin
                strict_lower = local[:, None] > local[None, :]
                source_grid = source_positions[:, None]
                query_block = query_store[
                    batch,
                    source_grid,
                    head,
                    qk_dimension_indices[None, :],
                ]
                key_block = key_store[
                    batch,
                    source_grid,
                    head,
                    qk_dimension_indices[None, :],
                ]
                value_block = value[
                    batch,
                    source_grid,
                    head,
                    value_dimension_indices[None, :],
                ]
                decay_input = adt[batch, head, source_positions] * I.LOG2E
                decay = I.cumsum(
                    decay_input,
                    axis=0,
                )
                decay_sum = I.reduce.sum(
                    decay_input,
                    axis=0,
                )
                carried = I.matmul(
                    query_block,
                    I.cast(state, I.bf16),
                    transpose_rhs=True,
                    acc_dtype=I.f32,
                ) * I.exp2(decay)[:, None]
                scores = I.matmul(
                    query_block,
                    key_block,
                    transpose_rhs=True,
                    acc_dtype=I.f32,
                )
                scores = I.mask(
                    scores
                    * I.exp2(
                        I.minimum(decay[:, None] - decay[None, :], 0.0)
                    ),
                    valid=strict_lower,
                    fill=0.0,
                )
                current = I.matmul(
                    I.cast(scores, I.bf16),
                    value_block,
                    acc_dtype=I.f32,
                )
                skip = (
                    residual + qk_store[batch, head, source_positions]
                )[:, None] * I.cast(value_block, I.f32)
                result = carried + current + skip
                gate_value = I.cast(
                    gate[
                        batch,
                        source_grid,
                        head,
                        value_dimension_indices[None, :],
                    ],
                    I.f32,
                )
                result = result * gate_value * I.sigmoid(gate_value)
                output[
                    batch,
                    source_grid,
                    head,
                    value_dimension_indices[None, :],
                ] = I.cast(result, I.bf16)
                reverse_decay = decay_sum - decay
                weighted_value = I.cast(value_block, I.f32) * I.exp2(
                    reverse_decay
                )[:, None]
                update = I.matmul(
                    I.cast(weighted_value, I.bf16),
                    key_block,
                    transpose_lhs=True,
                    acc_dtype=I.f32,
                )
                state = state * I.exp2(decay_sum) + update
