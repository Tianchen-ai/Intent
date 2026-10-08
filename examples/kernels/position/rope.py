import intent
import intent.language as I


HEAD_DIMENSION = 128
HALF_DIMENSION = HEAD_DIMENSION // 2
ROPE_BATCH = 4
ROPE_SEQUENCE = 2048
ROPE_QUERY_HEADS = 32
ROPE_KEY_HEADS = 8
PARTIAL_ROTARY_DIMENSION = HEAD_DIMENSION // 2
PARTIAL_HALF_DIMENSION = PARTIAL_ROTARY_DIMENSION // 2


@intent.kernel
def rotary_embedding_flat(
    values: I.In[I.f16, ("R", HEAD_DIMENSION)],
    cosine: I.In[I.f16, ("U", HALF_DIMENSION)],
    sine: I.In[I.f16, ("U", HALF_DIMENSION)],
    output: I.Out[I.f16, ("R", 2, HALF_DIMENSION)],
    HEADS: I.Constexpr[int],
):
    R, D = values.shape
    _, half_dimension = cosine.shape
    dimensions = I.domain(0, D)
    for row in I.parallel(I.domain(0, R)):
        token = row // HEADS
        dimension_index = I.indices(dimensions)
        paired_dimension = (dimension_index - half_dimension) % D
        phase_dimension = dimension_index % half_dimension
        rotate_sign = I.cast(
            I.cast(dimension_index >= half_dimension, I.i32) * 2 - 1,
            I.f16,
        )
        rotated = (
            values[row, dimensions] * cosine[token, phase_dimension]
            + values[row, paired_dimension]
            * sine[token, phase_dimension]
            * rotate_sign
        )
        output[row, :, :] = I.reshape(
            rotated,
            (2, HALF_DIMENSION),
        )


@intent.fn
def rotate_pair(first, second, cosine, sine):
    return (
        first * cosine - second * sine,
        second * cosine + first * sine,
    )


@intent.kernel
def rotary_qk_inplace(
    query: I.InOut[
        I.f16,
        ("B", ROPE_QUERY_HEADS, "S", HEAD_DIMENSION),
    ],
    key: I.InOut[
        I.f16,
        ("B", ROPE_KEY_HEADS, "S", HEAD_DIMENSION),
    ],
    cosine: I.In[I.f16, (1, "S", HEAD_DIMENSION)],
    sine: I.In[I.f16, (1, "S", HEAD_DIMENSION)],
):
    B, _, S, _ = query.shape
    query_heads = I.domain(0, ROPE_QUERY_HEADS)
    key_heads = I.domain(0, ROPE_KEY_HEADS)
    phase = I.domain(0, HALF_DIMENSION)

    for batch in I.parallel(I.domain(0, B)):
        for token in I.parallel(I.domain(0, S)):
            paired_phase = I.indices(phase) + HALF_DIMENSION
            cosine_row = cosine[0, token, phase]
            sine_row = sine[0, token, phase]

            query_first = query[batch, query_heads, token, phase]
            query_second = query[batch, query_heads, token, paired_phase]
            rotated_query_first, rotated_query_second = rotate_pair(
                query_first,
                query_second,
                cosine_row,
                sine_row,
            )
            query[batch, query_heads, token, phase] = rotated_query_first
            query[batch, query_heads, token, paired_phase] = rotated_query_second

            key_first = key[batch, key_heads, token, phase]
            key_second = key[batch, key_heads, token, paired_phase]
            rotated_key_first, rotated_key_second = rotate_pair(
                key_first,
                key_second,
                cosine_row,
                sine_row,
            )
            key[batch, key_heads, token, phase] = rotated_key_first
            key[batch, key_heads, token, paired_phase] = rotated_key_second


@intent.kernel
def rotary_qk_partial_inplace(
    query: I.InOut[
        I.f16,
        ("B", ROPE_QUERY_HEADS, "S", HEAD_DIMENSION),
    ],
    key: I.InOut[
        I.f16,
        ("B", ROPE_KEY_HEADS, "S", HEAD_DIMENSION),
    ],
    cosine: I.In[I.f16, (1, "S", PARTIAL_ROTARY_DIMENSION)],
    sine: I.In[I.f16, (1, "S", PARTIAL_ROTARY_DIMENSION)],
):
    B, _, S, _ = query.shape
    query_heads = I.domain(0, ROPE_QUERY_HEADS)
    key_heads = I.domain(0, ROPE_KEY_HEADS)
    phase = I.domain(0, PARTIAL_HALF_DIMENSION)

    for batch in I.parallel(I.domain(0, B)):
        for token in I.parallel(I.domain(0, S)):
            paired_phase = I.indices(phase) + PARTIAL_HALF_DIMENSION
            cosine_row = cosine[0, token, phase]
            sine_row = sine[0, token, phase]

            query_first = query[batch, query_heads, token, phase]
            query_second = query[batch, query_heads, token, paired_phase]
            rotated_query_first, rotated_query_second = rotate_pair(
                query_first,
                query_second,
                cosine_row,
                sine_row,
            )
            query[batch, query_heads, token, phase] = rotated_query_first
            query[batch, query_heads, token, paired_phase] = rotated_query_second

            key_first = key[batch, key_heads, token, phase]
            key_second = key[batch, key_heads, token, paired_phase]
            rotated_key_first, rotated_key_second = rotate_pair(
                key_first,
                key_second,
                cosine_row,
                sine_row,
            )
            key[batch, key_heads, token, phase] = rotated_key_first
            key[batch, key_heads, token, paired_phase] = rotated_key_second


@intent.kernel
def rotary_embedding_bf16(
    values: I.In[I.bf16, ("B", "S", "H", HEAD_DIMENSION)],
    cosine: I.In[I.f32, ("S", HALF_DIMENSION)],
    sine: I.In[I.f32, ("S", HALF_DIMENSION)],
    output: I.Out[I.bf16, ("B", "S", "H", HEAD_DIMENSION)],
):
    B, S, H, D = values.shape
    halves = I.domain(0, HALF_DIMENSION)
    for batch in I.parallel(I.domain(0, B)):
        for token in I.parallel(I.domain(0, S)):
            cos = cosine[token, halves]
            sin = sine[token, halves]
            for head in I.parallel(I.domain(0, H)):
                first = I.cast(values[batch, token, head, halves], I.f32)
                second = I.cast(
                    values[batch, token, head, I.indices(halves) + HALF_DIMENSION],
                    I.f32,
                )
                output[batch, token, head, halves] = I.cast(
                    first * cos - second * sin,
                    I.bf16,
                )
                output[
                    batch,
                    token,
                    head,
                    I.indices(halves) + HALF_DIMENSION,
                ] = I.cast(second * cos + first * sin, I.bf16)


@intent.kernel
def rotary_qk_bf16_inplace(
    query: I.InOut[
        I.bf16,
        ("B", ROPE_QUERY_HEADS, "S", HEAD_DIMENSION),
    ],
    key: I.InOut[
        I.bf16,
        ("B", ROPE_KEY_HEADS, "S", HEAD_DIMENSION),
    ],
    cosine: I.In[I.bf16, (1, "S", HEAD_DIMENSION)],
    sine: I.In[I.bf16, (1, "S", HEAD_DIMENSION)],
):
    B, _, S, D = query.shape
    query_heads = I.domain(0, ROPE_QUERY_HEADS)
    key_heads = I.domain(0, ROPE_KEY_HEADS)
    halves = I.domain(0, HALF_DIMENSION)
    for batch in I.parallel(I.domain(0, B)):
        for token in I.parallel(I.domain(0, S)):
            cos = I.cast(cosine[0, token, halves], I.f32)
            sin = I.cast(sine[0, token, halves], I.f32)
            paired = I.indices(halves) + HALF_DIMENSION
            query_first = I.cast(
                query[batch, query_heads, token, halves], I.f32
            )
            query_second = I.cast(
                query[batch, query_heads, token, paired], I.f32
            )
            query[batch, query_heads, token, halves] = I.cast(
                query_first * cos - query_second * sin,
                I.bf16,
            )
            query[batch, query_heads, token, paired] = I.cast(
                query_second * cos + query_first * sin,
                I.bf16,
            )
            key_first = I.cast(key[batch, key_heads, token, halves], I.f32)
            key_second = I.cast(key[batch, key_heads, token, paired], I.f32)
            key[batch, key_heads, token, halves] = I.cast(
                key_first * cos - key_second * sin,
                I.bf16,
            )
            key[batch, key_heads, token, paired] = I.cast(
                key_second * cos + key_first * sin,
                I.bf16,
            )
