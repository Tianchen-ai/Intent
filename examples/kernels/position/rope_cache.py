import intent
import intent.language as I


BATCH = 32
QUERY_HEADS = 32
KV_HEADS = 8
HEAD_DIMENSION = 128
CACHE_LENGTH = 8192
HALF_DIMENSION = HEAD_DIMENSION // 2


@intent.kernel
def padded_rope_cache_update(
    packed_input: I.In[
        I.f16, ("B", QUERY_HEADS + 2 * KV_HEADS, HEAD_DIMENSION)
    ],
    sequence_lengths: I.In[I.i32, ("B",)],
    output_storage: I.InOut[I.f16, ("O", HEAD_DIMENSION)],
    theta: I.f32,
    linear_scale: I.f32,
):
    B = packed_input.shape[0]
    halves = I.domain(0, HALF_DIMENSION)
    half_indices = I.indices(halves)
    query_rows = B * QUERY_HEADS
    cache_rows = B * (CACHE_LENGTH + 1) * KV_HEADS
    for batch in I.parallel(I.domain(0, B)):
        for head in I.parallel(I.domain(0, QUERY_HEADS + 2 * KV_HEADS)):
            position = I.cast(sequence_lengths[batch], I.index) - 1
            powers = I.cast(I.indices(halves) * 2, I.f32)
            frequencies = theta ** (powers / I.cast(-HEAD_DIMENSION, I.f32))
            angle = I.cast(position, I.f32) * frequencies / linear_scale
            cos = I.cos(angle)
            sin = I.sin(angle)
            first = I.cast(
                packed_input[batch, head, halves],
                I.f32,
            )
            second = I.cast(
                packed_input[batch, head, half_indices + HALF_DIMENSION],
                I.f32,
            )
            rotated_first = first * cos - second * sin
            rotated_second = second * cos + first * sin
            is_query = head < QUERY_HEADS
            is_value = head >= QUERY_HEADS + KV_HEADS
            cache_row = (batch * (CACHE_LENGTH + 1) + position) * KV_HEADS
            key_cache_row = cache_row + head - QUERY_HEADS
            value_cache_row = cache_row + head - QUERY_HEADS - KV_HEADS
            output_row = (
                batch * QUERY_HEADS + head
                if is_query
                else (
                    query_rows + cache_rows + value_cache_row
                    if is_value
                    else query_rows + key_cache_row
                )
            )
            I.assume_in_bounds(output_row, output_storage, axis=0)
            output_storage[output_row, halves] = I.cast(
                first if is_value else rotated_first,
                I.f16,
            )
            output_storage[output_row, half_indices + HALF_DIMENSION] = I.cast(
                second if is_value else rotated_second,
                I.f16,
            )
