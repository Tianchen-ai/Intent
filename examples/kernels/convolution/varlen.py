import intent
import intent.language as I


PACKED_TOKENS = 5120
HIDDEN = 4096
WIDTH = 4
SEQUENCES = 4
CHUNK_SIZE = 64


@intent.kernel
def varlen_aligned_causal_depthwise_conv1d(
    x: I.In[I.bf16, ("U", "D")],
    sequence_offsets: I.In[I.i64, (SEQUENCES + 1,)],
    chunk_indices: I.In[I.i32, ("C", 2)],
    weight: I.In[I.f32, ("D", WIDTH)],
    bias: I.In[I.f32, ("D",)],
    output: I.Out[I.bf16, ("U", "D")],
):
    U, D = x.shape
    C = chunk_indices.shape[0]
    packed_tokens = I.domain(0, U)
    channels = I.domain(0, D)
    taps = I.domain(0, WIDTH)
    for chunk in I.parallel(I.domain(0, C)):
        I.assume_in_bounds(chunk, chunk_indices, axis=0)
        sequence = I.cast(chunk_indices[chunk, 0], I.index)
        local_chunk = I.cast(chunk_indices[chunk, 1], I.index)
        next_sequence = sequence + 1
        I.assume_in_bounds(sequence, sequence_offsets, axis=0)
        I.assume_in_bounds(next_sequence, sequence_offsets, axis=0)
        sequence_start = I.cast(sequence_offsets[sequence], I.index)
        sequence_end = I.cast(sequence_offsets[next_sequence], I.index)
        token_start = sequence_start + local_chunk * CHUNK_SIZE
        token_end = I.minimum(token_start + CHUNK_SIZE, sequence_end)
        tokens = packed_tokens[token_start:token_end]
        token_indices = I.indices(tokens)
        channel_indices = I.indices(channels)
        tap_indices = I.indices(taps)
        source_index = (
            token_indices[:, None]
            - I.cast(WIDTH - 1, I.index)
            + tap_indices[None, :]
        )
        valid = source_index >= sequence_start
        value = I.gather(
            x,
            index=(
                source_index[:, :, None],
                channel_indices[None, None, :],
            ),
            valid=valid[:, :, None],
            fill=I.cast(0.0, I.bf16),
        )
        tap_weight = I.transpose(
            I.cast(weight[channels, taps], I.f32), permutation=(1, 0)
        )
        accumulation = I.reduce.sum(
            I.cast(value, I.f32) * tap_weight[None, :, :],
            axis=1,
        )
        accumulation = accumulation + I.cast(
            bias[channels], I.f32
        )[None, :]
        sigmoid = I.sigmoid(accumulation)
        I.scatter_unique(
            output,
            index=(token_indices[:, None], channel_indices[None, :]),
            value=I.cast(accumulation * sigmoid, I.bf16),
        )


@intent.kernel
def varlen_causal_conv1d_final_state(
    x: I.In[I.bf16, ("U", "D")],
    sequence_offsets: I.In[I.i64, (SEQUENCES + 1,)],
    final_state: I.Out[I.bf16, (SEQUENCES, WIDTH, "D")],
):
    _, D = x.shape
    channels = I.domain(0, D)
    for sequence in I.parallel(I.domain(0, SEQUENCES)):
        next_sequence = sequence + 1
        I.assume_in_bounds(sequence, sequence_offsets, axis=0)
        I.assume_in_bounds(next_sequence, sequence_offsets, axis=0)
        sequence_start = I.cast(sequence_offsets[sequence], I.index)
        sequence_end = I.cast(sequence_offsets[next_sequence], I.index)
        channel_indices = I.indices(channels)
        for history in I.parallel(I.domain(0, WIDTH)):
            source_index = (
                sequence_end
                - I.cast(WIDTH, I.index)
                + I.cast(history, I.index)
            )
            values = I.gather(
                x,
                index=(source_index, channel_indices),
                valid=source_index >= sequence_start,
                fill=I.cast(0.0, I.bf16),
            )
            final_state[sequence, history, channels] = values
