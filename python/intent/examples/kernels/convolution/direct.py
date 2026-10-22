import intent
import intent.language as I


CONV1D_BATCH = 64
CONV1D_LENGTH = 16384
CONV1D_FILTER = 5
CAUSAL_CONV_BATCH = 8
CAUSAL_CONV_CHANNELS = 2048
CAUSAL_CONV_LENGTH = 4096
CAUSAL_CONV_WIDTH = 4
CAUSAL_UPDATE_BATCH = 64
CAUSAL_UPDATE_CHANNELS = 4096

CONV2D_BATCH = 16
CONV2D_HEIGHT = 256
CONV2D_WIDTH = 256
CONV2D_FILTER_HEIGHT = 3
CONV2D_FILTER_WIDTH = 3
CONV2D_NHWC_BATCH = 32
CONV2D_NHWC_HEIGHT = 128
CONV2D_NHWC_WIDTH = 128
CONV2D_NHWC_INPUT_CHANNELS = 256
CONV2D_NHWC_OUTPUT_CHANNELS = 512


@intent.kernel
def conv1d_same(
    x: I.In[I.f16, ("B", "L")],
    weight: I.In[I.f16, ("K",)],
    output: I.Out[I.f16, ("B", "L")],
):
    B, L = x.shape
    length = I.domain(0, L)
    taps = I.domain(0, CONV1D_FILTER)
    for batch in I.parallel(I.domain(0, B)):
        output_indices = I.indices(length)
        tap_indices = I.indices(taps)
        input_indices = (
            output_indices[:, None]
            + tap_indices[None, :]
            - CONV1D_FILTER // 2
        )
        valid = (input_indices >= 0) & (input_indices < L)
        safe_input_indices = I.select(valid, input_indices, 0)
        patch = I.cast(
            I.mask(
                x[batch, safe_input_indices],
                valid=valid,
                fill=I.cast(0.0, I.f16),
            ),
            I.f32,
        )
        filter_values = I.cast(weight[taps], I.f32)
        products = patch * filter_values
        reduced = I.reduce.sum(
            products,
            axis=1,
        )
        output[batch, length] = I.cast(reduced, I.f16)


@intent.kernel
def causal_depthwise_conv1d(
    x: I.In[I.f16, ("B", "D", "L")],
    weight: I.In[I.f16, ("D", "W")],
    bias: I.In[I.f16, ("D",)],
    output: I.Out[I.f16, ("B", "D", "L")],
    SILU: I.Constexpr[bool],
):
    B, D, L = x.shape
    W = weight.shape[1]
    positions = I.domain(0, L)
    taps = I.domain(0, W)
    for batch in I.parallel(I.domain(0, B)):
        for channel in I.parallel(I.domain(0, D)):
            output_index = I.indices(positions)[:, None]
            tap_index = I.indices(taps)[None, :]
            input_index = output_index - (W - 1) + tap_index
            valid = (input_index >= 0) & (input_index < L)
            safe_input_index = I.select(valid, input_index, 0)
            patch = x[batch, channel, safe_input_index]
            patch = I.mask(
                patch,
                valid=valid,
                fill=I.cast(0.0, I.f16),
            )
            products = (
                I.cast(patch, I.f32)
                * I.cast(weight[channel, taps], I.f32)[None, :]
            )
            reduced = I.reduce.sum(
                products,
                axis=1,
            ) + I.cast(bias[channel], I.f32)
            if SILU:
                reduced = reduced * I.sigmoid(reduced)
            output[batch, channel, positions] = I.cast(reduced, I.f16)


@intent.kernel
def causal_depthwise_conv1d_bf16(
    x: I.In[I.bf16, ("B", "D", "L")],
    weight: I.In[I.bf16, ("D", "W")],
    bias: I.In[I.bf16, ("D",)],
    output: I.Out[I.bf16, ("B", "D", "L")],
    SILU: I.Constexpr[bool],
):
    B, D, L = x.shape
    W = weight.shape[1]
    positions = I.domain(0, L)
    channels = I.domain(0, D)
    taps = I.domain(0, W)
    for batch in I.parallel(I.domain(0, B)):
        output_index = I.indices(positions)[:, None]
        tap_index = I.indices(taps)[None, :]
        input_index = output_index - (W - 1) + tap_index
        valid = (input_index >= 0) & (input_index < L)
        safe_input_index = I.select(valid, input_index, 0)
        patch = I.mask(
            x[batch, channels, safe_input_index],
            valid=valid[None, :, :],
            fill=I.cast(0.0, I.bf16),
        )
        reduced = I.reduce.sum(
            I.cast(patch, I.f32)
            * I.cast(weight[channels, taps], I.f32)[:, None, :],
            axis=2,
        ) + I.cast(bias[channels], I.f32)[:, None]
        if SILU:
            reduced = reduced * I.sigmoid(reduced)
        output[batch, channels, positions] = I.cast(
            reduced, I.bf16
        )


@intent.kernel
def causal_depthwise_conv1d_update(
    x: I.In[I.f16, ("B", "D")],
    state: I.InOut[I.f16, ("B", "D", CAUSAL_CONV_WIDTH)],
    weight: I.In[I.f16, ("D", CAUSAL_CONV_WIDTH)],
    bias: I.In[I.f16, ("D",)],
    output: I.Out[I.f16, ("B", "D")],
    SILU: I.Constexpr[bool],
):
    B, D = x.shape
    channels = I.domain(0, D)
    for batch in I.parallel(I.domain(0, B)):
        accumulator = I.cast(bias[channels], I.f32)
        for tap in range(CAUSAL_CONV_WIDTH - 1):
            shifted = state[batch, channels, tap + 1]
            state[batch, channels, tap] = shifted
            accumulator = accumulator + I.cast(shifted, I.f32) * I.cast(
                weight[channels, tap],
                I.f32,
            )
        current = x[batch, channels]
        state[batch, channels, CAUSAL_CONV_WIDTH - 1] = current
        accumulator = accumulator + I.cast(current, I.f32) * I.cast(
            weight[channels, CAUSAL_CONV_WIDTH - 1],
            I.f32,
        )
        if SILU:
            accumulator = accumulator * I.sigmoid(accumulator)
        output[batch, channels] = I.cast(accumulator, I.f16)


@intent.kernel
def causal_depthwise_conv1d_update_bf16(
    x: I.In[I.bf16, ("B", "D")],
    state: I.InOut[I.bf16, ("B", "D", CAUSAL_CONV_WIDTH)],
    weight: I.In[I.f32, ("D", CAUSAL_CONV_WIDTH)],
    bias: I.In[I.f32, ("D",)],
    output: I.Out[I.bf16, ("B", "D")],
    SILU: I.Constexpr[bool],
):
    B, D = x.shape
    channels = I.domain(0, D)
    taps = I.domain(0, CAUSAL_CONV_WIDTH)
    channel_coordinates = I.indices(channels)
    tap_coordinates = I.indices(taps)
    source_coordinates = I.minimum(
        tap_coordinates + 1, CAUSAL_CONV_WIDTH - 1
    )
    shifted = tap_coordinates < CAUSAL_CONV_WIDTH - 1
    for batch in I.parallel(I.domain(0, B)):
        current = x[batch, channels]
        cache = I.gather(
            state,
            index=(
                batch,
                channel_coordinates[:, None],
                source_coordinates[None, :],
            ),
            valid=shifted[None, :],
            fill=current[:, None],
        )
        accumulator = bias[channels] + I.reduce.sum(
            I.cast(cache, I.f32) * weight[channels, taps],
            axis=1,
        )
        if SILU:
            accumulator = accumulator * I.sigmoid(accumulator)
        state[batch, channels, taps] = cache
        output[batch, channels] = I.cast(accumulator, I.bf16)


@intent.kernel
def conv2d_same(
    x: I.In[I.f16, ("B", "H", "W")],
    weight: I.In[I.f16, ("R", "S")],
    output: I.Out[I.f16, ("B", "H", "W")],
):
    B, H, W = x.shape
    height = I.domain(0, H)
    width = I.domain(0, W)
    kernel_rows = I.domain(0, CONV2D_FILTER_HEIGHT)
    kernel_columns = I.domain(0, CONV2D_FILTER_WIDTH)
    for batch in I.parallel(I.domain(0, B)):
        output_row_indices = I.reshape(
            I.indices(height),
            (H, 1, 1, 1),
        )
        output_column_indices = I.reshape(
            I.indices(width),
            (1, W, 1, 1),
        )
        kernel_row_indices = I.indices(kernel_rows)[:, None]
        kernel_column_indices = I.indices(kernel_columns)
        input_rows = (
            output_row_indices
            + kernel_row_indices
            - CONV2D_FILTER_HEIGHT // 2
        )
        input_columns = (
            output_column_indices
            + kernel_column_indices
            - CONV2D_FILTER_WIDTH // 2
        )
        row_valid = (input_rows >= 0) & (input_rows < H)
        column_valid = (input_columns >= 0) & (input_columns < W)
        safe_input_rows = I.select(row_valid, input_rows, 0)
        safe_input_columns = I.select(column_valid, input_columns, 0)
        patch = I.cast(
            I.mask(
                x[batch, safe_input_rows, safe_input_columns],
                valid=row_valid & column_valid,
                fill=I.cast(0.0, I.f16),
            ),
            I.f32,
        )
        filter_values = I.cast(
            weight[kernel_rows, kernel_columns],
            I.f32,
        )
        products = patch * filter_values
        reduced_columns = I.reduce.sum(
            products,
            axis=3,
        )
        reduced = I.reduce.sum(
            reduced_columns,
            axis=2,
        )
        output[batch, height, width] = I.cast(
            reduced,
            I.f16,
        )


@intent.kernel
def conv2d_nhwc(
    x: I.In[I.f16, ("B", "H", "W", "CI")],
    weight: I.In[I.f16, (3, 3, "CI", "CO")],
    output: I.Out[I.f16, ("B", "H", "W", "CO")],
):
    B, H, W, CI = x.shape
    CO = weight.shape[3]
    width = I.domain(0, W)
    input_channels = I.domain(0, CI)
    output_channels = I.domain(0, CO)
    for batch in I.parallel(I.domain(0, B)):
        for output_row in I.parallel(I.domain(0, H)):
            accumulator = I.zeros(
                (W, CO), dtype=I.f32
            )
            for kernel_row in range(3):
                input_row = output_row + kernel_row - 1
                row_valid = (input_row >= 0) & (input_row < H)
                safe_input_row = I.select(row_valid, input_row, 0)
                for kernel_column in range(3):
                    input_column = (
                        I.indices(width) + kernel_column - 1
                    )
                    column_valid = (input_column >= 0) & (input_column < W)
                    safe_input_column = I.select(
                        column_valid,
                        input_column,
                        0,
                    )
                    patch = I.mask(
                        x[
                            batch,
                            safe_input_row,
                            safe_input_column,
                            input_channels,
                        ],
                        valid=(row_valid & column_valid)[:, None],
                        fill=I.cast(0.0, I.f16),
                    )
                    filter_values = weight[
                        kernel_row,
                        kernel_column,
                        input_channels,
                        output_channels,
                    ]
                    accumulator = accumulator + I.matmul(
                        patch,
                        filter_values,
                        acc_dtype=I.f32,
                    )
            output[
                batch,
                output_row,
                width,
                output_channels,
            ] = I.cast(accumulator, I.f16)
