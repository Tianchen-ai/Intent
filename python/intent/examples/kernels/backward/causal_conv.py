import intent
import intent.language as I


BATCH = 8
CHANNELS = 2048
LENGTH = 4096
WIDTH = 4


@intent.kernel
def causal_conv1d_backward_partials(
    x: I.In[I.f16, ("B", "D", "L")],
    weight: I.In[I.f16, ("D", "W")],
    grad_output: I.In[I.f16, ("B", "D", "L")],
    grad_x: I.Out[I.f16, ("B", "D", "L")],
    grad_weight_partial: I.Out[I.f32, ("B", "D", "W")],
    grad_bias_partial: I.Out[I.f32, ("B", "D")],
):
    B, D, L = x.shape
    W = weight.shape[1]
    positions = I.domain(0, L)
    taps = I.domain(0, W)
    for batch in I.parallel(I.domain(0, B)):
        for channel in I.parallel(I.domain(0, D)):
            position_index = I.indices(positions)[:, None]
            tap_index = I.indices(taps)[None, :]
            input_index = position_index - (W - 1) + tap_index
            valid_input = (input_index >= 0) & (input_index < L)
            safe_input_index = I.select(valid_input, input_index, 0)
            input_patch = x[batch, channel, safe_input_index]
            input_patch = I.mask(
                input_patch,
                valid=valid_input,
                fill=I.cast(0.0, I.f16),
            )
            output_gradient = I.cast(
                grad_output[batch, channel, positions],
                I.f32,
            )
            grad_weight_partial[batch, channel, taps] = I.reduce.sum(
                I.cast(input_patch, I.f32) * output_gradient[:, None],
                axis=0,
            )
            grad_bias_partial[batch, channel] = I.reduce.sum(
                output_gradient,
                axis=0,
            )

            contributing_output = position_index + (W - 1) - tap_index
            valid_output = (contributing_output >= 0) & (contributing_output < L)
            safe_output = I.select(valid_output, contributing_output, 0)
            output_patch = grad_output[batch, channel, safe_output]
            output_patch = I.mask(
                output_patch,
                valid=valid_output,
                fill=I.cast(0.0, I.f16),
            )
            grad_x_value = I.reduce.sum(
                I.cast(output_patch, I.f32)
                * I.cast(weight[channel, taps], I.f32)[None, :],
                axis=1,
            )
            grad_x[batch, channel, positions] = I.cast(
                grad_x_value,
                I.f16,
            )


@intent.kernel
def causal_conv1d_backward_reduce(
    grad_weight_partial: I.In[I.f32, ("B", "D", "W")],
    grad_bias_partial: I.In[I.f32, ("B", "D")],
    grad_weight: I.Out[I.f32, ("D", "W")],
    grad_bias: I.Out[I.f32, ("D",)],
):
    B, D, W = grad_weight_partial.shape
    batches = I.domain(0, B)
    taps = I.domain(0, W)
    for channel in I.parallel(I.domain(0, D)):
        grad_bias[channel] = I.reduce.sum(
            grad_bias_partial[batches, channel],
            axis=0,
        )
        grad_weight[channel, taps] = I.reduce.sum(
            grad_weight_partial[batches, channel, taps],
            axis=0,
        )
