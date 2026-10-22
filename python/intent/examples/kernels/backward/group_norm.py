import intent
import intent.language as I


BATCH = 32
CHANNELS = 256
SPATIAL = 1024
GROUPS = 32
CHANNELS_PER_GROUP = CHANNELS // GROUPS


@intent.kernel
def group_norm_backward_dx(
    x: I.In[I.f16, (BATCH, CHANNELS, SPATIAL)],
    grad_y: I.In[I.f16, (BATCH, CHANNELS, SPATIAL)],
    weight: I.In[I.f16, (CHANNELS,)],
    mean: I.In[I.f16, (BATCH, GROUPS)],
    rstd: I.In[I.f16, (BATCH, GROUPS)],
    grad_x: I.Out[I.f16, (BATCH, CHANNELS, SPATIAL)],
    inverse_group_elements: I.f32,
):
    positions = I.domain(0, SPATIAL)
    channel_offsets = I.domain(0, CHANNELS_PER_GROUP)
    for batch in I.parallel(I.domain(0, BATCH)):
        for group in I.parallel(I.domain(0, GROUPS)):
            channels = group * CHANNELS_PER_GROUP + I.indices(channel_offsets)
            values = I.cast(x[batch, channels, positions], I.f32)
            normalized = (
                values - I.cast(mean[batch, group], I.f32)
            ) * I.cast(rstd[batch, group], I.f32)
            normalized_gradient = (
                I.cast(grad_y[batch, channels, positions], I.f32)
                * I.cast(weight[channels, None], I.f32)
            )
            group_sum = I.reduce.sum(
                I.reduce.sum(normalized_gradient, axis=1),
                axis=0,
            )
            group_projection = I.reduce.sum(
                I.reduce.sum(
                    normalized_gradient * normalized,
                    axis=1,
                ),
                axis=0,
            )
            result = I.cast(rstd[batch, group], I.f32) * (
                normalized_gradient
                - group_sum * inverse_group_elements
                - normalized * group_projection * inverse_group_elements
            )
            grad_x[batch, channels, positions] = I.cast(result, I.f16)


@intent.kernel
def group_norm_backward_weight_bias(
    x: I.In[I.f16, (BATCH, CHANNELS, SPATIAL)],
    grad_y: I.In[I.f16, (BATCH, CHANNELS, SPATIAL)],
    mean: I.In[I.f16, (BATCH, GROUPS)],
    rstd: I.In[I.f16, (BATCH, GROUPS)],
    grad_weight: I.Out[I.f16, (CHANNELS,)],
    grad_bias: I.Out[I.f16, (CHANNELS,)],
):
    batches = I.domain(0, BATCH)
    positions = I.domain(0, SPATIAL)
    for channel in I.parallel(I.domain(0, CHANNELS)):
        group = channel // CHANNELS_PER_GROUP
        values = I.cast(x[batches, channel, positions], I.f32)
        gradient = I.cast(grad_y[batches, channel, positions], I.f32)
        normalized = (
            values - I.cast(mean[batches, group, None], I.f32)
        ) * I.cast(rstd[batches, group, None], I.f32)
        grad_weight[channel] = I.cast(
            I.reduce.sum(
                I.reduce.sum(gradient * normalized, axis=1),
                axis=0,
            ),
            I.f16,
        )
        grad_bias[channel] = I.cast(
            I.reduce.sum(
                I.reduce.sum(gradient, axis=1),
                axis=0,
            ),
            I.f16,
        )
