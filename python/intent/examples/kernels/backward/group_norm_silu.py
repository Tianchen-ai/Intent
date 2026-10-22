import intent
import intent.language as I


BATCH = 32
CHANNELS = 256
SPATIAL = 1024
GROUPS = 32
CHANNELS_PER_GROUP = CHANNELS // GROUPS


@intent.kernel
def group_norm_silu_backward(
    x: I.In[I.bf16, ("B", CHANNELS, "S")],
    upstream: I.In[I.bf16, ("B", CHANNELS, "S")],
    weight: I.In[I.f32, (CHANNELS,)],
    bias: I.In[I.f32, (CHANNELS,)],
    mean: I.In[I.f32, ("B", GROUPS)],
    rstd: I.In[I.f32, ("B", GROUPS)],
    dx: I.Out[I.bf16, ("B", CHANNELS, "S")],
    dweight: I.InOut[I.f32, (CHANNELS,)],
    dbias: I.InOut[I.f32, (CHANNELS,)],
    inverse_group_elements: I.f32,
):
    B, _, S = x.shape
    positions = I.domain(0, S)
    channel_offsets = I.domain(0, CHANNELS_PER_GROUP)
    for batch in I.parallel(I.domain(0, B)):
        for group in I.parallel(I.domain(0, GROUPS)):
            channels = (
                group * CHANNELS_PER_GROUP
                + I.indices(channel_offsets)
            )
            x_values = I.cast(x[batch, channels, positions], I.f32)
            normalized = (x_values - mean[batch, group]) * rstd[batch, group]
            affine = normalized * weight[channels, None] + bias[channels, None]
            sigmoid = I.sigmoid(affine)
            silu_gradient = sigmoid + affine * sigmoid * (1.0 - sigmoid)
            gradient = I.cast(upstream[batch, channels, positions], I.f32) * silu_gradient
            normalized_gradient = gradient * weight[channels, None]
            reduced_positions = I.reduce.sum(
                normalized_gradient,
                axis=1,
            )
            group_sum = I.reduce.sum(
                reduced_positions,
                axis=0,
            )
            reduced_projection_positions = I.reduce.sum(
                normalized_gradient * normalized,
                axis=1,
            )
            group_projection = I.reduce.sum(
                reduced_projection_positions,
                axis=0,
            )
            dx_values = rstd[batch, group] * (
                normalized_gradient
                - group_sum * inverse_group_elements
                - normalized * group_projection * inverse_group_elements
            )
            dx[batch, channels, positions] = I.cast(dx_values, I.bf16)
            I.scatter_reduce(
                dweight,
                index=(channels,),
                value=I.reduce.sum(gradient * normalized, axis=1),
                combine=I.add,
            )
            I.scatter_reduce(
                dbias,
                index=(channels,),
                value=I.reduce.sum(gradient, axis=1),
                combine=I.add,
            )
