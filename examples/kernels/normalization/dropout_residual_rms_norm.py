import intent
import intent.language as I


ROWS = 4096
FEATURES = 4096
KEEP_PROBABILITY = 0.9
SEED = 17


@intent.kernel
def dropout_residual_rms_norm_forward(
    x: I.In[I.bf16, ("M", "N")],
    residual: I.In[I.bf16, ("M", "N")],
    weight: I.In[I.bf16, ("N",)],
    normalized: I.Out[I.bf16, ("M", "N")],
    residual_out: I.Out[I.bf16, ("M", "N")],
    seed: I.u64,
    keep_probability: I.f32,
    inverse_keep_probability: I.f32,
    inverse_features: I.f32,
    epsilon: I.f32,
    weight_offset: I.f32,
):
    M, N = x.shape
    columns = I.domain(0, N)
    for row in I.parallel(I.domain(0, M)):
        counter = I.indices(columns) + row * N
        keep = I.random.uniform(seed, counter, dtype=I.f32) < keep_probability
        dropped = (
            I.cast(x[row, columns], I.f32)
            * I.cast(keep, I.f32)
            * inverse_keep_probability
        )
        summed = I.cast(
            dropped + I.cast(residual[row, columns], I.f32),
            I.bf16,
        )
        residual_out[row, columns] = summed
        summed_f32 = I.cast(summed, I.f32)
        mean_square = (
            I.reduce.sum(summed_f32 * summed_f32, axis=0)
            * inverse_features
        )
        normalized[row, columns] = I.cast(
            summed_f32
            * I.rsqrt(mean_square + epsilon)
            * (I.cast(weight[columns], I.f32) + weight_offset),
            I.bf16,
        )


@intent.kernel
def dropout_residual_rms_norm_backward_data(
    x: I.In[I.bf16, ("M", "N")],
    residual: I.In[I.bf16, ("M", "N")],
    weight: I.In[I.bf16, ("N",)],
    dnormalized: I.In[I.bf16, ("M", "N")],
    dresidual_out: I.In[I.bf16, ("M", "N")],
    dx: I.Out[I.bf16, ("M", "N")],
    dresidual: I.Out[I.bf16, ("M", "N")],
    seed: I.u64,
    keep_probability: I.f32,
    inverse_keep_probability: I.f32,
    inverse_features: I.f32,
    epsilon: I.f32,
    weight_offset: I.f32,
):
    M, N = x.shape
    columns = I.domain(0, N)
    for row in I.parallel(I.domain(0, M)):
        counter = I.indices(columns) + row * N
        keep = I.random.uniform(seed, counter, dtype=I.f32) < keep_probability
        keep_f32 = I.cast(keep, I.f32)
        summed = I.cast(
            I.cast(x[row, columns], I.f32)
            * keep_f32
            * inverse_keep_probability
            + I.cast(residual[row, columns], I.f32),
            I.bf16,
        )
        summed_f32 = I.cast(summed, I.f32)
        inverse_rms = I.rsqrt(
            I.reduce.sum(summed_f32 * summed_f32, axis=0)
            * inverse_features
            + epsilon
        )
        normalized_gradient = I.cast(dnormalized[row, columns], I.f32) * (
            I.cast(weight[columns], I.f32) + weight_offset
        )
        projection = I.reduce.sum(
            normalized_gradient * summed_f32,
            axis=0,
        ) * inverse_features
        summed_gradient = (
            normalized_gradient * inverse_rms
            - summed_f32
            * (inverse_rms * inverse_rms * inverse_rms)
            * projection
            + I.cast(dresidual_out[row, columns], I.f32)
        )
        dx[row, columns] = I.cast(
            summed_gradient * keep_f32 * inverse_keep_probability,
            I.bf16,
        )
        dresidual[row, columns] = I.cast(summed_gradient, I.bf16)
