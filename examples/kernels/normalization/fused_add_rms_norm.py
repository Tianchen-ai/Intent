import intent
import intent.language as I


ROWS = 8192
FEATURES = 4096


@intent.kernel
def fused_add_rms_norm(
    x: I.In[I.bf16, ("M", "N")],
    residual: I.In[I.bf16, ("M", "N")],
    weight: I.In[I.bf16, ("N",)],
    normalized: I.Out[I.bf16, ("M", "N")],
    residual_out: I.Out[I.bf16, ("M", "N")],
    inverse_features: I.f32,
    epsilon: I.f32,
    weight_offset: I.f32,
):
    M, N = x.shape
    columns = I.domain(0, N)
    for row in I.parallel(I.domain(0, M)):
        summed = I.cast(
            I.cast(x[row, columns], I.f32)
            + I.cast(residual[row, columns], I.f32),
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
