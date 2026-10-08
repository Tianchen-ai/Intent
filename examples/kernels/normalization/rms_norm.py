import intent
import intent.language as I


ROWS = 8192
FEATURES = 4096


@intent.kernel
def weighted_rms_norm(
    x: I.In[I.f32, ("M", "N")],
    weight: I.In[I.f32, ("N",)],
    y: I.Out[I.f32, ("M", "N")],
    inverse_features: I.f32,
    epsilon: I.f32,
):
    M, N = x.shape
    columns = I.domain(0, N)
    for row in I.parallel(I.domain(0, M)):
        values = x[row, columns]
        mean_square = (
            I.reduce.sum(values * values, axis=0)
            * inverse_features
        )
        y[row, columns] = values * I.rsqrt(mean_square + epsilon) * weight[columns]


@intent.kernel
def rms_norm_f32(
    x: I.In[I.f32, ("M", "N")],
    y: I.Out[I.f32, ("M", "N")],
    inverse_features: I.f32,
    epsilon: I.f32,
):
    M, N = x.shape
    columns = I.domain(0, N)
    for row in I.parallel(I.domain(0, M)):
        values = x[row, columns]
        mean_square = (
            I.reduce.sum(values * values, axis=0)
            * inverse_features
        )
        y[row, columns] = values * I.rsqrt(mean_square + epsilon)


@intent.kernel
def rms_norm_bf16(
    x: I.In[I.bf16, ("M", "N")],
    weight: I.In[I.bf16, ("N",)],
    y: I.Out[I.bf16, ("M", "N")],
    inverse_features: I.f32,
    epsilon: I.f32,
):
    M, N = x.shape
    columns = I.domain(0, N)
    for row in I.parallel(I.domain(0, M)):
        values = I.cast(x[row, columns], I.f32)
        mean_square = (
            I.reduce.sum(values * values, axis=0)
            * inverse_features
        )
        y[row, columns] = I.cast(
            values
            * I.rsqrt(mean_square + epsilon)
            * I.cast(weight[columns], I.f32),
            I.bf16,
        )
