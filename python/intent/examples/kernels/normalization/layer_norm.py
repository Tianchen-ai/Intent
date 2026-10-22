import intent
import intent.language as I


ROWS = 8192
FEATURES = 4096


@intent.kernel
def weighted_layer_norm(
    x: I.In[I.f32, ("M", "N")],
    weight: I.In[I.f32, ("N",)],
    bias: I.In[I.f32, ("N",)],
    y: I.Out[I.f32, ("M", "N")],
    inverse_features: I.f32,
    epsilon: I.f32,
):
    M, N = x.shape
    columns = I.domain(0, N)
    for row in I.parallel(I.domain(0, M)):
        values = x[row, columns]
        mean = I.reduce.sum(values, axis=0) * inverse_features
        centered = values - mean
        second_moment = (
            I.reduce.sum(values * values, axis=0) * inverse_features
        )
        variance = second_moment - mean * mean
        normalized = centered * I.rsqrt(variance + epsilon)
        y[row, columns] = normalized * weight[columns] + bias[columns]


@intent.kernel
def layer_norm_f16(
    x: I.In[I.f16, ("M", "N")],
    weight: I.In[I.f16, ("N",)],
    bias: I.In[I.f16, ("N",)],
    y: I.Out[I.f16, ("M", "N")],
    inverse_features: I.f32,
    epsilon: I.f32,
):
    M, N = x.shape
    columns = I.domain(0, N)
    for row in I.parallel(I.domain(0, M)):
        values = I.cast(x[row, columns], I.f32)
        mean = I.reduce.sum(values, axis=0) * inverse_features
        centered = values - mean
        variance = (
            I.reduce.sum(centered * centered, axis=0)
            * inverse_features
        )
        y[row, columns] = I.cast(
            centered
            * I.rsqrt(variance + epsilon)
            * I.cast(weight[columns], I.f32)
            + I.cast(bias[columns], I.f32),
            I.f16,
        )


@intent.kernel
def layer_norm_bf16(
    x: I.In[I.bf16, ("M", "N")],
    weight: I.In[I.bf16, ("N",)],
    bias: I.In[I.bf16, ("N",)],
    y: I.Out[I.bf16, ("M", "N")],
    inverse_features: I.f32,
    epsilon: I.f32,
):
    M, N = x.shape
    columns = I.domain(0, N)
    for row in I.parallel(I.domain(0, M)):
        values = I.cast(x[row, columns], I.f32)
        mean = I.reduce.sum(values, axis=0) * inverse_features
        centered = values - mean
        variance = (
            I.reduce.sum(centered * centered, axis=0)
            * inverse_features
        )
        y[row, columns] = I.cast(
            centered
            * I.rsqrt(variance + epsilon)
            * I.cast(weight[columns], I.f32)
            + I.cast(bias[columns], I.f32),
            I.bf16,
        )
