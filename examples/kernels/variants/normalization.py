import intent
import intent.language as I

from kernels.streaming.online_softmax import online_softmax_summary


@intent.kernel
def stable_softmax_online(
    x: I.In[I.f32, ("M", "N")],
    y: I.Out[I.f32, ("M", "N")],
):
    M, N = x.shape
    columns = I.domain(0, N)
    for row in I.parallel(I.domain(0, M)):
        values = x[row, columns]
        summary = online_softmax_summary(values)
        safe_denominator = I.select(summary.valid, summary.denominator, 1.0)
        y[row, columns] = I.select(
            summary.valid,
            I.exp(values - summary.maximum) / safe_denominator,
            0.0,
        )


@intent.kernel
def weighted_layer_norm_second_moment(
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
        normalized = (values - mean) * I.rsqrt(
            I.reduce.sum(values * values, axis=0)
            * inverse_features
            - mean * mean
            + epsilon
        )
        y[row, columns] = normalized * weight[columns] + bias[columns]
