import intent
import intent.language as I


BATCH = 32
CHANNELS = 64
SPATIAL = 4096


@intent.fn
def welford_combine(left, right):
    count = left.count + right.count
    delta = right.mean - left.mean
    safe_count = I.maximum(count, I.cast(1, I.i32))
    safe_count_f32 = I.cast(safe_count, I.f32)
    left_count = I.cast(left.count, I.f32)
    right_count = I.cast(right.count, I.f32)
    mean_correction = delta * right_count / safe_count_f32
    mean = left.mean + mean_correction
    m2 = (
        left.m2
        + right.m2
        + delta * left_count * mean_correction
    )
    return I.record(count=count, mean=mean, m2=m2)


@intent.kernel
def batch_norm_training(
    x: I.In[I.f16, ("B", "C", "S")],
    weight: I.In[I.f32, ("C",)],
    bias: I.In[I.f32, ("C",)],
    running_mean: I.InOut[I.f32, ("C",)],
    running_variance: I.InOut[I.f32, ("C",)],
    output: I.Out[I.f16, ("B", "C", "S")],
    saved_mean: I.Out[I.f32, ("C",)],
    saved_rstd: I.Out[I.f32, ("C",)],
    epsilon: I.f32,
    momentum: I.f32,
):
    B, C, S = x.shape
    # Interface precondition: B * S > 1 for the unbiased running variance.
    batches = I.domain(0, B)
    spatials = I.domain(0, S)
    for channel in I.parallel(I.domain(0, C)):
        values = I.cast(x[batches, channel, spatials], I.f32)
        identity = I.record(
            count=I.cast(0, I.i32),
            mean=I.cast(0.0, I.f32),
            m2=I.cast(0.0, I.f32),
        )
        summary = I.reduce(
            I.record(
                count=I.full(values.shape, fill=1, dtype=I.i32),
                mean=values,
                m2=I.zeros(values.shape, dtype=I.f32),
            ),
            axis=(0, 1),
            identity=identity,
            combine=welford_combine,
        )
        count, mean, m2 = summary.count, summary.mean, summary.m2
        total_count = I.cast(count, I.f32)
        variance = m2 / total_count
        rstd = I.rsqrt(variance + epsilon)
        running_mean[channel] = (
            (1.0 - momentum) * running_mean[channel] + momentum * mean
        )
        running_variance[channel] = (
            (1.0 - momentum) * running_variance[channel]
            + momentum
            * variance
            * total_count
            / (total_count - 1.0)
        )
        output_values = I.cast(x[batches, channel, spatials], I.f32)
        output[batches, channel, spatials] = I.cast(
            (output_values - mean) * rstd * weight[channel] + bias[channel],
            I.f16,
        )
        saved_mean[channel] = mean
        saved_rstd[channel] = rstd
