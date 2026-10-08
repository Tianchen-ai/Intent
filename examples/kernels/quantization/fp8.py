import intent
import intent.language as I


ROWS = 8192
FEATURES = 4096
GROUP_SIZE = 128
FP8_MAX = 448.0


@intent.kernel
def bf16_groupwise_fp8_quantize(
    x: I.In[I.bf16, ("M", "N")],
    quantized: I.Out[I.f8e4m3fn, ("M", "N")],
    scales: I.InOut[I.f32, ("M", "G")],
):
    M, N = x.shape
    groups = I.domain(0, scales.shape[1])
    group_members = I.domain(0, GROUP_SIZE)
    for row in I.parallel(I.domain(0, M)):
        for group in I.parallel(groups):
            columns = group * GROUP_SIZE + I.indices(group_members)
            values = I.cast(
                I.mask(
                    x[row, columns],
                    valid=columns < N,
                    fill=I.cast(0.0, I.bf16),
                ),
                I.f32,
            )
            absolute = I.mask(
                I.maximum(values, -values),
                valid=columns < N,
                fill=-I.inf,
            )
            maximum = I.reduce.max(absolute, axis=0)
            scale = I.maximum(maximum, 1.0e-12) / FP8_MAX
            normalized = I.minimum(I.maximum(values / scale, -FP8_MAX), FP8_MAX)
            quantized[row, columns] = I.cast(normalized, I.f8e4m3fn)
            scales[row, group] = scale


@intent.kernel
def f32_groupwise_fp8_quantize(
    x: I.In[I.f32, ("M", "N")],
    quantized: I.Out[I.f8e4m3fn, ("M", "N")],
    scales: I.InOut[I.f32, ("M", "G")],
):
    M, N = x.shape
    groups = I.domain(0, scales.shape[1])
    group_members = I.domain(0, GROUP_SIZE)
    for row in I.parallel(I.domain(0, M)):
        for group in I.parallel(groups):
            columns = group * GROUP_SIZE + I.indices(group_members)
            values = I.mask(
                x[row, columns],
                valid=columns < N,
                fill=0.0,
            )
            absolute = I.mask(
                I.maximum(values, -values),
                valid=columns < N,
                fill=-I.inf,
            )
            maximum = I.reduce.max(absolute, axis=0)
            scale = I.maximum(maximum, 1.0e-4) / FP8_MAX
            normalized = I.minimum(I.maximum(values / scale, -FP8_MAX), FP8_MAX)
            quantized[row, columns] = I.cast(normalized, I.f8e4m3fn)
            scales[row, group] = scale
