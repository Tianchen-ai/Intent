import intent
import intent.language as I


ROWS = 8192
FEATURES = 4096
GATED_ROWS = 4096
GATED_FEATURES = 14336


@intent.fn
def tanh_value(value):
    return I.tanh(I.cast(value, I.f32))


@intent.fn
def gelu_tanh_value(value):
    value_f32 = I.cast(value, I.f32)
    inner = 0.7978845608028654 * (
        value_f32 + 0.044715 * value_f32 * value_f32 * value_f32
    )
    return 0.5 * value_f32 * (1.0 + tanh_value(inner))


@intent.kernel
def gelu_tanh(
    x: I.In[I.f16, ("M", "N")],
    output: I.Out[I.f16, ("M", "N")],
):
    M, N = x.shape
    columns = I.domain(0, N)
    for row in I.parallel(I.domain(0, M)):
        output[row, columns] = I.cast(gelu_tanh_value(x[row, columns]), I.f16)


@intent.kernel
def geglu_tanh(
    x: I.In[I.f16, ("M", "TWO_N")],
    output: I.InOut[I.f16, ("M", "N")],
):
    M, _ = x.shape
    N = output.shape[1]
    columns = I.domain(0, N)
    for row in I.parallel(I.domain(0, M)):
        left = I.cast(x[row, columns], I.f32)
        right = x[row, I.indices(columns) + N]
        output[row, columns] = I.cast(left * gelu_tanh_value(right), I.f16)


@intent.kernel
def relu_forward(
    x: I.In[I.f16, ("M", "N")],
    output: I.Out[I.f16, ("M", "N")],
):
    M, N = x.shape
    columns = I.domain(0, N)
    for row in I.parallel(I.domain(0, M)):
        output[row, columns] = I.maximum(x[row, columns], I.cast(0.0, I.f16))


@intent.kernel
def addcmul_broadcast_bf16(
    x: I.In[I.bf16, ("B", "C", "L")],
    scale: I.In[I.bf16, ("B", "C")],
    bias: I.In[I.bf16, ("B", "C")],
    output: I.Out[I.bf16, ("B", "C", "L")],
):
    B, C, L = x.shape
    channels = I.domain(0, C)
    positions = I.domain(0, L)
    for batch in I.parallel(I.domain(0, B)):
        output[batch, channels, positions] = (
            bias[batch, channels, None]
            + x[batch, channels, positions] * scale[batch, channels, None]
        )
