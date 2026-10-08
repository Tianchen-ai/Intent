import intent
import intent.language as I

ROWS = 8192
COLUMNS = 8192
ROW_MAJOR_NOALIAS = I.constraints(
    strides=(None, 1),
    noalias=True,
)


@intent.fn
def maximum_number(lhs, rhs):
    return I.maximum_num(lhs, rhs)


@intent.fn
def softmax_maximum(values):
    # NaN inputs still propagate through the exponential sum to every output.
    return I.reduce(values, axis=0, identity=-I.inf, combine=maximum_number)


@intent.kernel
def stable_softmax(
    x: I.In[I.f32, ("M", "N"), ROW_MAJOR_NOALIAS],
    y: I.Out[I.f32, ("M", "N"), ROW_MAJOR_NOALIAS],
):
    M, N = x.shape
    columns = I.domain(0, N)
    for row in I.parallel(I.domain(0, M)):
        values = x[row, columns]
        maximum = softmax_maximum(values)
        numerator = I.exp(values - maximum)
        denominator = I.reduce.sum(numerator, axis=0)
        inverse_denominator = 1.0 / denominator
        y[row, columns] = numerator * inverse_denominator


@intent.kernel
def stable_softmax_f16(
    x: I.In[I.f16, ("M", "N")],
    y: I.Out[I.f16, ("M", "N")],
):
    M, N = x.shape
    columns = I.domain(0, N)
    for row in I.parallel(I.domain(0, M)):
        values = I.cast(x[row, columns], I.f32)
        maximum = softmax_maximum(values)
        numerator = I.exp(values - maximum)
        denominator = I.reduce.sum(numerator, axis=0)
        y[row, columns] = I.cast(numerator / denominator, I.f16)


@intent.kernel
def chunked_softmax_bf16(
    x: I.In[I.bf16, ("M", "N")],
    y: I.Out[I.bf16, ("M", "N")],
):
    M, N = x.shape
    columns = I.domain(0, N)
    for row in I.parallel(I.domain(0, M)):
        values = I.cast(x[row, columns], I.f32)
        maximum = softmax_maximum(values)
        numerator = I.exp(values - maximum)
        denominator = I.reduce.sum(numerator, axis=0)
        y[row, columns] = I.cast(numerator / denominator, I.bf16)
