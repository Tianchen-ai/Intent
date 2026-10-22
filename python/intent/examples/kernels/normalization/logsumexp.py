import intent
import intent.language as I


ROWS = 8192
COLUMNS = 8192


@intent.kernel
def row_logsumexp(
    x: I.In[I.f32, ("M", "N")],
    y: I.Out[I.f32, ("M",)],
):
    M, N = x.shape
    columns = I.domain(0, N)
    for row in I.parallel(I.domain(0, M)):
        values = x[row, columns]
        maximum = I.reduce.max(values, axis=0)
        shifted = I.exp(values - maximum)
        denominator = I.reduce.sum(shifted, axis=0)
        y[row] = maximum + I.log(denominator)
