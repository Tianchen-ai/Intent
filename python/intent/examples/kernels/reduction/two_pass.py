import intent
import intent.language as I


ROWS = 128
COLUMNS = 257
PARTS = 300


@intent.kernel
def partitioned_max_partial(
    x: I.In[I.f32, ("M", "N")],
    partial: I.Out[I.f32, ("M", "P")],
):
    M, N = x.shape
    _, P = partial.shape
    columns = I.domain(0, N)
    parts = I.domain(0, P)
    width = (N + P - 1) // P
    for row in I.parallel(I.domain(0, M)):
        for part in I.parallel(parts):
            begin = I.minimum(part * width, N)
            end = I.minimum((part + 1) * width, N)
            region = columns[begin:end]
            partial[row, part] = I.reduce.max(
                x[row, region], axis=0
            )


@intent.kernel
def partitioned_max_reduce(
    partial: I.In[I.f32, ("M", "P")],
    output: I.Out[I.f32, ("M",)],
):
    M, P = partial.shape
    parts = I.domain(0, P)
    for row in I.parallel(I.domain(0, M)):
        output[row] = I.reduce.max(
            partial[row, parts], axis=0
        )
