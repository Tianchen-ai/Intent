import intent
import intent.language as I


ROWS = 64
VALUES = 4096


@intent.kernel
def compact_nonzero_rows(
    values: I.In[I.f32, ("M", "N")],
    indices: I.Out[I.i32, ("M", "N")],
    counts: I.Out[I.i32, ("M",)],
):
    M, N = values.shape
    columns = I.domain(0, N)
    for row in I.parallel(I.domain(0, M)):
        for column in columns:
            indices[row, column] = -1
        flags = I.cast(values[row, columns] != 0.0, I.i32)
        inclusive = I.cumsum(
            flags,
            axis=0,
        )
        for column in columns:
            if flags[column] != 0:
                indices[row, inclusive[column] - 1] = I.cast(column, I.i32)
        counts[row] = inclusive[N - 1]
