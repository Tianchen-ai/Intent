import intent
import intent.language as I


ROWS = 4093
COLUMNS = 8191


@intent.kernel
def matrix_transpose(
    x: I.In[I.f16, ("M", "N")],
    output: I.Out[I.f16, ("N", "M")],
):
    M, N = x.shape
    rows = I.domain(0, M)
    columns = I.domain(0, N)
    tile = x[rows, columns]
    output[columns, rows] = I.transpose(
        tile, permutation=(1, 0)
    )
