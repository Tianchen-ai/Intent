import intent
import intent.language as I


BATCH = 512
ROWS = 17
COLUMNS = 31
ROW_MAJOR = I.constraints(
    strides=(ROWS * COLUMNS, COLUMNS, 1),
)


@intent.kernel
def ordered_product_prefix(
    x: I.In[I.f32, (BATCH, ROWS, COLUMNS), ROW_MAJOR],
    output: I.Out[I.f32, (BATCH, ROWS, COLUMNS), ROW_MAJOR],
):
    elements = I.domain(0, ROWS * COLUMNS)
    for batch in I.parallel(I.domain(0, BATCH)):
        element = I.indices(elements)
        row = element // COLUMNS
        column = element % COLUMNS
        prefix = I.cumsum(
            x[batch, row, column],
            axis=0,
        )
        I.scatter_unique(output, index=(batch, row, column), value=prefix)


@intent.kernel
def row_cumsum_f32(
    x: I.In[I.f32, ("M", "N")],
    output: I.Out[I.f32, ("M", "N")],
):
    M, N = x.shape
    columns = I.domain(0, N)
    for row in I.parallel(I.domain(0, M)):
        output[row, columns] = I.cumsum(
            x[row, columns],
            axis=0,
        )
