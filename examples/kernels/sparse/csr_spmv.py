import intent
import intent.language as I


ROWS = 32768
COLUMNS = 32768
NONZEROS_PER_ROW = 32
NONZEROS = ROWS * NONZEROS_PER_ROW


@intent.kernel
def csr_spmv(
    row_offsets: I.In[I.i32, (ROWS + 1,)],
    column_indices: I.In[I.i32, (NONZEROS,)],
    values: I.In[I.f32, (NONZEROS,)],
    vector: I.In[I.f32, (COLUMNS,)],
    output: I.Out[I.f32, (ROWS,)],
):
    for row in I.parallel(I.domain(0, ROWS)):
        start = row_offsets[row]
        stop = row_offsets[row + 1]
        accumulation = I.cast(0.0, I.f32)
        for nonzero in range(start, stop):
            I.assume_in_bounds(nonzero, column_indices, axis=0)
            I.assume_in_bounds(nonzero, values, axis=0)
            column = column_indices[nonzero]
            I.assume_in_bounds(column, vector, axis=0)
            accumulation = accumulation + values[nonzero] * vector[column]
        output[row] = accumulation
