import intent
import intent.language as I


BATCH = 17
ROWS = 257
COLUMNS = 4093


@intent.kernel
def batched_row_affine(
    x: I.In[I.f32, ("B", "M", "N")],
    scale: I.In[I.f32, ("B", "M")],
    bias: I.In[I.f32, ("B", "M")],
    output: I.Out[I.f32, ("B", "M", "N")],
):
    B, M, N = x.shape
    batch_axis = I.domain(0, B)
    row_axis = I.domain(0, M)
    columns = I.domain(0, N)
    for batch in I.parallel(batch_axis):
        for row in I.parallel(row_axis):
            output[batch, row, columns] = (
                x[batch, row, columns] * scale[batch, row] + bias[batch, row]
            )
