import intent
import intent.language as I


ROWS = 4093
COLUMNS = 8191


@intent.kernel
def alternating_signed_indices(
    shape_source: I.In[I.f32, ("M", "N")],
    output: I.Out[I.i32, ("M", "N")],
):
    M, N = shape_source.shape
    columns = I.domain(0, N)
    for row in I.parallel(I.domain(0, M)):
        positions = I.cast(I.indices(columns), I.i32)
        negative_positions = -positions
        selected = positions if row % 2 == 0 else negative_positions
        output[row, columns] = selected
