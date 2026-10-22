import intent
import intent.language as I


ROWS = 8192
COLUMNS = 4093


@intent.kernel
def row_boolean_reduction(
    shape_source: I.In[I.f32, ("M", "N")],
    output: I.Out[I.i32, ("M",)],
):
    M, N = shape_source.shape
    columns = I.domain(0, N)
    for row in I.parallel(I.domain(0, M)):
        column_indices = I.indices(columns)
        contains_origin = column_indices == 0
        stays_nonnegative = column_indices >= 0
        any_origin = I.reduce.any(contains_origin, axis=0)
        all_nonnegative = I.reduce.all(
            stays_nonnegative,
            axis=0,
        )
        encoded = I.cast(any_origin, I.i32) + 2 * I.cast(all_nonnegative, I.i32)
        output[row] = encoded
