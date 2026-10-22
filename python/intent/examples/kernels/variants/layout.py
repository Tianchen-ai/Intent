import intent
import intent.language as I


@intent.kernel
def matrix_transpose_scalar_domains(
    x: I.In[I.f16, ("M", "N")],
    output: I.Out[I.f16, ("N", "M")],
):
    M, N = x.shape
    for row in I.parallel(I.domain(0, M)):
        for column in I.parallel(I.domain(0, N)):
            output[column, row] = x[row, column]
