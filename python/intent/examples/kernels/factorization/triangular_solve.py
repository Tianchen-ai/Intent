import intent
import intent.language as I


BATCH = 4096
SIZE = 16


@intent.kernel
def batched_lower_triangular_solve(
    lower: I.In[I.f32, (BATCH, SIZE, SIZE)],
    solution: I.InOut[I.f32, (BATCH, SIZE)],
):
    for batch in I.parallel(I.domain(0, BATCH)):
        for row in range(SIZE):
            residual = solution[batch, row]
            for column in range(row):
                residual = residual - (
                    lower[batch, row, column] * solution[batch, column]
                )
            solution[batch, row] = residual / lower[batch, row, row]
