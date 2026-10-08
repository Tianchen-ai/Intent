import intent
import intent.language as I


BATCH = 256
SIZE = 16


@intent.kernel
def batched_cholesky_lower(
    matrices: I.InOut[I.f32, ("B", SIZE, SIZE)],
):
    B = matrices.shape[0]
    for batch in I.parallel(I.domain(0, B)):
        for column in range(SIZE):
            diagonal_sum = I.cast(0.0, I.f32)
            for inner in range(column):
                value = matrices[batch, column, inner]
                diagonal_sum = diagonal_sum + value * value
            diagonal = matrices[batch, column, column] - diagonal_sum
            factor = 1.0 / I.rsqrt(diagonal)
            matrices[batch, column, column] = factor
            for row in range(column + 1, SIZE):
                product_sum = I.cast(0.0, I.f32)
                for inner in range(column):
                    product_sum = product_sum + (
                        matrices[batch, row, inner]
                        * matrices[batch, column, inner]
                    )
                matrices[batch, row, column] = (
                    matrices[batch, row, column] - product_sum
                ) / factor
            for upper in range(column + 1, SIZE):
                matrices[batch, column, upper] = 0.0
