import intent
import intent.language as I


BATCH = 128
ROWS = 32
COLUMNS = 16


@intent.kernel
def batched_householder_qr(
    matrices: I.InOut[I.f32, ("B", ROWS, COLUMNS)],
    tau: I.Out[I.f32, ("B", COLUMNS)],
):
    B = matrices.shape[0]
    for batch in I.parallel(I.domain(0, B)):
        for reflector in range(COLUMNS):
            norm_square = I.cast(0.0, I.f32)
            for row in range(reflector, ROWS):
                value = matrices[batch, row, reflector]
                norm_square = norm_square + value * value
            norm = 1.0 / I.rsqrt(norm_square)
            alpha = matrices[batch, reflector, reflector]
            beta = norm
            if alpha >= 0.0:
                beta = -norm
            scale = 1.0 / (alpha - beta)
            tau_value = (beta - alpha) / beta
            matrices[batch, reflector, reflector] = beta
            tau[batch, reflector] = tau_value
            for row in range(reflector + 1, ROWS):
                matrices[batch, row, reflector] = (
                    matrices[batch, row, reflector] * scale
                )
            for column in range(reflector + 1, COLUMNS):
                dot = matrices[batch, reflector, column]
                for row in range(reflector + 1, ROWS):
                    dot = dot + (
                        matrices[batch, row, reflector]
                        * matrices[batch, row, column]
                    )
                dot = tau_value * dot
                matrices[batch, reflector, column] = (
                    matrices[batch, reflector, column] - dot
                )
                for row in range(reflector + 1, ROWS):
                    matrices[batch, row, column] = (
                        matrices[batch, row, column]
                        - matrices[batch, row, reflector] * dot
                    )
