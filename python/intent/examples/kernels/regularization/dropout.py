import intent
import intent.language as I


ROWS = 8192
FEATURES = 4096
SEED = 12345
DROP_PROBABILITY = 0.1


@intent.kernel
def xor_shift_dropout(
    x: I.In[I.f16, ("M", "N")],
    output: I.Out[I.f16, ("M", "N")],
    mixed_seed: I.i32,
    drop_probability: I.f32,
    inverse_keep_probability: I.f32,
):
    M, N = x.shape
    columns = I.domain(0, N)
    for row in I.parallel(I.domain(0, M)):
        offsets = I.cast(row * N + I.indices(columns), I.i32)
        hashed = offsets * 1103515245 + mixed_seed
        hashed = hashed ^ (hashed >> 16)
        hashed = hashed ^ (hashed << 8)
        hashed = hashed ^ (hashed >> 4)
        positive = hashed & 0x7FFFFFFF
        random = I.cast(positive, I.f32) / 2147483647.0
        keep = random > drop_probability
        output[row, columns] = I.mask(
            x[row, columns] * I.cast(inverse_keep_probability, I.f16),
            valid=keep,
            fill=I.cast(0.0, I.f16),
        )
