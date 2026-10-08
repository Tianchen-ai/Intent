import intent
import intent.language as I


BATCH = 512
FEATURES = 128
MAX_LENGTH = 128
TOKENS = 33024


@intent.kernel
def jagged_mean(
    values: I.In[I.f32, ("T", "D")],
    offsets: I.In[I.i32, ("B_PLUS_1",)],
    output: I.Out[I.f32, (BATCH, "D")],
):
    D = output.shape[1]
    B = output.shape[0]
    T = values.shape[0]
    batches = I.domain(0, B)
    features = I.domain(0, D)
    rows = I.ragged(
        outer=batches,
        members=I.domain(0, T),
        offsets=offsets,
    )
    for batch in I.parallel(rows.outer):
        total = I.reduce.sum(
            values[rows[batch], features],
            axis=0,
        )
        count = I.cast(offsets[batch + 1] - offsets[batch], I.f32)
        output[batch, features] = total / count
