import intent
import intent.language as I


BATCH = 128
QUERY_LENGTH = 128
REFERENCE_LENGTH = 128
MATCH_SCORE = 2
MISMATCH_SCORE = -1
GAP_SCORE = -1


@intent.kernel
def smith_waterman_score(
    query: I.In[I.i32, (BATCH, QUERY_LENGTH)],
    reference: I.In[I.i32, (BATCH, REFERENCE_LENGTH)],
    output: I.Out[I.i32, (BATCH,)],
):
    for batch in I.parallel(I.domain(0, BATCH)):
        previous = I.buffer((REFERENCE_LENGTH + 1,), I.i32, init=0)
        current = I.buffer((REFERENCE_LENGTH + 1,), I.i32, init=0)
        maximum_score = I.cast(0, I.i32)
        for row in range(1, QUERY_LENGTH + 1):
            I.store(current, 0, I.cast(0, I.i32))
            for column in range(1, REFERENCE_LENGTH + 1):
                substitution = I.cast(
                    MATCH_SCORE
                    if query[batch, row - 1] == reference[batch, column - 1]
                    else MISMATCH_SCORE,
                    I.i32,
                )
                diagonal = (
                    I.mutable_load(previous, column - 1) + substitution
                )
                deletion = I.mutable_load(previous, column) + GAP_SCORE
                insertion = I.mutable_load(current, column - 1) + GAP_SCORE
                cell = I.maximum(
                    I.cast(0, I.i32),
                    I.maximum(diagonal, I.maximum(deletion, insertion)),
                )
                I.store(current, column, cell)
                maximum_score = I.maximum(maximum_score, cell)
            for column in range(REFERENCE_LENGTH + 1):
                I.store(previous, column, I.mutable_load(current, column))
        output[batch] = maximum_score
