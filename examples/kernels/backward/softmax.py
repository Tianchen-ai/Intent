import intent
import intent.language as I


ROWS = 4096
COLUMNS = 4097


@intent.kernel
def softmax_backward(
    probabilities: I.In[I.f32, (ROWS, COLUMNS)],
    upstream: I.In[I.f32, (ROWS, COLUMNS)],
    gradient: I.Out[I.f32, (ROWS, COLUMNS)],
):
    columns = I.domain(0, COLUMNS)
    for row in I.parallel(I.domain(0, ROWS)):
        row_probabilities = probabilities[row, columns]
        row_upstream = upstream[row, columns]
        projection = I.reduce.sum(
            row_probabilities * row_upstream,
            axis=0,
        )
        gradient[row, columns] = row_probabilities * (
            row_upstream - projection
        )
