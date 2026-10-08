import intent
import intent.language as I


ROWS = 4096
COLUMNS = 4096


@intent.kernel
def adafactor_update_rows(
    gradient: I.In[I.f32, ("M", "N")],
    row_state: I.InOut[I.f32, ("M",)],
    row_mean: I.InOut[I.f32, (1,)],
    decay: I.f32,
    inverse_columns: I.f32,
    inverse_rows: I.f32,
):
    M, N = gradient.shape
    columns = I.domain(0, N)
    for row in I.parallel(I.domain(0, M)):
        values = gradient[row, columns]
        square_mean = (
            I.reduce.sum(values * values, axis=0)
            * inverse_columns
        )
        updated = decay * row_state[row] + (1.0 - decay) * square_mean
        row_state[row] = updated
        I.atomic.add(
            row_mean,
            index=(0,),
            value=updated * inverse_rows,
            order="relaxed",
        )


@intent.kernel
def adafactor_update_columns(
    gradient: I.In[I.f32, ("M", "N")],
    column_state: I.InOut[I.f32, ("N",)],
    decay: I.f32,
    inverse_rows: I.f32,
):
    M, N = gradient.shape
    rows = I.domain(0, M)
    for column in I.parallel(I.domain(0, N)):
        values = gradient[rows, column]
        square_mean = (
            I.reduce.sum(values * values, axis=0) * inverse_rows
        )
        column_state[column] = (
            decay * column_state[column] + (1.0 - decay) * square_mean
        )


@intent.kernel
def adafactor_apply(
    gradient: I.In[I.f32, ("M", "N")],
    row_state: I.In[I.f32, ("M",)],
    column_state: I.In[I.f32, ("N",)],
    row_mean: I.In[I.f32, (1,)],
    parameter: I.InOut[I.f32, ("M", "N")],
    learning_rate: I.f32,
    epsilon: I.f32,
):
    M, N = parameter.shape
    columns = I.domain(0, N)
    for row in I.parallel(I.domain(0, M)):
        variance = (
            row_state[row]
            * column_state[columns]
            / I.maximum(row_mean[0], epsilon)
        )
        update = gradient[row, columns] * I.rsqrt(variance + epsilon)
        parameter[row, columns] = parameter[row, columns] - learning_rate * update
