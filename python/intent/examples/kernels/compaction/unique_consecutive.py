import intent
import intent.language as I


ROWS = 64
VALUES = 4096


@intent.kernel
def unique_consecutive_rows(
    values: I.In[I.i32, ("M", "N")],
    unique_values: I.Out[I.i32, ("M", "N")],
    run_lengths: I.InOut[I.i32, ("M", "N")],
    inverse: I.Out[I.i32, ("M", "N")],
    counts: I.Out[I.i32, ("M",)],
):
    M, N = values.shape
    columns = I.domain(0, N)
    for row in I.parallel(I.domain(0, M)):
        for column in columns:
            unique_values[row, column] = 0
        positions = I.indices(columns)
        previous_positions = I.maximum(positions - 1, 0)
        I.assume_in_bounds(previous_positions, values, axis=1)
        run_starts = (
            I.cast(positions == 0, I.i32)
            + I.cast(
                values[row, columns] != values[row, previous_positions],
                I.i32,
            )
            > 0
        )
        groups = I.cumsum(
            I.cast(run_starts, I.i32),
            axis=0,
        )
        for column in columns:
            group = groups[column] - 1
            inverse[row, column] = group
            I.atomic.add(
                run_lengths,
                index=(row, group),
                value=I.cast(1, I.i32),
                order="relaxed",
            )
            if run_starts[column]:
                unique_values[row, group] = values[row, column]
        counts[row] = groups[N - 1]
