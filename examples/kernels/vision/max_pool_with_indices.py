import intent
import intent.language as I

from .max_pool import BATCH
from .max_pool import CHANNELS
from .max_pool import HEIGHT
from .max_pool import OUTPUT_HEIGHT
from .max_pool import OUTPUT_WIDTH
from .max_pool import PADDING
from .max_pool import STRIDE
from .max_pool import WIDTH


KERNEL_HEIGHT = 3
KERNEL_WIDTH = 3
KERNEL_ELEMENTS = KERNEL_HEIGHT * KERNEL_WIDTH


@intent.kernel
def max_pool2d_with_indices(
    x: I.In[I.f16, (BATCH, CHANNELS, HEIGHT, WIDTH)],
    output: I.Out[I.f16, (BATCH, CHANNELS, OUTPUT_HEIGHT, OUTPUT_WIDTH)],
    indices: I.Out[I.i64, (BATCH, CHANNELS, OUTPUT_HEIGHT, OUTPUT_WIDTH)],
):
    output_rows = I.domain(0, OUTPUT_HEIGHT)
    output_columns = I.domain(0, OUTPUT_WIDTH)
    filter_elements = I.domain(0, KERNEL_ELEMENTS)
    for batch in I.parallel(I.domain(0, BATCH)):
        for channel in I.parallel(I.domain(0, CHANNELS)):
            local_index = I.indices(filter_elements)
            filter_row = local_index // KERNEL_WIDTH
            filter_column = local_index % KERNEL_WIDTH
            input_rows = I.reshape(
                I.indices(output_rows) * STRIDE - PADDING,
                (output_rows, 1, 1),
            ) + I.reshape(filter_row, (1, 1, KERNEL_ELEMENTS))
            input_columns = I.reshape(
                I.indices(output_columns) * STRIDE - PADDING,
                (1, output_columns, 1),
            ) + I.reshape(filter_column, (1, 1, KERNEL_ELEMENTS))
            row_valid = (input_rows >= 0) & (input_rows < HEIGHT)
            column_valid = (input_columns >= 0) & (input_columns < WIDTH)
            safe_input_rows = I.select(row_valid, input_rows, 0)
            safe_input_columns = I.select(column_valid, input_columns, 0)
            patch = I.mask(
                x[batch, channel, safe_input_rows, safe_input_columns],
                valid=row_valid & column_valid,
                fill=I.cast(-I.inf, I.f16),
            )
            maximum, winner = I.arg_reduce.max(
                patch,
                axis=2,
            )
            winner_index = I.cast(winner, I.index)
            winner_row = winner_index // KERNEL_WIDTH
            winner_column = winner_index % KERNEL_WIDTH
            global_row = (
                I.indices(output_rows)[:, None] * STRIDE
                - PADDING
                + winner_row
            )
            global_column = (
                I.indices(output_columns)[None, :] * STRIDE
                - PADDING
                + winner_column
            )
            winner_valid = (
                (global_row >= 0)
                & (global_row < HEIGHT)
                & (global_column >= 0)
                & (global_column < WIDTH)
            )
            first_valid_row = I.maximum(
                I.indices(output_rows)[:, None] * STRIDE - PADDING,
                0,
            )
            first_valid_column = I.maximum(
                I.indices(output_columns)[None, :] * STRIDE - PADDING,
                0,
            )
            global_row = I.select(winner_valid, global_row, first_valid_row)
            global_column = I.select(
                winner_valid,
                global_column,
                first_valid_column,
            )
            output[batch, channel, output_rows, output_columns] = maximum
            indices[batch, channel, output_rows, output_columns] = I.cast(
                global_row * WIDTH + global_column,
                I.i64,
            )
