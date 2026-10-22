import intent
import intent.language as I


BATCH = 8
CHANNELS = 32
HEIGHT = 128
WIDTH = 128
KERNEL_HEIGHT = 3
KERNEL_WIDTH = 3
STRIDE = 2
PADDING = 1
OUTPUT_HEIGHT = (HEIGHT + 2 * PADDING - KERNEL_HEIGHT) // STRIDE + 1
OUTPUT_WIDTH = (WIDTH + 2 * PADDING - KERNEL_WIDTH) // STRIDE + 1


@intent.kernel
def max_pool2d(
    x: I.In[I.f16, (BATCH, CHANNELS, HEIGHT, WIDTH)],
    output: I.Out[I.f16, (BATCH, CHANNELS, OUTPUT_HEIGHT, OUTPUT_WIDTH)],
):
    output_rows = I.domain(0, OUTPUT_HEIGHT)
    output_columns = I.domain(0, OUTPUT_WIDTH)
    kernel_rows = I.domain(0, KERNEL_HEIGHT)
    kernel_columns = I.domain(0, KERNEL_WIDTH)
    for batch in I.parallel(I.domain(0, BATCH)):
        for channel in I.parallel(I.domain(0, CHANNELS)):
            row_indices = I.reshape(
                I.indices(output_rows) * STRIDE - PADDING,
                (OUTPUT_HEIGHT, 1, 1, 1),
            )
            column_indices = I.reshape(
                I.indices(output_columns) * STRIDE - PADDING,
                (1, OUTPUT_WIDTH, 1, 1),
            )
            input_rows = row_indices + I.indices(kernel_rows)[:, None]
            input_columns = column_indices + I.indices(kernel_columns)
            row_valid = (input_rows >= 0) & (input_rows < HEIGHT)
            column_valid = (input_columns >= 0) & (input_columns < WIDTH)
            safe_input_rows = I.select(row_valid, input_rows, 0)
            safe_input_columns = I.select(column_valid, input_columns, 0)
            patch = I.mask(
                x[batch, channel, safe_input_rows, safe_input_columns],
                valid=row_valid & column_valid,
                fill=I.cast(-I.inf, I.f16),
            )
            column_maxima = I.reduce.max(
                patch,
                axis=3,
            )
            maxima = I.reduce.max(
                column_maxima,
                axis=2,
            )
            output[batch, channel, output_rows, output_columns] = maxima
