import intent
import intent.language as I


BATCH = 8
CHANNELS = 64
HEIGHT = 128
WIDTH = 128
ROIS = 2048
POOLED_HEIGHT = 7
POOLED_WIDTH = 7


@intent.kernel
def roi_align_center_sample(
    feature: I.In[I.f32, (BATCH, CHANNELS, HEIGHT, WIDTH)],
    rois: I.In[I.f32, (ROIS, 5)],
    output: I.Out[
        I.f32,
        (ROIS, CHANNELS, POOLED_HEIGHT, POOLED_WIDTH),
    ],
):
    for roi in I.parallel(I.domain(0, ROIS)):
        batch = I.cast(rois[roi, 0], I.i32)
        start_x = rois[roi, 1]
        start_y = rois[roi, 2]
        end_x = rois[roi, 3]
        end_y = rois[roi, 4]
        I.assume_in_bounds(batch, feature, axis=0)
        bin_width = I.maximum(end_x - start_x, 1.0) / POOLED_WIDTH
        bin_height = I.maximum(end_y - start_y, 1.0) / POOLED_HEIGHT
        for channel in I.parallel(I.domain(0, CHANNELS)):
            for pooled_y in range(POOLED_HEIGHT):
                sample_y = start_y + (I.cast(pooled_y, I.f32) + 0.5) * bin_height
                y_low = I.cast(sample_y, I.i32)
                y_low = I.maximum(y_low, I.cast(0, I.i32))
                y_low = I.minimum(y_low, I.cast(HEIGHT - 1, I.i32))
                y_high = I.minimum(
                    y_low + 1,
                    I.cast(HEIGHT - 1, I.i32),
                )
                I.assume_in_bounds(y_low, feature, axis=2)
                I.assume_in_bounds(y_high, feature, axis=2)
                y_fraction = sample_y - I.cast(y_low, I.f32)
                for pooled_x in range(POOLED_WIDTH):
                    sample_x = start_x + (
                        I.cast(pooled_x, I.f32) + 0.5
                    ) * bin_width
                    x_low = I.cast(sample_x, I.i32)
                    x_low = I.maximum(x_low, I.cast(0, I.i32))
                    x_low = I.minimum(x_low, I.cast(WIDTH - 1, I.i32))
                    x_high = I.minimum(
                        x_low + 1,
                        I.cast(WIDTH - 1, I.i32),
                    )
                    I.assume_in_bounds(x_low, feature, axis=3)
                    I.assume_in_bounds(x_high, feature, axis=3)
                    x_fraction = sample_x - I.cast(x_low, I.f32)
                    top = (
                        feature[batch, channel, y_low, x_low]
                        * (1.0 - x_fraction)
                        + feature[batch, channel, y_low, x_high]
                        * x_fraction
                    )
                    bottom = (
                        feature[batch, channel, y_high, x_low]
                        * (1.0 - x_fraction)
                        + feature[batch, channel, y_high, x_high]
                        * x_fraction
                    )
                    output[roi, channel, pooled_y, pooled_x] = (
                        top * (1.0 - y_fraction) + bottom * y_fraction
                    )
