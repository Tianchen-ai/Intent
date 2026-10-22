import intent
import intent.language as I


@intent.fn
def encode_e2m1(value):
    magnitude = I.maximum(value, -value)
    code = I.select(magnitude > 0.25, I.cast(1, I.u8), I.cast(0, I.u8))
    code = I.select(magnitude >= 0.75, I.cast(2, I.u8), code)
    code = I.select(magnitude > 1.25, I.cast(3, I.u8), code)
    code = I.select(magnitude >= 1.75, I.cast(4, I.u8), code)
    code = I.select(magnitude > 2.5, I.cast(5, I.u8), code)
    code = I.select(magnitude >= 3.5, I.cast(6, I.u8), code)
    code = I.select(magnitude > 5.0, I.cast(7, I.u8), code)
    sign = I.cast(I.bitcast(value, I.u32) >> I.cast(31, I.u32), I.u8)
    return code | (sign << I.cast(3, I.u8))


@intent.fn
def quantize_nvfp4_groups(values, encoding_scale):
    absolute = I.maximum(values, -values)
    maximum = I.reduce.max(absolute, axis=1)
    scale = I.cast(
        I.maximum((maximum / 6.0) * encoding_scale, 1.5258789e-05),
        I.f8e4m3fn,
    )
    multiplier = encoding_scale / I.cast(scale, I.f32)
    return encode_e2m1(values * multiplier[:, None]), I.bitcast(scale, I.u8)


@intent.fn
def quantize_nvfp4_group(x, packed, scales, encoding_scale, row, group):
    members = I.indices(I.domain(0, 16))
    pairs = I.indices(I.domain(0, 8))
    values = I.reshape(I.cast(x[row, group * 16 + members], I.f32), (1, 16))
    encoded, scale = quantize_nvfp4_groups(values, encoding_scale)
    scales[row // 128, group // 4, row % 32, (row % 128) // 32, group % 4] = scale[0]
    paired = I.reshape(encoded, (8, 2))
    packed[row, group * 8 + pairs] = paired[:, 0] | (paired[:, 1] << I.cast(4, I.u8))
    return


@intent.kernel
def nvfp4_quantize(
    x: I.In[I.bf16, ("M", "N")],
    global_scale: I.In[I.f32, (1,)],
    packed: I.InOut[I.u8, ("M", "P")],
    scales: I.InOut[I.u8, ("MT", "NT", 32, 4, 4)],
):
    M, N = x.shape
    groups = N // 16
    encoding_scale = global_scale[0]

    # The scale ABI stores 128 rows and four group16 scales in one swizzled block.
    local_rows = I.indices(I.domain(0, 128))
    local_columns = I.indices(I.domain(0, 64))
    packed_columns = I.indices(I.domain(0, 32))
    for row_block in I.parallel(I.domain(0, (M + 127) // 128)):
        for group_block in I.parallel(I.domain(0, (groups + 3) // 4)):
            row_start = row_block * 128
            group_start = group_block * 4
            if (row_start + 128 <= M) and (group_start + 4 <= groups):
                rows = row_start + local_rows
                columns = group_start * 16 + local_columns
                values = I.reshape(
                    I.cast(x[rows[:, None], columns[None, :]], I.f32), (512, 16)
                )
                encoded, scale = quantize_nvfp4_groups(values, encoding_scale)
                scales[row_block, group_block, :, :, :] = I.transpose(
                    I.reshape(scale, (4, 32, 4)), (1, 0, 2)
                )
                paired = I.reshape(encoded, (128, 32, 2))
                packed[rows[:, None], (group_start * 8 + packed_columns)[None, :]] = (
                    paired[:, :, 0] | (paired[:, :, 1] << I.cast(4, I.u8))
                )
            else:
                # Only valid groups are written; InOut padding is unchanged.
                row_count = I.maximum(0, I.minimum(128, M - row_start))
                group_count = I.maximum(0, I.minimum(4, groups - group_start))
                for row_offset in I.domain(0, row_count):
                    for group_offset in I.domain(0, group_count):
                        quantize_nvfp4_group(
                            x, packed, scales, encoding_scale,
                            row_start + row_offset, group_start + group_offset,
                        )
