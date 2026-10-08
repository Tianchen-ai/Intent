import intent
import intent.language as I


TOKENS = 2048
HIDDEN = 4096
STREAMS = 4
MIX_COMPONENTS = STREAMS * (STREAMS + 2)


@intent.fn
def empty_mhc_gemm_rms_summary(tokens, columns):
    return I.record(
        linear=I.zeros((tokens, columns), dtype=I.f32),
        square_sum=I.zeros((tokens,), dtype=I.f32),
    )


@intent.fn
def summarize_mhc_gemm_rms_chunk(x_chunk, weight_chunk, coordinates, stop):
    values = I.transpose(x_chunk, permutation=(1, 0))
    active = coordinates < stop
    valid = I.full(values.shape, fill=True, dtype=I.bool) & active[None, :]
    values = I.mask(values, valid=valid, fill=I.cast(0.0, I.bf16))
    weight_chunk = I.mask(
        weight_chunk,
        valid=active[:, None],
        fill=I.cast(0.0, I.bf16),
    )
    float_values = I.cast(values, I.f32)
    return I.record(
        linear=I.matmul(
            values,
            weight_chunk,
            acc_dtype=I.f32,
        ),
        square_sum=I.reduce.sum(
            float_values * float_values,
            axis=1,
        ),
    )


@intent.fn
def merge_mhc_gemm_rms_summaries(lhs, rhs):
    return I.record(
        linear=lhs.linear + rhs.linear,
        square_sum=lhs.square_sum + rhs.square_sum,
    )


@intent.kernel
def mhc_gemm_rms_partial(
    x: I.In[I.bf16, ("T", "K")],
    weight: I.In[I.bf16, ("K", "N")],
    partial_linear: I.Out[I.f32, ("P", "T", "N")],
    partial_square_sum: I.Out[I.f32, ("P", "T")],
    P: I.Constexpr[int],
):
    _, T, N = partial_linear.shape
    K = x.shape[1]
    tokens = I.domain(0, T)
    columns = I.domain(0, N)
    reduction_axis = I.domain(0, K)
    parts = I.domain(0, P)
    width = (K + P - 1) // P
    for part in I.parallel(parts):
        begin = I.minimum(part * width, K)
        end = I.minimum((part + 1) * width, K)
        reduction = reduction_axis[begin:end]
        summary = I.region_fold(
            source=(
                I.transpose(x[tokens, reduction], permutation=(1, 0)),
                weight[reduction, columns],
                I.indices(reduction),
            ),
            axis=0,
            summarize=summarize_mhc_gemm_rms_chunk,
            combine=merge_mhc_gemm_rms_summaries,
            identity=empty_mhc_gemm_rms_summary(T, N),
            operands=(end,),
        )
        partial_linear[part, tokens, columns] = summary.linear
        partial_square_sum[part, tokens] = summary.square_sum


@intent.kernel
def mhc_gemm_rms_finalize(
    partial_linear: I.In[I.f32, ("P", "T", "N")],
    partial_square_sum: I.In[I.f32, ("P", "T")],
    bias: I.In[I.bf16, ("N",)],
    mixed: I.Out[I.bf16, ("T", "N")],
    rms: I.Out[I.f32, ("T", 1)],
    reduction_size: I.i64,
    P: I.Constexpr[int],
    STREAMS: I.Constexpr[int],
    ALPHA_PRE: I.Constexpr[float],
    ALPHA_POST: I.Constexpr[float],
    ALPHA_RESIDUAL: I.Constexpr[float],
):
    _, T, N = partial_linear.shape
    parts = I.domain(0, P)
    tokens = I.domain(0, T)
    columns = I.domain(0, N)
    linear = I.reduce.sum(
        partial_linear[parts, tokens, columns],
        axis=0,
    )
    square_sum = I.reduce.sum(
        partial_square_sum[parts, tokens],
        axis=0,
    )
    root_mean_square = I.rsqrt(
        square_sum / I.cast(reduction_size, I.f32)
    )
    column_index = I.indices(columns)
    pre = column_index < STREAMS
    post = (column_index >= STREAMS) & (column_index < 2 * STREAMS)
    zero = I.cast(column_index, I.f32) * 0.0
    scale = I.mask(
        zero + ALPHA_PRE,
        valid=pre,
        fill=zero + ALPHA_RESIDUAL,
    )
    scale = I.mask(
        zero + ALPHA_POST,
        valid=post,
        fill=scale,
    )
    normalized = (
        linear * scale * root_mean_square[:, None]
        + I.cast(bias[columns], I.f32)
    )
    sigmoid = 1.0 / (1.0 + I.exp(-normalized))
    result = I.mask(
        normalized,
        valid=(pre == False) & (post == False),
        fill=sigmoid,
    )
    result = I.mask(2.0 * sigmoid, valid=post, fill=result)
    mixed[tokens, columns] = I.cast(result, I.bf16)
    rms[tokens, 0] = 1.0 / root_mean_square


@intent.kernel
def mhc_apply_residual(
    residual: I.In[I.bf16, ("T", "S", "D")],
    layer_output: I.In[I.bf16, ("T", "D")],
    post_mix: I.In[I.f32, ("T", "S")],
    residual_mix: I.In[I.f32, ("T", "S", "S")],
    output: I.Out[I.bf16, ("T", "S", "D")],
):
    T, S, D = residual.shape
    output_streams = I.domain(0, S)
    dimensions = I.domain(0, D)
    for token in I.parallel(I.domain(0, T)):
        mixed_residual = I.zeros((S, D), dtype=I.f32)
        for source_stream in range(STREAMS):
            source_values = I.cast(
                residual[token, source_stream, dimensions], I.f32
            )
            mix = residual_mix[token, output_streams, source_stream]
            mixed_residual = mixed_residual + source_values[None, :] * mix[:, None]
        output[token, output_streams, dimensions] = I.cast(
            mixed_residual
            + post_mix[token, output_streams][:, None]
            * I.cast(layer_output[token, dimensions], I.f32)[None, :],
            I.bf16,
        )


@intent.kernel
def mhc_sinkhorn(
    logits: I.InOut[I.f32, ("T", "S", "S")],
):
    T, S, _ = logits.shape
    rows = I.domain(0, S)
    columns = I.domain(0, S)
    for token in I.parallel(I.domain(0, T)):
        matrix = I.exp(logits[token, rows, columns])
        for _ in range(20):
            row_sum = I.reduce.sum(matrix, axis=1)
            matrix = matrix / row_sum[:, None]
            column_sum = I.reduce.sum(matrix, axis=0)
            matrix = matrix / column_sum[None, :]
        logits[token, rows, columns] = matrix


@intent.kernel
def mhc_pre_gemm_sqrsum(
    residual_flat: I.In[I.bf16, ("T", "K")],
    weight: I.In[I.f32, ("C", "K")],
    mixes: I.Out[I.f32, ("T", "C")],
    square_sum: I.Out[I.f32, ("T",)],
):
    T, K = residual_flat.shape
    C = weight.shape[0]
    tokens = I.domain(0, T)
    reduction = I.domain(0, K)
    components = I.domain(0, C)
    values = I.cast(residual_flat[tokens, reduction], I.f32)
    square_sum[tokens] = I.reduce.sum(
        values * values,
        axis=1,
    )
    mixes[tokens, components] = I.matmul(
        residual_flat[tokens, reduction],
        I.cast(weight[components, reduction], I.bf16),
        transpose_rhs=True,
        acc_dtype=I.f32,
    )


@intent.kernel
def mhc_pre_fuse(
    mixes: I.In[I.f32, ("T", "C")],
    square_sum: I.In[I.f32, ("T",)],
    scale: I.In[I.f32, (3,)],
    base: I.In[I.f32, ("C",)],
    residual: I.In[I.bf16, ("T", "S", "D")],
    post_mix: I.Out[I.f32, ("T", "S")],
    residual_mix: I.Out[I.f32, ("T", "S", "S")],
    layer_input: I.Out[I.bf16, ("T", "D")],
    RMS_EPS: I.Constexpr[float],
    PRE_EPS: I.Constexpr[float],
    SINKHORN_EPS: I.Constexpr[float],
    POST_MULTIPLIER: I.Constexpr[float],
    SINKHORN_REPEATS: I.Constexpr[int],
):
    T, S, D = residual.shape
    streams = I.domain(0, S)
    hidden = I.domain(0, D)
    for token in I.parallel(I.domain(0, T)):
        normalization = I.rsqrt(
            square_sum[token] / I.cast(S * D, I.f32) + RMS_EPS
        )
        pre_logits = (
            mixes[token, streams] * normalization * scale[0]
            + base[streams]
        )
        pre = 1.0 / (1.0 + I.exp(-pre_logits)) + PRE_EPS
        post_indices = I.indices(streams) + S
        post_logits = (
            mixes[token, post_indices] * normalization * scale[1]
            + base[post_indices]
        )
        post_mix[token, streams] = (
            1.0 / (1.0 + I.exp(-post_logits))
        ) * POST_MULTIPLIER
        matrix_indices = (
            2 * S
            + I.indices(streams)[:, None] * S
            + I.indices(streams)[None, :]
        )
        matrix = (
            mixes[token, matrix_indices] * normalization * scale[2]
            + base[matrix_indices]
        )
        row_maximum = I.reduce.max(matrix, axis=1)
        matrix = I.exp(matrix - row_maximum[:, None])
        row_sum = I.reduce.sum(matrix, axis=1)
        matrix = matrix / row_sum[:, None] + SINKHORN_EPS
        column_sum = I.reduce.sum(matrix, axis=0)
        matrix = matrix / (column_sum[None, :] + SINKHORN_EPS)
        for _ in range(SINKHORN_REPEATS - 1):
            row_sum = I.reduce.sum(matrix, axis=1)
            matrix = matrix / (row_sum[:, None] + SINKHORN_EPS)
            column_sum = I.reduce.sum(matrix, axis=0)
            matrix = matrix / (column_sum[None, :] + SINKHORN_EPS)
        residual_mix[token, streams, streams] = matrix
        layer_input[token, hidden] = I.cast(
            I.reduce.sum(
                I.cast(residual[token, streams, hidden], I.f32)
                * pre[:, None],
                axis=0,
            ),
            I.bf16,
        )
