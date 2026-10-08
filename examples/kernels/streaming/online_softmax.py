import intent
import intent.language as I


ROWS = 8192
COLUMNS = 8192


@intent.fn
def merge_softmax_summary(lhs, rhs):
    valid = lhs.valid | rhs.valid
    maximum = I.select(lhs.valid, lhs.maximum, rhs.maximum)
    maximum = I.select(
        rhs.valid,
        I.maximum(maximum, rhs.maximum),
        maximum,
    )
    lhs_maximum = I.select(lhs.valid, lhs.maximum, maximum)
    rhs_maximum = I.select(rhs.valid, rhs.maximum, maximum)
    lhs_scale = I.select(
        lhs.valid,
        I.exp(lhs_maximum - maximum),
        0.0,
    )
    rhs_scale = I.select(
        rhs.valid,
        I.exp(rhs_maximum - maximum),
        0.0,
    )
    return I.record(
        valid=valid,
        maximum=maximum,
        denominator=(
            lhs_scale * lhs.denominator
            + rhs_scale * rhs.denominator
        ),
    )


@intent.fn
def online_softmax_summary(values):
    return I.reduce(
        I.record(
            valid=I.full(values.shape, fill=True, dtype=I.bool),
            maximum=values,
            denominator=I.full(values.shape, fill=1.0, dtype=I.f32),
        ),
        axis=0,
        identity=I.record(valid=False, maximum=0.0, denominator=0.0),
        combine=merge_softmax_summary,
    )


@intent.kernel
def streamed_online_softmax(
    x: I.In[I.f32, ("M", "N")],
    y: I.Out[I.f32, ("M", "N")],
):
    M, N = x.shape
    columns = I.domain(0, N)
    for row in I.parallel(I.domain(0, M)):
        values = x[row, columns]
        summary = online_softmax_summary(values)
        safe_denominator = I.select(summary.valid, summary.denominator, 1.0)
        inverse_denominator = 1.0 / safe_denominator
        y[row, columns] = I.select(
            summary.valid,
            I.exp(values - summary.maximum) * inverse_denominator,
            0.0,
        )


@intent.kernel
def streamed_online_softmax_f16(
    x: I.In[I.f16, ("M", "N")],
    y: I.Out[I.f16, ("M", "N")],
):
    M, N = x.shape
    columns = I.domain(0, N)
    for row in I.parallel(I.domain(0, M)):
        values = I.cast(x[row, columns], I.f32)
        summary = online_softmax_summary(values)
        safe_denominator = I.select(summary.valid, summary.denominator, 1.0)
        inverse_denominator = 1.0 / safe_denominator
        normalized = I.select(
            summary.valid,
            I.exp(values - summary.maximum) * inverse_denominator,
            0.0,
        )
        y[row, columns] = I.cast(normalized, I.f16)
