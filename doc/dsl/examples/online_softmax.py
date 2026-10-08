import intent
import intent.language as I


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
    denominator = (
        lhs_scale * lhs.denominator
        + rhs_scale * rhs.denominator
    )
    return I.record(
        valid=valid,
        maximum=maximum,
        denominator=denominator,
    )


@intent.kernel
def online_softmax(
    x: I.In[I.f32, ("M", "N")],
    output: I.Out[I.f32, ("M", "N")],
):
    M, N = x.shape
    columns = I.domain(0, N)

    for row in I.parallel(I.domain(0, M)):
        values = x[row, columns]
        element_summaries = I.record(
            valid=I.full((N,), fill=True, dtype=I.bool),
            maximum=values,
            denominator=I.full((N,), fill=1.0, dtype=I.f32),
        )
        summary = I.reduce(
            element_summaries,
            axis=0,
            identity=I.record(
                valid=False,
                maximum=0.0,
                denominator=0.0,
            ),
            combine=merge_softmax_summary,
        )
        safe_denominator = I.select(
            summary.valid,
            summary.denominator,
            1.0,
        )
        output[row, columns] = (
            I.exp(values - summary.maximum) / safe_denominator
        )
