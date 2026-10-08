import intent
import intent.language as I


TOKENS = 4096
VOCABULARY = 16384
IGNORE_INDEX = -100


@intent.fn
def merge_cross_entropy_summary(lhs, rhs):
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
    lhs_wins = (lhs.maximum > rhs.maximum) | (
        (lhs.maximum == rhs.maximum) & (lhs.predicted <= rhs.predicted)
    )
    predicted = I.select(lhs.valid, lhs.predicted, rhs.predicted)
    predicted = I.select(
        lhs.valid & rhs.valid,
        I.select(lhs_wins, lhs.predicted, rhs.predicted),
        predicted,
    )
    return I.record(
        valid=valid,
        maximum=maximum,
        denominator=(
            lhs_scale * lhs.denominator
            + rhs_scale * rhs.denominator
        ),
        predicted=predicted,
    )


@intent.fn
def minimum_index(lhs, rhs):
    return I.minimum(lhs, rhs)


@intent.fn
def summarize_cross_entropy_chunk(values, coordinates, empty_prediction):
    members = coordinates < empty_prediction
    raw_maximum = I.reduce.max(
        I.select(members, values, -I.inf),
        axis=0,
    )
    denominator = I.reduce.sum(
        I.select(
            members,
            I.exp(values - raw_maximum),
            0.0,
        ),
        axis=0,
    )
    chunk_valid = denominator > 0.0
    chunk_maximum = I.select(chunk_valid, raw_maximum, 0.0)
    predicted = I.reduce(
        I.select(
            members & (values == raw_maximum),
            coordinates,
            empty_prediction,
        ),
        axis=0,
        identity=empty_prediction,
        combine=minimum_index,
    )
    return I.record(
        valid=chunk_valid,
        maximum=chunk_maximum,
        denominator=denominator,
        predicted=I.select(
            chunk_valid,
            predicted,
            empty_prediction * 0,
        ),
    )


@intent.fn
def cross_entropy_summary(values, coordinates, empty_prediction):
    return I.region_fold(
        source=(values, coordinates),
        axis=0,
        summarize=summarize_cross_entropy_chunk,
        combine=merge_cross_entropy_summary,
        identity=I.record(
            valid=False,
            maximum=I.cast(0.0, I.f32),
            denominator=I.cast(0.0, I.f32),
            predicted=empty_prediction * 0,
        ),
        operands=(empty_prediction,),
    )


@intent.kernel
def fused_cross_entropy(
    logits: I.InOut[I.f32, ("M", "V")],
    labels: I.In[I.i32, ("M",)],
    loss: I.Out[I.f32, ("M",)],
    prediction: I.Out[I.i32, ("M",)],
    IGNORE_INDEX: I.Constexpr[int],
):
    M, V = logits.shape
    classes = I.domain(0, V)
    for row in I.parallel(I.domain(0, M)):
        label = labels[row]
        if label == IGNORE_INDEX:
            loss[row] = 0.0
            prediction[row] = -1
            logits[row, classes] = 0.0
        else:
            I.assume_in_bounds(label, logits, axis=1)
            target = logits[row, label]
            values = logits[row, classes]
            summary = cross_entropy_summary(
                values,
                I.cast(I.indices(classes), I.i32),
                I.cast(V, I.i32),
            )
            safe_denominator = I.select(summary.valid, summary.denominator, 1.0)
            loss[row] = summary.maximum + I.log(safe_denominator) - target
            prediction[row] = summary.predicted
            probability = I.exp(values - summary.maximum) / safe_denominator
            gradient = probability - I.cast(
                I.cast(I.indices(classes), I.i32) == label,
                I.f32,
            )
            logits[row, classes] = gradient


@intent.kernel
def fused_cross_entropy_bf16(
    logits: I.InOut[I.bf16, ("M", "V")],
    labels: I.In[I.i64, ("M",)],
    loss: I.Out[I.f32, ("M",)],
    prediction: I.Out[I.i64, ("M",)],
    IGNORE_INDEX: I.Constexpr[int],
):
    M, V = logits.shape
    classes = I.domain(0, V)
    for row in I.parallel(I.domain(0, M)):
        label = labels[row]
        if label == IGNORE_INDEX:
            loss[row] = 0.0
            prediction[row] = -1
            logits[row, classes] = I.cast(0.0, I.bf16)
        else:
            I.assume_in_bounds(label, logits, axis=1)
            target = I.cast(I.gather(logits, index=(row, label)), I.f32)
            values = I.cast(logits[row, classes], I.f32)
            summary = cross_entropy_summary(
                values,
                I.cast(I.indices(classes), I.i64),
                I.cast(V, I.i64),
            )
            safe_denominator = I.select(summary.valid, summary.denominator, 1.0)
            loss[row] = summary.maximum + I.log(safe_denominator) - target
            prediction[row] = summary.predicted
            class_index = I.cast(I.indices(classes), I.i64)
            probability = I.exp(values - summary.maximum) / safe_denominator
            logits[row, classes] = I.cast(
                probability - I.cast(class_index == label, I.f32),
                I.bf16,
            )


@intent.kernel
def flash_cross_entropy_bf16(
    logits: I.In[I.bf16, ("M", "V")],
    labels: I.In[I.i64, ("M",)],
    loss: I.Out[I.f32, ("M",)],
    z_loss: I.Out[I.f32, ("M",)],
    IGNORE_INDEX: I.Constexpr[int],
    Z_LOSS_SCALE: I.Constexpr[float],
):
    M, V = logits.shape
    classes = I.domain(0, V)
    for row in I.parallel(I.domain(0, M)):
        label = labels[row]
        if label == IGNORE_INDEX:
            loss[row] = 0.0
            z_loss[row] = 0.0
        else:
            I.assume_in_bounds(label, logits, axis=1)
            target = I.cast(I.gather(logits, index=(row, label)), I.f32)
            values = I.cast(logits[row, classes], I.f32)
            summary = cross_entropy_summary(
                values,
                I.cast(I.indices(classes), I.i64),
                I.cast(V, I.i64),
            )
            safe_denominator = I.select(summary.valid, summary.denominator, 1.0)
            lse = summary.maximum + I.log(safe_denominator)
            regularizer = Z_LOSS_SCALE * lse * lse
            z_loss[row] = regularizer
            loss[row] = lse - target + regularizer
