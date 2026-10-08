import intent
import intent.language as I


@intent.fn
def summarize_attention_chunk(
    key_chunk,
    value_chunk,
    key_coordinates,
    queries,
    query_coordinates,
    scale,
    causal,
):
    # key_chunk/value_chunk are slices chosen by region_fold. Their leading
    # extent is intentionally absent from the author program.
    scores = I.matmul(
        queries,
        key_chunk,
        transpose_rhs=True,
        acc_dtype=I.f32,
    )
    scores = scores * (scale * I.LOG2E)

    valid = I.full(scores.shape, fill=True, dtype=I.bool)
    if causal:
        valid = (
            query_coordinates[:, None]
            >= key_coordinates[None, :]
        )

    masked_scores = I.select(valid, scores, -I.inf)
    chunk_valid = I.reduce.any(
        valid,
        axis=1,
    )
    raw_maximum = I.reduce.max(
        masked_scores,
        axis=1,
    )

    # An all-masked row is the identity contribution. Use a finite temporary
    # maximum so that invalid arithmetic cannot create NaN before selection.
    maximum = I.select(chunk_valid, raw_maximum, 0.0)
    probabilities = I.select(
        valid,
        I.exp2(masked_scores - maximum[:, None]),
        0.0,
    )
    denominator = I.reduce.sum(
        probabilities,
        axis=1,
    )
    accumulator = I.matmul(
        probabilities,
        value_chunk,
        acc_dtype=I.f32,
    )

    return I.record(
        valid=chunk_valid,
        maximum=maximum,
        denominator=denominator,
        accumulator=accumulator,
    )


@intent.fn
def merge_attention_summaries(lhs, rhs):
    valid = lhs.valid | rhs.valid

    # Select a finite maximum when either side is the empty/invalid summary.
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
        I.exp2(lhs_maximum - maximum),
        0.0,
    )
    rhs_scale = I.select(
        rhs.valid,
        I.exp2(rhs_maximum - maximum),
        0.0,
    )

    return I.record(
        valid=valid,
        maximum=maximum,
        denominator=(
            lhs_scale * lhs.denominator
            + rhs_scale * rhs.denominator
        ),
        accumulator=(
            lhs_scale[:, None] * lhs.accumulator
            + rhs_scale[:, None] * rhs.accumulator
        ),
    )


@intent.kernel
def flash_attention_forward(
    q: I.In[I.f16, ("B", "H", "Q", "D")],
    k: I.In[I.f16, ("B", "H", "K", "D")],
    v: I.In[I.f16, ("B", "H", "K", "DV")],
    output: I.Out[I.f16, ("B", "H", "Q", "DV")],
    scale: I.f32,
    CAUSAL: I.Constexpr[bool],
):
    B, H, Q, D = q.shape
    _, _, K, _ = k.shape
    _, _, _, DV = v.shape

    query_axis = I.domain(0, Q)
    key_axis = I.domain(0, K)
    query_coordinates = I.indices(query_axis)
    key_coordinates = I.indices(key_axis)

    for batch in I.parallel(I.domain(0, B)):
        for head in I.parallel(I.domain(0, H)):
            queries = q[batch, head, query_axis, :]
            keys = k[batch, head, key_axis, :]
            values = v[batch, head, key_axis, :]

            empty_summary = I.record(
                valid=I.full((Q,), fill=False, dtype=I.bool),
                maximum=I.full((Q,), fill=0.0, dtype=I.f32),
                denominator=I.zeros((Q,), dtype=I.f32),
                accumulator=I.zeros((Q, DV), dtype=I.f32),
            )

            summary = I.region_fold(
                source=(keys, values, key_coordinates),
                axis=0,
                summarize=summarize_attention_chunk,
                combine=merge_attention_summaries,
                identity=empty_summary,
                operands=(
                    queries,
                    query_coordinates,
                    scale,
                    CAUSAL,
                ),
            )

            safe_denominator = I.select(
                summary.valid,
                summary.denominator,
                1.0,
            )
            normalized = I.select(
                summary.valid[:, None],
                summary.accumulator / safe_denominator[:, None],
                0.0,
            )
            output[batch, head, query_axis, :] = I.cast(
                normalized,
                I.f16,
            )


# The author specifies QK, local normalization, PV, and summary merge. No
# program id, Q/K block extent, pointer, mask tensor shape, storage, warp, or
# pipeline appears in the source. Physical lowering may choose both query
# fragments and key segments. The second contract in summarize_attention_chunk
# is the canonical source of the provider's dot(probabilities, values).
# Because key_coordinates retains its source-coordinate provenance through the
# region-fold helper, a physical predicate-range pass may later project
# query_coordinate >= key_coordinate onto the chosen query fragment: remove the
# all-invalid key suffix, drop the mask on the all-valid prefix, and preserve it
# only on the mixed diagonal range. This changes physical traversal, not the
# logical key source or causal predicate written above.
