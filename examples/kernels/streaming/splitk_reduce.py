import intent
import intent.language as I

from kernels.streaming.attention import reduce_score_maximum


BATCH = 8
HEADS = 32
SPLITS = 16
HEAD_DIMENSION = 128


@intent.kernel
def splitk_attention_reduce(
    partial: I.In[I.bf16, ("B", "H", "S", "D")],
    partial_lse: I.In[I.f32, ("B", "H", "S")],
    output: I.Out[I.bf16, ("B", "H", "D")],
):
    B, H, S, D = partial.shape
    splits = I.domain(0, S)
    dimensions = I.domain(0, D)
    for batch in I.parallel(I.domain(0, B)):
        for head in I.parallel(I.domain(0, H)):
            lse = partial_lse[batch, head, splits]
            maximum = reduce_score_maximum(lse, axis=0)
            weights = I.exp2(lse - maximum, approximate=True, flush_to_zero=True)
            denominator = I.reduce.sum(
                weights,
                axis=0,
            )
            numerator = I.vecmat(
                weights,
                I.cast(partial[batch, head, splits, dimensions], I.f32),
                acc_dtype=I.f32,
            )
            output[batch, head, dimensions] = I.cast(
                numerator * I.fdiv(1.0, denominator, approximate=True, flush_to_zero=True),
                I.bf16,
            )


@intent.kernel
def splitk_attention_reduce_f16(
    partial: I.In[I.f16, ("B", "H", "S", "D")],
    partial_lse: I.In[I.f32, ("B", "H", "S")],
    output: I.Out[I.f16, ("B", "H", "D")],
):
    B, H, S, D = partial.shape
    splits = I.domain(0, S)
    dimensions = I.domain(0, D)
    for batch in I.parallel(I.domain(0, B)):
        for head in I.parallel(I.domain(0, H)):
            lse = partial_lse[batch, head, splits]
            maximum = reduce_score_maximum(lse, axis=0)
            weights = I.exp2(lse - maximum, approximate=True, flush_to_zero=True)
            denominator = I.reduce.sum(
                weights,
                axis=0,
            )
            numerator = I.vecmat(
                weights,
                I.cast(partial[batch, head, splits, dimensions], I.f32),
                acc_dtype=I.f32,
            )
            output[batch, head, dimensions] = I.cast(
                numerator * I.fdiv(1.0, denominator, approximate=True, flush_to_zero=True),
                I.f16,
            )


@intent.kernel
def splitk_attention_weighted_sum_reduce(
    partial: I.In[I.bf16, ("B", "H", "S", "D")],
    partial_lse: I.In[I.f32, ("B", "H", "S")],
    output: I.Out[I.bf16, ("B", "H", "D")],
):
    B, H, S, D = partial.shape
    splits = I.domain(0, S)
    dimensions = I.domain(0, D)
    for batch in I.parallel(I.domain(0, B)):
        for head in I.parallel(I.domain(0, H)):
            lse = partial_lse[batch, head, splits]
            maximum = reduce_score_maximum(lse, axis=0)
            weights = I.exp2(lse - maximum, approximate=True, flush_to_zero=True)
            denominator = I.reduce.sum(
                weights,
                axis=0,
            )
            numerator = I.vecmat(
                weights,
                I.cast(partial[batch, head, splits, dimensions], I.f32),
                acc_dtype=I.f32,
            )
            output[batch, head, dimensions] = I.cast(
                numerator * I.fdiv(1.0, denominator, approximate=True, flush_to_zero=True),
                I.bf16,
            )


@intent.kernel
def splitk_attention_bf16_to_f16_reduce(
    partial: I.In[I.bf16, ("B", "H", "S", "D")],
    partial_lse: I.In[I.f32, ("B", "H", "S")],
    output: I.Out[I.f16, ("B", "H", "D")],
):
    B, H, S, D = partial.shape
    splits = I.domain(0, S)
    dimensions = I.domain(0, D)
    for batch in I.parallel(I.domain(0, B)):
        for head in I.parallel(I.domain(0, H)):
            lse = partial_lse[batch, head, splits]
            maximum = reduce_score_maximum(lse, axis=0)
            weights = I.exp2(lse - maximum, approximate=True, flush_to_zero=True)
            denominator = I.reduce.sum(
                weights,
                axis=0,
            )
            numerator = I.vecmat(
                weights,
                I.cast(partial[batch, head, splits, dimensions], I.f32),
                acc_dtype=I.f32,
            )
            output[batch, head, dimensions] = I.cast(
                numerator * I.fdiv(1.0, denominator, approximate=True, flush_to_zero=True),
                I.f16,
            )


@intent.kernel
def splitk_attention_f32_to_f16_reduce(
    partial: I.In[I.f32, ("B", "H", "S", "D")],
    partial_lse: I.In[I.f32, ("B", "H", "S")],
    output: I.Out[I.f16, ("B", "H", "D")],
):
    B, H, S, D = partial.shape
    splits = I.domain(0, S)
    dimensions = I.domain(0, D)
    for batch in I.parallel(I.domain(0, B)):
        for head in I.parallel(I.domain(0, H)):
            lse = partial_lse[batch, head, splits]
            maximum = reduce_score_maximum(lse, axis=0)
            weights = I.exp(lse - maximum)
            denominator = I.reduce.sum(
                weights,
                axis=0,
            )
            numerator = I.vecmat(
                weights,
                partial[batch, head, splits, dimensions],
                acc_dtype=I.f32,
            )
            output[batch, head, dimensions] = I.cast(
                numerator * I.fdiv(1.0, denominator, approximate=True, flush_to_zero=True),
                I.f16,
            )
