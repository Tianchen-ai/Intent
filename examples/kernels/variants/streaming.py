import intent
import intent.language as I

from kernels.streaming.attention import empty_attention_summary
from kernels.streaming.attention import merge_attention_summaries
from kernels.streaming.attention import normalize_attention_summary
from kernels.streaming.attention import summarize_attention_chunk_f16
from kernels.streaming.online_softmax import online_softmax_summary


@intent.kernel
def streamed_online_softmax_inline(
    x: I.In[I.f32, ("M", "N")],
    y: I.Out[I.f32, ("M", "N")],
):
    M, N = x.shape
    columns = I.domain(0, N)
    for row in I.parallel(I.domain(0, M)):
        values = x[row, columns]
        summary = online_softmax_summary(values)
        safe_denominator = I.select(summary.valid, summary.denominator, 1.0)
        y[row, columns] = I.select(
            summary.valid,
            I.exp(values - summary.maximum) / safe_denominator,
            0.0,
        )


@intent.fn
def run_attention_variant(q, k, v, scale, causal):
    query_count = q.shape[0]
    key_count = k.shape[0]
    value_width = v.shape[1]
    query_axis = I.domain(0, query_count)
    key_axis = I.domain(0, key_count)
    return I.region_fold(
        source=(k, v, I.indices(key_axis)),
        axis=0,
        summarize=summarize_attention_chunk_f16,
        combine=merge_attention_summaries,
        identity=empty_attention_summary(query_count, value_width),
        operands=(q, I.indices(query_axis), scale, causal),
    )


@intent.kernel
def flash_attention_inline_fwd(
    q: I.In[I.f16, ("B", "H", "Q", "D")],
    k: I.In[I.f16, ("B", "H", "K", "D")],
    v: I.In[I.f16, ("B", "H", "K", "DV")],
    output: I.Out[I.f16, ("B", "H", "Q", "DV")],
    scale: I.f32,
    CAUSAL: I.Constexpr[bool],
):
    B, H, Q, _ = q.shape
    K = k.shape[2]
    query_axis = I.domain(0, Q)
    key_axis = I.domain(0, K)
    for batch in I.parallel(I.domain(0, B)):
        for head in I.parallel(I.domain(0, H)):
            summary = run_attention_variant(
                q[batch, head, query_axis, :],
                k[batch, head, key_axis, :],
                v[batch, head, key_axis, :],
                scale,
                CAUSAL,
            )
            output[batch, head, query_axis, :] = I.cast(
                normalize_attention_summary(summary),
                I.f16,
            )


@intent.kernel
def flash_attention_select_fwd(
    q: I.In[I.f16, ("B", "H", "Q", "D")],
    k: I.In[I.f16, ("B", "H", "K", "D")],
    v: I.In[I.f16, ("B", "H", "K", "DV")],
    output: I.Out[I.f16, ("B", "H", "Q", "DV")],
    scale: I.f32,
    CAUSAL: I.Constexpr[bool],
):
    B, H, Q, _ = q.shape
    K = k.shape[2]
    query_axis = I.domain(0, Q)
    key_axis = I.domain(0, K)
    for batch in I.parallel(I.domain(0, B)):
        for head in I.parallel(I.domain(0, H)):
            summary = run_attention_variant(
                q[batch, head, query_axis, :],
                k[batch, head, key_axis, :],
                v[batch, head, key_axis, :],
                scale,
                CAUSAL,
            )
            output[batch, head, query_axis, :] = I.cast(
                normalize_attention_summary(summary),
                I.f16,
            )


@intent.kernel
def flash_attention_full_causal_stream_fwd(
    q: I.In[I.f16, ("B", "H", "Q", "D")],
    k: I.In[I.f16, ("B", "H", "K", "D")],
    v: I.In[I.f16, ("B", "H", "K", "DV")],
    output: I.Out[I.f16, ("B", "H", "Q", "DV")],
    scale: I.f32,
    CAUSAL: I.Constexpr[bool],
):
    B, H, Q, _ = q.shape
    K = k.shape[2]
    query_axis = I.domain(0, Q)
    key_axis = I.domain(0, K)
    for batch in I.parallel(I.domain(0, B)):
        for head in I.parallel(I.domain(0, H)):
            summary = run_attention_variant(
                q[batch, head, query_axis, :],
                k[batch, head, key_axis, :],
                v[batch, head, key_axis, :],
                scale,
                CAUSAL,
            )
            output[batch, head, query_axis, :] = I.cast(
                normalize_attention_summary(summary),
                I.f16,
            )
