import intent
import intent.language as I


MQA_QUERIES = 256
MQA_KEYS = 4096
MQA_HEADS = 32
MQA_HEAD_DIMENSION = 128


@intent.kernel
def fp8_mqa_logits(
    q: I.In[I.f8e4m3fn, ("Q", "H", "D")],
    kv: I.In[I.f8e4m3fn, ("K", "D")],
    kv_scale: I.In[I.f32, ("K",)],
    head_weight: I.In[I.f32, ("Q", "H")],
    key_start: I.In[I.i32, ("Q",)],
    key_end: I.In[I.i32, ("Q",)],
    output: I.Out[I.f32, ("Q", "K")],
):
    Q, H, D = q.shape
    K = kv.shape[0]
    head_axis = I.domain(0, H)
    reduction_axis = I.domain(0, D)
    key_axis = I.domain(0, K)
    for query in I.parallel(I.domain(0, Q)):
        head_logits = I.matmul(
            q[query, head_axis, reduction_axis],
            kv[key_axis, reduction_axis],
            transpose_rhs=True,
            acc_dtype=I.f32,
        )
        weighted = I.maximum(head_logits, 0.0) * head_weight[
            query, head_axis, None
        ]
        logits = I.reduce.sum(
            weighted,
            axis=0,
        ) * kv_scale[key_axis]
        key_index = I.indices(key_axis)
        begin = I.cast(key_start[query], I.index)
        end = I.cast(key_end[query], I.index)
        valid = (key_index >= begin) & (key_index < end)
        output[query, key_axis] = I.mask(
            logits,
            valid=valid,
            fill=-I.inf,
        )
