import intent
import intent.language as I


@intent.kernel
def mla_head_projection(
    source: I.In[I.f16, ("B", "Q", "H", "I")],
    weight: I.In[I.f16, ("H", "O", "I")],
    output: I.Out[I.f16, ("B", "Q", "H", "O")],
):
    B, Q, H, _ = source.shape
    I_DIMENSION = source.shape[3]
    O = weight.shape[1]
    query_axis = I.domain(0, Q)
    output_axis = I.domain(0, O)
    reduction_axis = I.domain(0, I_DIMENSION)
    for batch in I.parallel(I.domain(0, B)):
        for head in I.parallel(I.domain(0, H)):
            projected = I.matmul(
                source[batch, query_axis, head, reduction_axis],
                weight[head, output_axis, reduction_axis],
                transpose_rhs=True,
                acc_dtype=I.f32,
            )
            output[batch, query_axis, head, output_axis] = I.cast(
                projected,
                I.f16,
            )
