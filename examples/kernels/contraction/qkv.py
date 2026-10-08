import intent
import intent.language as I


TOKENS = 4096
HIDDEN = 4096
PROJECTION = 4096


@intent.kernel
def fused_qkv_projection(
    x: I.In[I.f16, ("M", "K")],
    weights: I.In[I.f16, ("P", "K", "N")],
    output: I.Out[I.f16, ("P", "M", "N")],
):
    M, K = x.shape
    P, _, N = weights.shape
    projections = I.domain(0, P)
    rows = I.domain(0, M)
    reduction = I.domain(0, K)
    columns = I.domain(0, N)
    for projection in I.parallel(projections):
        output[projection, rows, columns] = I.cast(
            I.matmul(
                x[rows, reduction],
                weights[projection, reduction, columns],
                acc_dtype=I.f32,
            ),
            I.f16,
        )
