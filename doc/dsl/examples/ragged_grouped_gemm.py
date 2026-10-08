import intent
import intent.language as I


@intent.kernel
def ragged_grouped_gemm(
    x: I.In[I.f16, ("R", "K")],
    offsets: I.In[I.index, ("G_PLUS_1",)],
    weight: I.In[I.f16, ("G", "K", "N")],
    output: I.Out[I.f16, ("R", "N")],
):
    R, K = x.shape
    G, _, N = weight.shape
    reduction = I.domain(0, K)
    columns = I.domain(0, N)
    groups = I.ragged(
        outer=I.domain(0, G),
        members=I.domain(0, R),
        offsets=offsets,
    )

    for group in I.parallel(groups.outer):
        rows = I.members(groups[group])
        values = I.gather(x, index=(rows, reduction))
        result = I.matmul(
            values,
            weight[group, reduction, columns],
            acc_dtype=I.f32,
        )
        I.scatter_unique(
            output,
            index=(rows, columns),
            value=I.cast(result, I.f16),
        )
