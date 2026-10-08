import intent
import intent.language as I


ROWS = 8192
K = 4096
N = 4096
GROUPS = 8


@intent.kernel
def ragged_grouped_gemm(
    x: I.In[I.f16, ("R", "K")],
    group_offsets: I.In[I.i32, ("G_PLUS_1",)],
    weight: I.In[I.f16, ("G", "K", "N")],
    y: I.Out[I.f16, ("R", "N")],
):
    R, K = x.shape
    G, _, N = weight.shape
    reduction = I.domain(0, K)
    columns = I.domain(0, N)
    groups = I.ragged(
        outer=I.domain(0, G),
        members=I.domain(0, R),
        offsets=group_offsets,
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
            y,
            index=(rows, columns),
            value=I.cast(result, I.f16),
        )


@intent.kernel
def ragged_grouped_gemm_bf16(
    x: I.In[I.bf16, ("R", "K")],
    group_offsets: I.In[I.i32, ("G_PLUS_1",)],
    weight: I.In[I.bf16, ("G", "K", "N")],
    y: I.Out[I.bf16, ("R", "N")],
):
    R, K = x.shape
    G, _, N = weight.shape
    reduction = I.domain(0, K)
    columns = I.domain(0, N)
    groups = I.ragged(
        outer=I.domain(0, G),
        members=I.domain(0, R),
        offsets=group_offsets,
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
            y,
            index=(rows, columns),
            value=I.cast(result, I.bf16),
        )


@intent.kernel
def ragged_grouped_gemm_backward_weight(
    left: I.In[I.f16, ("R", "K")],
    right: I.In[I.f16, ("R", "N")],
    group_offsets: I.In[I.i32, ("G_PLUS_1",)],
    grad_weight: I.Out[I.f16, ("G", "K", "N")],
):
    R, K = left.shape
    N = right.shape[1]
    G = grad_weight.shape[0]
    groups = I.ragged(
        outer=I.domain(0, G),
        members=I.domain(0, R),
        offsets=group_offsets,
    )
    k_axis = I.domain(0, K)
    n_axis = I.domain(0, N)
    for group in I.parallel(groups.outer):
        members = groups[group]
        result = I.matmul(
            left[members, k_axis],
            right[members, n_axis],
            transpose_lhs=True,
            acc_dtype=I.f32,
        )
        grad_weight[group, k_axis, n_axis] = I.cast(result, I.f16)


@intent.kernel
def routed_expert_projection_bf16(
    x: I.In[I.bf16, ("T", "K")],
    expert_offsets: I.In[I.i32, ("E_PLUS_1",)],
    member_routes: I.In[I.i32, ("R",)],
    weight: I.In[I.bf16, ("E", "N", "K")],
    output: I.Out[I.bf16, ("T", "TOP_K", "N")],
    TOP_K: I.Constexpr[int],
):
    E = weight.shape[0]
    N = weight.shape[1]
    K = weight.shape[2]
    R = member_routes.shape[0]
    reduction = I.domain(0, K)
    columns = I.domain(0, N)
    experts = I.ragged(
        outer=I.domain(0, E),
        members=I.domain(0, R),
        offsets=expert_offsets,
        indices=member_routes,
    )
    for expert in I.parallel(experts.outer):
        routes = I.members(experts[expert])
        token = routes // TOP_K
        slot = routes % TOP_K
        I.assume_in_bounds(token, x, axis=0)
        values = I.gather(x, index=(token, reduction))
        result = I.matmul(
            values,
            weight[expert, columns, reduction],
            transpose_rhs=True,
            acc_dtype=I.f32,
        )
        I.scatter_unique(
            output,
            index=(token, slot, columns),
            value=I.cast(result, I.bf16),
        )
