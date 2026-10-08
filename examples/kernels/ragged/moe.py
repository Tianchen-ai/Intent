import intent
import intent.language as I


TOKENS = 4096
HIDDEN = 4096
INTERMEDIATE = 14336
EXPERTS = 8
TOP_K = 2


@intent.kernel
def moe_expert_ffn(
    x: I.In[I.f16, ("T", "D")],
    route_offsets: I.In[I.i32, ("E_PLUS_1",)],
    member_routes: I.In[I.i32, ("R",)],
    route_token: I.In[I.i32, ("NR",)],
    route_weights: I.In[I.f32, ("NR",)],
    w1: I.In[I.f16, ("E", "D", "F")],
    w2: I.In[I.f16, ("E", "F", "D")],
    y: I.InOut[I.f32, ("T", "D")],
):
    T, D = x.shape
    E, _, F = w1.shape
    R = member_routes.shape[0]
    groups = I.ragged(
        outer=I.domain(0, E),
        members=I.domain(0, R),
        offsets=route_offsets,
        indices=member_routes,
    )
    for expert in I.parallel(groups.outer):
        routes = I.members(groups[expert])
        token = I.gather(route_token, index=routes)
        weight = I.gather(route_weights, index=routes)
        values = I.gather(x, index=(token, slice(None)))
        hidden = I.matmul(
            values,
            w1[expert, :, :],
            acc_dtype=I.f32,
        )
        hidden = I.cast(I.maximum(hidden, 0.0), I.f16)
        route_output = I.matmul(
            hidden,
            w2[expert, :, :],
            acc_dtype=I.f32,
        )
        I.scatter_reduce(
            y,
            index=(token, slice(None)),
            value=weight[:, None] * route_output,
            combine=I.add,
        )
