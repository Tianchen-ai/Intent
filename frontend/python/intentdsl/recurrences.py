"""Algorithms with explicit order inside independent output owners."""

import intentdsl as I


@I.kernel
def gated_delta(
    q: I.In[I.bf16, ("B", "T", "H", "K")],
    k: I.In[I.bf16, ("B", "T", "H", "K")],
    v: I.In[I.bf16, ("B", "T", "HV", "V")],
    gate: I.In[I.bf16, ("B", "T", "HV")],
    beta: I.In[I.bf16, ("B", "T", "HV")],
    out: I.Out[I.bf16, ("B", "T", "HV", "V")],
    final: I.Out[I.f32, ("B", "HV", "K", "V")],
    scale: I.f32,
):
    for batch in I.parallel(v.dim(0)):
        for head in I.parallel(v.dim(2)):
            for feature in I.parallel(v.dim(3)):
                hidden = I.zeros(k.dim(3), I.f32)
                qhead = head // 2
                for time in I.ordered(v.dim(1)):
                    key = I.cast(k[batch, time, qhead, :], I.f32)
                    query = I.cast(q[batch, time, qhead, :], I.f32) * scale
                    decayed = hidden * I.exp(I.cast(gate[batch, time, head], I.f32))
                    residual = I.cast(v[batch, time, head, feature], I.f32) - I.sum(
                        key * decayed
                    )
                    hidden = decayed + key * (
                        residual * I.cast(beta[batch, time, head], I.f32)
                    )
                    out[batch, time, head, feature] = I.cast(
                        I.sum(hidden * query), I.bf16
                    )
                final[batch, head, :, feature] = hidden


@I.kernel
def causal_linear_attention(
    q: I.In[I.f32, ("B", "S", "K")],
    k: I.In[I.f32, ("B", "S", "K")],
    v: I.In[I.f32, ("B", "S", "V")],
    out: I.Out[I.f32, ("B", "S", "V")],
    final: I.Out[I.f32, ("B", "K", "V")],
):
    for batch in I.parallel(q.dim(0)):
        for feature in I.parallel(v.dim(2)):
            state = I.zeros(q.dim(2), I.f32)
            for time in I.ordered(q.dim(1)):
                state = state + k[batch, time, :] * v[batch, time, feature]
                out[batch, time, feature] = I.sum(q[batch, time, :] * state)
            final[batch, :, feature] = state


@I.kernel
def selective_state_scan(
    x: I.In[I.f32, ("B", "S")],
    decay: I.In[I.f32, ("B", "S")],
    drive: I.In[I.f32, ("B", "S")],
    y: I.Out[I.f32, ("B", "S")],
):
    for b in I.parallel(x.dim(0)):
        state = I.f32(0)
        for s in I.ordered(x.dim(1)):
            state = decay[b, s] * state + drive[b, s] * x[b, s]
            y[b, s] = state


@I.kernel
def cholesky(matrix: I.InOut[I.f32, ("B", "N", "N")]):
    for b in I.parallel(matrix.dim(0)):
        for c in I.ordered(matrix.dim(1)):
            square = I.f32(0)
            for k in I.ordered(c):
                element = matrix[b, c, k]
                square = square + element * element
            diagonal = I.sqrt(matrix[b, c, c] - square)
            matrix[b, c, c] = diagonal
            for r in I.ordered(c + 1, matrix.dim(1)):
                dot = I.f32(0)
                for k in I.ordered(c):
                    dot = dot + matrix[b, r, k] * matrix[b, c, k]
                matrix[b, r, c] = (matrix[b, r, c] - dot) / diagonal
                matrix[b, c, r] = I.f32(0)


@I.kernel
def causal_convolution(
    x: I.In[I.f16, ("B", "C", "S")],
    w: I.In[I.f16, ("C", "W")],
    bias: I.In[I.f16, ("C",)],
    y: I.Out[I.f16, ("B", "C", "S")],
):
    for b in I.parallel(x.dim(0)):
        for c in I.parallel(x.dim(1)):
            for s in I.parallel(x.dim(2)):
                total = I.f32(0)
                for tap in I.ordered(w.dim(1)):
                    pos = s + tap + 1 - w.dim(1)
                    sample = I.cast(x[b, c, pos], I.f32) if pos >= 0 else I.f32(0)
                    total = total + sample * I.cast(w[c, tap], I.f32)
                z = total + I.cast(bias[c], I.f32)
                y[b, c, s] = I.cast(z / (I.f32(1) + I.exp(I.f32(0) - z)), I.f16)
