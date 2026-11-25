"""Group 7 CPU-authored operators and alternate-expression LayerNorm variant."""

import intentdsl as I


@I.kernel
def embedding_bag_rows(
    table: I.In[I.f32, ("V", "D")],
    indices: I.In[I.i32, ("T",)],
    offsets: I.In[I.i32, ("O",)],
    out: I.Out[I.f32, ("B", "D")],
):
    for bag in I.parallel(out.dim(0)):
        lo_i32 = offsets[bag]
        hi_i32 = offsets[bag + 1]
        lo = I.cast(lo_i32, I.index)
        hi = I.cast(hi_i32, I.index)
        acc = I.zeros(out.dim(1), I.f32)
        for p in I.fold(lo, hi):
            idx = I.cast(indices[p], I.index)
            acc = acc + table[idx, :]
        count = I.cast(hi_i32 - lo_i32, I.f32)
        inv = I.f32(1) / count if hi > lo else I.f32(0)
        out[bag, :] = acc * inv


@I.program
def embedding_bag(table, indices, offsets):
    out = I.output(I.f32, (offsets.dim(0) - 1, table.dim(1)))
    embedding_bag_rows(table, indices, offsets, out)
    return out


@I.kernel
def index_add_copy(
    base: I.In[I.f32, ("M", "D")],
    out: I.Out[I.f32, ("M", "D")],
):
    for row in I.parallel(base.dim(0)):
        out[row, :] = base[row, :]


@I.kernel
def index_add_accumulate(
    indices: I.In[I.i32, ("N",)],
    src: I.In[I.f32, ("N", "D")],
    out: I.InOut[I.f32, ("M", "D")],
):
    for d in I.parallel(out.dim(1)):
        for n in I.fold(src.dim(0)):
            idx = I.cast(indices[n], I.index)
            out[idx, d] = out[idx, d] + src[n, d]


@I.program
def index_add(base, indices, src):
    out = I.output(I.f32, (base.dim(0), base.dim(1)))
    index_add_copy(base, out)
    index_add_accumulate(indices, src, out)
    return out


@I.kernel
def triangular_solve_copy(
    rhs: I.In[I.f32, ("B", "N", "K")],
    x: I.Out[I.f32, ("B", "N", "K")],
):
    for b, r in I.parallel((rhs.dim(0), rhs.dim(1))):
        k = I.domain(0, rhs.dim(2))
        x[b, r, k] = rhs[b, r, k]


@I.kernel
def triangular_solve_inplace(
    l: I.In[I.f32, ("B", "N", "N")],
    x: I.InOut[I.f32, ("B", "N", "K")],
):
    for b in I.parallel(x.dim(0)):
        for col in I.parallel(x.dim(2)):
            for r in I.ordered(x.dim(1)):
                dot = I.f32(0)
                for c in I.ordered(r):
                    dot = dot + l[b, r, c] * x[b, c, col]
                x[b, r, col] = (x[b, r, col] - dot) / l[b, r, r]


@I.program
def triangular_solve(l, rhs):
    x = I.output(I.f32, (rhs.dim(0), rhs.dim(1), rhs.dim(2)))
    triangular_solve_copy(rhs, x)
    triangular_solve_inplace(l, x)
    return x


@I.kernel
def log_softmax_kl(
    logits: I.In[I.f32, ("M", "V")],
    target: I.In[I.f32, ("M", "V")],
    log_probs: I.Out[I.f32, ("M", "V")],
    kl: I.Out[I.f32, ("M",)],
):
    for row in I.parallel(logits.dim(0)):
        peak = -I.inf
        for col in I.fold(logits.dim(1)):
            peak = I.maximum_num(peak, logits[row, col])
        denom = I.f32(0)
        for col in I.fold(logits.dim(1)):
            denom = denom + I.exp(logits[row, col] - peak)
        lse = peak + I.log(denom)
        log_probs[row, :] = logits[row, :] - lse
        total_kl = I.f32(0)
        for col in I.fold(logits.dim(1)):
            lp = logits[row, col] - lse
            p = target[row, col]
            term = p * (I.log(p) - lp) if p > I.f32(0) else I.f32(0)
            total_kl = total_kl + term
        kl[row] = total_kl


@I.kernel
def retention_recurrence(
    q: I.In[I.bf16, ("B", "T", "H", "K")],
    k: I.In[I.bf16, ("B", "T", "H", "K")],
    v: I.In[I.bf16, ("B", "T", "H", "V")],
    gamma: I.In[I.f32, ("H",)],
    out: I.Out[I.bf16, ("B", "T", "H", "V")],
    final: I.Out[I.f32, ("B", "H", "K", "V")],
    scale: I.f32,
):
    for batch in I.parallel(v.dim(0)):
        for head in I.parallel(v.dim(2)):
            for feature in I.parallel(v.dim(3)):
                decay = gamma[head]
                state = I.zeros(k.dim(3), I.f32)
                for time in I.ordered(v.dim(1)):
                    key = I.cast(k[batch, time, head, :], I.f32)
                    query = I.cast(q[batch, time, head, :], I.f32) * scale
                    val = I.cast(v[batch, time, head, feature], I.f32)
                    state = state * decay + key * val
                    out[batch, time, head, feature] = I.cast(
                        I.sum(query * state), I.bf16
                    )
                final[batch, head, :, feature] = state


@I.kernel
def radix_pass_rows(
    keys: I.In[I.i32, ("B", "N")],
    values: I.In[I.i32, ("B", "N")],
    hist: I.InOut[I.i32, ("B", 256)],
    cursors: I.InOut[I.i32, ("B", 256)],
    offsets: I.Out[I.i32, ("B", 257)],
    out_keys: I.Out[I.i32, ("B", "N")],
    out_values: I.Out[I.i32, ("B", "N")],
    divisor: I.i32,
):
    for row in I.parallel(keys.dim(0)):
        for b in I.ordered(hist.dim(1)):
            hist[row, b] = I.i32(0)
        for i in I.ordered(keys.dim(1)):
            digit = (keys[row, i] // divisor) % I.i32(256)
            b = I.cast(digit, I.index)
            hist[row, b] = hist[row, b] + I.i32(1)
        prefix = I.i32(0)
        for b in I.ordered(hist.dim(1)):
            cnt = hist[row, b]
            offsets[row, b] = prefix
            cursors[row, b] = prefix
            prefix = prefix + cnt
        offsets[row, 256] = prefix
        for i in I.ordered(keys.dim(1)):
            k = keys[row, i]
            v = values[row, i]
            digit = (k // divisor) % I.i32(256)
            b = I.cast(digit, I.index)
            dst = cursors[row, b]
            cursors[row, b] = dst + I.i32(1)
            pos = I.cast(dst, I.index)
            out_keys[row, pos] = k
            out_values[row, pos] = v


@I.program
def radix_pass(keys, values, divisor):
    hist = I.workspace(I.i32, (keys.dim(0), 256))
    cursors = I.workspace(I.i32, (keys.dim(0), 256))
    offsets = I.output(I.i32, (keys.dim(0), 257))
    out_keys = I.output(I.i32, (keys.dim(0), keys.dim(1)))
    out_values = I.output(I.i32, (keys.dim(0), keys.dim(1)))
    I.fill(hist, 0)
    I.fill(cursors, 0)
    radix_pass_rows(
        keys, values, hist, cursors, offsets, out_keys, out_values, divisor
    )
    return out_keys, out_values, offsets


@I.kernel
def layer_norm_welford(
    x: I.In[I.f32, ("R", "C")],
    gain: I.In[I.f32, ("C",)],
    offset: I.In[I.f32, ("C",)],
    out: I.Out[I.f32, ("R", "C")],
    epsilon: I.f32,
):
    for row in I.parallel(x.dim(0)):
        v = x[row, :]
        count, center, spread = I.welford(v)
        variance = spread / I.cast(count, I.f32)
        out[row, :] = (v - center) * I.rsqrt(variance + epsilon) * gain[:] + offset[:]
