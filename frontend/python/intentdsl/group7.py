"""Group 7 GPU-authored operators and alternate-expression attention variant."""

import intentdsl as I


@I.kernel(numerics="relaxed_normalization")
def sliding_window_attention(
    q: I.In[I.f16, ("B", "HQ", "Q", "D")],
    k: I.In[I.f16, ("B", "HK", "K", "D")],
    v: I.In[I.f16, ("B", "HK", "K", "DV")],
    y: I.Out[I.f16, ("B", "HQ", "Q", "DV")],
    scale: I.f32,
    window: I.i32,
):
    for batch in I.parallel(q.dim(0)):
        for head in I.parallel(q.dim(1)):
            for query in I.parallel(q.dim(2)):
                kv = head // 4
                w = I.cast(window, I.index)
                lo = query + 1 - w if query + 1 >= w else 0
                hi = query + 1 if query + 1 < k.dim(2) else k.dim(2)
                query_vector = I.cast(q[batch, head, query, :], I.f32)
                log2_scale = scale * I.f32(1.4426950408889634)
                peak = -I.inf
                for token in I.fold(lo, hi):
                    key = I.cast(k[batch, kv, token, :], I.f32)
                    score = I.sum(query_vector * key) * log2_scale
                    peak = I.maximum_num(peak, score)
                mass = I.f32(0)
                weighted = I.zeros(v.dim(3), I.f32)
                for token in I.fold(lo, hi):
                    key = I.cast(k[batch, kv, token, :], I.f32)
                    score = I.sum(query_vector * key) * log2_scale
                    weight = I.exp2(
                        score - peak, approximate=True, flush_to_zero=True
                    )
                    mass = mass + weight
                    factor = I.cast(I.cast(weight, I.f16), I.f32)
                    weighted = weighted + I.cast(v[batch, kv, token, :], I.f32) * factor
                inverse = (
                    I.fdiv(I.f32(1), mass, approximate=True, flush_to_zero=True)
                    if lo < hi
                    else I.f32(0)
                )
                y[batch, head, query, :] = I.cast(weighted * inverse, I.f16)


@I.kernel(numerics="relaxed_normalization")
def alibi_attention(
    q: I.In[I.bf16, ("B", "HQ", "Q", "D")],
    k: I.In[I.bf16, ("B", "HK", "K", "D")],
    v: I.In[I.bf16, ("B", "HK", "K", "DV")],
    slopes: I.In[I.f32, ("HQ",)],
    y: I.Out[I.bf16, ("B", "HQ", "Q", "DV")],
    scale: I.f32,
):
    for batch in I.parallel(q.dim(0)):
        for head in I.parallel(q.dim(1)):
            for query in I.parallel(q.dim(2)):
                kv = head // 4
                slope = slopes[head]
                q_pos = I.cast(query, I.f32)
                query_vector = I.cast(q[batch, head, query, :], I.f32)
                log2_e = I.f32(1.4426950408889634)
                peak = -I.inf
                for token in I.fold(k.dim(2)):
                    key = I.cast(k[batch, kv, token, :], I.f32)
                    bias = slope * (I.cast(token, I.f32) - q_pos)
                    score = (I.sum(query_vector * key) * scale + bias) * log2_e
                    peak = I.maximum_num(peak, score if query >= token else -I.inf)
                mass = I.f32(0)
                weighted = I.zeros(v.dim(3), I.f32)
                for token in I.fold(k.dim(2)):
                    key = I.cast(k[batch, kv, token, :], I.f32)
                    bias = slope * (I.cast(token, I.f32) - q_pos)
                    score = (I.sum(query_vector * key) * scale + bias) * log2_e
                    weight = (
                        I.exp2(score - peak, approximate=True, flush_to_zero=True)
                        if query >= token
                        else I.f32(0)
                    )
                    mass = mass + weight
                    factor = I.cast(I.cast(weight, I.bf16), I.f32)
                    weighted = weighted + I.cast(v[batch, kv, token, :], I.f32) * factor
                inverse = (
                    I.f32(0)
                    if k.dim(2) == 0
                    else I.fdiv(I.f32(1), mass, approximate=True, flush_to_zero=True)
                )
                y[batch, head, query, :] = I.cast(weighted * inverse, I.bf16)


@I.kernel(numerics="relaxed_normalization")
def cross_entropy(
    logits: I.In[I.bf16, ("M", "V")],
    targets: I.In[I.i32, ("M",)],
    grad_loss: I.In[I.f32, ("M",)],
    loss: I.Out[I.f32, ("M",)],
    grad_logits: I.Out[I.bf16, ("M", "V")],
):
    for row in I.parallel(logits.dim(0)):
        target = I.cast(targets[row], I.index)
        target_logit = I.cast(logits[row, target], I.f32)
        peak = -I.inf
        for col in I.fold(logits.dim(1)):
            peak = I.maximum_num(peak, I.cast(logits[row, col], I.f32))
        denom = I.f32(0)
        for col in I.fold(logits.dim(1)):
            denom = denom + I.exp(I.cast(logits[row, col], I.f32) - peak)
        loss[row] = peak + I.log(denom) - target_logit
        upstream = grad_loss[row]
        inv_denom = I.f32(1) / denom
        for col in I.ordered(logits.dim(1)):
            prob = I.exp(I.cast(logits[row, col], I.f32) - peak) * inv_denom
            diff = prob - I.f32(1) if col == target else prob
            grad_logits[row, col] = I.cast(diff * upstream, I.bf16)


@I.kernel
def topk_rows(
    logits: I.In[I.f32, ("M", "V")],
    values: I.Out[I.f32, ("M", 8)],
    indices: I.Out[I.i32, ("M", 8)],
):
    for row in I.parallel(logits.dim(0)):
        v0 = -I.inf
        v1 = -I.inf
        v2 = -I.inf
        v3 = -I.inf
        v4 = -I.inf
        v5 = -I.inf
        v6 = -I.inf
        v7 = -I.inf
        i0 = I.i32(-1)
        i1 = I.i32(-1)
        i2 = I.i32(-1)
        i3 = I.i32(-1)
        i4 = I.i32(-1)
        i5 = I.i32(-1)
        i6 = I.i32(-1)
        i7 = I.i32(-1)
        for col in I.ordered(logits.dim(1)):
            x = logits[row, col]
            idx = I.cast(col, I.i32)
            c0 = x > v0
            nv0 = x if c0 else v0
            ni0 = idx if c0 else i0
            x = v0 if c0 else x
            idx = i0 if c0 else idx
            v0 = nv0
            i0 = ni0
            c1 = x > v1
            nv1 = x if c1 else v1
            ni1 = idx if c1 else i1
            x = v1 if c1 else x
            idx = i1 if c1 else idx
            v1 = nv1
            i1 = ni1
            c2 = x > v2
            nv2 = x if c2 else v2
            ni2 = idx if c2 else i2
            x = v2 if c2 else x
            idx = i2 if c2 else idx
            v2 = nv2
            i2 = ni2
            c3 = x > v3
            nv3 = x if c3 else v3
            ni3 = idx if c3 else i3
            x = v3 if c3 else x
            idx = i3 if c3 else idx
            v3 = nv3
            i3 = ni3
            c4 = x > v4
            nv4 = x if c4 else v4
            ni4 = idx if c4 else i4
            x = v4 if c4 else x
            idx = i4 if c4 else idx
            v4 = nv4
            i4 = ni4
            c5 = x > v5
            nv5 = x if c5 else v5
            ni5 = idx if c5 else i5
            x = v5 if c5 else x
            idx = i5 if c5 else idx
            v5 = nv5
            i5 = ni5
            c6 = x > v6
            nv6 = x if c6 else v6
            ni6 = idx if c6 else i6
            x = v6 if c6 else x
            idx = i6 if c6 else idx
            v6 = nv6
            i6 = ni6
            c7 = x > v7
            v7 = x if c7 else v7
            i7 = idx if c7 else i7
        values[row, 0] = v0
        values[row, 1] = v1
        values[row, 2] = v2
        values[row, 3] = v3
        values[row, 4] = v4
        values[row, 5] = v5
        values[row, 6] = v6
        values[row, 7] = v7
        indices[row, 0] = i0
        indices[row, 1] = i1
        indices[row, 2] = i2
        indices[row, 3] = i3
        indices[row, 4] = i4
        indices[row, 5] = i5
        indices[row, 6] = i6
        indices[row, 7] = i7


@I.kernel
def qk_norm_rope(
    q: I.In[I.bf16, ("B", 32, "S", 128)],
    k: I.In[I.bf16, ("B", 8, "S", 128)],
    q_weight: I.In[I.bf16, (128,)],
    k_weight: I.In[I.bf16, (128,)],
    cos: I.In[I.bf16, ("S", 64)],
    sin: I.In[I.bf16, ("S", 64)],
    q_out: I.Out[I.bf16, ("B", 32, "S", 128)],
    k_out: I.Out[I.bf16, ("B", 8, "S", 128)],
    inv_d: I.f32,
    eps: I.f32,
):
    for b, s, g in I.parallel((q.dim(0), q.dim(2), 8)):
        j = I.domain(0, 64)
        c = I.cast(cos[s, j], I.f32)
        sn = I.cast(sin[s, j], I.f32)
        qw0 = I.cast(q_weight[j], I.f32)
        qw1 = I.cast(q_weight[j + 64], I.f32)
        kw0 = I.cast(k_weight[j], I.f32)
        kw1 = I.cast(k_weight[j + 64], I.f32)
        for h in I.static_range(4):
            q0 = I.cast(q[b, g * 4 + h, s, j], I.f32)
            q1 = I.cast(q[b, g * 4 + h, s, j + 64], I.f32)
            q_sq = I.sum(q0 * q0, axis=0, acc=I.f32) + I.sum(
                q1 * q1, axis=0, acc=I.f32
            )
            q_inv = I.rsqrt(q_sq * inv_d + eps)
            qn0 = q0 * q_inv * qw0
            qn1 = q1 * q_inv * qw1
            q_out[b, g * 4 + h, s, j] = I.cast(qn0 * c - qn1 * sn, I.bf16)
            q_out[b, g * 4 + h, s, j + 64] = I.cast(qn1 * c + qn0 * sn, I.bf16)
        k0 = I.cast(k[b, g, s, j], I.f32)
        k1 = I.cast(k[b, g, s, j + 64], I.f32)
        k_sq = I.sum(k0 * k0, axis=0, acc=I.f32) + I.sum(
            k1 * k1, axis=0, acc=I.f32
        )
        k_inv = I.rsqrt(k_sq * inv_d + eps)
        kn0 = k0 * k_inv * kw0
        kn1 = k1 * k_inv * kw1
        k_out[b, g, s, j] = I.cast(kn0 * c - kn1 * sn, I.bf16)
        k_out[b, g, s, j + 64] = I.cast(kn1 * c + kn0 * sn, I.bf16)


@I.kernel(numerics="relaxed_normalization")
def attention_unscaled_max(
    q: I.In[I.bf16, ("B", "HQ", "Q", "D")],
    k: I.In[I.bf16, ("B", "HK", "K", "D")],
    v: I.In[I.bf16, ("B", "HK", "K", "DV")],
    y: I.Out[I.bf16, ("B", "HQ", "Q", "DV")],
    scale: I.f32,
):
    for batch in I.parallel(q.dim(0)):
        for head in I.parallel(q.dim(1)):
            for query in I.parallel(q.dim(2)):
                kv = head // 4
                query_vector = I.cast(q[batch, head, query, :], I.f32)
                peak = -I.inf
                for token in I.fold(k.dim(2)):
                    key = I.cast(k[batch, kv, token, :], I.f32)
                    raw = I.sum(query_vector * key)
                    peak = I.maximum_num(peak, raw if query >= token else -I.inf)
                mass = I.f32(0)
                weighted = I.zeros(v.dim(3), I.f32)
                for token in I.fold(k.dim(2)):
                    key = I.cast(k[batch, kv, token, :], I.f32)
                    raw = I.sum(query_vector * key)
                    weight = (
                        I.exp((raw - peak) * scale) if query >= token else I.f32(0)
                    )
                    mass = mass + weight
                    factor = I.cast(I.cast(weight, I.bf16), I.f32)
                    weighted = weighted + I.cast(v[batch, kv, token, :], I.f32) * factor
                inverse = I.f32(0) if k.dim(2) == 0 else I.f32(1) / mass
                y[batch, head, query, :] = I.cast(weighted * inverse, I.bf16)
