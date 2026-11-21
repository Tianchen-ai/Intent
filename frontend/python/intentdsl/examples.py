import intentdsl as I


@I.kernel
def softmax(x: I.In[I.f16, ("M", "N")], out: I.Out[I.f16, ("M", "N")]):
    for r in I.parallel(x.dim(0)):
        v = I.cast(x[r, :], I.f32)
        m = I.reduce(v, axis=0, combine=I.max_num, init=I.neg_inf(I.f32))
        w = I.exp(v - m)
        out[r, :] = I.cast(w / I.sum(w, axis=0, acc=I.f32), I.f16)


@I.kernel
def softmax_f32(x: I.In[I.f32, ("M", "N")], out: I.Out[I.f32, ("M", "N")]):
    for r in I.parallel(x.dim(0)):
        v = x[r, :]
        m = I.reduce(v, axis=0, combine=I.max_num, init=I.neg_inf(I.f32))
        w = I.exp(v - m)
        out[r, :] = w / I.sum(w, axis=0, acc=I.f32)


@I.kernel
def rms_norm(
    x: I.In[I.f32, ("M", "N")],
    gain: I.In[I.f32, ("N",)],
    out: I.Out[I.f32, ("M", "N")],
    inv_c: I.f32,
    eps: I.f32,
):
    for r in I.parallel(x.dim(0)):
        u = x[r, :]
        s = I.sum(u * u, axis=0, acc=I.f32)
        out[r, :] = (u * I.rsqrt(s * inv_c + eps)) * gain[:]


@I.kernel
def relu(x: I.In[I.f16, ("M", "N")], out: I.Out[I.f16, ("M", "N")]):
    for row in I.parallel(x.dim(0)):
        out[row, :] = I.maximum(x[row, :], I.cast(0.0, I.f16))


@I.kernel
def transpose(x: I.In[I.f16, ("M", "N")], out: I.Out[I.f16, ("N", "M")]):
    for row in I.parallel(x.dim(0)):
        out[:, row] = x[row, :]


@I.kernel
def swiglu(
    gate: I.In[I.bf16, ("M", "N")],
    up: I.In[I.bf16, ("M", "N")],
    out: I.Out[I.bf16, ("M", "N")],
):
    for row in I.parallel(gate.dim(0)):
        x = I.cast(gate[row, :], I.f32)
        one = I.cast(1.0, I.f32)
        sigmoid = one / (one + I.exp(I.cast(0.0, I.f32) - x))
        rounded = I.cast(x * sigmoid, I.bf16)
        out[row, :] = rounded * up[row, :]


@I.kernel
def dropout(
    x: I.In[I.f16, ("M", "N")],
    out: I.Out[I.f16, ("M", "N")],
    seed: I.i32,
    probability: I.f32,
    scale: I.f32,
):
    for row in I.parallel(x.dim(0)):
        h = I.linear_index(x, row) * I.cast(1103515245, I.i32) + seed
        a = h ^ (h >> I.cast(16, I.i32))
        b = a ^ (a << I.cast(8, I.i32))
        c = b ^ (b >> I.cast(4, I.i32))
        u = I.cast(c & I.cast(2147483647, I.i32), I.f32) / I.cast(2147483647.0, I.f32)
        out[row, :] = I.where(
            u > probability, x[row, :] * I.cast(scale, I.f16), I.cast(0.0, I.f16)
        )


@I.kernel
def adamw(
    g: I.In[I.f32, ("N",)],
    p: I.InOut[I.f32, ("N",)],
    m: I.InOut[I.f32, ("N",)],
    v: I.InOut[I.f32, ("N",)],
    lr: I.f32,
    beta1: I.f32,
    beta2: I.f32,
    correction1: I.f32,
    correction2: I.f32,
    eps: I.f32,
    decay: I.f32,
):
    for i in I.parallel(g.dim(0)):
        one = I.cast(1.0, I.f32)
        grad = g[i]
        mean = beta1 * m[i] + (one - beta1) * grad
        square = beta2 * v[i] + ((one - beta2) * grad) * grad
        step = (mean / correction1) / (I.sqrt(square / correction2) + eps) + decay * p[
            i
        ]
        updated = p[i] - lr * step
        m[i] = mean
        v[i] = square
        p[i] = updated


@I.kernel
def dropout_norm_forward(
    x: I.In[I.bf16, ("M", "N")],
    residual: I.In[I.bf16, ("M", "N")],
    gain: I.In[I.bf16, ("N",)],
    normalized: I.Out[I.bf16, ("M", "N")],
    summed: I.Out[I.bf16, ("M", "N")],
    seed: I.u64,
    keep: I.f32,
    scale: I.f32,
    inv_n: I.f32,
    eps: I.f32,
    bias: I.f32,
):
    for row in I.parallel(x.dim(0)):
        uniform = I.uniform(seed, I.member_counter(x, row))
        active = I.cast(keep > uniform, I.f32)
        value = I.cast(
            I.cast(x[row, :], I.f32) * active * scale + I.cast(residual[row, :], I.f32),
            I.bf16,
        )
        z = I.cast(value, I.f32)
        inv = I.rsqrt(I.sum(z * z, axis=0, acc=I.f32) * inv_n + eps)
        y = I.cast(z * inv * (I.cast(gain[:], I.f32) + bias), I.bf16)
        normalized[row, :] = y
        summed[row, :] = value


@I.kernel
def dropout_norm_backward(
    x: I.In[I.bf16, ("M", "N")],
    residual: I.In[I.bf16, ("M", "N")],
    gain: I.In[I.bf16, ("N",)],
    dy: I.In[I.bf16, ("M", "N")],
    carry: I.In[I.bf16, ("M", "N")],
    dx: I.Out[I.bf16, ("M", "N")],
    dr: I.Out[I.bf16, ("M", "N")],
    seed: I.u64,
    keep: I.f32,
    scale: I.f32,
    inv_n: I.f32,
    eps: I.f32,
    bias: I.f32,
):
    for row in I.parallel(x.dim(0)):
        uniform = I.uniform(seed, I.member_counter(x, row))
        active = I.cast(keep > uniform, I.f32)
        rounded = I.cast(
            I.cast(x[row, :], I.f32) * active * scale + I.cast(residual[row, :], I.f32),
            I.bf16,
        )
        z = I.cast(rounded, I.f32)
        wdy = I.cast(dy[row, :], I.f32) * (I.cast(gain[:], I.f32) + bias)
        inv = I.rsqrt(I.sum(z * z, axis=0, acc=I.f32) * inv_n + eps)
        dot = I.sum(wdy * z, axis=0, acc=I.f32) * inv_n
        dz = wdy * inv - z * (inv * inv * inv) * dot + I.cast(carry[row, :], I.f32)
        dx[row, :] = I.cast(dz * active * scale, I.bf16)
        dr[row, :] = I.cast(dz, I.bf16)
