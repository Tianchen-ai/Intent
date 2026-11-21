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
