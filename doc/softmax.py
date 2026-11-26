@I.kernel
def softmax(x: I.In[I.f16, ("M", "N")], out: I.Out[I.f16, ("M", "N")]):
    for row in I.parallel(x.dim(0)):
        v = I.cast(x[row, :], I.f32)
        peak = I.reduce(v, axis=0, combine=I.max_num, init=I.neg_inf(I.f32))
        w = I.exp(v - peak)
        out[row, :] = I.cast(w / I.sum(w, axis=0, acc=I.f32), I.f16)
