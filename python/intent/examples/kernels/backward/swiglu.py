import intent
import intent.language as I


TOKENS = 4096
FEATURES = 4096


@intent.kernel
def swiglu_backward(
    dc: I.In[I.bf16, ("M", "N")],
    a: I.In[I.bf16, ("M", "N")],
    b: I.In[I.bf16, ("M", "N")],
    da: I.Out[I.bf16, ("M", "N")],
    db: I.Out[I.bf16, ("M", "N")],
):
    M, N = dc.shape
    columns = I.domain(0, N)
    for row in I.parallel(I.domain(0, M)):
        dc_values = I.cast(dc[row, columns], I.f32)
        a_values = I.cast(a[row, columns], I.f32)
        b_values = I.cast(b[row, columns], I.f32)
        sigmoid = I.sigmoid(a_values)
        silu = a_values * sigmoid
        da[row, columns] = I.cast(
            dc_values * (silu * (1.0 - sigmoid) + sigmoid) * b_values,
            I.bf16,
        )
        db[row, columns] = I.cast(dc_values * silu, I.bf16)
