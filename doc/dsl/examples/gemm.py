import intent
import intent.language as I


class Activation(I.Enum):
    NONE = 0
    RELU = 1


@intent.kernel
def gemm(
    a: I.In[I.f16, ("M", "K")],
    b: I.In[I.f16, ("K", "N")],
    c: I.Out[I.f16, ("M", "N")],
    ACTIVATION: I.Constexpr[Activation] = Activation.NONE,
):
    M, K = a.shape
    _, N = b.shape
    m = I.domain(0, M)
    n = I.domain(0, N)
    k = I.domain(0, K)

    accumulator = I.matmul(
        a[m, k],
        b[k, n],
        acc_dtype=I.f32,
    )
    if ACTIVATION == Activation.RELU:
        accumulator = I.maximum(accumulator, 0.0)
    c[m, n] = I.cast(accumulator, I.f16)
