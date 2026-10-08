import intent
import intent.language as I


BATCH = 32
M = 512
N = 512
K = 1024


@intent.kernel
def batched_gemm_nn(
    a: I.In[I.bf16, ("Q", "M", "K")],
    b: I.In[I.bf16, ("Q", "K", "N")],
    c: I.Out[I.bf16, ("Q", "M", "N")],
):
    Q, M, K = a.shape
    _, _, N = b.shape
    m_axis = I.domain(0, M)
    n_axis = I.domain(0, N)
    k_axis = I.domain(0, K)
    batch = I.domain(0, Q)
    accumulator = I.matmul(
        a[batch, m_axis, k_axis],
        b[batch, k_axis, n_axis],
        acc_dtype=I.f32,
    )
    c[batch, m_axis, n_axis] = I.cast(accumulator, I.bf16)


@intent.kernel
def batched_gemm_tn(
    a: I.In[I.bf16, ("Q", "K", "M")],
    b: I.In[I.bf16, ("Q", "K", "N")],
    c: I.Out[I.bf16, ("Q", "M", "N")],
):
    Q, K, M = a.shape
    _, _, N = b.shape
    m_axis = I.domain(0, M)
    n_axis = I.domain(0, N)
    k_axis = I.domain(0, K)
    for batch in I.parallel(I.domain(0, Q)):
        accumulator = I.matmul(
            a[batch, k_axis, m_axis],
            b[batch, k_axis, n_axis],
            transpose_lhs=True,
            acc_dtype=I.f32,
        )
        c[batch, m_axis, n_axis] = I.cast(accumulator, I.bf16)


@intent.kernel
def batched_gemm_nt(
    a: I.In[I.bf16, ("Q", "M", "K")],
    b: I.In[I.bf16, ("Q", "N", "K")],
    c: I.Out[I.bf16, ("Q", "M", "N")],
):
    Q, M, K = a.shape
    _, N, _ = b.shape
    m_axis = I.domain(0, M)
    n_axis = I.domain(0, N)
    k_axis = I.domain(0, K)
    for batch in I.parallel(I.domain(0, Q)):
        accumulator = I.matmul(
            a[batch, m_axis, k_axis],
            b[batch, n_axis, k_axis],
            transpose_rhs=True,
            acc_dtype=I.f32,
        )
        c[batch, m_axis, n_axis] = I.cast(accumulator, I.bf16)


@intent.kernel
def batched_gemm_tt(
    a: I.In[I.bf16, ("Q", "K", "M")],
    b: I.In[I.bf16, ("Q", "N", "K")],
    c: I.Out[I.bf16, ("Q", "M", "N")],
):
    Q, K, M = a.shape
    _, N, _ = b.shape
    m_axis = I.domain(0, M)
    n_axis = I.domain(0, N)
    k_axis = I.domain(0, K)
    for batch in I.parallel(I.domain(0, Q)):
        accumulator = I.matmul(
            a[batch, k_axis, m_axis],
            b[batch, n_axis, k_axis],
            transpose_lhs=True,
            transpose_rhs=True,
            acc_dtype=I.f32,
        )
        c[batch, m_axis, n_axis] = I.cast(accumulator, I.bf16)
