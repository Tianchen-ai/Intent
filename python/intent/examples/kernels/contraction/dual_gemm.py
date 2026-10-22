import intent
import intent.language as I


M = 2048
K = 4096
N = 4096


@intent.kernel
def gated_dual_gemm(
    x: I.In[I.f16, ("M", "K")],
    gate_weight: I.In[I.f16, ("K", "N")],
    value_weight: I.In[I.f16, ("K", "N")],
    y: I.Out[I.f16, ("M", "N")],
):
    M, K = x.shape
    _, N = gate_weight.shape
    m_axis = I.domain(0, M)
    n_axis = I.domain(0, N)
    k_axis = I.domain(0, K)
    gate = I.matmul(
        x[m_axis, k_axis],
        gate_weight[k_axis, n_axis],
        acc_dtype=I.f32,
    )
    value = I.matmul(
        x[m_axis, k_axis],
        value_weight[k_axis, n_axis],
        acc_dtype=I.f32,
    )
    y[m_axis, n_axis] = I.cast(I.maximum(gate, 0.0) * value, I.f16)
