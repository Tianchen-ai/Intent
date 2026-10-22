import intent
import intent.language as I


@intent.kernel
def vector_dot(
    lhs: I.In[I.f32, ("K",)],
    rhs: I.In[I.f32, ("K",)],
    output: I.Out[I.f32, (1,)],
):
    k = I.domain(0, lhs.shape[0])
    output[0] = I.dot(lhs[k], rhs[k], acc_dtype=I.f32)


@intent.kernel
def matrix_vector(
    matrix: I.In[I.f32, ("M", "K")],
    vector: I.In[I.f32, ("K",)],
    output: I.Out[I.f32, ("M",)],
):
    m = I.domain(0, matrix.shape[0])
    k = I.domain(0, matrix.shape[1])
    output[m] = I.matvec(matrix[m, k], vector[k], acc_dtype=I.f32)


@intent.kernel
def vector_matrix(
    vector: I.In[I.f32, ("K",)],
    matrix: I.In[I.f32, ("K", "N")],
    output: I.Out[I.f32, ("N",)],
):
    k = I.domain(0, matrix.shape[0])
    n = I.domain(0, matrix.shape[1])
    output[n] = I.vecmat(vector[k], matrix[k, n], acc_dtype=I.f32)


@intent.kernel
def vector_outer(
    lhs: I.In[I.f32, ("M",)],
    rhs: I.In[I.f32, ("N",)],
    output: I.Out[I.f32, ("M", "N")],
):
    m = I.domain(0, lhs.shape[0])
    n = I.domain(0, rhs.shape[0])
    output[m, n] = I.outer(lhs[m], rhs[n])
