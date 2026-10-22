import intent
import intent.language as I


TOKENS = 32768
VOCABULARY = 8192
FEATURES = 1021


@intent.kernel
def embedding_forward_lookup(
    embedding_table: I.In[I.f32, ("V", "D")],
    indices: I.In[I.i32, ("M",)],
    output: I.Out[I.f32, ("M", "D")],
):
    M = indices.shape[0]
    D = embedding_table.shape[1]
    tokens = I.domain(0, M)
    features = I.domain(0, D)
    rows = indices[tokens]
    I.assume_in_bounds(rows, embedding_table, axis=0)
    output[tokens, features] = embedding_table[rows, features]


@intent.kernel
def embedding_forward_lookup_bf16(
    embedding_table: I.In[I.bf16, ("V", "D")],
    indices: I.In[I.i64, ("M",)],
    output: I.Out[I.bf16, ("M", "D")],
):
    M = indices.shape[0]
    D = embedding_table.shape[1]
    tokens = I.domain(0, M)
    features = I.domain(0, D)
    rows = indices[tokens]
    I.assume_in_bounds(rows, embedding_table, axis=0)
    output[tokens, features] = embedding_table[rows, features]


@intent.kernel
def embedding_backward_atomic(
    indices: I.In[I.i32, ("M",)],
    grad_output: I.In[I.f32, ("M", "D")],
    grad_weight: I.InOut[I.f32, ("V", "D")],
):
    M, D = grad_output.shape
    features = I.domain(0, D)
    for token in I.parallel(I.domain(0, M)):
        embedding = indices[token]
        I.assume_in_bounds(embedding, grad_weight, axis=0)
        I.atomic.add(
            grad_weight,
            index=(embedding, features),
            value=grad_output[token, features],
            order="relaxed",
        )
