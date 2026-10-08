import intent
import intent.language as I


ROWS = 1024
VOCABULARY = 4093
TOP_K = 8


@intent.kernel
def insertion_top_k(
    logits: I.In[I.f32, ("M", "V")],
    top_values: I.Out[I.f32, ("M", TOP_K)],
    top_indices: I.Out[I.i32, ("M", TOP_K)],
):
    M, V = logits.shape
    for row in I.parallel(I.domain(0, M)):
        values = I.buffer((TOP_K,), I.f32, init=-I.inf)
        indices = I.buffer((TOP_K,), I.i32, init=-1)
        for candidate in range(V):
            carry_value = logits[row, candidate]
            carry_index = I.cast(candidate, I.i32)
            for slot in range(TOP_K):
                current_value = I.mutable_load(values, slot)
                current_index = I.mutable_load(indices, slot)
                if carry_value > current_value:
                    I.store(values, slot, carry_value)
                    I.store(indices, slot, carry_index)
                    carry_value = current_value
                    carry_index = current_index
        for slot in range(TOP_K):
            top_values[row, slot] = I.mutable_load(values, slot)
            top_indices[row, slot] = I.mutable_load(indices, slot)
