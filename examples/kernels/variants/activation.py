import intent
import intent.language as I


@intent.fn
def sigmoid_helper(values):
    return 1.0 / (1.0 + I.exp(-values))


@intent.kernel
def swiglu_forward_helper(
    gate: I.In[I.bf16, ("M", "N")],
    up: I.In[I.bf16, ("M", "N")],
    output: I.Out[I.bf16, ("M", "N")],
):
    M, N = gate.shape
    columns = I.domain(0, N)
    for row in I.parallel(I.domain(0, M)):
        gate_values = I.cast(gate[row, columns], I.f32)
        up_values = I.cast(up[row, columns], I.f32)
        output[row, columns] = I.cast(
            gate_values * sigmoid_helper(gate_values) * up_values,
            I.bf16,
        )
