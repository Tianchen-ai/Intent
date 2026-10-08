import intent
import intent.language as I


TOKENS = 4096
FEATURES = 14336


@intent.kernel
def swiglu_forward(
    gate: I.In[I.bf16, ("M", "N")],
    up: I.In[I.bf16, ("M", "N")],
    output: I.Out[I.bf16, ("M", "N")],
):
    M, N = gate.shape
    columns = I.domain(0, N)
    for row in I.parallel(I.domain(0, M)):
        gate_values = I.cast(gate[row, columns], I.f32)
        up_values = up[row, columns]
        sigmoid = I.fdiv(
            1.0, 1.0 + I.exp(-gate_values), approximate=True, flush_to_zero=True
        )
        silu = I.cast(gate_values * sigmoid, I.bf16)
        output[row, columns] = silu * up_values


@intent.kernel
def silu_and_mul_packed(
    packed: I.In[I.bf16, ("M", 2 * FEATURES)],
    output: I.Out[I.bf16, ("M", FEATURES)],
):
    M = packed.shape[0]
    columns = I.domain(0, FEATURES)
    for row in I.parallel(I.domain(0, M)):
        gate_values = I.cast(packed[row, columns], I.f32)
        up_values = I.cast(
            packed[row, I.indices(columns) + FEATURES],
            I.f32,
        )
        sigmoid = I.fdiv(
            1.0, 1.0 + I.exp(-gate_values), approximate=True, flush_to_zero=True
        )
        output[row, columns] = I.cast(
            gate_values * sigmoid * up_values,
            I.bf16,
        )
