import intent
import intent.language as I


@intent.kernel
def fused_bias_gelu(
    x: I.In[I.bf16, ("M", "N")],
    bias: I.In[I.bf16, ("N",)],
    output: I.Out[I.bf16, ("M", "N")],
):
    M, N = x.shape
    rows = I.domain(0, M)
    columns = I.domain(0, N)

    value = I.cast(x[rows, columns], I.f32)
    shifted = value + I.cast(bias[columns], I.f32)
    activated = 0.5 * shifted * (
        1.0 + I.erf(shifted * 0.7071067811865476)
    )
    output[rows, columns] = I.cast(activated, I.bf16)


def compile_bias_gelu(*, compiler, target):
    artifact = intent.compile(fused_bias_gelu, compiler=compiler, target=target)
    return artifact.run
