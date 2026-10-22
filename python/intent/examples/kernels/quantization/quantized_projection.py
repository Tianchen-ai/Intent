import intent
import intent.language as I


@intent.kernel
def quantized_projection(
    weights: I.In[I.u8, ("N", "G", 144)],
    activation: I.In[I.f32, ("G", 256)],
    output: I.Out[I.f32, ("N",)],
):
    records = I.domain(0, activation.shape[0])
    elements = I.domain(0, 256)
    weight_bytes = I.domain(0, 144)
    quantized = I.quantize(activation[records, elements], format=I.quant.q8_k)
    for row in I.parallel(I.domain(0, weights.shape[0])):
        output[row] = I.quantized_dot(
            weights[row, records, weight_bytes], quantized,
            lhs_format=I.quant.q4_k, rhs_format=I.quant.q8_k, acc_dtype=I.f32,
        )
