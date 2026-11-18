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


@intent.kernel
def q4_projection(
    weights: I.In[I.u8, ("N", "G", 144)],
    activation: I.In[I.f32, ("M", "G", 256)],
    output: I.Out[I.f32, ("M", "N")],
):
    records = I.domain(0, weights.shape[1])
    elements = I.domain(0, 256)
    weight_bytes = I.domain(0, 144)
    for row in I.parallel(I.domain(0, activation.shape[0])):
        quantized = I.quantize(activation[row, records, elements], format=I.quant.q8_k)
        for column in I.parallel(I.domain(0, weights.shape[0])):
            output[row, column] = I.quantized_dot(
                weights[column, records, weight_bytes], quantized,
                lhs_format=I.quant.q4_k, rhs_format=I.quant.q8_k, acc_dtype=I.f32,
            )


@intent.kernel
def q4_dot(
    weights: I.In[I.u8, ("G", 144)],
    activation: I.In[I.u8, ("G", 292)],
    output: I.Out[I.f32, (1, 1)],
):
    records = I.domain(0, weights.shape[0])
    left_bytes = I.domain(0, 144)
    right_bytes = I.domain(0, 292)
    output[0, 0] = I.quantized_dot(
        weights[records, left_bytes], activation[records, right_bytes],
        lhs_format=I.quant.q4_k, rhs_format=I.quant.q8_k, acc_dtype=I.f32,
    )


@intent.kernel
def q8_quantize(values: I.In[I.f32, ("G", 256)], output: I.Out[I.u8, ("G", 292)]):
    records = I.domain(0, values.shape[0])
    elements = I.domain(0, 256)
    packed_bytes = I.domain(0, 292)
    encoded = I.quantize(values[records, elements], format=I.quant.q8_k)
    output[records, packed_bytes] = encoded
