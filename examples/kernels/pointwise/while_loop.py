import intent
import intent.language as I


ELEMENTS = 262144


@intent.kernel
def integer_log2_floor(
    values: I.In[I.i32, ("N",)],
    output: I.Out[I.i32, ("N",)],
):
    N = values.shape[0]
    for index in I.parallel(I.domain(0, N)):
        value = values[index]
        exponent = I.cast(0, I.i32)
        while value > 1:
            value = value // 2
            exponent = exponent + 1
        output[index] = exponent
