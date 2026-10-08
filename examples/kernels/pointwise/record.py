import intent
import intent.language as I


ELEMENTS = 1048576


@intent.kernel
def paired_sum_product(
    x: I.In[I.f32, ("N",)],
    y: I.In[I.f32, ("N",)],
    output: I.Out[I.f32, ("N",)],
):
    N = x.shape[0]
    for index in I.parallel(I.domain(0, N)):
        pair = I.record(
            summed=x[index] + y[index],
            multiplied=x[index] * y[index],
        )
        output[index] = pair.summed + pair.multiplied
