import torch
import intent
import intent.language as I


@intent.kernel
def sum_partials(
    input: I.In[I.f32, ("N",)],
    partial: I.Out[I.f32, ("P",)],
):
    N = input.shape[0]
    P = partial.shape[0]
    source = I.domain(0, N)
    width = (N + P - 1) // P
    for part in I.parallel(I.domain(0, P)):
        begin = I.minimum(part * width, N)
        end = I.minimum((part + 1) * width, N)
        region = source[begin:end]
        partial[part] = I.reduce.sum(input[region], axis=0, acc_dtype=I.f32)


@intent.kernel
def sum_finalize(
    partial: I.In[I.f32, ("P",)],
    output: I.Out[I.f32, ()],
):
    output[()] = I.reduce.sum(partial, axis=0, acc_dtype=I.f32)


def build(context):
    partial_kernel = context.compile("sum_partials", sum_partials)
    final_kernel = context.compile("sum_finalize", sum_finalize)

    def sum(input, dim, keepdim=False, *, dtype=None):
        if input.ndim != 1 or dim not in (0, -1):
            raise NotImplementedError("partitioned sum requires a one-dimensional input")
        if dtype not in (None, torch.float32):
            raise NotImplementedError("partitioned sum accumulates and returns float32")
        partial = torch.empty((256,), device=input.device, dtype=torch.float32)
        partial_kernel(input, partial)
        output = final_kernel.run(partial)
        return output.reshape(1) if keepdim else output

    return sum
