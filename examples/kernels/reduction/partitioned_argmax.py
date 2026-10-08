import torch
import intent
import intent.language as I


@intent.kernel
def argmax_partials(
    input: I.In[I.f16, ("N",)],
    values: I.Out[I.f16, ("P",)],
    indices: I.Out[I.i64, ("P",)],
):
    N = input.shape[0]
    P = values.shape[0]
    source = I.domain(0, N)
    for part in I.parallel(I.domain(0, P)):
        begin = part * N // P
        end = (part + 1) * N // P
        value, index = I.arg_reduce.max(input[source[begin:end]], axis=0)
        values[part] = value
        indices[part] = I.cast(begin + index, I.i64)


@intent.kernel
def argmax_finalize(
    values: I.In[I.f16, ("P",)],
    indices: I.In[I.i64, ("P",)],
    output: I.Out[I.i64, ()],
):
    value, part = I.arg_reduce.max(values, axis=0)
    output[()] = indices[part]


def build(context):
    partial_kernel = context.compile("argmax_partials", argmax_partials)
    final_kernel = context.compile("argmax_finalize", argmax_finalize)

    def argmax(input, dim, keepdim=False):
        if input.ndim != 1 or dim not in (0, -1):
            raise NotImplementedError("partitioned argmax requires a one-dimensional input")
        if input.numel() == 0:
            raise ValueError("argmax requires a nonempty input")
        parts = min(input.numel(), 1024)
        values = torch.empty((parts,), device=input.device, dtype=torch.float16)
        indices = torch.empty((parts,), device=input.device, dtype=torch.int64)
        partial_kernel(input, values, indices)
        output = final_kernel.run(values, indices)
        return output.reshape(1) if keepdim else output

    return argmax
