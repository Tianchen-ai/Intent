import torch
import intent
import intent.language as I


@intent.kernel
def std_partials(
    input: I.In[I.f32, ("N",)],
    partial: I.Out[I.f32, ("P", 3)],
):
    N = input.shape[0]
    P = partial.shape[0]
    source = I.domain(0, N)
    for part in I.parallel(I.domain(0, P)):
        begin = part * N // P
        end = (part + 1) * N // P
        values = input[source[begin:end]]
        count = I.cast(end - begin, I.f32)
        mean = I.reduce.sum(values, axis=0, acc_dtype=I.f32) / count
        centered = values - mean
        m2 = I.reduce.sum(centered * centered, axis=0, acc_dtype=I.f32)
        partial[part, 0] = mean
        partial[part, 1] = m2
        partial[part, 2] = count


@intent.kernel
def std_finalize(
    partial: I.In[I.f32, ("P", 3)],
    output: I.Out[I.f32, ()],
    correction: I.f32,
):
    parts = I.domain(0, partial.shape[0])
    means = partial[parts, 0]
    m2 = partial[parts, 1]
    counts = partial[parts, 2]
    total_count = I.reduce.sum(counts, axis=0, acc_dtype=I.f32)
    mean = I.reduce.sum(means * counts, axis=0, acc_dtype=I.f32) / total_count
    centered = means - mean
    between = I.reduce.sum(counts * centered * centered, axis=0, acc_dtype=I.f32)
    variance = (I.reduce.sum(m2, axis=0, acc_dtype=I.f32) + between) / I.maximum(
        total_count - correction, 0.0
    )
    output[()] = I.sqrt(variance)


def build(context):
    partial_kernel = context.compile("std_partials", std_partials)
    final_kernel = context.compile("std_finalize", std_finalize)

    def std(input, dim=None, *, correction=1, keepdim=False, out=None):
        if input.ndim != 1 or dim not in (None, 0, -1):
            raise NotImplementedError("partitioned std requires a one-dimensional input")
        if input.numel() == 0:
            raise NotImplementedError("partitioned std requires a nonempty input")
        parts = min(input.numel(), 1024)
        partial = torch.empty((parts, 3), device=input.device, dtype=torch.float32)
        output = out if out is not None else torch.empty(
            (1,) if keepdim else (), device=input.device, dtype=torch.float32
        )
        partial_kernel(input, partial)
        final_kernel(partial, output.reshape(()), float(correction))
        return output

    return std
