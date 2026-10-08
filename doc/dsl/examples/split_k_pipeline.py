import intent
import intent.language as I
import torch


@intent.kernel
def split_k_partial(
    a: I.In[I.f16, ("M", "K")],
    b: I.In[I.f16, ("K", "N")],
    partial: I.Out[I.f32, ("P", "M", "N")],
):
    P, M, N = partial.shape
    K = a.shape[1]
    parts = I.domain(0, P)
    rows = I.domain(0, M)
    columns = I.domain(0, N)
    reduction_axis = I.domain(0, K)
    width = (K + P - 1) // P

    for part in I.parallel(parts):
        begin = I.minimum(part * width, K)
        end = I.minimum((part + 1) * width, K)
        reduction = reduction_axis[begin:end]
        partial[part, rows, columns] = I.matmul(
            a[rows, reduction],
            b[reduction, columns],
            acc_dtype=I.f32,
        )


@intent.kernel
def split_k_combine(
    partial: I.In[I.f32, ("P", "M", "N")],
    output: I.Out[I.f16, ("M", "N")],
):
    P, M, N = partial.shape
    parts = I.domain(0, P)
    rows = I.domain(0, M)
    columns = I.domain(0, N)
    output[rows, columns] = I.cast(
        I.reduce.sum(partial[parts, rows, columns], axis=0),
        I.f16,
    )


def compile_split_k(*, compiler, target):
    """Compile both kernels before invoking or timing the returned operator."""
    partial_kernel = intent.compile(split_k_partial, compiler=compiler, target=target)
    combine_kernel = intent.compile(split_k_combine, compiler=compiler, target=target)

    def run(a, b, parts):
        partial = torch.empty((parts, a.shape[0], b.shape[1]), device=a.device, dtype=torch.float32)
        output = torch.empty((a.shape[0], b.shape[1]), device=a.device, dtype=torch.float16)
        partial_kernel(a, b, partial)
        combine_kernel(partial, output)
        return output

    return run
