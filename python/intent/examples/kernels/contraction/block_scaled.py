import intent
import intent.language as I


M = 512
N = 512
K_BLOCKS = 24
BLOCK_SIZE = 32


@intent.kernel
def block_scaled_matmul(
    lhs: I.In[I.f8e4m3fn, ("M", "KB", 32)],
    lhs_scale: I.In[I.u8, ("M", "KB")],
    rhs: I.In[I.f8e4m3fn, ("KB", 32, "N")],
    rhs_scale: I.In[I.u8, ("KB", "N")],
    output: I.Out[I.f32, ("M", "N")],
):
    M, KB, KI = lhs.shape
    N = rhs.shape[2]
    rows = I.domain(0, M)
    columns = I.domain(0, N)
    k_blocks = I.domain(0, KB)
    k_inner = I.domain(0, KI)
    output[rows, columns] = I.scaled_matmul(
        lhs[rows, k_blocks, k_inner],
        lhs_scale[rows, k_blocks],
        rhs[k_blocks, k_inner, columns],
        I.transpose(rhs_scale[k_blocks, columns], (1, 0)),
        lhs_format=I.e4m3,
        rhs_format=I.e4m3,
        group_size=32,
        acc_dtype=I.f32,
    )


@intent.kernel
def scaled_fp8_matmul(
    lhs: I.In[I.f8e4m3fn, ("M", "K")],
    rhs: I.In[I.f8e4m3fn, ("K", "N")],
    output: I.Out[I.f16, ("M", "N")],
    lhs_scale: I.f32,
    rhs_scale: I.f32,
):
    M, K = lhs.shape
    N = rhs.shape[1]
    rows = I.domain(0, M)
    columns = I.domain(0, N)
    reduction = I.domain(0, K)
    value = I.matmul(
        lhs[rows, reduction],
        rhs[reduction, columns],
        acc_dtype=I.f32,
    )
    output[rows, columns] = I.cast(
        value * lhs_scale * rhs_scale,
        I.f16,
    )


@intent.kernel
def scaled_fp8_splitk_matmul(
    lhs: I.In[I.f8e4m3fn, ("M", "IT", "SP", 256)],
    rhs: I.In[I.f8e4m3fn, ("IT", "SP", 256, "N")],
    output: I.InOut[I.f16, ("M", "N")],
    lhs_scale: I.f32,
    rhs_scale: I.f32,
):
    M, IT, SP, _ = lhs.shape
    N = rhs.shape[3]
    rows = I.domain(0, M)
    columns = I.domain(0, N)
    iterations = I.domain(0, IT)
    splits = I.domain(0, SP)
    reduction = I.domain(0, 256)
    for split in I.parallel(splits):
        partial = I.contract(
            lhs[rows, iterations, split, reduction],
            rhs[iterations, split, reduction, columns],
            reduce=((1, 0), (2, 1)),
            acc_dtype=I.f32,
        )
        I.atomic.add(
            output,
            index=(rows, columns),
            value=I.cast(
                partial * lhs_scale * rhs_scale,
                I.f16,
            ),
            order="relaxed",
        )


@intent.kernel
def deepgemm_fp8_2xacc(
    lhs: I.In[I.f8e4m3fn, ("M", "KB", 128)],
    rhs: I.In[I.f8e4m3fn, ("N", "KB", 128)],
    lhs_scale: I.In[I.f32, ("M", "KB")],
    rhs_scale: I.In[I.f32, ("NB", "KB")],
    output: I.Out[I.bf16, ("M", "N")],
):
    M, KB, KI = lhs.shape
    N = rhs.shape[0]
    rows = I.domain(0, M)
    columns = I.domain(0, N)
    blocks = I.domain(0, KB)
    reduction = I.domain(0, KI)
    column_blocks = I.indices(columns) // 128
    I.assume_in_bounds(column_blocks, rhs_scale, axis=0)
    result = I.scaled_matmul(
        lhs[rows, blocks, reduction],
        lhs_scale[rows, blocks],
        I.transpose(rhs[columns, blocks, reduction], (1, 2, 0)),
        rhs_scale[column_blocks, blocks],
        lhs_format=I.e4m3,
        rhs_format=I.e4m3,
        group_size=128,
        acc_dtype=I.f32,
    )
    output[rows, columns] = I.cast(
        result,
        I.bf16,
    )
