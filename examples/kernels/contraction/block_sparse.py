import intent
import intent.language as I


M = 4096
N = 4096
K = 4096
BLOCK_M = 128
BLOCK_N = 128
BLOCK_K = 32


@intent.kernel
def block_sparse_matmul(
    lhs: I.In[I.f16, ("M", "K")],
    rhs: I.In[I.f16, ("K", "N")],
    block_mask: I.In[I.u8, ("MB", "NB", "KB")],
    output: I.Out[I.f16, ("M", "N")],
):
    M, K = lhs.shape
    N = rhs.shape[1]
    MB, NB, KB = block_mask.shape
    rows = I.domain(0, M)
    columns = I.domain(0, N)
    reduction = I.domain(0, K)
    row_blocks = I.domain(0, MB)
    column_blocks = I.domain(0, NB)
    reduction_blocks = I.domain(0, KB)
    for row_block in I.parallel(row_blocks):
        for column_block in I.parallel(column_blocks):
            row_begin = row_block * BLOCK_M
            row_end = I.minimum(row_begin + BLOCK_M, M)
            column_begin = column_block * BLOCK_N
            column_end = I.minimum(column_begin + BLOCK_N, N)
            row_region = rows[row_begin:row_end]
            column_region = columns[column_begin:column_end]
            accumulator = I.zeros(
                (row_region, column_region),
                dtype=I.f32,
            )
            for reduction_block in reduction_blocks:
                reduction_begin = reduction_block * BLOCK_K
                reduction_end = I.minimum(reduction_begin + BLOCK_K, K)
                reduction_region = reduction[reduction_begin:reduction_end]
                enabled = block_mask[row_block, column_block, reduction_block] != 0
                if enabled:
                    partial = I.matmul(
                        lhs[row_region, reduction_region],
                        rhs[reduction_region, column_region],
                        acc_dtype=I.f32,
                    )
                    accumulator = accumulator + partial
            output[row_region, column_region] = I.cast(accumulator, I.f16)
