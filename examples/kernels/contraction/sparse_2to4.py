import intent
import intent.language as I


M = 2048
N = 4096
K = 4096


@intent.kernel
def sparse_2to4_gemm(
    compressed: I.In[I.f16, ("M", "KC")],
    metadata: I.In[I.i16, ("M", "KE")],
    rhs: I.In[I.f16, ("K", "N")],
    output: I.Out[I.f32, ("M", "N")],
):
    M, _ = compressed.shape
    K, N = rhs.shape
    rows = I.domain(0, M)
    columns = I.domain(0, N)
    reduction = I.domain(0, K)
    groups = I.domain(0, K // 4)
    compressed_axis = I.domain(0, K // 2)
    group_coordinates = I.indices(groups)
    metadata_word = metadata[rows, group_coordinates // 4]
    shift = I.cast((group_coordinates % 4) * 4, I.i16)
    metadata_nibble = (metadata_word >> shift[None, :]) & 0xF
    logical_positions = I.record(
        first=I.cast(metadata_nibble & 0x3, I.index),
        second=I.cast((metadata_nibble >> 2) & 0x3, I.index),
    )
    output[rows, columns] = I.sparse_matmul(
        compressed[rows, compressed_axis],
        logical_positions,
        rhs[reduction, columns],
        format=I.sparse.two_of_four(compression_axis=1, logical_extent=K),
        acc_dtype=I.f32,
    )
