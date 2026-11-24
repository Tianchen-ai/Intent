@always_inline
def welford_vec_step[W: Int](
    x: SIMD[DType.float32, W],
    mean: SIMD[DType.float32, W],
    m2: SIMD[DType.float32, W],
    step_n: Int32,
) -> Tuple[SIMD[DType.float32, W], SIMD[DType.float32, W]]:
    var inv_n = SIMD[DType.float32, W](Float32(1.0) / Float32(step_n))
    var delta = x - mean
    var next_mean = mean + delta * inv_n
    var next_m2 = m2 + delta * (x - next_mean)
    return (next_mean, next_m2)


@always_inline
def welford_vec_reduce[W: Int](
    step_n: Int32,
    mean_in: SIMD[DType.float32, W],
    m2_in: SIMD[DType.float32, W],
) -> Tuple[Int32, Float32, Float32]:
    if step_n == 0:
        return (Int32(0), Float32(0.0), Float32(0.0))
    var cur_mean = mean_in
    var cur_m2 = m2_in
    var cur_n = step_n
    comptime for s in range(6):
        comptime stride = 1 << s
        comptime if stride < W:
            var half_w = Float32(0.5) * Float32(cur_n)
            var next_mean = SIMD[DType.float32, W](0)
            var next_m2 = SIMD[DType.float32, W](0)
            comptime for i in range(W // (2 * stride)):
                var ma = cur_mean[i]
                var mb = cur_mean[i + (W // (2 * stride))]
                var m2a = cur_m2[i]
                var m2b = cur_m2[i + (W // (2 * stride))]
                var delta = mb - ma
                next_mean[i] = ma + Float32(0.5) * delta
                next_m2[i] = m2a + m2b + delta * delta * half_w
            cur_mean = next_mean
            cur_m2 = next_m2
            cur_n = cur_n * 2
    return (cur_n, cur_mean[0], cur_m2[0])


@always_inline
def welford_scalar_step(
    n: Int32, mean: Float32, m2: Float32, x: Float32
) -> Tuple[Int32, Float32, Float32]:
    var total = n + 1
    var delta = x - mean
    var corr = delta / Float32(total)
    if n == 0:
        return (total, x, Float32(0.0))
    return (total, mean + corr, m2 + delta * Float32(n) * corr)
