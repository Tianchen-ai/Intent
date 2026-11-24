@always_inline
def simd_scan_sum[dt: DType, W: Int](
    x: SIMD[dt, W], carry: Scalar[dt]
) -> Tuple[SIMD[dt, W], Scalar[dt]]:
    var cur = x
    comptime for s in range(6):
        comptime jump = 1 << s
        comptime if jump < W:
            var shifted = SIMD[dt, W](0)
            comptime for lane in range(jump, W):
                shifted[lane] = cur[lane - jump]
            cur += shifted
    cur += carry
    return (cur, cur[W - 1])
