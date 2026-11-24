@always_inline
def _pair_hadd_inlane(
    a: SIMD[DType.float32, 16], b: SIMD[DType.float32, 16]
) -> SIMD[DType.float32, 16]:
    var ev = a.shuffle[
        0, 2, 16, 18, 4, 6, 20, 22, 8, 10, 24, 26, 12, 14, 28, 30
    ](b)
    var od = a.shuffle[
        1, 3, 17, 19, 5, 7, 21, 23, 9, 11, 25, 27, 13, 15, 29, 31
    ](b)
    return ev + od


@always_inline
def reduce_sum_16x16(
    v0: SIMD[DType.float32, 16], v1: SIMD[DType.float32, 16],
    v2: SIMD[DType.float32, 16], v3: SIMD[DType.float32, 16],
    v4: SIMD[DType.float32, 16], v5: SIMD[DType.float32, 16],
    v6: SIMD[DType.float32, 16], v7: SIMD[DType.float32, 16],
    v8: SIMD[DType.float32, 16], v9: SIMD[DType.float32, 16],
    v10: SIMD[DType.float32, 16], v11: SIMD[DType.float32, 16],
    v12: SIMD[DType.float32, 16], v13: SIMD[DType.float32, 16],
    v14: SIMD[DType.float32, 16], v15: SIMD[DType.float32, 16],
) -> SIMD[DType.float32, 16]:
    var s0 = _pair_hadd_inlane(v0, v1)
    var s1 = _pair_hadd_inlane(v2, v3)
    var s2 = _pair_hadd_inlane(v4, v5)
    var s3 = _pair_hadd_inlane(v6, v7)
    var s4 = _pair_hadd_inlane(v8, v9)
    var s5 = _pair_hadd_inlane(v10, v11)
    var s6 = _pair_hadd_inlane(v12, v13)
    var s7 = _pair_hadd_inlane(v14, v15)

    var c0 = _pair_hadd_inlane(s0, s1)
    var c1 = _pair_hadd_inlane(s2, s3)
    var c2 = _pair_hadd_inlane(s4, s5)
    var c3 = _pair_hadd_inlane(s6, s7)

    var y0 = c0.slice[8, offset=0]() + c0.slice[8, offset=8]()
    var y1 = c1.slice[8, offset=0]() + c1.slice[8, offset=8]()
    var y2 = c2.slice[8, offset=0]() + c2.slice[8, offset=8]()
    var y3 = c3.slice[8, offset=0]() + c3.slice[8, offset=8]()

    var x0 = y0.slice[4, offset=0]() + y0.slice[4, offset=4]()
    var x1 = y1.slice[4, offset=0]() + y1.slice[4, offset=4]()
    var x2 = y2.slice[4, offset=0]() + y2.slice[4, offset=4]()
    var x3 = y3.slice[4, offset=0]() + y3.slice[4, offset=4]()
    return x0.join(x1).join(x2.join(x3))
