@always_inline
def transpose_tile_8x8[dt: DType](
    src: Pointer[Scalar[dt], MutUntrackedOrigin],
    src_off: Int,
    src_s0: Int,
    dst: Pointer[Scalar[dt], MutUntrackedOrigin],
    dst_off: Int,
    dst_s0: Int,
):
    var r0 = src.unsafe_offset(src_off + 0 * src_s0).unsafe_load[width=8]()
    var r1 = src.unsafe_offset(src_off + 1 * src_s0).unsafe_load[width=8]()
    var r2 = src.unsafe_offset(src_off + 2 * src_s0).unsafe_load[width=8]()
    var r3 = src.unsafe_offset(src_off + 3 * src_s0).unsafe_load[width=8]()
    var r4 = src.unsafe_offset(src_off + 4 * src_s0).unsafe_load[width=8]()
    var r5 = src.unsafe_offset(src_off + 5 * src_s0).unsafe_load[width=8]()
    var r6 = src.unsafe_offset(src_off + 6 * src_s0).unsafe_load[width=8]()
    var r7 = src.unsafe_offset(src_off + 7 * src_s0).unsafe_load[width=8]()
    dst.unsafe_offset(dst_off + 0 * dst_s0).unsafe_store(
        SIMD[dt, 8](r0[0], r1[0], r2[0], r3[0], r4[0], r5[0], r6[0], r7[0])
    )
    dst.unsafe_offset(dst_off + 1 * dst_s0).unsafe_store(
        SIMD[dt, 8](r0[1], r1[1], r2[1], r3[1], r4[1], r5[1], r6[1], r7[1])
    )
    dst.unsafe_offset(dst_off + 2 * dst_s0).unsafe_store(
        SIMD[dt, 8](r0[2], r1[2], r2[2], r3[2], r4[2], r5[2], r6[2], r7[2])
    )
    dst.unsafe_offset(dst_off + 3 * dst_s0).unsafe_store(
        SIMD[dt, 8](r0[3], r1[3], r2[3], r3[3], r4[3], r5[3], r6[3], r7[3])
    )
    dst.unsafe_offset(dst_off + 4 * dst_s0).unsafe_store(
        SIMD[dt, 8](r0[4], r1[4], r2[4], r3[4], r4[4], r5[4], r6[4], r7[4])
    )
    dst.unsafe_offset(dst_off + 5 * dst_s0).unsafe_store(
        SIMD[dt, 8](r0[5], r1[5], r2[5], r3[5], r4[5], r5[5], r6[5], r7[5])
    )
    dst.unsafe_offset(dst_off + 6 * dst_s0).unsafe_store(
        SIMD[dt, 8](r0[6], r1[6], r2[6], r3[6], r4[6], r5[6], r6[6], r7[6])
    )
    dst.unsafe_offset(dst_off + 7 * dst_s0).unsafe_store(
        SIMD[dt, 8](r0[7], r1[7], r2[7], r3[7], r4[7], r5[7], r6[7], r7[7])
    )
