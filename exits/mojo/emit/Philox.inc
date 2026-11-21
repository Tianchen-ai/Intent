R"mojo(
@always_inline
def uniform_at[width: Int](key: UInt64, position: SIMD[DType.uint64, width]) -> SIMD[DType.float32, width]:
    var block = position >> 2
    var a = block.cast[DType.uint32]()
    var b = (block >> 32).cast[DType.uint32]()
    var c = SIMD[DType.uint32, width](0)
    var d = SIMD[DType.uint32, width](0)
    var k = key.cast[DType.uint32]()
    var l = (key >> 32).cast[DType.uint32]()
    comptime for i in range(10):
        var ac = a.cast[DType.uint64]() * 0xd2511f53
        var cd = c.cast[DType.uint64]() * 0xcd9e8d57
        a = (cd >> 32).cast[DType.uint32]() ^ b ^ k
        c = (ac >> 32).cast[DType.uint32]() ^ d ^ l
        b = cd.cast[DType.uint32]()
        d = ac.cast[DType.uint32]()
        k += 0x9e3779b9
        l += 0xbb67ae85
    var lane = (position & 3).cast[DType.uint32]()
    var sample = lane.eq(0).select(a,lane.eq(1).select(b,lane.eq(2).select(c,d)))
    return (sample >> 8).cast[DType.float32]() * 0.000000059604644775390625

)mojo"
