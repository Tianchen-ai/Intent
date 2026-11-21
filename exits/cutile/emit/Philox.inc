R"python(
@ct.function
def uniform_at(seed, position):
    counter = ct.astype(position, ct.uint64) >> 2
    a = ct.astype(counter, ct.uint32)
    b = ct.astype(counter >> 32, ct.uint32)
    c = a ^ a
    d = c
    key = ct.astype(seed, ct.uint64)
    lo = ct.astype(key, ct.uint32)
    hi = ct.astype(key >> 32, ct.uint32)
    for i in range(10):
        left = ct.astype(a, ct.uint64) * 0xd2511f53
        right = ct.astype(c, ct.uint64) * 0xcd9e8d57
        a = ct.astype(right >> 32, ct.uint32) ^ b ^ lo
        c = ct.astype(left >> 32, ct.uint32) ^ d ^ hi
        b = ct.astype(right, ct.uint32)
        d = ct.astype(left, ct.uint32)
        lo = ct.astype(lo + 0x9e3779b9, ct.uint32)
        hi = ct.astype(hi + 0xbb67ae85, ct.uint32)
    word = position & 3
    bits = ct.where(word == 0, a, ct.where(word == 1, b, ct.where(word == 2, c, d)))
    return ct.astype(bits >> 8,ct.float32) * 0.000000059604644775390625
)python"
