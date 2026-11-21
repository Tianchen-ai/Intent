R"python(
@ct.function
def merge_moments(an, am, av, bn, bm, bv):
    count = an + bn
    delta = bm - am
    correction = ct.truediv(delta * ct.astype(bn,ct.float32),ct.astype(ct.maximum(count,1),ct.float32),rounding_mode=ct.RoundingMode.RN)
    merged = am + correction
    spread = av + bv + delta * ct.astype(an, ct.float32) * correction
    return count, ct.where(an == 0, bm, ct.where(bn == 0, am, merged)), ct.where(an == 0, bv, ct.where(bn == 0, av, spread))
)python"
