import triton
import triton.language as tl

@triton.jit
def merge_moments(an, am, av, bn, bm, bv):
    count = an + bn
    delta = bm - am
    correction = tl.div_rn(delta * bn.to(tl.float32), tl.maximum(count, 1).to(tl.float32))
    merged = am + correction
    spread = av + bv + delta * (an).to(tl.float32) * correction
    return count, tl.where(an == 0, bm, tl.where(bn == 0, am, merged)), tl.where(an == 0, bv, tl.where(bn == 0, av, spread))
