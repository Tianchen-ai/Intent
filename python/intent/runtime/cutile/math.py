"""Tile implementations of ordinary floating-point library operations."""

# The asin/erf/erfc/lgamma rational coefficients and intervals are from fdlibm:
# https://github.com/JuliaMath/openlibm/blob/master/src/e_asinf.c
# https://github.com/JuliaMath/openlibm/blob/master/src/s_erff.c
# https://github.com/JuliaMath/openlibm/blob/master/src/s_erf.c
# https://github.com/JuliaMath/openlibm/blob/master/src/e_lgammaf_r.c
# Float conversion by Ian Lance Taylor, Cygnus Support.
# Copyright (C) 1993 by Sun Microsystems, Inc. All rights reserved.
# Developed at SunPro, a Sun Microsystems, Inc. business.
# Permission to use, copy, modify, and distribute this software is freely
# granted, provided that this notice is preserved.

import cuda.tile as ct


@ct.function
def mma_chunks(lhs, rhs, accumulator, chunk: ct.Constant[int]):
    ct.static_assert(chunk > 0 and lhs.shape[-1] >= chunk
                     and lhs.shape[-1] % chunk == 0,
                     "MMA reduction extent must contain complete chunks")
    if len(lhs.shape) == 2:
        for part in ct.static_iter(range(lhs.shape[-1] // chunk)):
            left = ct.extract(lhs, (0, part), (lhs.shape[0], chunk))
            right = ct.extract(rhs, (part, 0), (chunk, rhs.shape[1]))
            accumulator = ct.mma(left, right, accumulator)
    else:
        for part in ct.static_iter(range(lhs.shape[-1] // chunk)):
            left = ct.extract(lhs, (0, 0, part), (lhs.shape[0], lhs.shape[1], chunk))
            right = ct.extract(rhs, (0, part, 0), (rhs.shape[0], chunk, rhs.shape[2]))
            accumulator = ct.mma(left, right, accumulator)
    return accumulator


@ct.function
def asin(x):
    ordinary = ct.astype(x, ct.float32)
    magnitude = ct.abs(ordinary)
    small = magnitude < 0.5
    reduced = ct.where(small, ordinary * ordinary,
                       (1.0 - ct.minimum(magnitude, 1.0)) * 0.5)
    numerator = reduced * (1.6666586697e-01 + reduced * (
        -4.2743422091e-02 + reduced * -8.6563630030e-03))
    denominator = 1.0 + reduced * -7.0662963390e-01
    correction = ct.truediv(numerator, denominator,
                            rounding_mode=ct.RoundingMode.RN)
    near_zero = ordinary + ordinary * correction
    # fdlibm evaluates the square root and endpoint subtraction in double.
    root = ct.sqrt(ct.astype(reduced, ct.float64),
                   rounding_mode=ct.RoundingMode.RN)
    near_one = ct.astype(ct.float64(1.570796326794896558) -
                         2.0 * (root + root * ct.astype(correction, ct.float64)),
                         ct.float32)
    near_one = ct.where(ordinary < 0.0, -near_one, near_one)
    value = ct.where(small, near_zero, near_one)
    value = ct.where(magnitude > 1.0, float("nan"), value)
    value = ct.where((magnitude < 2.0 ** -12) | ct.isnan(ordinary),
                     ordinary, value)
    return ct.astype(value, x.dtype)


@ct.function
def _erf_small_ratio(x):
    square = x * x
    numerator = square * -1.86260219e-03 - 3.36030394e-01
    numerator = numerator * square + 1.28379166e-01
    denominator = square * -1.98859419e-03 + 2.16070302e-02
    denominator = denominator * square + 3.12324286e-01
    denominator = denominator * square + 1.0
    return numerator / denominator


@ct.function
def _erf_small(x):
    value = x + x * _erf_small_ratio(x)
    tiny = (8.0 * x + 1.0270333290 * x) / 8.0
    return ct.where(x < 2.0 ** -119, tiny, value)


@ct.function
def _erf_near_one_ratio(x):
    shifted = x - 1.0
    numerator = shifted * 1.10914491e-01 - 1.65179938e-01
    numerator = numerator * shifted + 4.15109694e-01
    numerator = numerator * shifted + 3.64939137e-06
    denominator = shifted * 5.62181212e-02 + 1.68576106e-01
    denominator = denominator * shifted + 5.35934687e-01
    denominator = denominator * shifted + 6.02074385e-01
    denominator = denominator * shifted + 1.0
    return numerator / denominator


@ct.function
def _erf_near_one(x):
    return 8.42697144e-01 + _erf_near_one_ratio(x)


@ct.function
def _erfc_tail(x):
    inverse_square = 1.0 / (x * x)
    r1 = inverse_square * -1.43268085 - 2.17589188
    r1 = r1 * inverse_square - 5.53605914e-01
    r1 = r1 * inverse_square - 9.87132732e-03
    s1 = inverse_square * -5.77397496e-02 + 1.43113089
    s1 = s1 * inverse_square + 6.69798088
    s1 = s1 * inverse_square + 5.45995426
    s1 = s1 * inverse_square + 1.0
    r2 = inverse_square * -9.53764343 - 1.66696873e+01
    r2 = r2 * inverse_square - 6.16498327
    r2 = r2 * inverse_square - 6.25171244e-01
    r2 = r2 * inverse_square - 9.86494310e-03
    s2 = inverse_square * 8.93033314 + 4.72810211e+01
    s2 = s2 * inverse_square + 4.51839523e+01
    s2 = s2 * inverse_square + 1.26884899e+01
    s2 = s2 * inverse_square + 1.0
    first_interval = x < 2.857142857142857
    rational = ct.where(first_interval, r1, r2) / ct.where(first_interval, s1, s2)
    high = ct.bitcast(ct.bitcast(x, ct.uint32) & 0xFFFFE000, ct.float32)
    return ct.exp(-high * high - 0.5625) * ct.exp(
        (high - x) * (high + x) + rational
    ) / x


@ct.function
def erf(x):
    # Use the single-precision library algorithm; narrower storage types use
    # the same f32-to-storage conversion as the other ordinary unary functions.
    wide = ct.astype(x, ct.float32)
    magnitude = ct.abs(wide)
    small = _erf_small(ct.minimum(magnitude, 0.84375))
    value = small
    if ct.max(ct.astype(magnitude >= 0.84375, ct.int32)) != 0:
        near = _erf_near_one(ct.minimum(ct.maximum(magnitude, 0.84375), 1.25))
        value = ct.where(magnitude < 0.84375, small, near)
        needs_tail = (magnitude >= 1.25) & (magnitude < 4.0)
        if ct.max(ct.astype(needs_tail, ct.int32)) != 0:
            tail = _erfc_tail(ct.minimum(ct.maximum(magnitude, 1.25), 4.0))
            value = ct.where(needs_tail, 1.0 - tail, value)
    value = ct.where(magnitude >= 4.0, 1.0, value)
    value = ct.where(wide < 0.0, -value, value)
    value = ct.where((wide == 0.0) | ct.isnan(wide), wide, value)
    return ct.astype(ct.astype(value, ct.float32), x.dtype)


@ct.function
def erfc(x):
    wide = ct.astype(x, ct.float32)
    magnitude = ct.abs(wide)
    small_x = ct.minimum(ct.maximum(wide, -0.84375), 0.84375)
    correction = small_x * _erf_small_ratio(small_x)
    small = ct.where(small_x < 0.25,
                     1.0 - (small_x + correction),
                     0.5 - ((small_x - 0.5) + correction))
    small = ct.where(magnitude < 2.0 ** -24, 1.0 - wide, small)
    value = small
    if ct.max(ct.astype(magnitude >= 0.84375, ct.int32)) != 0:
        near_ratio = _erf_near_one_ratio(
            ct.minimum(ct.maximum(magnitude, 0.84375), 1.25))
        near = ct.where(wide < 0.0,
                        1.0 + (8.42697144e-01 + near_ratio),
                        (1.0 - 8.42697144e-01) - near_ratio)
        value = ct.where(magnitude < 0.84375, small, near)
        # Evaluate the tail directly to retain erfc's subnormal results.
        needs_tail = (magnitude >= 1.25) & (magnitude < 11.0)
        if ct.max(ct.astype(needs_tail, ct.int32)) != 0:
            tail = _erfc_tail(ct.minimum(ct.maximum(magnitude, 1.25), 11.0))
            tail = ct.where(wide < 0.0, 2.0 - tail, tail)
            value = ct.where(needs_tail, tail, value)
    limit = ct.where(wide < 0.0, ct.float32(2.0), ct.float32(0.0))
    value = ct.where(magnitude >= 11.0, limit, value)
    value = ct.where(ct.isnan(wide), wide, value)
    return ct.astype(ct.astype(value, ct.float32), x.dtype)


@ct.function
def _chebyshev(x, coefficients: ct.Constant):
    current = ct.full(x.shape, coefficients[0], dtype=x.dtype)
    previous = ct.full(x.shape, 0.0, dtype=x.dtype)
    second_previous = previous
    for coefficient in ct.static_iter(coefficients[1:]):
        second_previous = previous
        previous = current
        current = x * previous - second_previous + coefficient
    return 0.5 * (current - second_previous)


@ct.function
def i0(x):
    # Cephes Chebyshev coefficients, also used by ATen's CUDA i0:
    # https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/cuda/Math.cuh
    # f64 evaluation avoids exp(f32) overflowing before the f32 I0 result.
    # On the reduced [-2, 2] intervals, omitted leading coefficients bound
    # the series error below 1e-9 (small) and 2e-12 (large), below f32 precision.
    wide = ct.astype(x, ct.float64)
    magnitude = ct.abs(wide)
    small_x = ct.minimum(magnitude, 8.0)
    small = ct.exp(small_x) * _chebyshev(0.5 * small_x - 2.0, (
        2.65982372468238665035e-09,
        -1.30002500998624804212e-08, 6.04699502254191894932e-08,
        -2.67079385394061173391e-07, 1.11738753912010371815e-06,
        -4.41673835845875056359e-06, 1.64484480707288970893e-05,
        -5.75419501008210370398e-05, 1.88502885095841655729e-04,
        -5.76375574538582365885e-04, 1.63947561694133579842e-03,
        -4.32430999505057594430e-03, 1.05464603945949983183e-02,
        -2.37374148058994688156e-02, 4.93052842396707084878e-02,
        -9.49010970480476444210e-02, 1.71620901522208775349e-01,
        -3.04682672343198398683e-01, 6.76795274409476084995e-01,
    ))
    value = small
    if ct.max(ct.astype(magnitude > 8.0, ct.int32)) != 0:
        large_x = ct.maximum(magnitude, 8.0)
        large = ct.exp(large_x) * _chebyshev(32.0 / large_x - 2.0, (
            -1.79417853150680611778e-12,
            -1.32158118404477131188e-11, -3.14991652796324136454e-11,
            1.18891471078464383424e-11, 4.94060238822496958910e-10,
            3.39623202570838634515e-09, 2.26666899049817806459e-08,
            2.04891858946906374183e-07, 2.89137052083475648297e-06,
            6.88975834691682398426e-05, 3.36911647825569408990e-03,
            8.04490411014108831608e-01,
        )) / ct.sqrt(large_x)
        value = ct.where(magnitude <= 8.0, small, large)
    value = ct.where(magnitude == 0.0, 1.0, value)
    value = ct.where(magnitude == float("inf"), float("inf"), value)
    value = ct.where(ct.isnan(wide), wide, value)
    return ct.astype(ct.astype(value, ct.float32), x.dtype)


@ct.function
def log1p(x):
    wide = ct.astype(x, ct.float64)
    shifted = 1.0 + wide
    # When double addition rounds to one, log1p(x) rounds to x in f32.
    value = ct.where(shifted == 1.0, wide, ct.log(shifted))
    return ct.astype(ct.astype(value, ct.float32), x.dtype)


@ct.function
def _horner(x, coefficients: ct.Constant):
    value = ct.full(x.shape, coefficients[0], dtype=x.dtype)
    for coefficient in ct.static_iter(coefficients[1:]):
        value = value * x + coefficient
    return value


@ct.function
def _lgamma_positive(x):
    small_x = ct.minimum(x, 2.0)
    bits = ct.bitcast(small_x, ct.uint32)
    shift = bits <= 0x3F666666
    near_two = ct.where(shift, bits >= 0x3F3B4A20, bits >= 0x3FDDA618)
    near_minimum = ct.where(shift, bits >= 0x3E6D3308, bits >= 0x3F9DA620)
    y = ct.where(shift, 1.0 - small_x, 2.0 - small_x)
    z = y * y
    p1 = _horner(z, (2.5214456400e-05, 2.2086278477e-04, 1.1927076848e-03,
                     7.3855509982e-03, 6.7352302372e-02, 7.7215664089e-02))
    p2 = z * _horner(z, (4.4864096708e-05, 1.0801156895e-04, 5.1006977446e-04,
                         2.8905137442e-03, 2.0580807701e-02, 3.2246702909e-01))
    around_two = (y * p1 + p2) - 0.5 * y

    center = ct.float32(1.4616321325)
    y = ct.where(shift, small_x - (center - 1.0), small_x - center)
    z = y * y
    w = z * y
    p1 = _horner(w, (3.1563205994e-04, -1.4034647029e-03, 6.1005386524e-03,
                     -3.2788541168e-02, 4.8383611441e-01))
    p2 = _horner(w, (-3.1275415677e-04, 8.8108185446e-04, -3.6845202558e-03,
                     1.7970675603e-02, -1.4758771658e-01))
    p3 = _horner(w, (3.3552918467e-04, -5.3859531181e-04, 2.2596477065e-03,
                     -1.0314224288e-02, 6.4624942839e-02))
    around_minimum = -1.2148628384e-01 + (
        z * p1 - (6.6971006518e-09 - w * (p2 + y * p3)))

    y = ct.where(shift, small_x, small_x - 1.0)
    p1 = y * _horner(y, (1.3381091878e-02, 2.2896373272e-01, 9.7771751881e-01,
                         1.4549225569e+00, 6.3282704353e-01, -7.7215664089e-02))
    p2 = _horner(y, (3.2170924824e-03, 1.0422264785e-01, 7.6928514242e-01,
                     2.1284897327e+00, 2.4559779167e+00, 1.0))
    around_one = -0.5 * y + p1 / p2
    small = ct.where(near_two, around_two,
                     ct.where(near_minimum, around_minimum, around_one))
    small_log = ct.log(small_x)
    small = small + ct.where(shift, -small_log, 0.0)

    value = small
    if ct.max(ct.astype(x >= 2.0, ct.int32)) != 0:
        middle_x = ct.minimum(ct.maximum(x, 2.0), 8.0)
        y = middle_x - ct.floor(middle_x)
        p = y * _horner(y, (3.1947532989e-05, 1.8402845599e-03, 2.6642270386e-02,
                            1.4635047317e-01, 3.2577878237e-01, 2.1498242021e-01,
                            -7.7215664089e-02))
        q = _horner(y, (7.3266842264e-06, 7.7794247773e-04, 1.8645919859e-02,
                        1.7193385959e-01, 7.2193557024e-01, 1.3920053244e+00, 1.0))
        product = ct.full(x.shape, 1.0, dtype=x.dtype)
        for offset in ct.static_iter((6, 5, 4, 3, 2)):
            product = product * ct.where(middle_x >= offset + 1, y + offset, 1.0)
        middle = 0.5 * y + p / q + ct.log(product)

        large_x = ct.maximum(x, 8.0)
        inverse = 1.0 / large_x
        correction = 4.1893854737e-01 + inverse * _horner(inverse * inverse, (
            -1.6309292987e-03, 8.3633989561e-04, -5.9518753551e-04,
            7.9365057172e-04, -2.7777778450e-03, 8.3333335817e-02))
        logarithm = ct.log(large_x) - 1.0
        large = ct.where(large_x < 2.0 ** 58,
                         (large_x - 0.5) * logarithm + correction,
                         large_x * logarithm)
        value = ct.where(x < 2.0, small, ct.where(x < 8.0, middle, large))
    value = ct.where(x < 2.0 ** -21, -small_log, value)
    return ct.where((x == 1.0) | (x == 2.0), 0.0, value)


@ct.function
def lgamma(x):
    ordinary = ct.astype(x, ct.float32)
    magnitude = ct.abs(ordinary)
    value = _lgamma_positive(magnitude)
    reflected_region = (ordinary < 0.0) & (magnitude >= 2.0 ** -21)
    if ct.max(ct.astype(reflected_region, ct.int32)) != 0:
        # Nonintegral negative f32 arguments have magnitude below 2**23.
        # Nearest-integer reduction keeps sin's argument within [-pi/2, pi/2].
        fraction = ordinary - ct.floor(ordinary + 0.5)
        sine = ct.abs(ct.sin(fraction * 3.1415927410))
        reflected = ct.log(3.1415927410 / (sine * magnitude)) - value
        value = ct.where(reflected_region, reflected, value)
    pole = (ordinary <= 0.0) & (ordinary == ct.floor(ordinary))
    value = ct.where(pole | (magnitude == float("inf")), float("inf"), value)
    value = ct.where(ct.isnan(ordinary), ordinary, value)
    return ct.astype(ct.astype(value, ct.float32), x.dtype)
