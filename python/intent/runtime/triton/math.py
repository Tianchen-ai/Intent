import triton
import triton.language as tl


@triton.constexpr_function
def _swap_last_axes(rank):
    return tuple(tl.constexpr(axis) for axis in range(rank - 2)) + (
        tl.constexpr(rank - 1), tl.constexpr(rank - 2)
    )


@triton.constexpr_function
def _split_levels(extent):
    return extent.bit_length() - 1


@triton.jit
def _contract_terms(value):
    extent: tl.constexpr = value.shape[-1]
    tl.static_assert(extent > 0 and extent & (extent - 1) == 0)
    terms = (value,)
    for level in tl.static_range(_split_levels(extent)):
        even_terms = ()
        odd_terms = ()
        for component in tl.static_range(len(terms)):
            pairs = tl.reshape(
                terms[component],
                terms[component].shape[:-1] + (terms[component].shape[-1] // 2, 2),
            )
            low, high = tl.split(pairs)
            even_terms += (low,)
            odd_terms += (high,)
        # Residue classes stay ordered as each new index bit is exposed.
        terms = even_terms + odd_terms
    return terms


@triton.jit
def contract_fma(lhs, rhs, accumulator):
    tl.static_assert(accumulator.dtype == tl.float32)
    left, right = tl.broadcast(
        tl.expand_dims(lhs.to(accumulator.dtype), len(lhs.shape)),
        tl.expand_dims(rhs.to(accumulator.dtype), len(rhs.shape) - 2),
    )
    order: tl.constexpr = _swap_last_axes(len(left.shape))
    left_terms = _contract_terms(tl.permute(left, tl.tuple(order)))
    right_terms = _contract_terms(tl.permute(right, tl.tuple(order)))
    for index in tl.static_range(len(left_terms)):
        left = tl.reshape(left_terms[index], accumulator.shape)
        right = tl.reshape(right_terms[index], accumulator.shape)
        accumulator = tl.fma(left, right, accumulator)
    return accumulator
