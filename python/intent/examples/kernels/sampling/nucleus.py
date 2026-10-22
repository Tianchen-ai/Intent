import intent
import intent.language as I


ROWS = 1024
CANDIDATES = 4093
THRESHOLD = 0.9


@intent.kernel
def sorted_nucleus_cutoff(
    probabilities: I.In[I.f32, ("M", "N")],
    cumulative: I.Out[I.f32, ("M", "N")],
    cutoff: I.Out[I.i32, ("M",)],
    threshold: I.f32,
):
    M, N = probabilities.shape
    candidates = I.domain(0, N)
    for row in I.parallel(I.domain(0, M)):
        prefix = I.cumsum(
            probabilities[row, candidates],
            axis=0,
        )
        cumulative[row, candidates] = prefix
        negative_positions = -I.cast(I.indices(candidates), I.f32)
        first_reached = I.mask(
            negative_positions,
            valid=prefix >= threshold,
            fill=-I.inf,
        )
        _, first_index = I.arg_reduce.max(
            first_reached,
            axis=0,
        )
        cutoff[row] = I.cast(first_index + 1, I.i32)
