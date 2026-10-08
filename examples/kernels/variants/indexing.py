import intent
import intent.language as I

from kernels.position.rope import HALF_DIMENSION
from kernels.position.rope import HEAD_DIMENSION


@intent.kernel
def rotary_embedding_equivalent_index(
    values: I.In[I.f16, ("R", HEAD_DIMENSION)],
    cosine: I.In[I.f16, ("U", HALF_DIMENSION)],
    sine: I.In[I.f16, ("U", HALF_DIMENSION)],
    output: I.Out[I.f16, ("R", 2, HALF_DIMENSION)],
    HEADS: I.Constexpr[int],
):
    R, D = values.shape
    _, half_dimension = cosine.shape
    dimensions = I.domain(0, D)
    for row in I.parallel(I.domain(0, R)):
        token = row // HEADS
        dimension_index = I.indices(dimensions)
        paired_dimension = (dimension_index + half_dimension) % D
        phase_dimension = dimension_index % half_dimension
        rotate_sign = I.cast(
            -1.0 if dimension_index < half_dimension else 1.0,
            I.f16,
        )
        rotated = (
            values[row, dimensions] * cosine[token, phase_dimension]
            + values[row, paired_dimension]
            * sine[token, phase_dimension]
            * rotate_sign
        )
        output[row, :, :] = I.reshape(rotated, (2, HALF_DIMENSION))
