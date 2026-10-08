import intent
import intent.language as I


ROWS = 4096
VALUES = 256


@intent.kernel
def bitonic_sort_rows(
    values: I.In[I.f32, (ROWS, VALUES)],
    output: I.Out[I.f32, (ROWS, VALUES)],
):
    for row in I.parallel(I.domain(0, ROWS)):
        local = I.buffer((VALUES,), I.f32, init=0.0)
        for index in range(VALUES):
            I.store(local, index, values[row, index])

        sequence = 2
        while sequence <= VALUES:
            stride = sequence // 2
            while stride > 0:
                for index in range(VALUES):
                    index_value = I.cast(index, I.i64)
                    partner = index_value ^ stride
                    I.assume_in_bounds(partner, local, axis=0)
                    if partner > index_value:
                        lhs = I.mutable_load(local, index)
                        rhs = I.mutable_load(local, partner)
                        ascending = (index_value & sequence) == 0
                        should_swap = (
                            lhs > rhs if ascending else lhs < rhs
                        )
                        if should_swap:
                            I.store(local, index, rhs)
                            I.store(local, partner, lhs)
                stride = stride // 2
            sequence = sequence * 2

        for index in range(VALUES):
            output[row, index] = I.mutable_load(local, index)
