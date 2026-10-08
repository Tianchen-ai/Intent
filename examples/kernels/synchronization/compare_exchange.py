import intent
import intent.language as I


SLOTS = 65536


@intent.kernel
def claim_zero_slots(
    state: I.InOut[I.i32, ("N",)],
    previous: I.Out[I.i32, ("N",)],
):
    N = state.shape[0]
    for index in I.parallel(I.domain(0, N)):
        result = I.atomic.compare_exchange(
            state,
            index=(index,),
            expected=0,
            desired=1,
            order="relaxed",
        )
        previous[index] = result.old_value
