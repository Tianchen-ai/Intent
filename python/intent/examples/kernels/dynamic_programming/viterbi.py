import intent
import intent.language as I


BATCH = 64
TIME_STEPS = 256
STATES = 64


@intent.kernel
def viterbi_decode(
    emissions: I.In[I.f32, (BATCH, TIME_STEPS, STATES)],
    transitions: I.In[I.f32, (STATES, STATES)],
    path: I.Out[I.i32, (BATCH, TIME_STEPS)],
    score: I.Out[I.f32, (BATCH,)],
):
    for batch in I.parallel(I.domain(0, BATCH)):
        previous = I.buffer((STATES,), I.f32, init=-I.inf)
        current = I.buffer((STATES,), I.f32, init=-I.inf)
        predecessors = I.buffer((TIME_STEPS, STATES), I.i32, init=0)

        for state in range(STATES):
            I.store(previous, state, emissions[batch, 0, state])

        for time in range(1, TIME_STEPS):
            for destination in range(STATES):
                best_value = I.cast(-I.inf, I.f32)
                best_source = I.cast(0, I.i32)
                for source in range(STATES):
                    candidate = (
                        I.mutable_load(previous, source)
                        + transitions[source, destination]
                    )
                    if candidate > best_value:
                        best_value = candidate
                        best_source = I.cast(source, I.i32)
                I.store(
                    current,
                    destination,
                    best_value + emissions[batch, time, destination],
                )
                I.store(predecessors, (time, destination), best_source)
            for state in range(STATES):
                I.store(previous, state, I.mutable_load(current, state))

        best_value = I.cast(-I.inf, I.f32)
        best_state = I.cast(0, I.i32)
        for state in range(STATES):
            candidate = I.mutable_load(previous, state)
            if candidate > best_value:
                best_value = candidate
                best_state = I.cast(state, I.i32)
        path[batch, TIME_STEPS - 1] = best_state
        time = TIME_STEPS - 1
        while time > 0:
            I.assume_in_bounds(time, predecessors, axis=0)
            I.assume_in_bounds(best_state, predecessors, axis=1)
            best_state = I.mutable_load(predecessors, (time, best_state))
            path[batch, time - 1] = best_state
            time = time - 1
        score[batch] = best_value
