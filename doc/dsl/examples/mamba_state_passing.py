import intent
import intent.language as I


@intent.kernel
def mamba_state_passing(
    chunk_updates: I.In[I.f32, ("B", "C", "H", "P", "N")],
    chunk_log_decay: I.In[I.f32, ("B", "C", "H")],
    initial_state: I.In[I.f32, ("B", "H", "P", "N")],
    incoming_states: I.Out[I.f32, ("B", "C", "H", "P", "N")],
    final_state: I.Out[I.f32, ("B", "H", "P", "N")],
):
    B, C, H, P, N = chunk_updates.shape
    chunks = I.domain(0, C)

    for batch in I.parallel(I.domain(0, B)):
        for head in I.parallel(I.domain(0, H)):
            state = initial_state[batch, head, :, :]

            # C and every incoming_states[chunk] are public algorithm data.
            # This is an ordinary ordered recurrence, not region_scan.
            for chunk in chunks:
                incoming_states[batch, chunk, head, :, :] = state
                decay = I.exp(
                    chunk_log_decay[batch, chunk, head]
                )
                state = (
                    decay * state
                    + chunk_updates[batch, chunk, head, :, :]
                )

            final_state[batch, head, :, :] = state


# The preceding Mamba chunk-state kernel and its host wrapper define what each
# chunk means and materialize chunk_updates/chunk_log_decay.  Their C dimension,
# incoming-state tensor and final state are observable ABI.  The compiler may
# block the loop physically, but it may not replace these logical chunks with
# compiler-selected region_scan segments.
