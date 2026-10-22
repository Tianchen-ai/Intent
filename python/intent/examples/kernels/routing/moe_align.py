import intent
import intent.language as I


TOKENS = 4096
TOP_K = 2
EXPERTS = 64
BLOCK_SIZE = 128
ROUTES = TOKENS * TOP_K
PADDED_ROUTES = ROUTES + EXPERTS * (BLOCK_SIZE - 1)
EXPERT_BLOCKS = PADDED_ROUTES // BLOCK_SIZE + 1


@intent.kernel
def moe_count_routes(
    topk_ids: I.In[I.i32, ("T", "K")],
    expert_counts: I.InOut[I.i32, (EXPERTS,)],
):
    T, K = topk_ids.shape
    for token in I.parallel(I.domain(0, T)):
        for choice in I.parallel(I.domain(0, K)):
            expert = topk_ids[token, choice]
            I.assume_in_bounds(expert, expert_counts, axis=0)
            I.atomic.add(
                expert_counts,
                index=(expert,),
                value=I.cast(1, I.i32),
                order="relaxed",
            )


@intent.kernel
def moe_prefix_routes(
    expert_counts: I.In[I.i32, (EXPERTS,)],
    expert_offsets: I.Out[I.i32, (EXPERTS + 1,)],
    total_padded: I.Out[I.i32, (1,)],
):
    experts = I.domain(0, expert_counts.shape[0])
    for singleton in I.parallel(I.domain(0, total_padded.shape[0])):
        counts = expert_counts[experts]
        padded = ((counts + BLOCK_SIZE - 1) // BLOCK_SIZE) * BLOCK_SIZE
        prefix = I.cumsum(
            padded,
            axis=0,
        )
        expert_offsets[singleton] = 0
        for expert in experts:
            expert_offsets[expert + 1] = prefix[expert]
        total_padded[singleton] = prefix[EXPERTS - 1]


@intent.kernel
def moe_scatter_routes(
    topk_ids: I.In[I.i32, ("T", "K")],
    expert_offsets: I.In[I.i32, (EXPERTS + 1,)],
    expert_cursors: I.InOut[I.i32, (EXPERTS,)],
    sorted_route_ids: I.InOut[I.i32, (PADDED_ROUTES,)],
):
    T, K = topk_ids.shape
    for token in I.parallel(I.domain(0, T)):
        for choice in I.parallel(I.domain(0, K)):
            expert = topk_ids[token, choice]
            I.assume_in_bounds(expert, expert_offsets, axis=0)
            I.assume_in_bounds(expert, expert_cursors, axis=0)
            local_position = I.atomic.add(
                expert_cursors,
                index=(expert,),
                value=I.cast(1, I.i32),
                order="relaxed",
            )
            position = expert_offsets[expert] + local_position
            I.assume_in_bounds(position, sorted_route_ids, axis=0)
            sorted_route_ids[position] = I.cast(token * TOP_K + choice, I.i32)


@intent.kernel
def moe_mark_expert_blocks(
    expert_offsets: I.In[I.i32, (EXPERTS + 1,)],
    expert_blocks: I.Out[I.i32, (EXPERT_BLOCKS,)],
):
    for expert in I.parallel(I.domain(0, EXPERTS)):
        first_block = expert_offsets[expert] // BLOCK_SIZE
        last_block = expert_offsets[expert + 1] // BLOCK_SIZE
        for block in range(first_block, last_block):
            I.assume_in_bounds(block, expert_blocks, axis=0)
            expert_blocks[block] = I.cast(expert, I.i32)
