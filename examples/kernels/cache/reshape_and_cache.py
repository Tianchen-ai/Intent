import intent
import intent.language as I


TOKENS = 4096
HEADS = 8
HEAD_DIMENSION = 128
BLOCKS = 2048
BLOCK_SIZE = 16


@intent.kernel
def reshape_and_cache(
    key: I.In[I.f16, ("T", "H", "D")],
    value: I.In[I.f16, ("T", "H", "D")],
    slot_mapping: I.In[I.i32, ("T",)],
    key_cache: I.InOut[I.f16, (BLOCKS, BLOCK_SIZE, "H", "D")],
    value_cache: I.InOut[I.f16, (BLOCKS, BLOCK_SIZE, "H", "D")],
):
    T, H, D = key.shape
    heads = I.domain(0, H)
    dimensions = I.domain(0, D)
    for token in I.parallel(I.domain(0, T)):
        slot = slot_mapping[token]
        block = slot // BLOCK_SIZE
        offset = slot % BLOCK_SIZE
        I.assume_in_bounds(block, key_cache, axis=0)
        I.assume_in_bounds(block, value_cache, axis=0)
        key_cache[block, offset, heads, dimensions] = key[token, heads, dimensions]
        value_cache[block, offset, heads, dimensions] = value[token, heads, dimensions]
