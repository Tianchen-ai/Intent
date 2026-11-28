"""Author constants shared across V2 program input builders and benchmarks."""

# MoE routing constants
TOKENS = 4096
TOP_K = 2
EXPERTS = 64
BLOCK_SIZE = 128
ROUTES = TOKENS * TOP_K
PADDED_ROUTES = ROUTES + EXPERTS * (BLOCK_SIZE - 1)
EXPERT_BLOCKS = PADDED_ROUTES // BLOCK_SIZE + 1

# RoPE constants
HEAD_DIMENSION = 128
ROPE_QUERY_HEADS = 32
ROPE_KEY_HEADS = 8

# GroupNorm backward constants
BATCH = 32
CHANNELS = 256
SPATIAL = 1024
GROUPS = 32
CHANNELS_PER_GROUP = CHANNELS // GROUPS

# Jagged mean constants
JAGGED_BATCH = 512

# CSR SpMV constants
ROWS = 32768
COLUMNS = 32768
NONZEROS_PER_ROW = 32
NONZEROS = ROWS * NONZEROS_PER_ROW

# Cholesky constants
SIZE = 16

# Histogram constants
BINS = 256
