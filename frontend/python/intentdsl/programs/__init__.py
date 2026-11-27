"""Pure-Python/NumPy author program descriptors and deterministic input builders."""

from importlib import import_module

from . import inputs
from .constants import (
    BLOCK_SIZE,
    EXPERT_BLOCKS,
    EXPERTS,
    PADDED_ROUTES,
    ROUTES,
    TOKENS,
    TOP_K,
)
from .inputs import Array, numpy_dtype

PROGRAMS = {
    "relu": ("elementwise", "Pointwise ReLU"),
    "swiglu": ("elementwise", "Two-input SwiGLU"),
    "rope": ("elementwise", "In-place rotary embedding"),
    "transpose": ("elementwise", "Matrix transpose"),
    "embedding": ("irregular", "BF16 embedding lookup"),
    "csr_spmv": ("irregular", "CSR sparse matrix-vector product"),
    "jagged_mean": ("irregular", "Segmented mean"),
    "softmax": ("collectives", "FP16 softmax"),
    "layer_norm": ("collectives", "Weighted LayerNorm"),
    "batch_norm": ("collectives", "Welford statistics and mutable state"),
    "group_norm_backward": ("collectives", "Two-kernel GroupNorm backward"),
    "cumsum": ("collectives", "Ordered row prefix"),
    "causal_linear_attention": ("collectives", "Region scan with matrix state"),
    "nonzero": ("irregular", "Prefix compaction"),
    "moe_alignment": ("irregular", "Count/prefix/scatter/mark"),
    "gemm": ("matrix", "FP16 GEMM"),
    "batched_gemm": ("matrix", "Batched BF16 GEMM"),
    "ragged_gemm": ("matrix", "Unequal groups and matrix supply"),
    "int8_gemm": ("matrix", "INT8 GEMM with i32 accumulation"),
    "q4_projection": ("matrix", "Q4_K weights and Q8_K activations"),
    "fp8_gemm": ("matrix", "Block-scaled E4M3 and E8M0"),
    "attention": ("streaming", "Causal BF16 grouped-query attention"),
    "paged_decode": ("streaming", "Paged partials and FP32-to-FP16 merge"),
    "mamba_chunk_state": ("streaming", "BF16 matrix supply and FP32 state"),
    "gated_delta": ("streaming", "Ordered gated recurrence"),
    "causal_convolution": ("structured", "Causal depthwise convolution"),
    "cholesky": ("structured", "In-place batched Cholesky"),
    "dropout": ("elementwise", "Integer-mixing dropout"),
    "adamw": ("elementwise", "Mutable optimizer state"),
    "histogram": ("collectives", "Histogram with sample masking"),
}


def load_program(name):
    """Load an author program callable by registered name."""
    module, _ = PROGRAMS[name]
    return getattr(import_module(f"intentdsl.programs.{module}"), name)


__all__ = [
    "Array",
    "BLOCK_SIZE",
    "EXPERTS",
    "EXPERT_BLOCKS",
    "PADDED_ROUTES",
    "PROGRAMS",
    "ROUTES",
    "TOKENS",
    "TOP_K",
    "inputs",
    "load_program",
    "numpy_dtype",
]
